# Identification search, strategy N1 (Phase 3 design, step 4): design-stage
# power with the shared harness, as declared in
# explorations/identification_search/N1/registry.json (design_power).
#
# Usage (repository root, after 02_dose_support.R and the registry freeze):
#   Rscript --vanilla explorations/identification_search/N1/03_power.R "$S" [runs]
# runs: comma-separated subset of R1,R2,R3,R4,R5 (default all).
# Writes $S/N1/n1_power_{support,calibration,mde,null,summary}_<run>.csv and
# $S/N1/n1_power_gate_<run>.csv (gate MDE in epsilon_cost at the declared and
# reported pass-through values).
#
# Integrity: checks the registry hash first. Only FY2020-FY2025 calibration
# outcomes enter (FE-only residual variance); no FY2026 outcome is read and no
# outcome is regressed on any dose.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr)})
source(path_code("designs", "border_discontinuity", "power.R"))
source("explorations/identification_search/shared/power_harness.R")
setFixest_nthreads(1)
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "N1")
RUNS <- if (length(args) > 1) strsplit(args[2], ",")[[1]] else c("R1", "R2", "R3", "R4", "R5")
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
reg_path <- "explorations/identification_search/N1/registry.json"
want <- strsplit(readLines("explorations/identification_search/N1/registry.sha256"), " +")[[1]][1]
got <- digest::digest(file = reg_path, algo = "sha256")
if (!identical(want, got)) stop("Registry hash mismatch: ", got)
reg <- jsonlite::read_json(reg_path, simplifyVector = TRUE)
settings <- ish_power_settings()
CAL <- 2020:2025
CBAR <- reg$estimand$C_bar_value          # ratio x w_bar, declared at freeze
PI_GATE <- reg$design_power$pi_gate        # declared gate pass-through
PI_GRID <- reg$design_power$pi_report_grid # reported pass-through values

ue <- read_parquet(file.path(out, "n1_units_employer.parquet"))
uc <- read_parquet(file.path(out, "n1_units_county.parquet"))
cal_e <- read_parquet(file.path(out, "n1_calibration_employer.parquet"))
cal_c <- read_parquet(file.path(out, "n1_calibration_county.parquet"))
stopifnot(max(cal_e$year) <= 2025, max(cal_c$year) <= 2025)
adj <- read_parquet(path_int("county_adjacency2010.parquet")) %>%
  transmute(a = substr(harmonize_county_fips_2010(county_fips), 1, 2), b = substr(harmonize_county_fips_2010(neighbor_county_fips), 1, 2)) %>%
  filter(a != b) %>% distinct()

# Design frames exactly as in 02_dose_support.R (registry dose.definition).
frames <- function(u, unitcol, base, ycol, cal, cal_years, dose = "T", resid = TRUE) {
  u <- u %>% filter(base_year == base) %>% mutate(tier_half = paste(state_fips, ifelse(s >= 0.5, "t2", "t1")))
  pre <- tidyr::crossing(u %>% select(all_of(c(unitcol, "state_fips", "aewr_region_id", "tier_half"))), year = cal_years) %>%
    left_join(cal %>% select(all_of(c(unitcol, "year", ycol))), by = c(unitcol, "year")) %>%
    mutate(across(all_of(ycol), ~ coalesce(.x, 0)), state_year = paste(state_fips, year))
  lin <- ish_ppml_linearize(as.data.frame(pre), as.data.frame(u), ycol, unitcol)
  uu <- lin$policy
  if (resid) {
    f <- fixest::feols(as.formula(paste(dose, "~ L | state_fips")), data = uu, weights = ~ppml_weight)
    uu$x_id <- uu[[dose]] - coef(f)[["L"]] * uu$L
  } else uu$x_id <- uu[[dose]]
  pol <- bind_rows(uu %>% mutate(year = base), uu %>% mutate(year = base + 1L)) %>%
    mutate(x = ifelse(year == base, 0, x_id), state_year = paste(state_fips, year))
  list(pre = lin$pre, policy = pol, y = lin$y, w = lin$weights)
}

run <- function(tag, fr, unitcol, node = "state_fips", group = "aewr_region_id", dgps = settings$dgp_ids, note = "") {
  message("== ", tag, " ", format(Sys.time()))
  fe <- c(unitcol, "state_year")
  ld <- if ("spatial_region_serial" %in% dgps) ish_spatial_loading(unique(fr$policy[[node]]), adj, settings$spatial_graph_coefficient) else NULL
  ns <- ish_node_period_share(as.data.frame(fr$pre), fr$y, unitcol, "year", fe, node, weights = fr$w)
  t0 <- Sys.time()
  r <- ish_power(as.data.frame(fr$policy), as.data.frame(fr$pre), y = fr$y, x = "x", unit = unitcol, period = "year", fe = fe,
    clusters = c("state_fips", "aewr_region_id"), jk_cluster = "state_fips", node = node, group = group, weights = fr$w,
    beta_e1 = 1 / CBAR, loading = ld, dgps = dgps, settings = settings, label = list(run = tag))
  gate_dgp <- if (ns$observed_node_period_share <= 1.5 * ns$pure_unit_noise_share) "county_serial" else "max_shared"
  sv <- r$summary
  shared <- sv %>% filter(dgp_id != "county_serial")
  gm <- if (gate_dgp == "county_serial") sv$size_valid_mde_elasticity[sv$dgp_id == "county_serial"] else
    if (any(is.na(shared$size_valid_mde_elasticity))) NA_real_ else max(shared$size_valid_mde_elasticity)
  gmeth <- if (gate_dgp == "county_serial") sv$preferred_method[sv$dgp_id == "county_serial"] else
    paste(unique(shared$preferred_method), collapse = "/")
  gate <- tibble(run = tag, node = node, observed_node_period_share = ns$observed_node_period_share,
    pure_unit_noise_share = ns$pure_unit_noise_share, gate_dgp = gate_dgp, gate_method = gmeth,
    itt_mde_eps_cost_pi1 = gm, pi = PI_GRID, mde_eps_cost = gm / PI_GRID, pi_gate = PI_GATE,
    mde_eps_cost_at_pi_gate = gm / PI_GATE, breakeven_pi_mde_1 = gm / 1, breakeven_pi_mde_0.5 = gm / 0.5,
    cbar = CBAR, seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")), note = note)
  print(r$support, width = 300); print(r$summary, width = 200)
  print(r$null %>% select(dgp_id, inference_method, null_rejection_rate, null_mc_high, size_gate_pass), n = 40)
  print(gate %>% select(run, gate_dgp, gate_method, itt_mde_eps_cost_pi1, pi, mde_eps_cost), n = 40)
  for (k in c("support", "calibration", "mde", "null", "summary"))
    write_csv(r[[k]], file.path(out, paste0("n1_power_", k, "_", tag, ".csv")))
  write_csv(gate, file.path(out, paste0("n1_power_gate_", tag, ".csv")))
  invisible(r)
}

if ("R1" %in% RUNS) run("R1", frames(ue, "unit", 2025, "cert_hours_rq13d", cal_e, CAL), "unit",
  note = "Primary: employer x state, cert_hours_rq13d PPML linearization, unit + state x FY FE, tier dose net of level control")
if ("R2" %in% RUNS) run("R2", frames(ue, "unit", 2025, "cert_hours_rq13d", cal_e, CAL), "unit", node = "tier_half", group = "state_fips",
  dgps = c("county_serial", "shared_region_serial", "shared_region_and_pair_serial"),
  note = "Stress: shared shocks at state x FY2025 tier half (s >= 0.5), a within-state shock correlated with the dose")
if ("R3" %in% RUNS) run("R3", frames(ue, "unit", 2025, "req_pos_rq13d", cal_e, CAL), "unit",
  note = "Lead secondary: requested positions rq13d, same design")
if ("R4" %in% RUNS) run("R4", frames(uc, "county_fips", 2025, "cert_hours_rq13d", cal_c, CAL), "county_fips",
  note = "Secondary (a): county + state x FY, county FY2025 tier dose (captures entry)")
if ("R5" %in% RUNS) run("R5", frames(ue, "unit", 2024, "cert_hours_rq13d", cal_e, 2020:2024), "unit",
  dgps = c("county_serial", "shared_region_serial", "shared_region_and_pair_serial"),
  note = "Falsification F1 power: FY2024 -> FY2025 placebo with FY2024 tier dose; calibration FY2020-FY2024")
message("done ", format(Sys.time()))
