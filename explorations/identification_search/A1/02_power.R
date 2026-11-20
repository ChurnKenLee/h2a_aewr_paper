# Identification search, strategy A1 (Phase 3 design, step 4): design-stage
# power with the shared harness, as declared in
# explorations/identification_search/A1/registry.json (design_power).
#
# Usage (repository root, after 01_support.R):
#   Rscript --vanilla explorations/identification_search/A1/02_power.R "$S"
# Writes $S/A1/a1_power_{support,calibration,mde,null,summary}.csv.
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
S <- normalizePath(args[1]); out <- file.path(S, "A1")
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
reg_path <- "explorations/identification_search/A1/registry.json"
want <- strsplit(readLines("explorations/identification_search/A1/registry.sha256"), " +")[[1]][1]
got <- digest::digest(file = reg_path, algo = "sha256")
if (!identical(want, got)) stop("Registry hash mismatch: ", got)
settings <- ish_power_settings()
CAL <- 2020:2025
CBAR <- 1.2401 * 15.2607  # registry estimand.C_bar (18.924)
PI_REPORT <- c(1, 0.8, 0.65)

doses <- read_parquet(file.path(out, "a1_county_doses.parquet"))
oc <- read_parquet(file.path(S, "shared", "outcomes_county_fy.parquet"),
  col_select = c("county_fips", "year", "cert_hours_rq13d", "req_pos_rq13d")) %>% filter(year %in% CAL)
adj <- read_parquet(path_int("county_adjacency2010.parquet")) %>%
  transmute(a = substr(harmonize_county_fips_2010(county_fips), 1, 2), b = substr(harmonize_county_fips_2010(neighbor_county_fips), 1, 2)) %>%
  filter(a != b) %>% distinct()
loading <- ish_spatial_loading(unique(doses$state_fips), adj, settings$spatial_graph_coefficient)

pre_c <- tidyr::crossing(doses %>% select(county_fips, state_fips, aewr_region_id, state_bhalf, multi_region), year = CAL) %>%
  left_join(oc, by = c("county_fips", "year")) %>%
  mutate(cert_hours_rq13d = coalesce(cert_hours_rq13d, 0), req_pos_rq13d = coalesce(req_pos_rq13d, 0),
    region_year = paste(aewr_region_id, year), state_year = paste(state_fips, year))
pol_c <- function(x25, x26) tidyr::crossing(doses, year = 2025:2026) %>%
  mutate(x = ifelse(year == 2025, .data[[x25]], .data[[x26]]), region_year = paste(aewr_region_id, year), state_year = paste(state_fips, year))
doses <- doses %>% mutate(lvl_d1 = e25 + d1_h)

res <- list()
run <- function(tag, policy, pre, ycount, unit, fe, clusters, jk, node, group = NULL, dgps = settings$dgp_ids, ri = NULL, ld = NULL, note = "") {
  message("== ", tag, " ", format(Sys.time()))
  lin <- ish_ppml_linearize(pre, policy, ycount, unit)
  pol <- lin$policy; pr <- lin$pre
  ns <- ish_node_period_share(pr, lin$y, unit, "year", fe, node, weights = lin$weights)
  t0 <- Sys.time()
  r <- ish_power(pol, pr, y = lin$y, x = "x", unit = unit, period = "year", fe = fe, clusters = clusters, jk_cluster = jk,
    node = node, group = group, weights = lin$weights, beta_e1 = 1 / CBAR, loading = ld, dgps = dgps,
    settings = settings, ri = ri, label = list(run = tag, outcome = ycount))
  gate_dgp <- if (ns$observed_node_period_share <= 1.5 * ns$pure_unit_noise_share) "county_serial" else "max_shared"
  sv <- r$summary
  gm <- if (gate_dgp == "county_serial") sv$size_valid_mde_elasticity[sv$dgp_id == "county_serial"] else
    max(sv$size_valid_mde_elasticity[sv$dgp_id != "county_serial"])  # NA if any shared DGP has no size-valid method
  gmeth <- if (gate_dgp == "county_serial") sv$preferred_method[sv$dgp_id == "county_serial"] else
    sv$preferred_method[sv$dgp_id != "county_serial"][which.max(sv$size_valid_mde_elasticity[sv$dgp_id != "county_serial"])]
  extra <- tibble(node = node, observed_node_period_share = ns$observed_node_period_share, pure_unit_noise_share = ns$pure_unit_noise_share,
    gate_dgp = gate_dgp, gate_method = if (length(gmeth)) gmeth else NA_character_, gate_mde_eps_cost_pi1 = gm,
    gate_mde_pi0.8 = gm / 0.8, gate_mde_pi0.65 = gm / 0.65, cbar = CBAR, seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")), note = note)
  r$support <- bind_cols(r$support, extra)
  print(r$support, width = 300); print(r$summary, width = 200); print(r$null %>% select(dgp_id, inference_method, null_rejection_rate, null_mc_high, size_gate_pass), n = 40)
  res[[tag]] <<- r
  invisible(r)
}

# P1 primary
pol1 <- pol_c("e25", "lvl_d1")
run("P1_primary_d1_within_state", pol1, pre_c, "cert_hours_rq13d", "county_fips", c("county_fips", "state_year"),
  c("state_fips", "aewr_region_id"), "state_fips", "state_fips", "aewr_region_id", ld = loading,
  note = "Registry primary: county FE + state x FY FE, dose e25 + b x (F - e25); pi = 1")
# P2 stress: shared shock at state x FY2025 binding half (not absorbed by state x FY FE)
pre2 <- pre_c %>% mutate(state_bhalf = state_bhalf); pol2 <- pol1
run("P2_stress_state_bhalf_node", pol2, pre2, "cert_hours_rq13d", "county_fips", c("county_fips", "state_year"),
  c("state_fips", "aewr_region_id"), "state_fips", "state_bhalf", NULL,
  dgps = c("county_serial", "shared_region_serial", "shared_region_and_pair_serial"),
  note = "Stress: node = state x binding half; all shared variance on that node")
# P3 lead secondary
run("P3_req_pos_d1_within_state", pol1, pre_c, "req_pos_rq13d", "county_fips", c("county_fips", "state_year"),
  c("state_fips", "aewr_region_id"), "state_fips", "state_fips", "aewr_region_id", ld = loading,
  note = "Lead secondary requested positions rq13d; same design")
# P4 cross-state D1 (46 states)
run("P4_SD1_d1_cross_state", pol1 %>% filter(multi_region), pre_c %>% filter(multi_region), "cert_hours_rq13d", "county_fips",
  c("county_fips", "region_year"), c("state_fips", "aewr_region_id"), "state_fips", "state_fips", "aewr_region_id", ld = loading,
  note = "SD1: county + former region x FY")
# P5 cross-state effective floor (46 states) with RI
run("P5_SD2_effective_floor_cross_state", pol_c("e25", "FH") %>% filter(multi_region), pre_c %>% filter(multi_region), "cert_hours_rq13d", "county_fips",
  c("county_fips", "region_year"), c("state_fips", "aewr_region_id"), "state_fips", "state_fips", "aewr_region_id", ld = loading,
  ri = list(assign = "state_fips", strata = "aewr_region_id", P = 199),
  note = "SD2: dose F (H-2A cash effective floor) vs e25; NY/OR sub-state MW areas make F vary within state, so RI may be skipped")
# P6 employer x state within-state
eb <- read_parquet(file.path(out, "a1_employer_doses.parquet"))
cre <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("fy_receipt", "in_rq13d", "state_fips", "employer_id_balanced", "cert_hours")) %>%
  filter(in_rq13d, fy_receipt %in% CAL, !is.na(employer_id_balanced)) %>%
  group_by(unit = paste(employer_id_balanced, state_fips), year = fy_receipt) %>% summarise(cert_hours_rq13d = sum(cert_hours, na.rm = TRUE), .groups = "drop")
pre_e <- tidyr::crossing(eb %>% select(unit, state_fips, aewr_region_id), year = CAL) %>% left_join(cre, by = c("unit", "year")) %>%
  mutate(cert_hours_rq13d = coalesce(cert_hours_rq13d, 0), state_year = paste(state_fips, year))
pol_e <- tidyr::crossing(eb, year = 2025:2026) %>% mutate(x = ifelse(year == 2025, e25, lvl_d1), state_year = paste(state_fips, year))
run("P6_SD5_employer_within_state", pol_e, pre_e, "cert_hours_rq13d", "unit", c("unit", "state_year"),
  c("state_fips", "aewr_region_id"), "state_fips", "state_fips", "aewr_region_id", ld = loading,
  note = "SD5: employer x state units with FY2025 hours; unit FE + state x FY")

for (k in c("support", "calibration", "mde", "null", "summary"))
  write_csv(bind_rows(lapply(res, `[[`, k)), file.path(out, paste0("a1_power_", k, ".csv")))
message("done ", format(Sys.time()))
