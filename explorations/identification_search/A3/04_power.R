# Identification search, strategy A3 (Phase 3 design, step 4): design-stage
# power with the shared harness, as declared in A3/registry.json (design_power).
#
# Usage (repository root, after 03_support.R and the registry freeze):
#   Rscript --vanilla explorations/identification_search/A3/04_power.R "$S"
# Writes $S/A3/a3_power_{support,calibration,mde,null,summary}.csv.
#
# Integrity: checks the registry hash first. Only FY2020-FY2022 outcomes enter
# (FE-only residual variance in the calibration frames); no outcome is
# regressed on any dose.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr)})
source(path_code("designs", "border_discontinuity", "power.R"))
source("explorations/identification_search/shared/power_harness.R")
setFixest_nthreads(1)
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "A3")
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
reg_path <- "explorations/identification_search/A3/registry.json"
want <- strsplit(readLines("explorations/identification_search/A3/registry.sha256"), " +")[[1]][1]
if (!identical(want, digest::digest(file = reg_path, algo = "sha256"))) stop("Registry hash mismatch")
reg <- jsonlite::read_json(reg_path)
RATIO <- reg$estimand$ratio_A3$central; PI_DESIGN <- reg$design_power$pi_design; PI_REPORT <- c(1, 0.9, 0.8, 0.6)
settings <- ish_power_settings()
fr <- readRDS(file.path(out, "a3_design_frames.rds"))
stopifnot(all(fr$primary$pre$year %in% 2020:2022))

# SD1 ratio: hours-weighted FY2020-FY2026 central (FY2026 row of cost_unit_by_fy is the H-2A cash basis)
cu <- read_csv(file.path(S, "shared", "cost_unit_by_fy.csv"), show_col_types = FALSE) %>% filter(year %in% 2020:2026)
RATIO_SD1 <- with(cu, sum(cert_hours * cost_ratio_central) / sum(cert_hours))

# Spatial loading over nodes (state x type): same type in adjacent states
adj <- read_parquet(path_int("county_adjacency2010.parquet")) %>%
  transmute(a = substr(harmonize_county_fips_2010(county_fips), 1, 2), b = substr(harmonize_county_fips_2010(neighbor_county_fips), 1, 2)) %>%
  filter(a != b) %>% distinct()
loading_for <- function(nodes) {
  nd <- tibble(node = sort(unique(nodes))) %>% mutate(st = substr(node, 1, 2), ty = substr(node, 4, 100))
  e <- nd %>% inner_join(adj, by = c(st = "a"), relationship = "many-to-many") %>%
    inner_join(nd %>% rename(node_b = node), by = c(b = "st", ty = "ty")) %>% transmute(a = node, b = node_b) %>% distinct()
  ish_spatial_loading(nd$node, as.data.frame(e), settings$spatial_graph_coefficient)
}

res <- list()
run <- function(tag, f, ratio, note = "") {
  message("== ", tag, " ", format(Sys.time()))
  pol <- f$pol; pre <- f$pre
  if (is.null(pol$group)) { pol$group <- sub("^.. ", "", pol$node); pre$group <- sub("^.. ", "", pre$node) }
  ld <- loading_for(pol$node)
  fe <- c("unit", "state_year")
  ns <- ish_node_period_share(pre, "y", "unit", "year", fe, "node", weights = "w")
  t0 <- Sys.time()
  r <- ish_power(pol, pre, y = "y", x = "x", unit = "unit", period = "year", fe = fe, clusters = c("state_fips", "node"),
    jk_cluster = "state_fips", node = "node", group = "group", weights = "w", beta_e1 = 1 / ratio, loading = ld,
    settings = settings, label = list(run = tag))
  gate_dgp <- if (ns$observed_node_period_share <= 1.5 * ns$pure_unit_noise_share) "county_serial" else "max_shared"
  sv <- r$summary
  shared <- sv[sv$dgp_id != "county_serial", ]
  gm <- if (gate_dgp == "county_serial") sv$size_valid_mde_elasticity[sv$dgp_id == "county_serial"] else
    if (any(is.na(shared$size_valid_mde_elasticity))) NA_real_ else max(shared$size_valid_mde_elasticity)
  gmeth <- if (gate_dgp == "county_serial") sv$preferred_method[sv$dgp_id == "county_serial"] else
    if (is.na(gm)) NA_character_ else shared$preferred_method[which.max(shared$size_valid_mde_elasticity)]
  extra <- tibble(node_def = "state x unit type", observed_node_period_share = ns$observed_node_period_share,
    pure_unit_noise_share = ns$pure_unit_noise_share, gate_dgp = gate_dgp, gate_method = gmeth, gate_mde_eps_cost_pi1 = gm,
    !!!setNames(lapply(PI_REPORT, function(p) gm / p), paste0("gate_mde_pi", PI_REPORT)), ratio = ratio,
    seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")), note = note)
  r$support <- bind_cols(r$support, extra)
  print(as.data.frame(r$support)); print(as.data.frame(r$summary))
  print(as.data.frame(r$null %>% select(dgp_id, inference_method, null_rejection_rate, null_mc_high, size_gate_pass)))
  res[[tag]] <<- r
  invisible(r)
}

run("P1_primary", fr$primary, RATIO, "registry primary: unit FE + state x FY FE, occupation-group share x FY controls partialled (FWL), pi = 1 in beta_e1")
run("P2_no_controls", fr$no_controls, RATIO, "S1: no occupation controls")
run("P3_nonfield_share_control", fr$nonfield_share_controls, RATIO, "S2: non-field share x FY control only")
run("P4_SD1_rq13d_fy2026", fr$rq13d_fy2026, RATIO_SD1, "SD1: rq13d FY2020-FY2026, IFR skill-I floors; ratio FY2020-FY2026")
cf <- fr$cells; cf$pol$group <- cf$pol$kind; cf$pre <- cf$pre %>% left_join(distinct(cf$pol, unit, kind), by = "unit") %>% mutate(group = kind)
run("P5_SD2_cells", cf, RATIO, "SD2: relabeling-exposed employer x state x {field, nonfield} cells, cell + state x FY FE, no controls")
run("P6_req_pos", fr$req_pos, RATIO, "lead secondary: requested positions (rfy), own pre-period weights")

for (k in c("support", "calibration", "mde", "null", "summary"))
  write_csv(bind_rows(lapply(res, `[[`, k)), file.path(out, paste0("a3_power_", k, ".csv")))
message("done ", format(Sys.time()))
