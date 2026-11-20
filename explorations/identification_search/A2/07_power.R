# Identification search, strategy A2 (Phase 3 design, step 4): design-stage
# power with the shared harness, as declared in A2/registry.json (design_power).
#
# Usage (repository root, after 06_support.R and the registry freeze):
#   Rscript --vanilla explorations/identification_search/A2/07_power.R "$S"
# Writes $S/A2/a2_power_{support,calibration,mde,null,summary}.csv.
#
# Integrity: checks the registry hash first. Only FY2006-FY2008 outcomes enter
# (FE-only residual variance in the calibration frame); no outcome is
# regressed on any dose.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr)})
source(path_code("designs", "border_discontinuity", "power.R"))
source("explorations/identification_search/shared/power_harness.R")
setFixest_nthreads(1)
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "A2")
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
reg_path <- "explorations/identification_search/A2/registry.json"
want <- strsplit(readLines("explorations/identification_search/A2/registry.sha256"), " +")[[1]][1]
if (!identical(want, digest::digest(file = reg_path, algo = "sha256"))) stop("Registry hash mismatch")
reg <- jsonlite::read_json(reg_path)
CBAR <- reg$estimand$C_bar; PI_DESIGN <- reg$design_power$pi_design; PI_REPORT <- c(1, 0.6, 0.4, 0.3)
settings <- ish_power_settings()
fr <- readRDS(file.path(out, "a2_design_frames.rds"))
stopifnot(all(fr$primary$pre$year %in% 2006:2008))

# Spatial loading over OES-area nodes from county adjacency
adj <- read_parquet(path_int("county_adjacency2010.parquet")) %>%
  transmute(a = harmonize_county_fips_2010(county_fips), b = harmonize_county_fips_2010(neighbor_county_fips))
amap <- distinct(fr$blank_filled$pol, county_fips, area_id)
aedges <- adj %>% inner_join(amap, by = c(a = "county_fips")) %>% rename(aa = area_id) %>%
  inner_join(amap, by = c(b = "county_fips")) %>% transmute(a = aa, b = area_id) %>% filter(a != b) %>% distinct()
loading_for <- function(areas) ish_spatial_loading(areas, aedges, settings$spatial_graph_coefficient)

res <- list()
run <- function(tag, f, unit, fe, clusters, jk, node, ri = NULL, note = "") {
  message("== ", tag, " ", format(Sys.time()))
  ld <- loading_for(sort(unique(f$pol[[node]])))
  ns <- ish_node_period_share(f$pre, f$y, unit, "year", fe, node, weights = f$w)
  t0 <- Sys.time()
  r <- ish_power(f$pol, f$pre, y = f$y, x = "x", unit = unit, period = "year", fe = fe, clusters = clusters, jk_cluster = jk,
    node = node, group = NULL, weights = f$w, beta_e1 = 1 / CBAR, loading = ld, settings = settings, ri = ri,
    label = list(run = tag))
  gate_dgp <- if (ns$observed_node_period_share <= 1.5 * ns$pure_unit_noise_share) "county_serial" else "max_shared"
  sv <- r$summary
  gm <- if (gate_dgp == "county_serial") sv$size_valid_mde_elasticity[sv$dgp_id == "county_serial"] else
    max(sv$size_valid_mde_elasticity[sv$dgp_id != "county_serial"])
  gmeth <- if (gate_dgp == "county_serial") sv$preferred_method[sv$dgp_id == "county_serial"] else
    sv$preferred_method[sv$dgp_id != "county_serial"][which.max(sv$size_valid_mde_elasticity[sv$dgp_id != "county_serial"])]
  extra <- tibble(node = node, observed_node_period_share = ns$observed_node_period_share, pure_unit_noise_share = ns$pure_unit_noise_share,
    gate_dgp = gate_dgp, gate_method = if (length(gmeth)) gmeth else NA_character_, gate_mde_eps_cost_pi1 = gm,
    !!!setNames(lapply(PI_REPORT, function(p) gm / p), paste0("gate_mde_pi", PI_REPORT)), cbar = CBAR,
    seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")), note = note)
  r$support <- bind_cols(r$support, extra)
  print(r$support, width = 300); print(r$summary, width = 200)
  print(r$null %>% select(dgp_id, inference_method, null_rejection_rate, null_mc_high, size_gate_pass), n = 40)
  res[[tag]] <<- r
  invisible(r)
}

FE <- c("county_fips", "state_year")
run("P1_primary", fr$primary, "county_fips", FE, c("state_fips", "area_id"), "state_fips", "area_id",
  note = "registry primary: county FE + state x FY FE, recession controls partialled (FWL), pi = 1 in beta_e1")
run("P2_no_controls_RI", fr$no_controls, "county_fips", FE, c("state_fips", "area_id"), "state_fips", "area_id",
  ri = list(assign = "assign_id", strata = "state_fips", P = 199), note = "S2: no controls; RI permutes area dose paths within state")
run("P3_area_unit", list(pre = fr$area$pre %>% mutate(area2 = area_id), pol = fr$area$policy %>% mutate(area2 = area_id),
  y = fr$area$y, w = fr$area$weights), "area_id", c("area_id", "state_year"), c("state_fips", "area2"), "state_fips", "area_id",
  note = "SD3: unit = OES area x state; node = area (= unit), so shared DGPs coincide with unit noise")
run("P4_blank_filled", fr$blank_filled, "county_fips", FE, c("state_fips", "area_id"), "state_fips", "area_id",
  note = "S3: blank 45-2092 L1 cells filled")

for (k in c("support", "calibration", "mde", "null", "summary"))
  write_csv(bind_rows(lapply(res, `[[`, k)), file.path(out, paste0("a2_power_", k, ".csv")))
message("done ", format(Sys.time()))
