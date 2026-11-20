# Identification search, Phase 0, step 3 test: does the shared harness
# reproduce explorations/identification/04_track1_power.R?
#
# Usage (repository root):
#   Rscript --vanilla explorations/identification_search/shared/02_power_harness_test.R "$S/shared" [track1_scratch_dir]
# The optional second argument is the scratch directory of the earlier 04 run
# (holds track1_pre_panel.csv and track1_power_mde.csv). Without it, only the
# shared-outcome run (test B) is made.
#
# Test A (exact): the harness on 04's own frozen pre panel and the same
#   treatment, seed and call order must give 04's MDEs exactly.
# Test B (noise): the harness on the shared outcome panel (01_outcomes.R,
#   window dq13) must give the county-only jackknife MDE within simulation
#   noise (declared tolerance: |difference| <= 0.10 elasticity units).
# Also: within-region RI and the PPML linearization on certified hours counts,
#   as usage demonstrations (no reference values exist for them).
# Reads only pre-FY2026 outcomes (FY2020-FY2025 calibration) and AEWR schedules.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr)})
source(path_code("designs", "border_discontinuity", "power.R"))
source("explorations/identification_search/shared/power_harness.R")
setFixest_nthreads(1)
args <- commandArgs(trailingOnly = TRUE)
out_dir <- normalizePath(args[1]); t1_dir <- if (length(args) > 1) args[2] else NA_character_
if (startsWith(paste0(out_dir, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
registry <- jsonlite::read_json("explorations/identification/track1_registry.json", simplifyVector = TRUE)
settings <- ish_power_settings()

# Track 1 sample (as in 04_track1_power.R) --------------------------------------
cp <- read_parquet(path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "state_fips", "aewr_region_id", "emp_farm_2011", "year")) %>% filter(year == 2011) %>% select(-year)
counties <- cp %>% filter(!state_fips %in% registry$geography$excluded_states, is.finite(emp_farm_2011), emp_farm_2011 > 0) %>%
  group_by(aewr_region_id) %>% filter(n_distinct(state_fips) > 1) %>% ungroup() %>% arrange(aewr_region_id, state_fips, county_fips)

regional <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"), worker_category == "covered_workers") %>%
  transmute(state_fips, aewr = as.numeric(aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
statewide <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  filter(trimws(source_sheet) == registry$policy_event$source_sheet, worker_category == "US", skill_level == 1) %>%
  transmute(state_fips, aewr = as.numeric(hourly_aewr), start = as.Date(effective_start), end = as.Date(effective_end_exclusive))
rate_on <- function(s, date) { r <- s %>% filter(start <= date, is.na(end) | end > date) %>% select(state_fips, aewr); stopifnot(!anyDuplicated(r$state_fips)); r }
treat <- bind_rows(lapply(2012:2026, function(fy) tibble(state_fips = unique(counties$state_fips), year = fy) %>%
  left_join(rate_on(if (fy >= 2026) statewide else regional, as.Date(sprintf("%d-04-01", fy))), by = "state_fips")))
stopifnot(all(is.finite(treat$aewr)))

adj <- read_parquet(path_int("county_adjacency2010.parquet")) %>%
  transmute(a = substr(harmonize_county_fips_2010(county_fips), 1, 2), b = substr(harmonize_county_fips_2010(neighbor_county_fips), 1, 2)) %>%
  filter(a != b) %>% distinct()
loading <- ish_spatial_loading(unique(counties$state_fips), adj, settings$spatial_graph_coefficient)

policy <- tidyr::crossing(counties, year = 2012:2026) %>% left_join(treat %>% rename(x = aewr), by = c("state_fips", "year")) %>%
  mutate(region_year = paste(aewr_region_id, year))
run <- function(pre_panel, yv, dgps, label, ri = NULL, weights = NULL, pol = policy, pre_years = 2020:2025) {
  pre <- pre_panel %>% semi_join(counties, by = "county_fips") %>% filter(year %in% pre_years) %>% mutate(region_year = paste(aewr_region_id, year))
  pol <- pol %>% semi_join(pre %>% distinct(county_fips), by = "county_fips")
  mean_y <- if (is.null(weights)) mean(pre[[yv]]) else NA_real_
  mean_p <- if (is.null(weights)) mean(pol$x) else sum(pol[[weights]] * pol$x) / sum(pol[[weights]])
  beta_e1 <- if (is.null(weights)) mean_y / mean_p else 1 / mean_p
  ish_power(pol, pre, y = yv, x = "x", unit = "county_fips", period = "year", fe = c("county_fips", "region_year"),
    clusters = c("state_fips", "aewr_region_id"), jk_cluster = "state_fips", node = "state_fips", group = "aewr_region_id",
    weights = weights, beta_e1 = beta_e1, loading = loading, dgps = dgps, settings = settings, ri = ri, label = label)
}
results <- list(); checks <- list()

# Test A: exact reproduction on 04's frozen pre panel --------------------------
if (!is.na(t1_dir) && file.exists(file.path(t1_dir, "track1_pre_panel.csv"))) {
  pre04 <- read_csv(file.path(t1_dir, "track1_pre_panel.csv"), col_types = cols(county_fips = "c", state_fips = "c", aewr_region_id = "c"))
  ref <- read_csv(file.path(t1_dir, "track1_power_mde.csv"), show_col_types = FALSE) %>%
    filter(label == "us_skill_1", scenario_id == "post_fy2026", calibration == "primary", outcome %in% c("cert", "hours"))
  for (oc in c("cert", "hours")) {
    yv <- c(cert = "y_cert_q", hours = "y_hours_q")[[oc]]
    dg <- if (oc == "cert") settings$dgp_ids else "county_serial"
    message("Test A: ", oc)
    r <- run(pre04, yv, dg, list(test = "A_exact_04_pre_panel", outcome = oc))
    results[[paste0("A_", oc)]] <- r
    cmp <- r$mde %>% transmute(outcome = oc, dgp_id, inference_method = sub("^state_fips_", "state_", sub("^aewr_region_id_", "former_region_", inference_method)),
      effect_sign, harness_mde_elasticity = mde_elasticity, harness_gate = size_gate_pass) %>%
      inner_join(ref %>% transmute(outcome, dgp_id, inference_method, effect_sign, ref_mde_elasticity = mde_elasticity, ref_gate = size_gate_pass),
        by = c("outcome", "dgp_id", "inference_method", "effect_sign")) %>%
      left_join(r$null %>% transmute(dgp_id, inference_method = sub("^state_fips_", "state_", sub("^aewr_region_id_", "former_region_", inference_method)), harness_null = null_rejection_rate), by = c("dgp_id", "inference_method")) %>%
      left_join(ref %>% filter(outcome == oc) %>% distinct(dgp_id, inference_method, ref_null = null_rejection_rate), by = c("dgp_id", "inference_method"))
    checks[[paste0("A_", oc)]] <- cmp
  }
}

# Test B: shared outcomes ------------------------------------------------------
oc <- read_parquet(file.path(out_dir, "outcomes_county_fy.parquet"), col_select = c("county_fips", "year", "cert_pos_dq13", "cert_hours_dq13")) %>%
  filter(year >= 2012, year <= 2025)
preB <- tidyr::crossing(counties, year = 2012:2025) %>% left_join(oc, by = c("county_fips", "year")) %>%
  mutate(across(c(cert_pos_dq13, cert_hours_dq13), ~ coalesce(.x, 0)),
    y_cert_q = cert_pos_dq13 / emp_farm_2011, y_hours_q = cert_hours_dq13 / emp_farm_2011)
if (exists("pre04")) {
  checks$panel_vs_04 <- preB %>% inner_join(pre04 %>% select(county_fips, year, c04 = y_cert_q, h04 = y_hours_q), by = c("county_fips", "year")) %>%
    summarise(rows = n(), max_abs_diff_cert = max(abs(y_cert_q - c04)), max_rel_diff_hours = max(abs(y_hours_q - h04) / pmax(1e-9, abs(h04))),
      total_rel_diff_hours = sum(y_hours_q * emp_farm_2011) / sum(h04 * emp_farm_2011) - 1)
}
message("Test B: cert, county only, with RI")
results$B_cert <- run(preB, "y_cert_q", "county_serial", list(test = "B_shared_outcomes", outcome = "cert"),
  ri = list(assign = "state_fips", strata = "aewr_region_id", P = 199))
message("Test B: hours, county only")
results$B_hours <- run(preB, "y_hours_q", "county_serial", list(test = "B_shared_outcomes", outcome = "hours"))

# PPML linearization demo: certified hours counts, county only ----------------
lin <- ish_ppml_linearize(preB %>% filter(year %in% 2020:2025), policy, "cert_hours_dq13", "county_fips")
message("Demo: PPML linearization, hours counts")
results$B_ppml_hours <- run(lin$pre, lin$y, "county_serial", list(test = "B_ppml_linearized", outcome = "hours_count"),
  weights = lin$weights, pol = lin$policy)

bind_or <- function(k) bind_rows(lapply(results, `[[`, k))
write_csv(bind_or("support"), file.path(out_dir, "harness_test_support.csv"))
write_csv(bind_or("mde"), file.path(out_dir, "harness_test_mde.csv"))
write_csv(bind_or("null"), file.path(out_dir, "harness_test_null.csv"))
write_csv(bind_or("summary"), file.path(out_dir, "harness_test_summary.csv"))
write_csv(bind_or("calibration"), file.path(out_dir, "harness_test_calibration.csv"))
if (length(checks)) for (k in names(checks)) write_csv(checks[[k]], file.path(out_dir, paste0("harness_test_check_", k, ".csv")))

# Verdicts ----------------------------------------------------------------------
verdict <- list()
if (!is.null(checks$A_cert)) {
  a <- bind_rows(checks$A_cert, checks$A_hours)
  verdict$A_exact <- all(abs(a$harness_mde_elasticity - a$ref_mde_elasticity) < 1e-9 | (is.na(a$harness_mde_elasticity) & is.na(a$ref_mde_elasticity))) &&
    all(abs(a$harness_null - a$ref_null) < 1e-12)
}
jk <- results$B_cert$mde %>% filter(inference_method == "state_fips_CV3J") %>% pull(mde_elasticity)
jkh <- results$B_hours$mde %>% filter(inference_method == "state_fips_CV3J") %>% pull(mde_elasticity)
verdict$B_cert_jackknife_upper <- max(jk); verdict$B_hours_jackknife_upper <- max(jkh)
verdict$B_within_noise <- abs(max(jk) - 1.51) <= 0.10 && abs(max(jkh) - 1.30) <= 0.10
jsonlite::write_json(verdict, file.path(out_dir, "harness_test_verdict.json"), auto_unbox = TRUE, pretty = TRUE)
print(checks); print(bind_or("support"), width = 250); print(bind_or("summary"), width = 250); print(bind_or("null"), n = 50); print(verdict)
