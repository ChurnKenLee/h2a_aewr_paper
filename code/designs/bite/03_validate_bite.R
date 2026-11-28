# Purpose: Independent validation of the standalone full-panel PPML bite design.
# Verifies panel keys, AEWR-001, algebraic exposure identities, all-real
# shortfall consistency, SHA-256 provenance, finite CV3J SEs, and domestic-floor
# 48-state coverage through 2025.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "bite", "helpers.R"))

suppressPackageStartupMessages({
  library(arrow)
  library(digest)
  library(dplyr)
  library(fixest)
  library(jsonlite)
  library(readr)
})

checks_passed <- 0L
assert_check <- function(name, cond) {
  if (!isTRUE(cond)) {
    stop("Bite validation FAILED: ", name, call. = FALSE)
  }
  checks_passed <<- checks_passed + 1L
}

shared_path <- path_processed("county_year_panel.parquet")
bite_path <- path_processed("bite_county_year_panel.parquet")
prov_path <- path_outputs("reports", "bite_provenance.json")

shared <- read_parquet(shared_path) %>% mutate(year = as.integer(year))
bite <- read_parquet(bite_path) %>% mutate(year = as.integer(year))
prov <- read_json(prov_path, simplifyVector = TRUE)

# 1. Geographic string keys, balanced shape, and AEWR-001
assert_geo_columns(bite, c("county_fips", "state_fips", "cz_id", "aewr_region_id"))
assert_check("exact_55944_rows", nrow(bite) == 55944L)
assert_check("unique_county_year_keys", anyDuplicated(bite[c("county_fips", "year")]) == 0L)
assert_check("exact_3108_counties_18_years", n_distinct(bite$county_fips) == 3108L && identical(sort(unique(bite$year)), 2008L:2025L))
assert_check("aewr_2009_strictly_null", all(is.na(bite$aewr[bite$year == 2009L])) && all(is.na(bite$aewr_ppi[bite$year == 2009L])) && all(is.na(bite$ln_aewr[bite$year == 2009L])))

# 2. Algebraic identity: max |E_p25 - (-std_wr(ln w_0))| < 1e-12 (both real and nominal)
base_check <- shared %>%
  filter(year %in% BITE_BASELINE_YEARS) %>%
  mutate(
    ln_w25_real = log(wage_p25),
    ln_w25_nom = log(wage_p25 * ppi_2012)
  ) %>%
  group_by(county_fips, aewr_region_id) %>%
  summarise(
    mean_ln_w25_real = mean(ln_w25_real),
    mean_ln_w25_nom = mean(ln_w25_nom),
    .groups = "drop"
  ) %>%
  mutate(
    neg_std_real = -bite_std_within_region(mean_ln_w25_real, aewr_region_id),
    neg_std_nom = -bite_std_within_region(mean_ln_w25_nom, aewr_region_id)
  )

stored_exposures <- bite %>%
  filter(year == 2011L) %>%
  select(county_fips, aewr_region_id, E_p25, E_oews, E_sf_log)

joined_id <- inner_join(stored_exposures, base_check, by = c("county_fips", "aewr_region_id"))
assert_check("algebraic_identity_E_p25_vs_neg_std_ln_w25_real", nrow(joined_id) == 3108L && max(abs(joined_id$E_p25 - joined_id$neg_std_real)) < 1e-12)
assert_check("algebraic_identity_E_p25_vs_neg_std_ln_w25_nom", max(abs(joined_id$E_p25 - joined_id$neg_std_nom)) < 1e-12)

# Verify within-region mean 0 and SD 1 across all 17 regions for all three exposures
for (col in c("E_p25", "E_oews", "E_sf_log")) {
  reg_stats <- stored_exposures %>%
    filter(is.finite(.data[[col]])) %>%
    group_by(aewr_region_id) %>%
    summarise(mu = mean(.data[[col]]), s = stats::sd(.data[[col]]), .groups = "drop")
  assert_check(
    paste0("within_region_std_", col),
    nrow(reg_stats) == 17L && max(abs(reg_stats$mu)) < 1e-12 && max(abs(reg_stats$s - 1)) < 1e-12
  )
}

# 3. All-real E_sf_log unit consistency (real PPI-2012 vs nominal parity)
sf_check <- shared %>%
  filter(year %in% BITE_BASELINE_YEARS)
sf_check$sf_real <- bite_compute_shortfall_log(sf_check)
sf_check <- sf_check %>%
  mutate(
    sf_nom = {
      ln_a <- log(aewr)
      ag_f <- prevailing_ag_min_wage
      0.175 * pmax(0, ln_a - log(pmax(wage_p10 * ppi_2012, ag_f))) +
        0.200 * pmax(0, ln_a - log(pmax(wage_p25 * ppi_2012, ag_f))) +
        0.250 * pmax(0, ln_a - log(pmax(wage_p50 * ppi_2012, ag_f))) +
        0.200 * pmax(0, ln_a - log(pmax(wage_p75 * ppi_2012, ag_f))) +
        0.175 * pmax(0, ln_a - log(pmax(wage_p90 * ppi_2012, ag_f)))
    }
  )

assert_check("shortfall_real_vs_nominal_unit_parity", max(abs(sf_check$sf_real - sf_check$sf_nom)) < 1e-12)

sf_county <- sf_check %>%
  group_by(county_fips, aewr_region_id) %>%
  summarise(raw_sf = mean(sf_real), .groups = "drop") %>%
  mutate(E_sf_recomp = bite_std_within_region(raw_sf, aewr_region_id)) %>%
  inner_join(stored_exposures, by = c("county_fips", "aewr_region_id"))

assert_check("E_sf_log_recomputation_parity", max(abs(sf_county$E_sf_recomp - sf_county$E_sf_log)) < 1e-12)

# 4. Input & output SHA-256 parity with bite_provenance.json
assert_check(
  "input_county_panel_sha256_parity",
  identical(digest::digest(file = shared_path, algo = "sha256"), prov$inputs$county_year_panel_parquet$sha256)
)
assert_check(
  "input_bite_panel_sha256_parity",
  identical(digest::digest(file = bite_path, algo = "sha256"), prov$inputs$bite_county_year_panel_parquet$sha256)
)

expected_outputs <- c(
  "outputs/tables/bite/bite_main_estimates.csv",
  "outputs/tables/bite/bite_diagnostics.csv",
  "outputs/tables/bite/bite_event_study.csv",
  "outputs/tables/bite/bite_loo_regions.csv",
  "outputs/tables/table_bite_main.tex",
  "outputs/tables/table_bite_diagnostics.tex",
  "outputs/figures/fig_bite_event_study.pdf",
  "outputs/figures/fig_bite_event_study.png",
  "outputs/reports/bite_design.md"
)

for (rel in expected_outputs) {
  abs_p <- path_root(rel)
  assert_check(paste0("output_exists_", basename(rel)), file.exists(abs_p) && file.info(abs_p)$size > 0)
  assert_check(
    paste0("output_sha256_", basename(rel)),
    identical(digest::digest(file = abs_p, algo = "sha256"), prov$outputs_sha256[[rel]])
  )
}

# 5. Output table completeness, finite CV3J SEs, 2023-2025 preservation, and 48-state domestic floor
main_df <- read_csv(path_tables("bite", "bite_main_estimates.csv"), show_col_types = FALSE)
diag_df <- read_csv(path_tables("bite", "bite_diagnostics.csv"), show_col_types = FALSE)
es_df <- read_csv(path_tables("bite", "bite_event_study.csv"), show_col_types = FALSE)
loo_df <- read_csv(path_tables("bite", "bite_loo_regions.csv"), show_col_types = FALSE)

assert_check("main_estimates_14_rows_all_estimated", nrow(main_df) == 14L && all(main_df$status == "estimated"))
assert_check(
  "main_estimates_finite_cv3j_and_mde",
  all(is.finite(main_df$estimate)) &&
    all(is.finite(main_df$se_cv3j) & main_df$se_cv3j > 0) &&
    all(is.finite(main_df$mde_cv3j) & abs(main_df$mde_cv3j - 2.8 * main_df$se_cv3j) < 1e-12) &&
    all(main_df$n_clusters == 17L)
)
assert_check(
  "xpre_preserves_2025",
  all(main_df$year_max[main_df$spec_group == "predetermined_controls"] == 2025L)
)

diag_est <- diag_df %>% filter(status == "estimated")
assert_check(
  "diagnostic_estimates_finite_cv3j",
  nrow(diag_est) > 0L &&
    all(is.finite(diag_est$estimate)) &&
    all(is.finite(diag_est$se_cv3j) & diag_est$se_cv3j > 0) &&
    all(is.finite(diag_est$mde_cv3j))
)

dom_rows <- diag_df %>% filter(spec_group == "standalone_domestic_floor")
assert_check(
  "domestic_floor_covers_all_48_states_through_2025",
  nrow(dom_rows) == 2L &&
    all(dom_rows$status == "estimated") &&
    all(dom_rows$cluster_var == "state_fips") &&
    all(dom_rows$n_clusters == 48L) &&
    all(dom_rows$year_min == 2011L) &&
    all(dom_rows$year_max == 2025L) &&
    all(grepl("Standalone domestic-floor margin", dom_rows$notes, fixed = TRUE))
)

assert_check(
  "event_study_36_rows_complete",
  nrow(es_df) == 36L &&
    all(is.finite(es_df$estimate)) &&
    all(is.finite(es_df$se_cv3j[es_df$year != 2011L]) & es_df$se_cv3j[es_df$year != 2011L] > 0)
)
assert_check("loo_regions_17_per_spec", nrow(loo_df) == 6L * 17L && all(is.finite(loo_df$estimate)))

# 6. Independent re-estimation spot-check of primary 1b-PPML hours specification
indep_fit <- fixest::fepois(
  man_hours_certified_start_year ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
  data = bite %>% filter(year >= 2011L, year <= 2025L),
  vcov = ~aewr_region_id,
  notes = FALSE
)
prim_h <- main_df %>% filter(spec_id == "main_p25_hours")
assert_check(
  "independent_primary_fepois_spotcheck",
  abs(unname(coef(indep_fit)["E_p25_x_ln_aewr"]) - prim_h$estimate[[1]]) < 1e-10 &&
    abs(unname(se(indep_fit)["E_p25_x_ln_aewr"]) - prim_h$se_crv1[[1]]) < 1e-10
)

cat(sprintf("Bite design validation passed: %d/%d checks OK.\n", checks_passed, checks_passed))
