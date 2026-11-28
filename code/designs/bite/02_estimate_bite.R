# Purpose: Estimate the standalone full-panel PPML bite design, diagnostics,
# farm outcomes, standalone domestic-floor margin, tables, figures, report,
# and SHA-256 provenance.
# Input:  data/processed/bite_county_year_panel.parquet
# Outputs:
#   outputs/tables/bite/bite_main_estimates.csv
#   outputs/tables/bite/bite_diagnostics.csv
#   outputs/tables/bite/bite_event_study.csv
#   outputs/tables/bite/bite_loo_regions.csv
#   outputs/tables/table_bite_main.tex
#   outputs/tables/table_bite_diagnostics.tex
#   outputs/figures/fig_bite_event_study.pdf
#   outputs/figures/fig_bite_event_study.png
#   outputs/reports/bite_design.md
#   outputs/reports/bite_provenance.json

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "bite", "helpers.R"))

suppressPackageStartupMessages({
  library(arrow)
  library(digest)
  library(dplyr)
  library(fixest)
  library(ggplot2)
  library(ggthemes)
  library(jsonlite)
  library(readr)
  library(tibble)
})

dir.create(path_tables("bite"), recursive = TRUE, showWarnings = FALSE)
dir.create(path_figures(), recursive = TRUE, showWarnings = FALSE)
dir.create(path_outputs("reports"), recursive = TRUE, showWarnings = FALSE)

shared_panel_path <- path_processed("county_year_panel.parquet")
bite_panel_path <- path_processed("bite_county_year_panel.parquet")

shared_panel_sha256 <- digest::digest(file = shared_panel_path, algo = "sha256")
bite_panel_sha256 <- digest::digest(file = bite_panel_path, algo = "sha256")

panel_full <- read_parquet(bite_panel_path) %>%
  mutate(year = as.integer(year))
assert_geo_columns(panel_full, c("county_fips", "state_fips", "cz_id", "aewr_region_id"))

panel_est <- panel_full %>%
  filter(year >= BITE_ESTIMATION_START_YEAR, year <= BITE_ESTIMATION_END_YEAR)
panel_pre19 <- panel_est %>% filter(year <= 2019L)
panel_post20 <- panel_est %>% filter(year >= 2020L)

# ==============================================================================
# 1. Main Full-Panel PPML Bite Specifications (2011-2025)
# ==============================================================================

res_main_h <- bite_fit_cv3j(
  data = panel_est,
  formula = man_hours_certified_start_year ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "main_p25_hours",
  spec_group = "primary_1b_ppml",
  label = "Primary 1b-PPML (E_p25, Certified Hours)",
  outcome = "man_hours_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 full panel"
)

res_main_w <- bite_fit_cv3j(
  data = panel_est,
  formula = nbr_workers_certified_start_year ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "main_p25_workers",
  spec_group = "primary_1b_ppml",
  label = "Primary 1b-PPML (E_p25, Certified Positions)",
  outcome = "nbr_workers_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 full panel"
)

res_main_ge5 <- bite_fit_cv3j(
  data = panel_est,
  formula = any_h2a_ge5 ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "feols",
  cluster_col = "aewr_region_id",
  spec_id = "main_p25_any_ge5",
  spec_group = "primary_1b_ppml",
  label = "Primary Extensive Margin (LPM, >=5 Positions)",
  outcome = "any_h2a_ge5",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 full panel"
)

res_main_dur <- bite_implied_difference(
  res_hours = res_main_h,
  res_workers = res_main_w,
  term = "E_p25_x_ln_aewr",
  spec_id = "main_p25_hours_per_pos",
  spec_group = "primary_1b_ppml",
  label = "Implied Hours per Position (theta_H - theta_N)",
  outcome = "implied_hours_per_position",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 full panel",
  notes = "Exact paired cluster-jackknife difference theta_hours - theta_workers"
)

res_sy_h <- bite_fit_cv3j(
  data = panel_est,
  formula = man_hours_certified_start_year ~ E_p25_x_ln_aewr | county_fips + state_fips^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "state_year_p25_hours",
  spec_group = "state_year_fe",
  label = "State x Year FE (E_p25, Certified Hours)",
  outcome = "man_hours_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + State x Year",
  sample_label = "2011-2025 full panel"
)

res_sy_w <- bite_fit_cv3j(
  data = panel_est,
  formula = nbr_workers_certified_start_year ~ E_p25_x_ln_aewr | county_fips + state_fips^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "state_year_p25_workers",
  spec_group = "state_year_fe",
  label = "State x Year FE (E_p25, Certified Positions)",
  outcome = "nbr_workers_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + State x Year",
  sample_label = "2011-2025 full panel"
)

res_xpre_h <- bite_fit_cv3j(
  data = panel_est,
  formula = man_hours_certified_start_year ~ E_p25_x_ln_aewr +
    i(year, ln_pop_2010, ref = 2011) +
    i(year, farm_emp_share_0810, ref = 2011) +
    i(year, ln_emp_farm_0810, ref = 2011) | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "xpre_p25_hours",
  spec_group = "predetermined_controls",
  label = "+ X_pre x Year FE (E_p25, Certified Hours)",
  outcome = "man_hours_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year + X_pre x Year",
  sample_label = "2011-2025 (predetermined 2008-10 covariates; preserves 2023-2025)"
)

res_xpre_w <- bite_fit_cv3j(
  data = panel_est,
  formula = nbr_workers_certified_start_year ~ E_p25_x_ln_aewr +
    i(year, ln_pop_2010, ref = 2011) +
    i(year, farm_emp_share_0810, ref = 2011) +
    i(year, ln_emp_farm_0810, ref = 2011) | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "xpre_p25_workers",
  spec_group = "predetermined_controls",
  label = "+ X_pre x Year FE (E_p25, Certified Positions)",
  outcome = "nbr_workers_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year + X_pre x Year",
  sample_label = "2011-2025 (predetermined 2008-10 covariates; preserves 2023-2025)"
)

res_oews_h <- bite_fit_cv3j(
  data = panel_est,
  formula = man_hours_certified_start_year ~ E_oews_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_oews_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "oews_hours",
  spec_group = "alt_exposure_oews",
  label = "OEWS Exposure (E_oews, Certified Hours)",
  outcome = "man_hours_certified_start_year",
  exposure_name = "E_oews",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 OEWS-covered counties"
)

res_oews_w <- bite_fit_cv3j(
  data = panel_est,
  formula = nbr_workers_certified_start_year ~ E_oews_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_oews_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "oews_workers",
  spec_group = "alt_exposure_oews",
  label = "OEWS Exposure (E_oews, Certified Positions)",
  outcome = "nbr_workers_certified_start_year",
  exposure_name = "E_oews",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 OEWS-covered counties"
)

res_sf_h <- bite_fit_cv3j(
  data = panel_est,
  formula = man_hours_certified_start_year ~ E_sf_log_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_sf_log_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "sf_log_hours",
  spec_group = "alt_exposure_shortfall",
  label = "All-Real Shortfall Exposure (E_sf_log, Certified Hours)",
  outcome = "man_hours_certified_start_year",
  exposure_name = "E_sf_log",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 full panel"
)

res_sf_w <- bite_fit_cv3j(
  data = panel_est,
  formula = nbr_workers_certified_start_year ~ E_sf_log_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_sf_log_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "sf_log_workers",
  spec_group = "alt_exposure_shortfall",
  label = "All-Real Shortfall Exposure (E_sf_log, Certified Positions)",
  outcome = "nbr_workers_certified_start_year",
  exposure_name = "E_sf_log",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 full panel"
)

# Variance split paired with 1a-PPML transparency
var_decomp <- bind_rows(
  bite_variance_decomposition(panel_est, "E_p25_x_ln_aewr", "E_p25", weights_vec = NULL),
  bite_variance_decomposition(
    res_main_h$used,
    "E_p25_x_ln_aewr",
    "E_p25",
    weights_vec = as.numeric(stats::fitted(res_main_h$fit))
  )
)

share_1a_unw <- var_decomp$share_1a_of_1b[var_decomp$weighting == "unweighted"][[1]]
share_1a_wt <- var_decomp$share_1a_of_1b[var_decomp$weighting == "ppml_fitted_hours"][[1]]
note_1a <- sprintf(
  "Retained residualized variance vs C+RY: %.2f%% unweighted, %.2f%% PPML-hours-weighted",
  100 * share_1a_unw,
  100 * share_1a_wt
)

res_1a_h <- bite_fit_cv3j(
  data = panel_est,
  formula = man_hours_certified_start_year ~ E_p25_x_ln_aewr +
    i(year, E_p25, ref = 2011) | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "spec1a_p25_hours",
  spec_group = "transparent_1a_ppml",
  label = "Transparent 1a-PPML (+ E_p25 x Year FE, Certified Hours)",
  outcome = "man_hours_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year + E_p25 x Year",
  sample_label = "2011-2025 full panel",
  notes = note_1a
)

res_1a_w <- bite_fit_cv3j(
  data = panel_est,
  formula = nbr_workers_certified_start_year ~ E_p25_x_ln_aewr +
    i(year, E_p25, ref = 2011) | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "spec1a_p25_workers",
  spec_group = "transparent_1a_ppml",
  label = "Transparent 1a-PPML (+ E_p25 x Year FE, Certified Positions)",
  outcome = "nbr_workers_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year + E_p25 x Year",
  sample_label = "2011-2025 full panel",
  notes = note_1a
)

main_estimates <- bind_rows(
  res_main_h$summary,
  res_main_w$summary,
  res_main_ge5$summary,
  res_main_dur$summary,
  res_sy_h$summary,
  res_sy_w$summary,
  res_xpre_h$summary,
  res_xpre_w$summary,
  res_oews_h$summary,
  res_oews_w$summary,
  res_sf_h$summary,
  res_sf_w$summary,
  res_1a_h$summary,
  res_1a_w$summary
)

write_csv(main_estimates, path_tables("bite", "bite_main_estimates.csv"))
bite_write_main_tex(main_estimates, var_decomp, path_tables("table_bite_main.tex"))

# ==============================================================================
# 2. Event Studies (2008-2025, ref = 2011) + Joint 2008-2010 Pre-Trend Tests
# ==============================================================================

event_years_nonref <- setdiff(BITE_EVENT_START_YEAR:BITE_EVENT_END_YEAR, BITE_EVENT_REF_YEAR)
event_terms <- paste0("year::", event_years_nonref, ":E_p25")
pre_terms <- paste0("year::", c(2008L, 2009L, 2010L), ":E_p25")

res_es_h <- bite_fit_cv3j(
  data = panel_full,
  formula = man_hours_certified_start_year ~ i(year, E_p25, ref = 2011) | county_fips + aewr_region_id^year,
  terms = event_terms,
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "event_study_hours",
  spec_group = "event_study",
  label = "FE-PPML Event Study (Certified Hours, 2008-2025)",
  outcome = "man_hours_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = "2008-2025 full panel (2008-10 weekly hours pooled-mean imputed)"
)

res_es_w <- bite_fit_cv3j(
  data = panel_full,
  formula = nbr_workers_certified_start_year ~ i(year, E_p25, ref = 2011) | county_fips + aewr_region_id^year,
  terms = event_terms,
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "event_study_workers",
  spec_group = "event_study",
  label = "FE-PPML Event Study (Certified Positions, 2008-2025)",
  outcome = "nbr_workers_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = "2008-2025 full panel"
)

build_es_rows <- function(res_obj, outcome_label, is_hours) {
  ref_row <- tibble(
    outcome = res_obj$summary$outcome[[1]],
    outcome_label = outcome_label,
    year = BITE_EVENT_REF_YEAR,
    term = "year::2011:E_p25",
    status = "reference",
    estimate = 0,
    se_crv1 = 0,
    p_crv1 = NA_real_,
    se_cv3j = 0,
    ci95_low_cv3j = 0,
    ci95_high_cv3j = 0,
    p_cv3j = NA_real_,
    loo_min = 0,
    loo_max = 0,
    n_obs = res_obj$summary$n_obs[[1]],
    n_counties = res_obj$summary$n_counties[[1]],
    n_clusters = res_obj$summary$n_clusters[[1]],
    hours_imputation_note = "reference_year"
  )

  est_rows <- res_obj$summary %>%
    transmute(
      outcome,
      outcome_label = outcome_label,
      year = as.integer(sub("^year::([0-9]{4}):E_p25$", "\\1", term)),
      term,
      status,
      estimate,
      se_crv1,
      p_crv1,
      se_cv3j,
      ci95_low_cv3j,
      ci95_high_cv3j,
      p_cv3j,
      loo_min,
      loo_max,
      n_obs,
      n_counties,
      n_clusters,
      hours_imputation_note = if_else(
        is_hours & year <= 2010L,
        "2008-2010_weekly_hours_pooled_mean_imputed",
        if_else(year == 2025L, "provisional_start_year_pending_fy2027", "observed")
      )
    )

  bind_rows(est_rows, ref_row) %>% arrange(year)
}

event_study_df <- bind_rows(
  build_es_rows(res_es_h, "Certified H-2A Hours", TRUE),
  build_es_rows(res_es_w, "Certified H-2A Positions", FALSE)
)

write_csv(event_study_df, path_tables("bite", "bite_event_study.csv"))

# Joint 2008-2010 Wald / F pre-trend tests (CRV1 and CV3J)
make_pretrend_row <- function(res_obj, spec_id, label, note_txt) {
  b_pre <- stats::coef(res_obj$fit)[pre_terms]
  G <- res_obj$summary$n_clusters[[1]]
  w_crv1 <- bite_joint_wald(b_pre, res_obj$vcov_crv1[pre_terms, pre_terms, drop = FALSE], df2 = G - 1L)
  w_cv3j <- bite_joint_wald(b_pre, res_obj$vcov_cv3j[pre_terms, pre_terms, drop = FALSE], df2 = G - 1L)
  tibble(
    spec_id = spec_id,
    spec_group = "pretrend_joint_test",
    label = label,
    estimator = "fepois_wald",
    outcome = res_obj$summary$outcome[[1]],
    exposure = "E_p25",
    fe = "County + Region x Year",
    sample = "2008-2025 event study (joint 2008-2010 = 0)",
    cluster_var = "aewr_region_id",
    term = "joint_2008_2010_F_cv3j",
    status = "diagnostic",
    estimate = w_cv3j$f_stat,
    se_crv1 = w_crv1$f_stat,
    p_crv1 = w_crv1$p_value,
    se_cv3j = NA_real_,
    ci95_low_cv3j = NA_real_,
    ci95_high_cv3j = NA_real_,
    p_cv3j = w_cv3j$p_value,
    mde_cv3j = NA_real_,
    loo_min = NA_real_,
    loo_max = NA_real_,
    n_obs = res_obj$summary$n_obs[[1]],
    n_complete_rows = res_obj$summary$n_complete_rows[[1]],
    n_counties = res_obj$summary$n_counties[[1]],
    n_clusters = G,
    year_min = 2008L,
    year_max = 2025L,
    notes = sprintf(
      "%s; F_CV3J(3,%d)=%.3f (p=%.4f), F_CRV1(3,%d)=%.3f (p=%.4f)",
      note_txt, G - 1L, w_cv3j$f_stat, w_cv3j$p_value, G - 1L, w_crv1$f_stat, w_crv1$p_value
    )
  )
}

pretrend_h <- make_pretrend_row(
  res_es_h,
  "pretrend_joint_hours",
  "Joint 2008-2010 Pre-Trend Test (Certified Hours)",
  "2008-2010 weekly hours are pooled-mean imputed"
)
pretrend_w <- make_pretrend_row(
  res_es_w,
  "pretrend_joint_workers",
  "Joint 2008-2010 Pre-Trend Test (Certified Positions)",
  "Positions directly observed 2008-2025"
)

# Event-study figure (ASCII labels for clean PDF + PNG rendering)
es_plot_df <- event_study_df %>%
  mutate(
    outcome_label = factor(
      outcome_label,
      levels = c("Certified H-2A Hours", "Certified H-2A Positions")
    ),
    ci95_low_crv1 = if_else(year == BITE_EVENT_REF_YEAR, 0, estimate - qt(0.975, df = n_clusters - 1L) * se_crv1),
    ci95_high_crv1 = if_else(year == BITE_EVENT_REF_YEAR, 0, estimate + qt(0.975, df = n_clusters - 1L) * se_crv1)
  )

es_plot <- ggplot(es_plot_df, aes(x = year, y = estimate)) +
  annotate("rect", xmin = 2007.5, xmax = 2010.5, ymin = -Inf, ymax = Inf, fill = "grey92", alpha = 0.65) +
  geom_hline(yintercept = 0, color = "grey35", linewidth = 0.5) +
  geom_vline(xintercept = 2011, linetype = "dashed", color = "grey35", linewidth = 0.5) +
  geom_ribbon(aes(ymin = ci95_low_cv3j, ymax = ci95_high_cv3j), fill = "steelblue", alpha = 0.18) +
  geom_errorbar(aes(ymin = ci95_low_crv1, ymax = ci95_high_crv1), width = 0.18, color = "steelblue4", linewidth = 0.45) +
  geom_line(color = "steelblue4", linewidth = 0.9) +
  geom_point(color = "steelblue4", size = 1.9) +
  facet_wrap(~outcome_label, ncol = 1L, scales = "free_y") +
  scale_x_continuous(breaks = seq(2008, 2025, by = 2)) +
  labs(
    title = "FE-PPML Event Study: Exposure E_p25 x Year (Ref. = 2011)",
    subtitle = "County + Region x Year FE; shaded band = 95% CV3J t(16) CI, bars = 95% CRV1 CI",
    x = "Start Year",
    y = "Log-Point Differential per +1 Within-Region SD of E_p25",
    caption = paste0(
      "Notes: Shaded 2008-2010 window is pre-estimation; 2008-2010 weekly hours are pooled-mean imputed.\n",
      "2025 start-year H-2A totals are provisional pending FY2027 decisions."
    )
  ) +
  theme_clean() +
  theme(
    plot.background = element_rect(color = NA, fill = "white"),
    strip.text = element_text(face = "bold", size = 10),
    plot.caption = element_text(size = 8, hjust = 0)
  )

ggsave(
  path_figures("fig_bite_event_study.pdf"),
  es_plot,
  width = 8.5,
  height = 6.8,
  device = "pdf"
)
ggsave(
  path_figures("fig_bite_event_study.png"),
  es_plot,
  width = 8.5,
  height = 6.8,
  dpi = 300,
  device = "png"
)

# ==============================================================================
# 3. Diagnostic Regressions, Farm Outcomes & Standalone Domestic Floor
# ==============================================================================

# 3a. Quartile SPT check (Q2..Q4 x ln_aewr, ref = Q1)
q_terms <- paste0("E_p25_quartile::Q", 2:4, ":ln_aewr")
res_q_h <- bite_fit_cv3j(
  data = panel_est,
  formula = man_hours_certified_start_year ~ i(E_p25_quartile, ln_aewr, ref = "Q1") | county_fips + aewr_region_id^year,
  terms = q_terms,
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "spt_quartile_hours",
  spec_group = "quartile_spt",
  label = "Quartile SPT Check (Certified Hours, vs Q1)",
  outcome = "man_hours_certified_start_year",
  exposure_name = "E_p25_quartile",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 full panel"
)

res_q_w <- bite_fit_cv3j(
  data = panel_est,
  formula = nbr_workers_certified_start_year ~ i(E_p25_quartile, ln_aewr, ref = "Q1") | county_fips + aewr_region_id^year,
  terms = q_terms,
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "spt_quartile_workers",
  spec_group = "quartile_spt",
  label = "Quartile SPT Check (Certified Positions, vs Q1)",
  outcome = "nbr_workers_certified_start_year",
  exposure_name = "E_p25_quartile",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 full panel"
)

# 3b. WPU01 PPI decoupling check
res_ppi_h <- bite_fit_cv3j(
  data = panel_est,
  formula = man_hours_certified_start_year ~ E_p25_x_ln_aewr + E_p25_x_ln_ppi_2012 | county_fips + aewr_region_id^year,
  terms = c("E_p25_x_ln_aewr", "E_p25_x_ln_ppi_2012"),
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "ppi_decouple_hours",
  spec_group = "ppi_decoupling",
  label = "WPU01 PPI Decoupling (Certified Hours)",
  outcome = "man_hours_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 full panel"
)

res_ppi_w <- bite_fit_cv3j(
  data = panel_est,
  formula = nbr_workers_certified_start_year ~ E_p25_x_ln_aewr + E_p25_x_ln_ppi_2012 | county_fips + aewr_region_id^year,
  terms = c("E_p25_x_ln_aewr", "E_p25_x_ln_ppi_2012"),
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "ppi_decouple_workers",
  spec_group = "ppi_decoupling",
  label = "WPU01 PPI Decoupling (Certified Positions)",
  outcome = "nbr_workers_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 full panel"
)

# 3c. Addendum-B extensive-margin proof (gt0 / ge1 / ge5 across 2011-2025 vs 2011-2019)
#     and balanced employer count strictly split 2011-2019 vs 2020-2025
add_specs <- list(
  list(id = "addendum_b_gt0_2011_2025", d = panel_est, y = "any_h2a_gt0", est = "feols", samp = "2011-2025 full panel", lbl = "Addendum-B Check: Any >0 (2011-2025)"),
  list(id = "addendum_b_gt0_2011_2019", d = panel_pre19, y = "any_h2a_gt0", est = "feols", samp = "2011-2019 pre-Addendum-B", lbl = "Addendum-B Check: Any >0 (2011-2019)"),
  list(id = "addendum_b_ge1_2011_2025", d = panel_est, y = "any_h2a_ge1", est = "feols", samp = "2011-2025 full panel", lbl = "Addendum-B Check: Any >=1 (2011-2025)"),
  list(id = "addendum_b_ge1_2011_2019", d = panel_pre19, y = "any_h2a_ge1", est = "feols", samp = "2011-2019 pre-Addendum-B", lbl = "Addendum-B Check: Any >=1 (2011-2019)"),
  list(id = "addendum_b_ge5_2011_2025", d = panel_est, y = "any_h2a_ge5", est = "feols", samp = "2011-2025 full panel", lbl = "Addendum-B Check: Any >=5 (2011-2025)"),
  list(id = "addendum_b_ge5_2011_2019", d = panel_pre19, y = "any_h2a_ge5", est = "feols", samp = "2011-2019 pre-Addendum-B", lbl = "Addendum-B Check: Any >=5 (2011-2019)"),
  list(id = "employers_balanced_2011_2019", d = panel_pre19, y = "nbr_employers_balanced_start_year", est = "fepois", samp = "2011-2019 pre-Addendum-B", lbl = "Balanced Worksite Employers (PPML, 2011-2019)"),
  list(id = "employers_balanced_2020_2025", d = panel_post20, y = "nbr_employers_balanced_start_year", est = "fepois", samp = "2020-2025 post-Addendum-B", lbl = "Balanced Worksite Employers (PPML, 2020-2025)")
)

add_results <- lapply(add_specs, function(s) {
  bite_fit_cv3j(
    data = s$d,
    formula = stats::as.formula(paste0(s$y, " ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year")),
    terms = "E_p25_x_ln_aewr",
    estimator = s$est,
    cluster_col = "aewr_region_id",
    spec_id = s$id,
    spec_group = "addendum_b_extensive",
    label = s$lbl,
    outcome = s$y,
    exposure_name = "E_p25",
    fe_label = "County + Region x Year",
    sample_label = s$samp
  )$summary
})

# 3d. Concentration, LOO region table, G_eff, CA Fisher-info share,
#     drop top 1% H-2A counties, variance split, state-year residual noise ratio
lev_h <- bite_cluster_leverage(
  fit = res_main_h$fit,
  used = res_main_h$used,
  regressor_col = "E_p25_x_ln_aewr",
  fe_rhs = "county_fips + aewr_region_id^year",
  cluster_col = "aewr_region_id",
  outcome_col = "man_hours_certified_start_year"
)

lev_w <- bite_cluster_leverage(
  fit = res_main_w$fit,
  used = res_main_w$used,
  regressor_col = "E_p25_x_ln_aewr",
  fe_rhs = "county_fips + aewr_region_id^year",
  cluster_col = "aewr_region_id",
  outcome_col = "nbr_workers_certified_start_year"
)

region_meta <- panel_est %>%
  group_by(aewr_region_id) %>%
  summarise(
    states = paste(sort(unique(state_abbrev)), collapse = ","),
    n_states = n_distinct(state_fips),
    n_counties_region = n_distinct(county_fips),
    hours_2011_2025 = sum(man_hours_certified_start_year, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(hours_share_2011_2025 = hours_2011_2025 / sum(hours_2011_2025))

loo_regions_tbl <- bind_rows(
  res_main_h$loo,
  res_main_w$loo,
  res_main_ge5$loo,
  res_main_dur$loo,
  res_1a_h$loo,
  res_1a_w$loo
) %>%
  left_join(region_meta, by = c("omitted_cluster" = "aewr_region_id")) %>%
  left_join(
    lev_h$by_cluster %>% transmute(omitted_cluster = cluster_id, fisher_info_share_hours = fisher_info_share),
    by = "omitted_cluster"
  ) %>%
  left_join(
    lev_w$by_cluster %>% transmute(omitted_cluster = cluster_id, fisher_info_share_workers = fisher_info_share),
    by = "omitted_cluster"
  ) %>%
  arrange(spec_id, as.integer(omitted_cluster))

write_csv(loo_regions_tbl, path_tables("bite", "bite_loo_regions.csv"))

# Drop top 1% H-2A counties by cumulative 2011-2025 certified hours
county_cum_hours <- panel_est %>%
  group_by(county_fips) %>%
  summarise(cum_hours_1125 = sum(man_hours_certified_start_year, na.rm = TRUE), .groups = "drop")
p99_hours_cutoff <- stats::quantile(county_cum_hours$cum_hours_1125, probs = 0.99, names = FALSE)
keep_notop1 <- county_cum_hours$county_fips[county_cum_hours$cum_hours_1125 <= p99_hours_cutoff]
n_dropped_top1 <- sum(county_cum_hours$cum_hours_1125 > p99_hours_cutoff)
panel_notop1 <- panel_est %>% filter(county_fips %in% keep_notop1)

res_notop1_h <- bite_fit_cv3j(
  data = panel_notop1,
  formula = man_hours_certified_start_year ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "drop_top1pct_hours",
  spec_group = "concentration_robustness",
  label = "Drop Top 1% H-2A Counties (Certified Hours)",
  outcome = "man_hours_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = sprintf("2011-2025 excluding top %d counties (>p99 cumulative hours)", n_dropped_top1)
)

res_notop1_w <- bite_fit_cv3j(
  data = panel_notop1,
  formula = nbr_workers_certified_start_year ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "drop_top1pct_workers",
  spec_group = "concentration_robustness",
  label = "Drop Top 1% H-2A Counties (Certified Positions)",
  outcome = "nbr_workers_certified_start_year",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = sprintf("2011-2025 excluding top %d counties (>p99 cumulative hours)", n_dropped_top1)
)

sy_noise_h <- bite_state_year_noise_ratio(res_main_h$fit, res_main_h$used, "man_hours_certified_start_year")
sy_noise_w <- bite_state_year_noise_ratio(res_main_w$fit, res_main_w$used, "nbr_workers_certified_start_year")

make_scalar_diag <- function(spec_id, spec_group, label, outcome, term, val, notes, n_obs = nrow(panel_est), n_cty = 3108L, n_cl = 17L) {
  tibble(
    spec_id = spec_id,
    spec_group = spec_group,
    label = label,
    estimator = "diagnostic_scalar",
    outcome = outcome,
    exposure = "E_p25",
    fe = "County + Region x Year",
    sample = "2011-2025",
    cluster_var = "aewr_region_id",
    term = term,
    status = "diagnostic",
    estimate = as.numeric(val),
    se_crv1 = NA_real_,
    p_crv1 = NA_real_,
    se_cv3j = NA_real_,
    ci95_low_cv3j = NA_real_,
    ci95_high_cv3j = NA_real_,
    p_cv3j = NA_real_,
    mde_cv3j = NA_real_,
    loo_min = NA_real_,
    loo_max = NA_real_,
    n_obs = as.integer(n_obs),
    n_complete_rows = as.integer(n_obs),
    n_counties = as.integer(n_cty),
    n_clusters = as.integer(n_cl),
    year_min = 2011L,
    year_max = 2025L,
    notes = notes
  )
}

scalar_diags <- bind_rows(
  make_scalar_diag(
    "geff_fisher_hours", "concentration_metrics", "Effective Clusters G_eff (Fisher Info, Hours)",
    "man_hours_certified_start_year", "G_eff_fisher", lev_h$g_eff_fisher,
    sprintf("G_eff_outcome=%.2f, G_eff_county=%.2f, hessian_rel_err=%.2e", lev_h$g_eff_outcome, lev_h$g_eff_county, lev_h$hessian_rel_err),
    n_obs = nrow(res_main_h$used), n_cty = n_distinct(res_main_h$used$county_fips)
  ),
  make_scalar_diag(
    "ca_fisher_share_hours", "concentration_metrics", "California (Region 17) Fisher Info Share (Hours)",
    "man_hours_certified_start_year", "ca_fisher_share", lev_h$ca_fisher_share,
    sprintf("CA hours share=%.4f", region_meta$hours_share_2011_2025[region_meta$aewr_region_id == "17"][[1]]),
    n_obs = nrow(res_main_h$used), n_cty = n_distinct(res_main_h$used$county_fips)
  ),
  make_scalar_diag(
    "geff_fisher_workers", "concentration_metrics", "Effective Clusters G_eff (Fisher Info, Positions)",
    "nbr_workers_certified_start_year", "G_eff_fisher", lev_w$g_eff_fisher,
    sprintf("G_eff_outcome=%.2f, hessian_rel_err=%.2e", lev_w$g_eff_outcome, lev_w$hessian_rel_err),
    n_obs = nrow(res_main_w$used), n_cty = n_distinct(res_main_w$used$county_fips)
  ),
  make_scalar_diag(
    "ca_fisher_share_workers", "concentration_metrics", "California (Region 17) Fisher Info Share (Positions)",
    "nbr_workers_certified_start_year", "ca_fisher_share", lev_w$ca_fisher_share,
    "Region 17 share of total FE-PPML Fisher information for positions",
    n_obs = nrow(res_main_w$used), n_cty = n_distinct(res_main_w$used$county_fips)
  ),
  make_scalar_diag(
    "var_split_1a_unweighted", "variance_decomposition", "Retained Variance Share (C+RY+E*Y / C+RY, Unweighted)",
    "E_p25_x_ln_aewr", "share_1a_of_1b", share_1a_unw,
    sprintf(
      "R2(C+RY)=%.6f, R2(C+RY+E*Y)=%.6f, R2(C+SY)=%.6f, within_SY_share=%.4f",
      var_decomp$r2_C_RY[[1]], var_decomp$r2_C_RY_EY[[1]], var_decomp$r2_C_SY[[1]], var_decomp$share_within_SY_of_1b[[1]]
    )
  ),
  make_scalar_diag(
    "var_split_1a_ppml_weighted", "variance_decomposition", "Retained Variance Share (C+RY+E*Y / C+RY, PPML-Hours Weighted)",
    "E_p25_x_ln_aewr", "share_1a_of_1b", share_1a_wt,
    sprintf(
      "R2(C+RY)=%.6f, R2(C+RY+E*Y)=%.6f, R2(C+SY)=%.6f, within_SY_share=%.4f",
      var_decomp$r2_C_RY[[2]], var_decomp$r2_C_RY_EY[[2]], var_decomp$r2_C_SY[[2]], var_decomp$share_within_SY_of_1b[[2]]
    ),
    n_obs = nrow(res_main_h$used), n_cty = n_distinct(res_main_h$used$county_fips)
  ),
  make_scalar_diag(
    "state_year_noise_ratio_hours", "variance_decomposition", "State-Year Residual Mean-Square Ratio vs County Noise (Hours)",
    "man_hours_certified_start_year", "ms_ratio_sy_to_county", sy_noise_h$ms_ratio_sy_to_county,
    sprintf(
      "R2_SY_within_RY=%.4f, RE_var_ratio=%.4f, df=(%d,%d), p=%.4g",
      sy_noise_h$r2_state_year_within_ry, sy_noise_h$re_var_ratio_sy_to_county,
      sy_noise_h$df_state_year_extra, sy_noise_h$df_county_noise, sy_noise_h$f_p_value
    ),
    n_obs = sy_noise_h$n_obs, n_cty = n_distinct(res_main_h$used$county_fips)
  ),
  make_scalar_diag(
    "state_year_noise_ratio_workers", "variance_decomposition", "State-Year Residual Mean-Square Ratio vs County Noise (Positions)",
    "nbr_workers_certified_start_year", "ms_ratio_sy_to_county", sy_noise_w$ms_ratio_sy_to_county,
    sprintf(
      "R2_SY_within_RY=%.4f, RE_var_ratio=%.4f, df=(%d,%d), p=%.4g",
      sy_noise_w$r2_state_year_within_ry, sy_noise_w$re_var_ratio_sy_to_county,
      sy_noise_w$df_state_year_extra, sy_noise_w$df_county_noise, sy_noise_w$f_p_value
    ),
    n_obs = sy_noise_w$n_obs, n_cty = n_distinct(res_main_w$used$county_fips)
  )
)

# 3e. Farm outcomes under E_p25 x ln(aewr) (County + Region x Year FE)
res_fisher <- bite_fit_cv3j(
  data = panel_est %>% filter(year <= 2024L),
  formula = fisher_index_ppi ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "feols",
  cluster_col = "aewr_region_id",
  spec_id = "farm_fisher_price",
  spec_group = "farm_outcomes",
  label = "Real Fisher Output Price Index (OLS, 2011-2024)",
  outcome = "fisher_index_ppi",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = "2011-2024 NASS Fisher price panel"
)

res_lshare <- bite_fit_cv3j(
  data = panel_est %>% filter(year <= 2022L),
  formula = share_farm_laborexp_prodexp ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "feols",
  cluster_col = "aewr_region_id",
  spec_id = "farm_labor_share",
  spec_group = "farm_outcomes",
  label = "Farm Labor Share of Production Expense (OLS, 2011-2022)",
  outcome = "share_farm_laborexp_prodexp",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = "2011-2022 BEA CAINC45 panel"
)

res_femp_ppml <- bite_fit_cv3j(
  data = panel_est %>% filter(year <= 2022L),
  formula = emp_farm ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "fepois",
  cluster_col = "aewr_region_id",
  spec_id = "farm_emp_ppml",
  spec_group = "farm_outcomes",
  label = "BEA Total Farm Employment (PPML, 2011-2022)",
  outcome = "emp_farm",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = "2011-2022 BEA CAEMP25N panel"
)

res_femp_ols <- bite_fit_cv3j(
  data = panel_est %>% filter(year <= 2022L),
  formula = emp_farm ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr",
  estimator = "feols",
  cluster_col = "aewr_region_id",
  spec_id = "farm_emp_ols",
  spec_group = "farm_outcomes",
  label = "BEA Total Farm Employment (OLS levels, 2011-2022)",
  outcome = "emp_farm",
  exposure_name = "E_p25",
  fe_label = "County + Region x Year",
  sample_label = "2011-2022 BEA CAEMP25N panel"
)

# 3f. Standalone domestic-floor margin (2011-2025, all 48 contiguous states; state clusters)
dom_note <- paste0(
  "Standalone domestic-floor margin (NOT an AEWR elasticity): all 48 contiguous states retained ",
  "through 2025 (46 multi-state-region states provide within-region variation; CA and FL absorbed by Region x Year FE)"
)

res_dom_h <- bite_fit_cv3j(
  data = panel_est,
  formula = man_hours_certified_start_year ~ ln_prevailing_ag_min_wage | county_fips + aewr_region_id^year,
  terms = "ln_prevailing_ag_min_wage",
  estimator = "fepois",
  cluster_col = "state_fips",
  spec_id = "domestic_floor_hours",
  spec_group = "standalone_domestic_floor",
  label = "Standalone Domestic-Floor Margin (Certified Hours, State Clusters)",
  outcome = "man_hours_certified_start_year",
  exposure_name = "ln_prevailing_ag_min_wage",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 all 48 contiguous states",
  notes = dom_note
)

res_dom_w <- bite_fit_cv3j(
  data = panel_est,
  formula = nbr_workers_certified_start_year ~ ln_prevailing_ag_min_wage | county_fips + aewr_region_id^year,
  terms = "ln_prevailing_ag_min_wage",
  estimator = "fepois",
  cluster_col = "state_fips",
  spec_id = "domestic_floor_workers",
  spec_group = "standalone_domestic_floor",
  label = "Standalone Domestic-Floor Margin (Certified Positions, State Clusters)",
  outcome = "nbr_workers_certified_start_year",
  exposure_name = "ln_prevailing_ag_min_wage",
  fe_label = "County + Region x Year",
  sample_label = "2011-2025 all 48 contiguous states",
  notes = dom_note
)

diagnostics_df <- bind_rows(
  pretrend_h,
  pretrend_w,
  res_q_h$summary,
  res_q_w$summary,
  res_ppi_h$summary,
  res_ppi_w$summary,
  bind_rows(add_results),
  res_notop1_h$summary,
  res_notop1_w$summary,
  scalar_diags,
  res_fisher$summary,
  res_lshare$summary,
  res_femp_ppml$summary,
  res_femp_ols$summary,
  res_dom_h$summary,
  res_dom_w$summary
)

write_csv(diagnostics_df, path_tables("bite", "bite_diagnostics.csv"))
bite_write_diag_tex(diagnostics_df, path_tables("table_bite_diagnostics.tex"))

# ==============================================================================
# 4. Auto-Generated Markdown Report & Provenance JSON
# ==============================================================================

report_lines <- c(
  "# Full-Panel FE-PPML Bite Design",
  "",
  sprintf("Generated %s from `data/processed/county_year_panel.parquet` (SHA-256 `%s`).", as.character(Sys.time()), shared_panel_sha256),
  "",
  "## Identification & Estimand Boundary",
  "",
  "- **Panel**: Balanced 3,108 contiguous U.S. counties across 48 states and 17 AEWR regions (`2008-2025`, 55,944 county-years). Estimation spans `2011-2025` (excluding `2009`, where statutory AEWR is strictly NULL under `AEWR-001`; `2025` start-year H-2A outcomes remain provisional pending FY2027 decisions).",
  "- **Frozen Baseline Exposure (`2008/2010`)**: Primary exposure $E_{p25}$ standardizes the 2008/2010 average of $\\ln(\\text{AEWR}^{\\text{PPI}} / w_{p25}^{\\text{PPI}})$ to mean 0, SD 1 within each AEWR region. Algebraically, within-region standardization cancels both the regional AEWR level and the national PPI deflator identically ($E_{p25} \\equiv -\\text{std}_{r}(\\overline{\\ln w_{p25}})$). Alternative frozen exposures use 2008/2010 OEWS Big-Six wages ($E_{\\text{oews}}$) and the all-real five-quantile agricultural shortfall ($E_{\\text{sf}}$).",
  "- **Estimand**: Coefficients on $E_c \\times \\ln(\\text{AEWR}_{rt})$ under `County + Region x Year FE` (`1b-PPML`) measure the **relative differential log-point (semi-elasticity) response** of high-bite vs low-bite counties within the same AEWR region per $+1$ SD of baseline exposure per unit increase in $\\ln(\\text{AEWR}_{rt})$. They are **not** aggregate AEWR elasticities.",
  "- **Inference**: Every regression reports both region-clustered CRV1 SEs and exact delete-one-cluster jackknife (**CV3J**) SEs with $t(G-1)$ critical values and MDE $= 2.8 \\times \\text{CV3J SE}$.",
  "",
  "## 1. Main Full-Panel Bite Estimates (`2011-2025`)",
  "",
  bite_md_table(
    main_estimates %>%
      select(spec_id, estimator, outcome, fe, estimate, se_crv1, se_cv3j, p_cv3j, mde_cv3j, loo_min, loo_max, n_obs, n_counties, n_clusters),
    digits = 4L
  ),
  "",
  "## 2. Identifying-Variation Split, Effective Clusters & Influence",
  "",
  bite_md_table(var_decomp, digits = 4L),
  "",
  sprintf(
    "- **Effective clusters ($G_{\\text{eff}}$, Fisher information)**: `%.2f` (hours; CA Region 17 share = `%.1f%%`, Hessian identity rel. err = `%.2e`), `%.2f` (positions; CA Region 17 share = `%.1f%%`).",
    lev_h$g_eff_fisher, 100 * lev_h$ca_fisher_share, lev_h$hessian_rel_err,
    lev_w$g_eff_fisher, 100 * lev_w$ca_fisher_share
  ),
  sprintf(
    "- **State-year residual variance ratio vs county noise (deviance residuals)**: Hours `MS_SY / MS_county = %.2f` (`R2_SY_within_RY = %.4f`, `p = %.3g`); Positions `MS_SY / MS_county = %.2f` (`R2_SY_within_RY = %.4f`, `p = %.3g`).",
    sy_noise_h$ms_ratio_sy_to_county, sy_noise_h$r2_state_year_within_ry, sy_noise_h$f_p_value,
    sy_noise_w$ms_ratio_sy_to_county, sy_noise_w$r2_state_year_within_ry, sy_noise_w$f_p_value
  ),
  "",
  "## 3. Event Study (`2008-2025`, Ref. = `2011`) & Joint Pre-Trend Tests",
  "",
  sprintf("- **Certified Hours joint `2008-2010` pre-trend**: %s", pretrend_h$notes[[1]]),
  sprintf("- **Certified Positions joint `2008-2010` pre-trend**: %s", pretrend_w$notes[[1]]),
  "",
  bite_md_table(
    event_study_df %>%
      select(outcome, year, estimate, se_crv1, se_cv3j, ci95_low_cv3j, ci95_high_cv3j, p_cv3j, hours_imputation_note),
    digits = 4L
  ),
  "",
  "## 4. Diagnostics, Farm Outcomes & Standalone Domestic-Floor Margin",
  "",
  bite_md_table(
    diagnostics_df %>%
      select(spec_id, spec_group, outcome, term, estimate, se_crv1, se_cv3j, p_cv3j, mde_cv3j, n_obs, n_clusters, notes),
    digits = 4L
  ),
  "",
  "## 5. Leave-One-Region-Out (`1b-PPML` Primary Hours & Positions)",
  "",
  bite_md_table(
    loo_regions_tbl %>%
      filter(spec_id %in% c("main_p25_hours", "main_p25_workers")) %>%
      select(spec_id, omitted_cluster, states, n_counties_region, hours_share_2011_2025, fisher_info_share_hours, estimate),
    digits = 4L
  )
)

report_path <- path_outputs("reports", "bite_design.md")
writeLines(report_lines, report_path)

output_rel_paths <- c(
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

output_sha256 <- setNames(
  lapply(output_rel_paths, function(rel) digest::digest(file = path_root(rel), algo = "sha256")),
  output_rel_paths
)

key_row <- function(df, id) {
  r <- df %>% filter(spec_id == !!id)
  stopifnot(nrow(r) == 1L)
  list(
    estimate = r$estimate[[1]],
    se_crv1 = r$se_crv1[[1]],
    se_cv3j = r$se_cv3j[[1]],
    p_cv3j = r$p_cv3j[[1]],
    mde_cv3j = r$mde_cv3j[[1]],
    loo_min = r$loo_min[[1]],
    loo_max = r$loo_max[[1]],
    n_obs = r$n_obs[[1]],
    n_counties = r$n_counties[[1]],
    n_clusters = r$n_clusters[[1]]
  )
}

provenance <- list(
  design = "bite_full_panel_ppml_v1",
  generated_at_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  inputs = list(
    county_year_panel_parquet = list(
      path = "data/processed/county_year_panel.parquet",
      sha256 = shared_panel_sha256
    ),
    bite_county_year_panel_parquet = list(
      path = "data/processed/bite_county_year_panel.parquet",
      sha256 = bite_panel_sha256
    )
  ),
  panel_counts = list(
    full_rows = nrow(panel_full),
    full_counties = n_distinct(panel_full$county_fips),
    full_years = c(min(panel_full$year), max(panel_full$year)),
    estimation_years = c(BITE_ESTIMATION_START_YEAR, BITE_ESTIMATION_END_YEAR),
    aewr_regions = n_distinct(panel_full$aewr_region_id),
    states = n_distinct(panel_full$state_fips)
  ),
  variance_decomposition = list(
    share_1a_of_1b_unweighted = share_1a_unw,
    share_1a_of_1b_ppml_hours_weighted = share_1a_wt,
    share_within_sy_unweighted = var_decomp$share_within_SY_of_1b[[1]],
    share_within_sy_ppml_hours_weighted = var_decomp$share_within_SY_of_1b[[2]]
  ),
  cluster_concentration = list(
    g_eff_fisher_hours = lev_h$g_eff_fisher,
    ca_fisher_share_hours = lev_h$ca_fisher_share,
    g_eff_fisher_workers = lev_w$g_eff_fisher,
    ca_fisher_share_workers = lev_w$ca_fisher_share
  ),
  key_estimates = list(
    primary_1b_hours = key_row(main_estimates, "main_p25_hours"),
    primary_1b_workers = key_row(main_estimates, "main_p25_workers"),
    primary_any_ge5 = key_row(main_estimates, "main_p25_any_ge5"),
    primary_implied_hours_per_pos = key_row(main_estimates, "main_p25_hours_per_pos"),
    state_year_hours = key_row(main_estimates, "state_year_p25_hours"),
    xpre_hours = key_row(main_estimates, "xpre_p25_hours"),
    oews_hours = key_row(main_estimates, "oews_hours"),
    sf_log_hours = key_row(main_estimates, "sf_log_hours"),
    spec1a_hours = key_row(main_estimates, "spec1a_p25_hours"),
    domestic_floor_hours = key_row(diagnostics_df, "domestic_floor_hours"),
    domestic_floor_workers = key_row(diagnostics_df, "domestic_floor_workers")
  ),
  outputs_sha256 = output_sha256
)

write_json(
  provenance,
  path_outputs("reports", "bite_provenance.json"),
  pretty = TRUE,
  auto_unbox = TRUE
)

cat(
  sprintf(
    "Estimated full-panel PPML bite design: %d main specs, %d diagnostic rows, %d event-study rows, %d LOO region rows.\n",
    nrow(main_estimates),
    nrow(diagnostics_df),
    nrow(event_study_df),
    nrow(loo_regions_tbl)
  )
)
