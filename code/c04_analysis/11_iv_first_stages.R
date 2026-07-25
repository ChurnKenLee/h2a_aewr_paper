# Purpose: Compare real AEWR and p10-bite TWFE first stages across IV designs.
# Input: data/processed/county_df_analysis_year_iv.parquet.
# Outputs: first-stage strength CSV/figure and printed strength summary.
# Run after: code/c03_iv/10_attach_instruments_to_panel.R.
rm(list = ls())
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(tidyverse)
library(janitor)
library(fixest)

county_df_iv <- read_parquet(path_processed(
  "county_df_analysis_year_iv.parquet"
)) %>%
  clean_names()

iv_long_specs <- read_parquet(path_int(
  "iv_oews_entropy_long.parquet"
))

iv_weight_specs <- iv_long_specs %>%
  distinct(
    weight_spec_label,
    prior_spec,
    calibration_mode,
    duration_analogue,
    soft_penalty
  ) %>%
  mutate(
    weight_spec_display = case_when(
      weight_spec_label == "wage_only_exact" ~ "Wage only",
      weight_spec_label == "wage_seasonal_exact" ~
        "Wage + seasonal (exact)",
      weight_spec_label == "wage_seasonal_qwi_duration_exact" ~
        "Wage + seasonal + QWI duration (exact)",
      weight_spec_label == "wage_seasonal_census_duration_exact" ~
        "Wage + seasonal + bridged Census duration (exact)",
      weight_spec_label == "wage_seasonal_interval" ~
        "Wage + seasonal (interval)",
      str_detect(weight_spec_label, "wage_seasonal_soft") ~
        paste0("Wage + seasonal (rho = ", soft_penalty, ")"),
      str_detect(weight_spec_label, "_prior_") ~ paste0(
        "Wage + seasonal (exact; ",
        str_replace(prior_spec, "_", " "),
        " prior)"
      ),
      .default = weight_spec_label
    ),
    weight_spec_order = case_when(
      weight_spec_label == "wage_only_exact" ~ 1,
      weight_spec_label == "wage_seasonal_exact" ~ 2,
      weight_spec_label ==
        "wage_seasonal_qwi_duration_exact" ~ 3,
      weight_spec_label ==
        "wage_seasonal_census_duration_exact" ~ 4,
      weight_spec_label == "wage_seasonal_interval" ~ 5,
      str_detect(weight_spec_label, "wage_seasonal_soft") ~
        6 + soft_penalty,
      .default = 20
    )
  ) %>%
  arrange(weight_spec_order, weight_spec_label)

# aewr_cz_p10 is constructed upstream in real 2012 dollars as
# aewr_ppi - wage_p10, where wage_p10 is the CZ prevailing-wage percentile.
iv_outcome_specs <- tribble(
  ~outcome_type,                ~outcome,
  "Real AEWR level",            "aewr_ppi",
  "Real AEWR bite (minus p10)", "aewr_cz_p10"
)

iv_design_specs <- iv_long_specs %>%
  distinct(iv_k, donor_cluster_count, weight_spec_label) %>%
  inner_join(
    iv_weight_specs,
    by = "weight_spec_label",
    relationship = "many-to-one"
  )

fs_specs <- crossing(
  iv_design_specs,
  iv_outcome_specs
) %>%
  mutate(
    design = "levels TWFE",
    fe_spec = "CZ-AEWR region + year",
    fe_terms = "cz_aewr_region_fe + year_fe",
    source = paste0(
      "OEWS ",
      weight_spec_display,
      ", 100% gap closure, k = ",
      iv_k,
      ", furthest ",
      donor_cluster_count,
      " donor cluster",
      if_else(donor_cluster_count == 1, "", "s")
    ),
    instrument = if_else(
      weight_spec_label == "wage_only_exact",
      paste0(
        "z_oews_entropy_agwage_l1_k",
        iv_k,
        "_d",
        donor_cluster_count,
        "_g100"
      ),
      paste0(
        "z_oews_entropy_agwage_l1_k",
        iv_k,
        "_d",
        donor_cluster_count,
        "_g100_",
        weight_spec_label
      )
    ),
    model_name = paste(
      "oews_entropy",
      paste0("k", iv_k),
      paste0("d", donor_cluster_count),
      weight_spec_label,
      "g100",
      outcome,
      "levels_twfe",
      sep = "_"
    )
  )

cat(
  "First-stage input:",
  path_processed(
    "county_df_analysis_year_iv.parquet"
  ),
  "\n\n"
)
cat("Instrument timing:\n")
cat("  Outcomes: real AEWR_t and real AEWR_t - CZ p10 wage_t\n")
cat("  Instrument: donor mean OEWS wage_{t-1}\n")
cat(
  "  Fixed effects: CZ-AEWR region and year\n\n"
)
cat("Clustered SE: cz_aewr_region_fe\n")
cat(
  "First-stage F below is the squared clustered t statistic for the single excluded instrument.\n\n"
)

fs_models <- list()
fs_strength <- tibble()

for (i in seq_len(nrow(fs_specs))) {
  spec <- fs_specs[i, ]
  fs_data <- county_df_iv %>%
    filter(
      any_cropland_2007 == 1,
      year > 2008,
      !is.na(.data[[spec$outcome]]),
      !is.na(.data[[spec$instrument]]),
      !is.na(aewr_region_fe),
      !is.na(year_fe),
      !is.na(cz_aewr_region_fe),
      is.finite(.data[[spec$outcome]]),
      is.finite(.data[[spec$instrument]])
    )

  fs_formula <- as.formula(paste(
    spec$outcome,
    "~",
    spec$instrument,
    "|",
    spec$fe_terms
  ))

  fs_models[[spec$model_name]] <- feols(
    fs_formula,
    data = fs_data,
    vcov = ~cz_aewr_region_fe
  )

  fs_ct <- coeftable(fs_models[[spec$model_name]])
  if (spec$instrument %in% rownames(fs_ct)) {
    fs_estimate <- fs_ct[spec$instrument, "Estimate"]
    fs_se <- fs_ct[spec$instrument, "Std. Error"]
    fs_t <- fs_ct[spec$instrument, "t value"]
    fs_p <- fs_ct[spec$instrument, "Pr(>|t|)"]
  } else {
    fs_estimate <- NA_real_
    fs_se <- NA_real_
    fs_t <- NA_real_
    fs_p <- NA_real_
  }

  fs_strength <- bind_rows(
    fs_strength,
    spec %>%
      select(
        model_name,
        design,
        source,
        weight_spec_label,
        weight_spec_display,
        soft_penalty,
        iv_k,
        donor_cluster_count,
        outcome_type,
        outcome,
        instrument,
        fe_spec
      ) %>%
      mutate(
        n = nobs(fs_models[[spec$model_name]]),
        cz_aewr_clusters = n_distinct(fs_data$cz_aewr_region_fe),
        estimate = fs_estimate,
        se = fs_se,
        t_stat = fs_t,
        first_stage_f = fs_t^2,
        p_value = fs_p,
        r2 = as.numeric(r2(fs_models[[spec$model_name]], "r2")),
        within_r2 = as.numeric(r2(fs_models[[spec$model_name]], "wr2"))
      )
  )
}

fs_strength_print <- fs_strength %>%
  mutate(
    across(
      c(estimate, se, t_stat, first_stage_f, p_value, r2, within_r2),
      ~ round(.x, 4)
    )
  ) %>%
  arrange(design, outcome_type, iv_k, donor_cluster_count)

cat("=== First-stage strength summary ===\n")
print(fs_strength_print, n = Inf, width = Inf)

dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)
dir.create(path_figures(), recursive = TRUE, showWarnings = FALSE)
write_csv(
  fs_strength,
  path_tables("iv_first_stage_strength.csv")
)

fs_strength_figure <- ggplot(
  fs_strength %>%
    mutate(
      weight_spec_display = factor(
        weight_spec_display,
        levels = iv_weight_specs$weight_spec_display
      ),
      outcome_type = factor(
        outcome_type,
        levels = iv_outcome_specs$outcome_type
      )
    ),
  aes(
    x = donor_cluster_count,
    y = first_stage_f,
    color = factor(iv_k),
    group = iv_k
  )
) +
  geom_hline(
    yintercept = 10,
    color = "grey55",
    linetype = "dashed",
    linewidth = 0.5
  ) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.8) +
  facet_grid(
    rows = vars(outcome_type),
    cols = vars(weight_spec_display),
    scales = "free_y"
  ) +
  scale_x_continuous(
    breaks = seq_len(max(iv_design_specs$donor_cluster_count))
  ) +
  scale_color_manual(
    values = c(
      `2` = "#0072B2",
      `3` = "#D55E00",
      `4` = "#009E73",
      `5` = "#CC79A7"
    )
  ) +
  labs(
    x = "Number of furthest donor clusters used",
    y = "First-stage F statistic",
    color = "Clusters per AEWR\nregion (k)",
    title = "TWFE first stages for real AEWR and its p10 bite",
    subtitle = paste0(
      "CZ–AEWR-region and year fixed effects; all weights target 100% of ",
      "the wage gap; dashed line marks F = 10"
    )
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom"
  )

ggsave(
  filename = path_figures(
    "fig_iv_first_stage_strength_by_cluster_count.png"
  ),
  plot = fs_strength_figure,
  width = 14,
  height = 7.5,
  dpi = 300
)

cat(
  "\nEstimated",
  length(fs_models),
  "first-stage regressions; complete coefficients and diagnostics are in",
  path_tables("iv_first_stage_strength.csv"),
  "\n"
)
