# Purpose: Compare real-wage TWFE first stages across cluster and donor counts.
# Input: data/processed/county_df_analysis_year_iv.parquet.
# Outputs: first-stage strength CSV/figure and printed strength summary.
# Run after: code/c03_iv/10_attach_instruments_to_panel.R.
rm(list = ls())
source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
library(arrow)
library(tidyverse)
library(janitor)
library(fixest)

county_df_iv <- read_parquet(path_processed(
  "county_df_analysis_year_iv.parquet"
)) %>%
  clean_names()

iv_design_specs <- crossing(
  iv_k = 2:5,
  donor_cluster_count = 1:4
) %>%
  filter(donor_cluster_count < iv_k)

fs_specs <- iv_design_specs %>%
  mutate(
    outcome_type = "real wage level",
    outcome = "aewr_ppi",
    design = "levels TWFE",
    fe_spec = "CZ-AEWR region + year",
    fe_terms = "cz_aewr_region_fe + year_fe",
    source = paste0(
      "OEWS wage-only, 100% gap closure, k = ",
      iv_k,
      ", furthest ",
      donor_cluster_count,
      " donor cluster",
      if_else(donor_cluster_count == 1, "", "s")
    ),
    instrument = paste0(
      "z_oews_entropy_agwage_l1_k",
      iv_k,
      "_d",
      donor_cluster_count,
      "_g100"
    ),
    model_name = paste(
      "oews_entropy",
      paste0("k", iv_k),
      paste0("d", donor_cluster_count),
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
cat("  Outcome: real AEWR_t (AEWR deflated by the PPI)\n")
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
  fs_strength,
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
    title = "Real-wage TWFE first stages by donor-set size",
    subtitle = "CZ–AEWR-region and year fixed effects; dashed line marks F = 10"
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
  width = 7.5,
  height = 4.8,
  dpi = 300
)

cat(
  "\nEstimated",
  length(fs_models),
  "first-stage regressions; complete coefficients and diagnostics are in",
  path_tables("iv_first_stage_strength.csv"),
  "\n"
)
