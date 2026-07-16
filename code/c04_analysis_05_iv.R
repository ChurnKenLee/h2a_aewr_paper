# Test CZ-level AEWR IV first stages
rm(list = ls())
if (!exists("path_code", mode = "function")) {
  source(file.path("code", "paths.R"))
}
library(arrow)
library(tidyverse)
library(janitor)
library(fixest)

county_df_iv <- read_parquet(path_processed(
  "county_df_analysis_year_iv.parquet"
)) %>%
  clean_names() %>%
  mutate(
    d_aewr = aewr - aewr_l1,
    d_aewr_ppi = aewr_ppi - aewr_ppi_l1
  )

gap_closure_specs <- tribble(
  ~gap_closure , ~gap_closure_label ,
  0.00         , "g000"             ,
  0.25         , "g025"             ,
  0.50         , "g050"             ,
  0.75         , "g075"             ,
  1.00         , "g100"
)

unit_iv_changes <- county_df_iv %>%
  select(
    cz_aewr_region_fe,
    year,
    starts_with("z_oews_entropy_agwage_l1_g")
  ) %>%
  distinct(cz_aewr_region_fe, year, .keep_all = TRUE) %>%
  pivot_longer(
    cols = starts_with("z_oews_entropy_agwage_l1_g"),
    names_to = "gap_closure_label",
    names_pattern = "z_oews_entropy_agwage_l1_(g[0-9]+)",
    values_to = "z_oews_entropy_agwage_l1"
  ) %>%
  arrange(cz_aewr_region_fe, gap_closure_label, year) %>%
  group_by(cz_aewr_region_fe, gap_closure_label) %>%
  mutate(
    year_l1 = lag(year),
    z_oews_entropy_d_agwage_l1 = if_else(
      year_l1 == year - 1,
      z_oews_entropy_agwage_l1 - lag(z_oews_entropy_agwage_l1),
      NA_real_
    )
  ) %>%
  ungroup() %>%
  select(
    cz_aewr_region_fe,
    year,
    gap_closure_label,
    z_oews_entropy_d_agwage_l1
  ) %>%
  pivot_wider(
    names_from = gap_closure_label,
    values_from = z_oews_entropy_d_agwage_l1,
    names_glue = "z_oews_entropy_d_agwage_l1_{gap_closure_label}"
  )

county_df_iv <- county_df_iv %>%
  left_join(unit_iv_changes, by = c("cz_aewr_region_fe", "year"))

fs_specs <- crossing(
  gap_closure_specs,
  tribble(
    ~outcome_type    , ~outcome     ,
    "nominal change" , "d_aewr"     ,
    "real change"    , "d_aewr_ppi"
  ),
  tribble(
    ~fe_spec                , ~fe_terms                     , ~fe_label       ,
    "AEWR region + year"    , "aewr_region_fe + year_fe"    , "region_year"   ,
    "CZ-AEWR region + year" , "cz_aewr_region_fe + year_fe" , "czregion_year"
  )
) %>%
  mutate(
    source = paste0(
      "OEWS entropy, ",
      as.integer(100 * gap_closure),
      "% gap closure"
    ),
    instrument = paste0(
      "z_oews_entropy_d_agwage_l1_",
      gap_closure_label
    ),
    model_name = paste(
      "oews_entropy",
      gap_closure_label,
      outcome,
      fe_label,
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
cat("Outcome d_aewr in year t: AEWR_t - AEWR_{t-1}\n")
cat("Outcome d_aewr_ppi in year t: AEWR_PPI_t - AEWR_PPI_{t-1}\n")
cat(
  "Instrument z_oews_entropy_d_agwage_l1_* in year t: donor mean OEWS wage_{t-1} - donor mean OEWS wage_{t-2}\n\n"
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
        source,
        gap_closure,
        gap_closure_label,
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
  arrange(outcome_type, fe_spec, gap_closure)

cat("=== First-stage strength summary ===\n")
print(fs_strength_print, n = Inf, width = Inf)

dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)
write_csv(
  fs_strength,
  path_tables("iv_first_stage_strength.csv")
)

cat("\n=== First-stage regressions: simple-change IVs ===\n")
print(
  etable(
    fs_models,
    fitstat = ~ n + r2 + wr2
  )
)
