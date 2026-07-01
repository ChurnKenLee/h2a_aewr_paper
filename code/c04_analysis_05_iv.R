# Test CZ-level AEWR IV first stages
rm(list = ls())
if (!exists("path_code", mode = "function")) {
  source(file.path("code", "paths.R"))
}
library(arrow)
library(tidyverse)
library(janitor)
library(fixest)

county_iv_path <- path_processed("county_df_analysis_year_iv.parquet")
if (!file.exists(county_iv_path)) {
  county_iv_path <- path_processed("county_df_analysis_year_iv_me.parquet")
}
if (!file.exists(county_iv_path)) {
  stop(
    "Missing final IV analysis parquet: ",
    county_iv_path,
    ". Run code/c03_build_06_dissimilarity_iv.R first."
  )
}

county_df_iv <- read_parquet(county_iv_path) %>%
  clean_names() %>%
  mutate(
    d_aewr = aewr - aewr_l1,
    d_aewr_ppi = aewr_ppi - aewr_ppi_l1
  )

unit_iv_changes <- county_df_iv %>%
  distinct(
    cz_aewr_region_fe,
    year,
    z_oews_agwage_l1,
    z_qcew_agwage_l1
  ) %>%
  arrange(cz_aewr_region_fe, year) %>%
  group_by(cz_aewr_region_fe) %>%
  mutate(
    year_l1 = lag(year),
    z_oews_d_agwage_l1 = if_else(
      year_l1 == year - 1,
      z_oews_agwage_l1 - lag(z_oews_agwage_l1),
      NA_real_
    ),
    z_qcew_d_agwage_l1 = if_else(
      year_l1 == year - 1,
      z_qcew_agwage_l1 - lag(z_qcew_agwage_l1),
      NA_real_
    )
  ) %>%
  ungroup() %>%
  select(
    cz_aewr_region_fe,
    year,
    z_oews_d_agwage_l1,
    z_qcew_d_agwage_l1
  )

county_df_iv <- county_df_iv %>%
  left_join(unit_iv_changes, by = c("cz_aewr_region_fe", "year"))

fs_specs <- tribble(
  ~model_name                     , ~source , ~outcome_type    , ~outcome     , ~instrument          , ~fe_spec                , ~fe_terms                     ,
  "oews_change_region_year"       , "OEWS"  , "nominal change" , "d_aewr"     , "z_oews_d_agwage_l1" , "AEWR region + year"    , "aewr_region_fe + year_fe"    ,
  "qcew_change_region_year"       , "QCEW"  , "nominal change" , "d_aewr"     , "z_qcew_d_agwage_l1" , "AEWR region + year"    , "aewr_region_fe + year_fe"    ,
  "oews_change_czregion_year"     , "OEWS"  , "nominal change" , "d_aewr"     , "z_oews_d_agwage_l1" , "CZ-AEWR region + year" , "cz_aewr_region_fe + year_fe" ,
  "qcew_change_czregion_year"     , "QCEW"  , "nominal change" , "d_aewr"     , "z_qcew_d_agwage_l1" , "CZ-AEWR region + year" , "cz_aewr_region_fe + year_fe" ,
  "oews_change_ppi_region_year"   , "OEWS"  , "real change"    , "d_aewr_ppi" , "z_oews_d_agwage_l1" , "AEWR region + year"    , "aewr_region_fe + year_fe"    ,
  "qcew_change_ppi_region_year"   , "QCEW"  , "real change"    , "d_aewr_ppi" , "z_qcew_d_agwage_l1" , "AEWR region + year"    , "aewr_region_fe + year_fe"    ,
  "oews_change_ppi_czregion_year" , "OEWS"  , "real change"    , "d_aewr_ppi" , "z_oews_d_agwage_l1" , "CZ-AEWR region + year" , "cz_aewr_region_fe + year_fe" ,
  "qcew_change_ppi_czregion_year" , "QCEW"  , "real change"    , "d_aewr_ppi" , "z_qcew_d_agwage_l1" , "CZ-AEWR region + year" , "cz_aewr_region_fe + year_fe"
)

cat("First-stage input:", county_iv_path, "\n\n")
cat("Instrument timing:\n")
cat("Outcome d_aewr in year t: AEWR_t - AEWR_{t-1}\n")
cat("Outcome d_aewr_ppi in year t: AEWR_PPI_t - AEWR_PPI_{t-1}\n")
cat(
  "Instrument z_*_d_agwage_l1 in year t: donor mean proxy_{t-1} - donor mean proxy_{t-2}\n\n"
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
  arrange(outcome_type, fe_spec, source)

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

cat("\nWrote:", path_tables("iv_first_stage_strength.csv"), "\n")
