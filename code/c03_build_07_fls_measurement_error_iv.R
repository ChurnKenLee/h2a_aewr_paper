# Construct AEWR change IV from FLS preliminary-to-revised measurement error
rm(list = ls())
if (!exists("path_code", mode = "function")) {
  source(file.path("code", "paths.R"))
}
ensure_project_dirs()

library(arrow)
library(dplyr)
library(readr)

fls_region <- read_parquet(path_int("fls_region.parquet")) %>%
  mutate(
    estimate_year = as.integer(estimate_year),
    aewr_region_num = as.integer(aewr_region_num),
    revised_year = as.integer(revised_year),
    preliminary_year = as.integer(preliminary_year)
  )

required_columns <- c(
  "estimate_year",
  "aewr_region_num",
  "region_name",
  "revised_year",
  "preliminary_year",
  "all_hired_revised",
  "all_hired_preliminary",
  "field_revised",
  "field_preliminary",
  "field_livestock_revised",
  "field_livestock_preliminary",
  "source_zip",
  "source_csv",
  "table_title"
)

missing_columns <- setdiff(required_columns, names(fls_region))
if (length(missing_columns) > 0) {
  stop(
    "fls_region.parquet is missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

fls_preliminary <- fls_region %>%
  transmute(
    fls_wage_year = preliminary_year,
    aewr_region_num,
    region_name,
    fls_all_hired_preliminary = all_hired_preliminary,
    fls_field_preliminary = field_preliminary,
    fls_field_livestock_preliminary = field_livestock_preliminary,
    fls_preliminary_source_zip = source_zip,
    fls_preliminary_source_csv = source_csv,
    fls_preliminary_table_title = table_title
  ) %>%
  filter(!is.na(fls_wage_year), !is.na(aewr_region_num))

fls_revised <- fls_region %>%
  transmute(
    fls_wage_year = revised_year,
    aewr_region_num,
    fls_all_hired_revised = all_hired_revised,
    fls_field_revised = field_revised,
    fls_field_livestock_revised = field_livestock_revised,
    fls_revised_source_zip = source_zip,
    fls_revised_source_csv = source_csv,
    fls_revised_table_title = table_title
  ) %>%
  filter(!is.na(fls_wage_year), !is.na(aewr_region_num))

fls_revision_region <- fls_preliminary %>%
  inner_join(fls_revised, by = c("fls_wage_year", "aewr_region_num")) %>%
  mutate(
    # FLS wage year s sets the AEWR in year s + 1.
    aewr_year = fls_wage_year + 1L,
    z_fls_all_hired_me = fls_all_hired_preliminary - fls_all_hired_revised,
    z_fls_field_me = fls_field_preliminary - fls_field_revised,
    z_fls_aewr_me = fls_field_livestock_preliminary -
      fls_field_livestock_revised,
    z_fls_ln_all_hired_me = log(fls_all_hired_preliminary) -
      log(fls_all_hired_revised),
    z_fls_ln_field_me = log(fls_field_preliminary) - log(fls_field_revised),
    z_fls_ln_aewr_me = log(fls_field_livestock_preliminary) -
      log(fls_field_livestock_revised),
    z_fls_pct_all_hired_me = fls_all_hired_preliminary /
      fls_all_hired_revised -
      1,
    z_fls_pct_field_me = fls_field_preliminary / fls_field_revised - 1,
    z_fls_pct_aewr_me = fls_field_livestock_preliminary /
      fls_field_livestock_revised -
      1
  ) %>%
  arrange(aewr_region_num, fls_wage_year) %>%
  group_by(aewr_region_num) %>%
  mutate(
    fls_wage_year_l1 = lag(fls_wage_year),
    fls_field_livestock_preliminary_l1 = lag(fls_field_livestock_preliminary),
    fls_field_livestock_revised_l1 = lag(fls_field_livestock_revised),
    z_fls_aewr_me_l1 = lag(z_fls_aewr_me),
    z_fls_ln_aewr_me_l1 = lag(z_fls_ln_aewr_me),
    fls_d_preliminary_aewr = if_else(
      fls_wage_year_l1 == fls_wage_year - 1L,
      fls_field_livestock_preliminary -
        fls_field_livestock_preliminary_l1,
      NA_real_
    ),
    fls_d_revised_aewr = if_else(
      fls_wage_year_l1 == fls_wage_year - 1L,
      fls_field_livestock_revised - fls_field_livestock_revised_l1,
      NA_real_
    ),
    fls_dln_preliminary_aewr = if_else(
      fls_wage_year_l1 == fls_wage_year - 1L,
      log(fls_field_livestock_preliminary) -
        log(fls_field_livestock_preliminary_l1),
      NA_real_
    ),
    fls_dln_revised_aewr = if_else(
      fls_wage_year_l1 == fls_wage_year - 1L,
      log(fls_field_livestock_revised) -
        log(fls_field_livestock_revised_l1),
      NA_real_
    ),
    # For AEWR changes, use the change in FLS measurement error between
    # wage years t - 1 and t - 2.
    z_fls_d_aewr_me = if_else(
      fls_wage_year_l1 == fls_wage_year - 1L,
      z_fls_aewr_me - z_fls_aewr_me_l1,
      NA_real_
    ),
    z_fls_dln_aewr_me = if_else(
      fls_wage_year_l1 == fls_wage_year - 1L,
      z_fls_ln_aewr_me - z_fls_ln_aewr_me_l1,
      NA_real_
    ),
    fls_delayed_2020_release = fls_wage_year == 2020L,
    fls_delayed_2020_release_in_change = fls_wage_year == 2020L |
      fls_wage_year_l1 == 2020L,
    aewr_2009_oes_exception = aewr_year == 2009L,
    aewr_change_oes_exception = aewr_year %in% c(2009L, 2010L)
  ) %>%
  ungroup()

if (file.exists(path_processed("aewr_data_full.parquet"))) {
  aewr_region <- read_parquet(path_processed("aewr_data_full.parquet")) %>%
    transmute(
      aewr_year = as.integer(year),
      aewr_region_num = as.integer(aewr_region_num),
      official_aewr = as.numeric(aewr),
      official_aewr_l1 = as.numeric(aewr_l1),
      official_d_aewr = official_aewr - official_aewr_l1,
      official_dln_aewr = log(official_aewr) - log(official_aewr_l1)
    )

  fls_revision_region <- fls_revision_region %>%
    left_join(aewr_region, by = c("aewr_year", "aewr_region_num")) %>%
    mutate(
      official_aewr_minus_fls_preliminary = official_aewr -
        fls_field_livestock_preliminary,
      official_dln_aewr_minus_fls_dln_preliminary = official_dln_aewr -
        fls_dln_preliminary_aewr
    )
}

fls_revision_region <- fls_revision_region %>%
  select(
    aewr_year,
    fls_wage_year,
    aewr_region_num,
    region_name,
    starts_with("z_fls_"),
    fls_d_preliminary_aewr,
    fls_d_revised_aewr,
    fls_dln_preliminary_aewr,
    fls_dln_revised_aewr,
    starts_with("fls_field_livestock"),
    starts_with("fls_all_hired"),
    starts_with("fls_field_"),
    starts_with("official_"),
    starts_with("aewr_"),
    starts_with("fls_delayed"),
    starts_with("fls_preliminary_source"),
    starts_with("fls_revised_source"),
    everything()
  ) %>%
  arrange(aewr_year, aewr_region_num)

fls_revision_summary <- fls_revision_region %>%
  summarise(
    min_aewr_year = min(aewr_year, na.rm = TRUE),
    max_aewr_year = max(aewr_year, na.rm = TRUE),
    region_years = n(),
    nonmissing_level_iv = sum(!is.na(z_fls_aewr_me)),
    nonmissing_change_iv = sum(!is.na(z_fls_dln_aewr_me)),
    min_level_iv = min(z_fls_aewr_me, na.rm = TRUE),
    max_level_iv = max(z_fls_aewr_me, na.rm = TRUE),
    mean_abs_level_iv = mean(abs(z_fls_aewr_me), na.rm = TRUE),
    min_change_iv = min(z_fls_dln_aewr_me, na.rm = TRUE),
    max_change_iv = max(z_fls_dln_aewr_me, na.rm = TRUE),
    mean_abs_change_iv = mean(abs(z_fls_dln_aewr_me), na.rm = TRUE)
  )

county_source_path <- path_int("county_df_analysis_year_iv.parquet")
if (!file.exists(county_source_path)) {
  county_source_path <- path_processed("county_df_analysis_year.parquet")
}

if (file.exists(county_source_path)) {
  county_df <- read_parquet(county_source_path) %>%
    mutate(
      year = as.integer(year),
      aewr_region_num = as.integer(aewr_region_num)
    )

  county_df_fls_iv <- county_df %>%
    left_join(
      fls_revision_region %>%
        select(
          year = aewr_year,
          aewr_region_num,
          region_name,
          starts_with("z_fls_"),
          fls_d_preliminary_aewr,
          fls_d_revised_aewr,
          fls_dln_preliminary_aewr,
          fls_dln_revised_aewr,
          starts_with("fls_delayed"),
          aewr_2009_oes_exception,
          aewr_change_oes_exception
        ),
      by = c("year", "aewr_region_num")
    )

  write_parquet(
    county_df_fls_iv,
    path_processed("county_df_analysis_year_iv_me.parquet")
  )

  cat(
    "county_df_analysis_year_fls_iv:",
    nrow(county_df_fls_iv),
    "rows,",
    ncol(county_df_fls_iv),
    "cols\n"
  )
  cat(
    "Nonmissing FLS measurement-error change IV rows:",
    sum(!is.na(county_df_fls_iv$z_fls_dln_aewr_me)),
    "\n"
  )
}

cat(
  "fls_aewr_measurement_error_iv:",
  nrow(fls_revision_region),
  "rows,",
  ncol(fls_revision_region),
  "cols\n"
)
print(fls_revision_summary)
