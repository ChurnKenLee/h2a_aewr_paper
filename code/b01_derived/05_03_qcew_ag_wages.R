# Purpose: Estimate state agricultural wage proxies from QCEW industry totals.
# Inputs: data/intermediate/qcew.parquet.
# Outputs: qcew_state_ag_wage.parquet and industry diagnostics.

if (!exists("path_code", mode = "function")) {
  source(
    if (file.exists(file.path("code", "bootstrap_paths.R"))) {
      file.path("code", "bootstrap_paths.R")
    } else {
      file.path("..", "bootstrap_paths.R")
    }
  )
}
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)

# QCEW has industry wages, not occupation wages, and no hours measure. Use crop
# and animal production as the FLS-like core, with support activities reported as
# a diagnostic extension.
qcew_assumed_hours_per_week <- 40
qcew_ag_core_industry_codes <- c("111", "112")
qcew_ag_support_industry_codes <- c("1151", "1152")

qcew_state_ag_industry <- open_dataset(path_int("qcew.parquet")) %>%
  select(
    area_fips,
    own_code,
    industry_code,
    agglvl_code,
    year,
    disclosure_code,
    annual_avg_emplvl,
    total_annual_wages
  ) %>%
  filter(
    own_code == "5",
    disclosure_code == "",
    (agglvl_code == "55" & industry_code %in% qcew_ag_core_industry_codes) |
      (agglvl_code == "56" &
        industry_code %in% qcew_ag_support_industry_codes),
    str_sub(area_fips, 3, 5) == "000"
  ) %>%
  collect() %>%
  mutate(
    state_fips_code = str_sub(area_fips, 1, 2),
    qcew_ag_industry_component = case_when(
      industry_code == "111" ~ "crop_production",
      industry_code == "112" ~ "animal_production",
      industry_code == "1151" ~ "support_crop_production",
      industry_code == "1152" ~ "support_animal_production",
      .default = "excluded"
    ),
    qcew_fls_core_industry = industry_code %in% qcew_ag_core_industry_codes,
    qcew_ag_support_industry = industry_code %in%
      qcew_ag_support_industry_codes
  )

qcew_state_ag_wage <- qcew_state_ag_industry %>%
  group_by(year, state_fips_code) %>%
  summarise(
    qcew_ag_workers = sum(
      annual_avg_emplvl[qcew_fls_core_industry],
      na.rm = TRUE
    ),
    qcew_ag_total_annual_wages = sum(
      total_annual_wages[qcew_fls_core_industry],
      na.rm = TRUE
    ),
    qcew_ag_crop_workers = sum(
      annual_avg_emplvl[industry_code == "111"],
      na.rm = TRUE
    ),
    qcew_ag_animal_workers = sum(
      annual_avg_emplvl[industry_code == "112"],
      na.rm = TRUE
    ),
    qcew_ag_support_workers = sum(
      annual_avg_emplvl[qcew_ag_support_industry],
      na.rm = TRUE
    ),
    qcew_ag_support_total_annual_wages = sum(
      total_annual_wages[qcew_ag_support_industry],
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  mutate(
    qcew_ag_support_incl_workers = qcew_ag_workers +
      qcew_ag_support_workers,
    qcew_ag_support_incl_total_annual_wages = qcew_ag_total_annual_wages +
      qcew_ag_support_total_annual_wages,
    qcew_ag_mean_annual_pay = if_else(
      qcew_ag_workers > 0,
      qcew_ag_total_annual_wages / qcew_ag_workers,
      NA_real_
    ),
    qcew_ag_mean_weekly_wage = qcew_ag_mean_annual_pay / 52,
    qcew_ag_mean_hourly_wage_40h = qcew_ag_mean_annual_pay /
      (52 * qcew_assumed_hours_per_week),
    qcew_ag_support_incl_mean_annual_pay = if_else(
      qcew_ag_support_incl_workers > 0,
      qcew_ag_support_incl_total_annual_wages /
        qcew_ag_support_incl_workers,
      NA_real_
    ),
    qcew_ag_support_incl_mean_weekly_wage = qcew_ag_support_incl_mean_annual_pay /
      52,
    qcew_ag_support_incl_mean_hourly_wage_40h = qcew_ag_support_incl_mean_annual_pay /
      (52 * qcew_assumed_hours_per_week),
    qcew_ag_support_worker_share = if_else(
      qcew_ag_support_incl_workers > 0,
      qcew_ag_support_workers / qcew_ag_support_incl_workers,
      NA_real_
    ),
    qcew_assumed_hours_per_week = qcew_assumed_hours_per_week
  ) %>%
  arrange(year, state_fips_code)

qcew_ag_wage_industry_diagnostics <- qcew_state_ag_industry %>%
  group_by(year, industry_code, qcew_ag_industry_component) %>%
  summarise(
    qcew_workers = sum(annual_avg_emplvl, na.rm = TRUE),
    qcew_total_annual_wages = sum(total_annual_wages, na.rm = TRUE),
    qcew_mean_annual_pay = if_else(
      qcew_workers > 0,
      qcew_total_annual_wages / qcew_workers,
      NA_real_
    ),
    qcew_mean_weekly_wage = qcew_mean_annual_pay / 52,
    qcew_mean_hourly_wage_40h = qcew_mean_annual_pay /
      (52 * qcew_assumed_hours_per_week),
    .groups = "drop"
  ) %>%
  arrange(year, industry_code)

write_parquet(
  qcew_state_ag_wage,
  path_int("qcew_state_ag_wage.parquet")
)
write_parquet(
  qcew_ag_wage_industry_diagnostics,
  path_int("qcew_ag_wage_industry_diagnostics.parquet")
)
