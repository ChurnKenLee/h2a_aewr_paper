# Purpose: Construct primary and alternative county priors for FLS calibration.
# Inputs: county panel, Census labor, QWI employment, and county CDL acreage.
# Output: data/intermediate/fls_county_weight.parquet.
# Run after: code/c02_build/04_finalize_county_panel.R and source extractors.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)

# The primary prior is BEA farm employment. Census hired-worker counts,
# Census hired-labor payroll, and QWI agricultural employment are separate
# sensitivity priors. They are never blended, and H-2A records are excluded
# from all prior construction.

interpolate_with_nearest_endpoint <- function(year, value) {
  observed <- is.finite(year) & is.finite(value) & value >= 0
  if (!any(observed)) {
    return(rep(NA_real_, length(year)))
  }
  if (sum(observed) == 1L) {
    return(rep(value[observed][[1]], length(year)))
  }
  approx(
    x = year[observed],
    y = value[observed],
    xout = year,
    method = "linear",
    rule = 2,
    ties = "ordered"
  )$y
}

county_year <- read_parquet(
  path_processed("county_df_analysis_year.parquet")
) %>%
  select(
    countyfips,
    year,
    statefips,
    state_abbrev,
    aewr_region_num,
    cz_out10,
    cz_aewr_region_fe,
    emp_farm
  ) %>%
  rename(bea_farm_emp = emp_farm) %>%
  distinct() %>%
  filter(!is.na(aewr_region_num))

census_prior_path <- path_int(
  "census_ag_hired_worker_duration_county.parquet"
)
census_county_prior <- if (file.exists(census_prior_path)) {
  census_source <- read_parquet(census_prior_path)
  if (!"census_hired_labor_expense" %in% names(census_source)) {
    census_source$census_hired_labor_expense <- NA_real_
  }
  census_source %>%
    transmute(
      countyfips = as.character(countyfips),
      year = as.integer(year),
      census_hired_workers_total = as.numeric(
        census_hired_workers_total
      ),
      census_hired_labor_expense = as.numeric(
        census_hired_labor_expense
      )
    )
} else {
  tibble(
    countyfips = character(),
    year = integer(),
    census_hired_workers_total = numeric(),
    census_hired_labor_expense = numeric()
  )
}

qwi_prior_path <- path_int(
  "qwi_county_ag_quarterly_employment.parquet"
)
qwi_county_prior <- if (file.exists(qwi_prior_path)) {
  read_parquet(qwi_prior_path) %>%
    mutate(
      countyfips = as.character(countyfips),
      year = as.integer(year)
    ) %>%
    group_by(countyfips, year, qtr) %>%
    summarise(
      qwi_quarter_ag_employment = if_else(
        all(is.na(qwi_beginning_quarter_employment)),
        NA_real_,
        sum(qwi_beginning_quarter_employment, na.rm = TRUE)
      ),
      .groups = "drop"
    ) %>%
    group_by(countyfips, year) %>%
    summarise(
      qwi_annual_employment = if_else(
        all(is.na(qwi_quarter_ag_employment)),
        NA_real_,
        mean(qwi_quarter_ag_employment, na.rm = TRUE)
      ),
      .groups = "drop"
    )
} else {
  tibble(
    countyfips = character(),
    year = integer(),
    qwi_annual_employment = numeric()
  )
}

# Count acreage of crop land within county. CDL remains in the output for
# diagnostics but is no longer blended into the primary prior.
cdl_ag_land <- read_parquet(path_int(
  "croplandcros_county_crop_type_acres.parquet"
))
crop_type_list <- c(
  "field crops",
  "fruit & tree nuts",
  "horticulture",
  "mixed crops",
  "mixed field crops/horticulture",
  "mixed field crops/horticulture/vegetables",
  "mixed field crops/vegetables",
  "mixed fruit & tree nuts/vegetables",
  "vegetables"
)
county_crop_acreage <- cdl_ag_land %>%
  mutate(cdl_is_ag = crop_type_label %in% crop_type_list) %>%
  group_by(countyfips, year) %>%
  summarise(
    cdl_ag_acres = sum(if_else(cdl_is_ag, acres, 0), na.rm = TRUE),
    cdl_non_ag_acres = sum(if_else(!cdl_is_ag, acres, 0), na.rm = TRUE),
    .groups = "drop"
  )

fls_county_weight <- county_year %>%
  left_join(county_crop_acreage, by = c("countyfips", "year")) %>%
  left_join(census_county_prior, by = c("countyfips", "year")) %>%
  group_by(countyfips) %>%
  arrange(year, .by_group = TRUE) %>%
  mutate(
    census_hired_workers_prior = interpolate_with_nearest_endpoint(
      year,
      census_hired_workers_total
    ),
    census_hired_payroll_prior = interpolate_with_nearest_endpoint(
      year,
      census_hired_labor_expense
    )
  ) %>%
  ungroup() %>%
  left_join(qwi_county_prior, by = c("countyfips", "year")) %>%
  mutate(
    bea_farm_emp_weight_base = if_else(
      is.finite(bea_farm_emp) & bea_farm_emp > 0,
      bea_farm_emp,
      0
    ),
    cdl_ag_acres_weight_base = if_else(
      is.finite(cdl_ag_acres) & cdl_ag_acres > 0,
      cdl_ag_acres,
      0
    ),
    census_workers_weight_base = if_else(
      is.finite(census_hired_workers_prior) &
        census_hired_workers_prior > 0,
      census_hired_workers_prior,
      0
    ),
    census_payroll_weight_base = if_else(
      is.finite(census_hired_payroll_prior) &
        census_hired_payroll_prior > 0,
      census_hired_payroll_prior,
      0
    ),
    qwi_employment_weight_base = if_else(
      is.finite(qwi_annual_employment) &
        qwi_annual_employment > 0,
      qwi_annual_employment,
      0
    )
  ) %>%
  group_by(aewr_region_num, year) %>%
  mutate(
    aewr_region_bea_farm_emp = sum(
      bea_farm_emp_weight_base,
      na.rm = TRUE
    ),
    aewr_region_cdl_ag_acres = sum(
      cdl_ag_acres_weight_base,
      na.rm = TRUE
    ),
    aewr_region_census_workers = sum(
      census_workers_weight_base,
      na.rm = TRUE
    ),
    aewr_region_census_payroll = sum(
      census_payroll_weight_base,
      na.rm = TRUE
    ),
    aewr_region_qwi_employment = sum(
      qwi_employment_weight_base,
      na.rm = TRUE
    )
  ) %>%
  ungroup() %>%
  mutate(
    fls_emp_share = bea_farm_emp_weight_base /
      aewr_region_bea_farm_emp,
    fls_area_share = cdl_ag_acres_weight_base /
      aewr_region_cdl_ag_acres,
    fls_county_weight_bea = fls_emp_share,
    fls_county_weight_census_workers = if_else(
      aewr_region_census_workers > 0,
      census_workers_weight_base / aewr_region_census_workers,
      NA_real_
    ),
    fls_county_weight_census_payroll = if_else(
      aewr_region_census_payroll > 0,
      census_payroll_weight_base / aewr_region_census_payroll,
      NA_real_
    ),
    fls_county_weight_qwi_employment = if_else(
      aewr_region_qwi_employment > 0,
      qwi_employment_weight_base / aewr_region_qwi_employment,
      NA_real_
    ),
    # Backward-compatible aliases used by the wage-only benchmark.
    fls_weight_raw = fls_county_weight_bea,
    fls_county_weight = fls_county_weight_bea
  ) %>%
  arrange(countyfips, year) %>%
  group_by(countyfips) %>%
  mutate(
    fls_county_weight_l1 = if_else(
      lag(year) == year - 1L,
      lag(fls_county_weight),
      NA_real_
    )
  ) %>%
  ungroup() %>%
  select(
    countyfips,
    year,
    statefips,
    state_abbrev,
    aewr_region_num,
    cz_out10,
    cz_aewr_region_fe,
    bea_farm_emp,
    cdl_ag_acres,
    cdl_non_ag_acres,
    census_hired_workers_prior,
    census_hired_payroll_prior,
    qwi_annual_employment,
    aewr_region_bea_farm_emp,
    aewr_region_cdl_ag_acres,
    fls_emp_share,
    fls_area_share,
    fls_weight_raw,
    fls_county_weight_bea,
    fls_county_weight_census_workers,
    fls_county_weight_census_payroll,
    fls_county_weight_qwi_employment,
    fls_county_weight,
    fls_county_weight_l1
  )

write_parquet(
  fls_county_weight,
  path_int("fls_county_weight.parquet")
)
