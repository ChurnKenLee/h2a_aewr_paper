# Purpose: Build the standalone multilevel Mundlak-Chamberlain-Wooldridge
# county-year panel.
# Input: data/processed/county_year_panel.parquet.
# Outputs:
#   data/processed/mundlak_chamberlain_county_year.parquet
#   data/intermediate/mundlak_chamberlain_metadata.rds
#   data/intermediate/mundlak_chamberlain_scaling.csv
#   data/intermediate/mundlak_chamberlain_variable_inventory.csv

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "mundlak_chamberlain", "design.R"))
source(path_code("designs", "mundlak_chamberlain", "helpers.R"))
library(arrow)
library(dplyr)
library(readr)
library(tibble)
library(tidyr)

shared_panel <- read_parquet(
  path_processed("county_year_panel.parquet")
) %>%
  mutate(
    year = as.integer(year),
    state_fips = as.integer(state_fips),
    aewr_region_id = as.integer(aewr_region_id),
    mc_market_id = mc_make_market_id(
      aewr_region_id,
      state_fips,
      cz_id
    )
  ) %>%
  filter(
    year >= min(MC_BASELINE_YEARS),
    year <= max(MC_TREATMENT_HISTORY_YEARS)
  )

required_columns <- unique(c(
  "county_fips",
  "year",
  "state_fips",
  "cz_id",
  "aewr_region_id",
  "mc_market_id",
  "aewr",
  "emp_farm",
  "emp_farm_2011",
  unname(MC_BASELINE_VARIABLES),
  "nbr_applications_start_year",
  "nbr_employers_balanced_start_year",
  "nbr_workers_requested_start_year",
  "nbr_workers_certified_start_year",
  "man_hours_certified_start_year"
))
missing_columns <- setdiff(required_columns, names(shared_panel))
if (length(missing_columns) > 0L) {
  stop(
    "Shared panel is missing required MC columns: ",
    paste(missing_columns, collapse = ", "),
    call. = FALSE
  )
}

if (anyDuplicated(shared_panel[c("county_fips", "year")]) > 0L) {
  stop("Shared panel must have unique county-year keys.", call. = FALSE)
}

state_region_contract <- shared_panel %>%
  distinct(state_fips, aewr_region_id) %>%
  count(state_fips, name = "regions")
if (any(state_region_contract$regions != 1L)) {
  stop("Every state must nest in exactly one AEWR region.", call. = FALSE)
}

regional_rate_contract <- shared_panel %>%
  filter(year >= 2010L) %>%
  group_by(aewr_region_id, year) %>%
  summarise(aewr_values = n_distinct(aewr), .groups = "drop")
if (
  nrow(regional_rate_contract) !=
    17L * length(2010:max(MC_TREATMENT_HISTORY_YEARS)) ||
    any(regional_rate_contract$aewr_values != 1L)
) {
  stop(
    "AEWR must be unique within every region-year from 2010 onward.",
    call. = FALSE
  )
}

eligible_counties <- shared_panel %>%
  filter(
    year == MC_REFERENCE_YEAR,
    is.finite(emp_farm_2011),
    emp_farm_2011 > 0,
    !is.na(mc_market_id)
  ) %>%
  distinct(
    county_fips,
    state_fips,
    aewr_region_id,
    mc_market_id,
    emp_farm_2011
  )

baseline_long <- shared_panel %>%
  semi_join(eligible_counties, by = "county_fips") %>%
  filter(year %in% MC_BASELINE_YEARS)

for (variable_name in names(MC_BASELINE_VARIABLES)) {
  source_column <- MC_BASELINE_VARIABLES[[variable_name]]
  baseline_long[[paste0("mc_raw_", variable_name)]] <-
    mc_transform_baseline(
      baseline_long[[source_column]],
      variable_name,
      baseline_long$emp_farm
    )
}

county_design <- eligible_counties
scaling_rows <- list()
inventory_rows <- list()
summary_sources <- character()
history_sources <- character()

for (variable_name in names(MC_BASELINE_VARIABLES)) {
  raw_column <- paste0("mc_raw_", variable_name)
  summary_table <- baseline_long %>%
    group_by(county_fips) %>%
    summarise(
      value_mean = mc_finite_mean(.data[[raw_column]]),
      value_trend = mc_linear_slope(.data[[raw_column]], year),
      .groups = "drop"
    )

  for (statistic in c("mean", "trend")) {
    source_name <- paste0("mc_b_", variable_name, "_", statistic)
    summary_table <- summary_table %>%
      rename(!!source_name := paste0("value_", statistic))
    county_design <- county_design %>%
      left_join(
        summary_table %>% select(county_fips, all_of(source_name)),
        by = "county_fips",
        relationship = "one-to-one"
      )
    county_design <- mc_impute_by_region(county_design, source_name)
    standardized <- mc_safe_standardize(county_design[[source_name]])
    standardized_name <- paste0(source_name, "_z")
    county_design[[standardized_name]] <- standardized$value
    summary_sources <- c(summary_sources, standardized_name)
    scaling_rows[[length(scaling_rows) + 1L]] <- tibble(
      constructed_column = standardized_name,
      source_column = MC_BASELINE_VARIABLES[[variable_name]],
      hierarchy_level = "county_input",
      center = standardized$center,
      scale = standardized$scale
    )
    inventory_rows[[length(inventory_rows) + 1L]] <- tibble(
      constructed_column = standardized_name,
      source_variable = variable_name,
      source_column = MC_BASELINE_VARIABLES[[variable_name]],
      history_year = NA_integer_,
      statistic = statistic,
      hierarchy_level = "county_input",
      role = "mundlak_source"
    )
  }

  if (variable_name %in% MC_CHAMBERLAIN_VARIABLES) {
    history_table <- baseline_long %>%
      select(county_fips, year, all_of(raw_column)) %>%
      pivot_wider(
        names_from = year,
        values_from = all_of(raw_column),
        names_prefix = "value_"
      )
    for (history_year in MC_BASELINE_YEARS) {
      value_column <- paste0("value_", history_year)
      source_name <- paste0(
        "mc_b_",
        variable_name,
        "_",
        history_year
      )
      history_piece <- history_table %>%
        transmute(
          county_fips,
          !!source_name := .data[[value_column]]
        )
      county_design <- county_design %>%
        left_join(
          history_piece,
          by = "county_fips",
          relationship = "one-to-one"
        )
      county_design <- mc_impute_by_region(county_design, source_name)
      standardized <- mc_safe_standardize(county_design[[source_name]])
      standardized_name <- paste0(source_name, "_z")
      county_design[[standardized_name]] <- standardized$value
      history_sources <- c(history_sources, standardized_name)
      scaling_rows[[length(scaling_rows) + 1L]] <- tibble(
        constructed_column = standardized_name,
        source_column = MC_BASELINE_VARIABLES[[variable_name]],
        hierarchy_level = "county_input",
        center = standardized$center,
        scale = standardized$scale
      )
      inventory_rows[[length(inventory_rows) + 1L]] <- tibble(
        constructed_column = standardized_name,
        source_variable = variable_name,
        source_column = MC_BASELINE_VARIABLES[[variable_name]],
        history_year = history_year,
        statistic = "period_value",
        hierarchy_level = "county_input",
        role = "chamberlain_source"
      )
    }
  }
}

component_scaling <- list()
summary_components <- list()
history_components <- list()

for (source_column in summary_sources) {
  stub <- sub("^mc_b_", "", sub("_z$", "", source_column))
  result <- mc_hierarchical_components(
    county_design,
    source_column,
    stub
  )
  county_design <- result$data
  summary_components[[source_column]] <- result$columns
  component_scaling[[length(component_scaling) + 1L]] <- result$scaling
  for (level in names(result$columns)) {
    variable_name <- sub("_(mean|trend)$", "", stub)
    statistic <- sub(paste0("^", variable_name, "_"), "", stub)
    inventory_rows[[length(inventory_rows) + 1L]] <- tibble(
      constructed_column = result$columns[[level]],
      source_variable = variable_name,
      source_column = source_column,
      history_year = NA_integer_,
      statistic = statistic,
      hierarchy_level = level,
      role = "mundlak_component"
    )
  }
}

for (source_column in history_sources) {
  stub <- sub("^mc_b_", "", sub("_z$", "", source_column))
  result <- mc_hierarchical_components(
    county_design,
    source_column,
    stub
  )
  county_design <- result$data
  history_components[[source_column]] <- result$columns
  component_scaling[[length(component_scaling) + 1L]] <- result$scaling
  history_year <- as.integer(sub("^.*_([0-9]{4})$", "\\1", stub))
  variable_name <- sub("_[0-9]{4}$", "", stub)
  for (level in names(result$columns)) {
    inventory_rows[[length(inventory_rows) + 1L]] <- tibble(
      constructed_column = result$columns[[level]],
      source_variable = variable_name,
      source_column = source_column,
      history_year = history_year,
      statistic = "period_value",
      hierarchy_level = level,
      role = "chamberlain_component"
    )
  }
}

scaling_table <- bind_rows(
  scaling_rows,
  bind_rows(component_scaling)
)
variable_inventory <- bind_rows(inventory_rows)

summary_component_columns <- unname(unlist(summary_components))
history_component_columns <- unname(unlist(history_components))

# Region-level AEWR histories are constructed once at the actual level of
# treatment assignment.  They enter the treatment-slope projection, while
# unrestricted region indicators saturate the random-intercept projection.
region_treatment <- shared_panel %>%
  filter(year >= 2010L) %>%
  distinct(aewr_region_id, year, aewr) %>%
  arrange(aewr_region_id, year) %>%
  group_by(aewr_region_id) %>%
  mutate(
    mc_dose_current = 100 * (log(aewr) - lag(log(aewr))),
    mc_dose_lag1 = lag(mc_dose_current),
    mc_dose_lag2 = lag(mc_dose_current, n = 2L),
    mc_dose_lead1 = lead(mc_dose_current)
  ) %>%
  ungroup()

region_history <- region_treatment %>%
  filter(year %in% MC_TREATMENT_HISTORY_YEARS) %>%
  select(aewr_region_id, year, mc_dose_current) %>%
  pivot_wider(
    names_from = year,
    values_from = mc_dose_current,
    names_prefix = "mc_r_dose_history_"
  )

region_history_terms <- setdiff(
  names(region_history),
  "aewr_region_id"
)
for (column in region_history_terms) {
  standardized <- mc_safe_standardize(region_history[[column]])
  region_history[[column]] <- standardized$value
  scaling_table <- bind_rows(
    scaling_table,
    tibble(
      constructed_column = column,
      source_column = "aewr",
      hierarchy_level = "region",
      center = standardized$center,
      scale = standardized$scale
    )
  )
  variable_inventory <- bind_rows(
    variable_inventory,
    tibble(
      constructed_column = column,
      source_variable = "aewr_change_history",
      source_column = "aewr",
      history_year = as.integer(sub("^.*_", "", column)),
      statistic = "period_value",
      hierarchy_level = "region",
      role = "treatment_slope_projection"
    )
  )
}

analysis_panel <- shared_panel %>%
  semi_join(eligible_counties, by = "county_fips") %>%
  filter(year %in% MC_ANALYSIS_YEARS) %>%
  select(
    -mc_market_id
  ) %>%
  left_join(
    county_design,
    by = c(
      "county_fips",
      "state_fips",
      "aewr_region_id",
      "emp_farm_2011"
    ),
    relationship = "many-to-one"
  ) %>%
  left_join(
    region_treatment %>%
      select(
        aewr_region_id,
        year,
        mc_dose_current,
        mc_dose_lag1,
        mc_dose_lag2,
        mc_dose_lead1
      ),
    by = c("aewr_region_id", "year"),
    relationship = "many-to-one"
  ) %>%
  left_join(
    region_history,
    by = "aewr_region_id",
    relationship = "many-to-one"
  ) %>%
  mutate(
    # Polynomial terms are scaled around a five-log-point change to improve
    # conditioning while retaining the linear term in log percentage points.
    mc_dose_current_sq = mc_dose_current^2 / 25,
    mc_dose_current_cu = mc_dose_current^3 / 125,
    mc_dose_lag1_sq = mc_dose_lag1^2 / 25,
    mc_dose_lag1_cu = mc_dose_lag1^3 / 125,
    mc_dose_lag2_sq = mc_dose_lag2^2 / 25,
    mc_dose_lag2_cu = mc_dose_lag2^3 / 125,
    mc_dose_lead1_sq = mc_dose_lead1^2 / 25,
    mc_dose_lead1_cu = mc_dose_lead1^3 / 125,
    mc_dose_current_x_lag1 = mc_dose_current * mc_dose_lag1 / 25,
    mc_dose_current_x_lag2 = mc_dose_current * mc_dose_lag2 / 25,
    mc_dose_lag1_x_lag2 = mc_dose_lag1 * mc_dose_lag2 / 25,
    mc_dose_current_x_lag1_x_lag2 = mc_dose_current *
      mc_dose_lag1 *
      mc_dose_lag2 /
      125,
    mc_y_applications = nbr_applications_start_year,
    mc_y_employers = nbr_employers_balanced_start_year,
    mc_y_requested_positions = nbr_workers_requested_start_year,
    mc_y_certified_positions = nbr_workers_certified_start_year,
    mc_y_certified_hours_thousands = man_hours_certified_start_year / 1000,
    mc_y_applications_per_1000 = 1000 *
      nbr_applications_start_year /
      emp_farm_2011,
    mc_y_requested_positions_per_1000 = 1000 *
      nbr_workers_requested_start_year /
      emp_farm_2011,
    mc_y_certified_positions_per_1000 = 1000 *
      nbr_workers_certified_start_year /
      emp_farm_2011,
    mc_y_certified_hours_per_worker = man_hours_certified_start_year /
      emp_farm_2011,
    mc_y_any_application = as.integer(nbr_applications_start_year > 0),
    mc_y_positions_per_application = if_else(
      nbr_applications_start_year > 0,
      nbr_workers_certified_start_year /
        nbr_applications_start_year,
      NA_real_
    ),
    mc_y_hours_per_position = if_else(
      nbr_workers_certified_start_year > 0,
      man_hours_certified_start_year /
        nbr_workers_certified_start_year,
      NA_real_
    ),
    mc_log_baseline_farm_employment = log(emp_farm_2011),
    mc_log_applications = if_else(
      nbr_applications_start_year > 0,
      log(nbr_applications_start_year),
      NA_real_
    ),
    mc_log_certified_positions = if_else(
      nbr_workers_certified_start_year > 0,
      log(nbr_workers_certified_start_year),
      NA_real_
    ),
    mc_z = .data[["mc_b_aewr_bite_mean_z"]]
  ) %>%
  group_by(year) %>%
  mutate(
    mc_binding_quartile = ntile(
      .data[["mc_b_aewr_bite_mean_z"]],
      4L
    ),
    mc_baseline_h2a_quartile = ntile(
      .data[["mc_b_h2a_cert_intensity_mean_z"]],
      4L
    )
  ) %>%
  ungroup() %>%
  arrange(county_fips, year) %>%
  mutate(mc_row_id = row_number())

if (
  nrow(analysis_panel) != nrow(eligible_counties) * length(MC_ANALYSIS_YEARS) ||
    anyDuplicated(analysis_panel[c("county_fips", "year")]) > 0L
) {
  stop(
    "MC analysis panel must be a balanced eligible-county panel.",
    call. = FALSE
  )
}
if (
  any(!is.finite(analysis_panel$mc_dose_current)) ||
    any(!is.finite(analysis_panel$mc_dose_lag1)) ||
    any(!is.finite(analysis_panel$mc_dose_lag2))
) {
  stop("Current and lagged AEWR changes must be complete.", call. = FALSE)
}

component_lookup <- variable_inventory %>%
  filter(role %in% c("mundlak_component", "chamberlain_component"))

mundlak_intercept_terms <- component_lookup %>%
  filter(
    role == "mundlak_component",
    hierarchy_level != "region"
  ) %>%
  pull(constructed_column)

mundlak_slope_terms <- component_lookup %>%
  filter(role == "mundlak_component") %>%
  pull(constructed_column)

mundlak_trend_terms <- component_lookup %>%
  filter(
    role == "mundlak_component",
    statistic == "mean",
    source_variable %in% MC_UNTREATED_TREND_VARIABLES,
    hierarchy_level %in% c("county", "market", "state")
  ) %>%
  pull(constructed_column)

chamberlain_replacement_variables <- MC_CHAMBERLAIN_VARIABLES
chamberlain_intercept_terms <- component_lookup %>%
  filter(
    hierarchy_level != "region",
    (role == "chamberlain_component" &
      source_variable %in% chamberlain_replacement_variables) |
      (role == "mundlak_component" &
        !source_variable %in% chamberlain_replacement_variables)
  ) %>%
  pull(constructed_column)

chamberlain_slope_terms <- component_lookup %>%
  filter(
    (role == "chamberlain_component" &
      source_variable %in% chamberlain_replacement_variables) |
      (role == "mundlak_component" &
        !source_variable %in% chamberlain_replacement_variables)
  ) %>%
  pull(constructed_column)

chamberlain_trend_terms <- component_lookup %>%
  filter(
    role == "mundlak_component",
    statistic == "mean",
    source_variable %in% MC_UNTREATED_TREND_VARIABLES,
    hierarchy_level %in% c("county", "market", "state")
  ) %>%
  pull(constructed_column)

# Region trends use four predetermined dimensions, leaving meaningful
# cross-region support in every year rather than mechanically saturating the
# 17 treatment-assignment units.
region_trend_terms <- component_lookup %>%
  filter(
    role == "mundlak_component",
    statistic == "mean",
    source_variable %in% MC_DYNAMIC_SLOPE_VARIABLES,
    hierarchy_level == "region"
  ) %>%
  pull(constructed_column)
chamberlain_trend_terms <- c(
  chamberlain_trend_terms,
  region_trend_terms
)

dynamic_slope_terms <- component_lookup %>%
  filter(
    role == "mundlak_component",
    statistic == "mean",
    source_variable %in% MC_DYNAMIC_SLOPE_VARIABLES
  ) %>%
  pull(constructed_column)

master_baseline_trend_terms <- component_lookup %>%
  filter(
    role == "mundlak_component",
    statistic == "mean",
    source_variable != MC_Z_VARIABLE,
    hierarchy_level %in% c("county", "market", "state")
  ) %>%
  pull(constructed_column)

region_treatment_history_map <- tibble(
  constructed_column = region_history_terms,
  history_year = as.integer(sub("^.*_", "", region_history_terms))
)

metadata <- list(
  design_version = MC_DESIGN_VERSION,
  baseline_years = MC_BASELINE_YEARS,
  treatment_history_years = MC_TREATMENT_HISTORY_YEARS,
  analysis_years = MC_ANALYSIS_YEARS,
  reference_year = MC_REFERENCE_YEAR,
  cluster_df = MC_CLUSTER_DF,
  ccv_method = MC_CCV_METHOD,
  ccv_reference_design = MC_CCV_REFERENCE_DESIGN,
  ccv_reference_states = MC_CCV_REFERENCE_STATES,
  outcomes = MC_OUTCOMES,
  mundlak_intercept_terms = unique(mundlak_intercept_terms),
  mundlak_slope_terms = unique(mundlak_slope_terms),
  mundlak_trend_terms = unique(mundlak_trend_terms),
  chamberlain_intercept_terms = unique(chamberlain_intercept_terms),
  chamberlain_slope_terms = unique(chamberlain_slope_terms),
  chamberlain_trend_terms = unique(chamberlain_trend_terms),
  dynamic_slope_terms = unique(dynamic_slope_terms),
  region_treatment_history_terms = region_history_terms,
  region_treatment_history_map = region_treatment_history_map,
  master_baseline_trend_terms = unique(master_baseline_trend_terms),
  z_column = MC_Z_COLUMN,
  z_variable = MC_Z_VARIABLE,
  z_label = MC_Z_LABEL,
  master_polynomial_degrees = MC_MASTER_POLYNOMIAL_DEGREES,
  summary_component_columns = summary_component_columns,
  history_component_columns = history_component_columns,
  variable_inventory = variable_inventory,
  scaling = scaling_table
)

write_parquet(
  analysis_panel,
  path_processed("mundlak_chamberlain_county_year.parquet")
)
saveRDS(
  metadata,
  path_int("mundlak_chamberlain_metadata.rds")
)
write_csv(
  scaling_table,
  path_int("mundlak_chamberlain_scaling.csv")
)
write_csv(
  variable_inventory,
  path_int("mundlak_chamberlain_variable_inventory.csv")
)

message(
  "Built MC panel: ",
  format(nrow(analysis_panel), big.mark = ","),
  " county-years, ",
  n_distinct(analysis_panel$county_fips),
  " counties, ",
  n_distinct(analysis_panel$mc_market_id),
  " nested market cells, ",
  n_distinct(analysis_panel$state_fips),
  " states, and ",
  n_distinct(analysis_panel$aewr_region_id),
  " AEWR regions."
)
