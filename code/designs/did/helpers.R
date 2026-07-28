did_sample <- function(panel) {
  panel |>
    dplyr::filter(
      any_cropland_2007,
      county_simple_treatment_groups != "always takers"
    )
}

did_cluster_formula <- ~cz_id^aewr_region_id

did_model <- function(data, outcome, controls = FALSE) {
  control_terms <- if (controls) {
    " + ln_pop_census + emp_pop_ratio"
  } else {
    ""
  }
  formula <- stats::as.formula(paste0(
    outcome,
    " ~ aewr_cz_p25_l1 * post",
    control_terms,
    " | county_fips + year"
  ))
  fixest::feols(
    formula,
    data = data,
    vcov = did_cluster_formula,
    notes = FALSE
  )
}

did_event_model <- function(data, controls = FALSE) {
  control_terms <- if (controls) {
    " + ln_pop_census + emp_pop_ratio"
  } else {
    ""
  }
  formula <- stats::as.formula(paste0(
    "h2a_cert_share_farm_workers_2011_start_year ~ ",
    "aewr_cz_p25_l1 + ",
    "i(year, aewr_cz_p25_l1, ref = 2011)",
    control_terms,
    " | county_fips + year"
  ))
  fixest::feols(
    formula,
    data = data,
    vcov = did_cluster_formula,
    notes = FALSE
  )
}

did_table_headers <- c(
  "No Controls",
  "Controls",
  "No Border, No Controls",
  "No Border, Controls"
)

did_table_dictionary <- c(
  h2a_cert_share_farm_workers_2011_start_year =
    "Normalized H-2A program usage",
  aewr_cz_p25_l1 = "Lagged AEWR vs 25th pct wage gap",
  "aewr_cz_p25_l1:postTRUE" =
    "Lagged AEWR vs 25th pct wage gap $\\times$ Post",
  post = "Post",
  postTRUE = "Post",
  ln_pop_census = "Log population",
  emp_pop_ratio = "Employment-to-pop ratio"
)
