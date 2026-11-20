# Plain helper for the Kitagawa instrument-decomposition diagnostic.
#
# For a target cluster g and source year t with balanced donor weights p_ct
# (summing to one) and donor wages w_ct, fix reference values p_bar_c (the
# time mean of p_ct) and w_bar_c (the predetermined base wage). Then exactly
#   Z_gt = sum p_bar w_bar + sum (p - p_bar) w_bar
#        + sum p_bar (w - w_bar) + sum (p - p_bar)(w - w_bar).
# The weight term does not depend on p_bar apart from a cluster constant.
# Callers are responsible for loading dplyr.

IV_DECOMPOSITION_BASE_WAGE_YEARS <- 2008:2010
IV_DECOMPOSITION_IDENTITY_TOLERANCE <- 1e-9

iv_kitagawa_components <- function(data, group_columns) {
  required <- c(
    group_columns,
    "county_fips",
    "source_year",
    "decomposition_weight",
    "donor_real_hourly_wage",
    "base_real_hourly_wage"
  )
  missing <- setdiff(required, names(data))
  if (length(missing)) {
    stop("Decomposition input missing: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  values <- data[c("decomposition_weight", "donor_real_hourly_wage", "base_real_hourly_wage")]
  if (!all(vapply(values, function(value) all(is.finite(value)), logical(1)))) {
    stop("Decomposition weights and wages must be finite.", call. = FALSE)
  }
  balance <- data %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(group_columns))) %>%
    dplyr::summarise(
      cells = dplyr::n(),
      counties = dplyr::n_distinct(.data$county_fips),
      years = dplyr::n_distinct(.data$source_year),
      .groups = "drop"
    )
  if (any(balance$cells != balance$counties * balance$years)) {
    stop("Decomposition donor sets must be balanced across source years.", call. = FALSE)
  }
  weight_sums <- data %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(c(group_columns, "source_year")))) %>%
    dplyr::summarise(weight_sum = sum(.data$decomposition_weight), .groups = "drop")
  if (any(abs(weight_sums$weight_sum - 1) > IV_DECOMPOSITION_IDENTITY_TOLERANCE)) {
    stop("Decomposition weights must sum to one within each group-year.", call. = FALSE)
  }

  components <- data %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(c(group_columns, "county_fips")))) %>%
    dplyr::mutate(reference_weight = mean(.data$decomposition_weight)) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      weight_deviation = .data$decomposition_weight - .data$reference_weight,
      wage_deviation = .data$donor_real_hourly_wage - .data$base_real_hourly_wage
    ) %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(c(group_columns, "source_year")))) %>%
    dplyr::summarise(
      z_balanced = sum(.data$decomposition_weight * .data$donor_real_hourly_wage),
      z_constant = sum(.data$reference_weight * .data$base_real_hourly_wage),
      z_weight = sum(.data$weight_deviation * .data$base_real_hourly_wage),
      z_wage = sum(.data$reference_weight * .data$wage_deviation),
      z_interaction = sum(.data$weight_deviation * .data$wage_deviation),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      identity_residual = .data$z_balanced -
        (.data$z_constant + .data$z_weight + .data$z_wage + .data$z_interaction)
    )
  if (any(abs(components$identity_residual) >
    IV_DECOMPOSITION_IDENTITY_TOLERANCE * pmax(1, abs(components$z_balanced)))) {
    stop("Kitagawa components do not add up to the balanced instrument.", call. = FALSE)
  }
  components
}

# Real donor wage by county and year, with stage 05's donor-wage selection and
# source-year PPI deflation. Callers source design.R and helpers.R.
iv_real_donor_wages <- function(county_panel, ppi) {
  wages <- dplyr::mutate(county_panel, source_year = as.integer(.data$year))
  dplyr::bind_cols(
    wages,
    select_panel_iv_donor_wage(
      wages$oews_big_six_mean_hourly_wage,
      wages$oews_wage_observed
    )
  ) %>%
    dplyr::left_join(ppi, by = "source_year", relationship = "many-to-one") %>%
    dplyr::transmute(
      .data$county_fips,
      .data$source_year,
      real_hourly_wage = dplyr::if_else(
        is.finite(.data$donor_nominal_hourly_wage) &
          is.finite(.data$source_year_ppi_2012) & .data$source_year_ppi_2012 > 0,
        .data$donor_nominal_hourly_wage / .data$source_year_ppi_2012,
        NA_real_
      )
    )
}

# Mean real donor wage in the base years.
iv_decomposition_base_wages <- function(county_panel, ppi) {
  iv_real_donor_wages(county_panel, ppi) %>%
    dplyr::filter(.data$source_year %in% IV_DECOMPOSITION_BASE_WAGE_YEARS) %>%
    dplyr::group_by(.data$county_fips) %>%
    dplyr::summarise(
      base_real_hourly_wage = finite_mean(.data$real_hourly_wage),
      .groups = "drop"
    )
}

# Balanced donor set and its Kitagawa components. A donor is balanced when it
# is `eligible` in every source year and has a base wage; its `weight` is
# renormalized over that set. A group is supported when every selected donor
# cluster contributes in every source year, mirroring the instrument's rule.
iv_balanced_components <- function(frame, group_columns, source_years, donor_clusters) {
  required <- c(
    group_columns, "county_fips", "source_year", "donor_cluster", "eligible",
    "weight", "donor_real_hourly_wage", "base_real_hourly_wage"
  )
  missing <- setdiff(required, names(frame))
  if (length(missing)) {
    stop("Balanced-donor input missing: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  balanced <- frame %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(c(group_columns, "county_fips")))) %>%
    dplyr::filter(
      dplyr::n_distinct(.data$source_year[.data$eligible]) == length(source_years),
      is.finite(dplyr::first(.data$base_real_hourly_wage))
    ) %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(c(group_columns, "source_year")))) %>%
    dplyr::mutate(
      decomposition_weight = .data$weight / sum(.data$weight),
      balanced_donor_clusters = dplyr::n_distinct(.data$donor_cluster),
      balanced_donor_counties = dplyr::n_distinct(.data$county_fips)
    ) %>%
    dplyr::ungroup()
  supported_groups <- balanced %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(group_columns))) %>%
    dplyr::summarise(
      decomposition_supported =
        all(.data$balanced_donor_clusters == donor_clusters) &
          dplyr::n_distinct(.data$source_year) == length(source_years),
      .groups = "drop"
    )
  supported <- dplyr::semi_join(
    balanced,
    dplyr::filter(supported_groups, .data$decomposition_supported),
    by = group_columns
  )
  list(
    balanced = balanced,
    supported_groups = supported_groups,
    components = iv_kitagawa_components(supported, group_columns)
  )
}
