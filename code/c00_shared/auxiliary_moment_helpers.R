# Pure helpers for constructing public-data auxiliary calibration moments.

weighted_median <- function(value, weight) {
  keep <- is.finite(value) & is.finite(weight) & weight > 0
  value <- value[keep]
  weight <- weight[keep]
  if (length(value) == 0L) {
    return(NA_real_)
  }
  ordering <- order(value)
  value <- value[ordering]
  weight <- weight[ordering] / sum(weight)
  value[[which(cumsum(weight) >= 0.5)[[1]]]]
}

safe_positive_ratio <- function(numerator, denominator) {
  ifelse(
    is.finite(numerator) &
      is.finite(denominator) &
      denominator > 0,
    numerator / denominator,
    NA_real_
  )
}

qwi_persistence_share <- function(
  stable_employment,
  beginning_quarter_employment
) {
  safe_positive_ratio(
    stable_employment,
    beginning_quarter_employment
  )
}

seasonal_employment_share <- function(
  reference_month_employment,
  annual_reference_month_total
) {
  safe_positive_ratio(
    reference_month_employment,
    annual_reference_month_total
  )
}

apply_public_odds_bridge <- function(
  census_share,
  odds_bridge_ratio
) {
  output <- rep(NA_real_, length(census_share))
  keep <- is.finite(census_share) &
    census_share >= 0 &
    census_share <= 1 &
    is.finite(odds_bridge_ratio) &
    odds_bridge_ratio > 0
  denominator <- 1 - census_share[keep] +
    odds_bridge_ratio * census_share[keep]
  output[keep] <- odds_bridge_ratio * census_share[keep] / denominator
  output
}

estimate_public_odds_bridge <- function(
  census_share,
  qwi_persistence_share,
  weight,
  boundary_epsilon = 1e-6
) {
  keep <- is.finite(census_share) &
    is.finite(qwi_persistence_share) &
    is.finite(weight) &
    weight > 0 &
    census_share > boundary_epsilon &
    census_share < 1 - boundary_epsilon &
    qwi_persistence_share > boundary_epsilon &
    qwi_persistence_share < 1 - boundary_epsilon
  if (!any(keep)) {
    return(NA_real_)
  }
  census_odds <- census_share[keep] / (1 - census_share[keep])
  qwi_odds <- qwi_persistence_share[keep] /
    (1 - qwi_persistence_share[keep])
  weighted_median(qwi_odds / census_odds, weight[keep])
}

fixed_standardized_discrepancy_band <- function(
  primary_value,
  comparison_value,
  weight,
  probability = 0.90,
  minimum_scale = 1e-8
) {
  keep <- is.finite(primary_value) &
    is.finite(comparison_value) &
    is.finite(weight) &
    weight > 0
  if (!any(keep)) {
    return(list(
      half_width = NA_real_,
      scale = NA_real_,
      matched_count = 0L
    ))
  }
  primary_value <- primary_value[keep]
  comparison_value <- comparison_value[keep]
  weight <- weight[keep] / sum(weight[keep])
  center <- sum(weight * primary_value)
  scale <- sqrt(sum(weight * (primary_value - center)^2))
  if (!is.finite(scale) || scale <= minimum_scale) {
    return(list(
      half_width = NA_real_,
      scale = scale,
      matched_count = length(primary_value)
    ))
  }
  standardized_discrepancy <- abs(
    primary_value - comparison_value
  ) / scale
  list(
    half_width = as.numeric(quantile(
      standardized_discrepancy,
      probs = probability,
      names = FALSE,
      type = 8,
      na.rm = TRUE
    )),
    scale = scale,
    matched_count = length(primary_value)
  )
}
