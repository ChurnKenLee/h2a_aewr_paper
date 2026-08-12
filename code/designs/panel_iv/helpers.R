# Pure numerical helpers for the panel-IV branch.
# Callers are responsible for loading any data-manipulation packages.

finite_mean <- function(value) {
  keep <- is.finite(value)
  if (!any(keep)) {
    return(NA_real_)
  }
  mean(value[keep])
}

finite_min <- function(value) {
  keep <- is.finite(value)
  if (!any(keep)) {
    return(NA_real_)
  }
  min(value[keep])
}

finite_max <- function(value) {
  keep <- is.finite(value)
  if (!any(keep)) {
    return(NA_real_)
  }
  max(value[keep])
}

positive_weighted_mean <- function(value, weight) {
  keep <- is.finite(value) & is.finite(weight) & weight > 0
  if (!any(keep)) {
    return(NA_real_)
  }
  stats::weighted.mean(value[keep], weight[keep])
}

select_panel_iv_donor_wage <- function(
  oews_hourly_wage,
  oews_wage_observed
) {
  oews_valid <-
    !is.na(oews_wage_observed) & oews_wage_observed &
      is.finite(oews_hourly_wage) & oews_hourly_wage > 0
  data.frame(
    donor_nominal_hourly_wage = ifelse(
      oews_valid,
      oews_hourly_wage,
      NA_real_
    ),
    donor_wage_source = ifelse(
      oews_valid,
      DISSIMILARITY_IV_DONOR_WAGE_SOURCE,
      "unavailable"
    ),
    stringsAsFactors = FALSE
  )
}
