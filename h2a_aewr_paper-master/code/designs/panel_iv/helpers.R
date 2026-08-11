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
