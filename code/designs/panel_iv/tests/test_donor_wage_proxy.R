# Synthetic checks for the OEWS-area hourly donor-wage proxy.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "helpers.R"))

selected <- select_panel_iv_donor_wage(
  oews_hourly_wage = c(18.50, 0, NA, 21.25, Inf),
  oews_wage_observed = c(TRUE, TRUE, TRUE, FALSE, TRUE)
)

stopifnot(
  identical(
    selected$donor_wage_source,
    c(
      DISSIMILARITY_IV_DONOR_WAGE_SOURCE,
      "unavailable",
      "unavailable",
      "unavailable",
      "unavailable"
    )
  ),
  isTRUE(all.equal(selected$donor_nominal_hourly_wage[[1]], 18.50)),
  all(is.na(selected$donor_nominal_hourly_wage[-1])),
  !"donor_nominal_annual_wage" %in% names(selected)
)
