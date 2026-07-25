#!/usr/bin/env Rscript

here::i_am("code/paths.R")

source(here::here("code", "c00_shared", "auxiliary_moment_helpers.R"))

stopifnot(
  weighted_median(c(1, 2, 10), c(0.2, 0.6, 0.2)) == 2,
  is.na(safe_positive_ratio(1, 0)),
  qwi_persistence_share(75, 100) == 0.75,
  seasonal_employment_share(250, 1000) == 0.25
)

census_share <- c(0.2, 0.4, 0.6)
known_bridge <- 2
qwi_share <- apply_public_odds_bridge(census_share, known_bridge)
estimated_bridge <- estimate_public_odds_bridge(
  census_share,
  qwi_share,
  c(1, 2, 1)
)
stopifnot(abs(estimated_bridge - known_bridge) <= 1e-12)
stopifnot(
  !grepl(
    "fls",
    paste(deparse(body(estimate_public_odds_bridge)), collapse = " "),
    ignore.case = TRUE
  )
)

band <- fixed_standardized_discrepancy_band(
  primary_value = c(0.1, 0.2, 0.4, 0.5),
  comparison_value = c(0.12, 0.18, 0.35, 0.55),
  weight = rep(1, 4)
)
stopifnot(
  is.finite(band$half_width),
  band$half_width >= 0,
  band$matched_count == 4L
)

cat("auxiliary moment tests passed\n")
