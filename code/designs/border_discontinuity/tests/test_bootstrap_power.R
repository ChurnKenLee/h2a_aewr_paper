# Locked-package equivalence for the scalar, pair-clustered FWL shortcut.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "border_discontinuity", "power.R"))
source(path_code("designs", "border_discontinuity", "bootstrap_power.R"))

near <- function(actual, expected, tolerance = 1e-8) {
  if (!identical(dim(actual), dim(expected)) ||
      any(!is.finite(c(actual, expected))) || max(abs(actual - expected)) > tolerance) {
    stop("Locked-package scalar bootstrap equivalence failed.")
  }
}
expect_error <- function(expression) {
  if (!inherits(try(force(expression), silent = TRUE), "try-error")) {
    stop("An invalid bootstrap specification was accepted.")
  }
}
stopifnot(as.character(utils::packageVersion("fwildclusterboot")) == "0.14.3")
settings <- list(repetitions = 9999L, seed = 201103L)

# Unequal pair sizes, unbalanced side sizes, fixed county weights, nested county
# and pair-year FE. Shuffling rows also challenges score/cluster ordering.
set.seed(412L)
rows <- list()
unit <- 0L
for (pair in seq_len(8L)) {
  for (side in seq_len(2L)) {
    for (county in seq_len(1L + ((pair + side) %% 3L))) {
      unit <- unit + 1L
      year <- 2008:2014
      rows[[length(rows) + 1L]] <- data.frame(
        county_fips = sprintf("%05d", unit), unit_fe_id = sprintf("u%03d", unit),
        region_pair_id = sprintf("pair_%02d", pair),
        aewr_region_id = sprintf("region_%02d", 2L * pair + side),
        stratum_id = sprintf("stratum_%02d", pair),
        stratum_year_id = paste(pair, year, sep = "_"), year = year,
        estimation_weight = 1 + county / 3,
        x = side * sin(year / (pair + 2)) + 0.07 * side * (year - 2008),
        y = rnorm(1L) + sin(year) + side * rnorm(length(year))
      )
    }
  }
}
data <- do.call(rbind, rows)
data <- data[sample.int(nrow(data)), ]
prepared <- border_power_projection(data, as.matrix(data["x"]), data$y)
h <- drop(rowsum(prepared$xw[, 1L]^2, prepared$pair_index, reorder = TRUE))
weights <- border_bootstrap_webb_weights(length(h), settings$repetitions, settings$seed)
transformed <- data.frame(x = prepared$xw[, 1L], y = prepared$yw,
                          pair = data$region_pair_id)

maximum_t_error <- 0
for (effect in c(0, -0.02, 0.01)) {
  transformed$injected_y <- transformed$y + effect * transformed$x
  model <- fixest::feols(injected_y ~ x - 1, transformed, nthreads = 1L)
  set.seed(settings$seed)
  dqrng::dqset.seed(settings$seed)
  package <- fwildclusterboot::boottest(
    model, param = "x", B = settings$repetitions, clustid = "pair",
    bootcluster = "pair", type = "webb", impose_null = TRUE,
    bootstrap_type = "fnw11", conf_int = FALSE, engine = "R", nthreads = 1L
  )
  q <- rowsum(transformed$x * transformed$injected_y,
              prepared$pair_index, reorder = TRUE)
  shortcut <- drop(border_bootstrap_scalar_t(q, h, weights))
  expected <- c(drop(package$t_stat), drop(package$t_boot))
  near(shortcut, expected)
  maximum_t_error <- max(maximum_t_error, max(abs(shortcut - expected)))
  probability <- border_bootstrap_pvalues(matrix(shortcut, ncol = 1L))
  stopifnot(probability$p_value == package$p_val,
            probability$invalid_bootstrap_draws == 0L, probability$valid)
}

# Explicit ties, invalid bootstrap draws, and infinity use the exact package
# p-value convention; invalid draws additionally fail the simulation validity.
ties <- border_bootstrap_pvalues(matrix(c(2, -2, 2, 3, NA_real_, Inf), ncol = 1L))
stopifnot(ties$p_value == 0.5, ties$invalid_bootstrap_draws == 1L,
          ties$infinite_bootstrap_draws == 1L, !ties$valid)
undefined <- border_bootstrap_pvalues(matrix(c(NaN, NaN, NaN), ncol = 1L))
stopifnot(is.na(undefined$p_value), !undefined$valid)
expect_error(border_bootstrap_scalar_t(matrix(0, 8L, 1L), rep(0, 8L), weights))

# A bounded runner test with two scales checks signed alternatives, exact null
# scale invariance, unchanged R RNG, and failures of the FWL/sample contract.
set.seed(77L)
scores <- matrix(rnorm(length(h) * 99L), nrow = length(h))
config <- list(
  bootstrap = list(bootstrap_type = "fnw11", weights = "webb",
                   cluster = "region_pair_id", repetitions = 99L, seed = 201103L),
  redesign = list(
    precision_assumptions = list(alpha = 0.05, size_maximum = 0.075),
    power = list(bootstrap_effect_grid = c(0, -0.01, 0.01),
                 bootstrap_model_ids = "toy", dgp_ids = "toy_dgp",
                 bootstrap_simulation_repetitions = 99L, repetitions = 199L,
                 chunk_size = 40L, seed = 202L)
  )
)
input <- list(
  model_id = "toy", dgp_id = "toy_dgp", pair_null_scores = scores,
  pair_treatment_ss = h, pair_ids = prepared$pairs,
  scales = list(preperiod_only = list(sigma = 1),
                policy_residual_sensitivity = list(sigma = 2)),
  n_obs = nrow(data), n_pairs = length(h), n_regions = length(prepared$regions),
  n_regressors = 1L, fixed_effects_nested_in_pair = TRUE
)
old_rng <- .Random.seed
result <- run_border_bootstrap_power(list(input), config)
stopifnot(identical(old_rng, .Random.seed), nrow(result) == 6L,
          all(result$repetitions == 99L), all(result$valid_repetitions == 99L),
          length(unique(result$rejections[result$effect == 0])) == 1L,
          all(result$invalid_bootstrap_draws == 0L))
non_nested <- input
non_nested$fixed_effects_nested_in_pair <- FALSE
expect_error(run_border_bootstrap_power(list(non_nested), config))
controlled <- input
controlled$n_regressors <- 2L
expect_error(run_border_bootstrap_power(list(controlled), config))
misordered <- input
misordered$pair_ids <- rev(misordered$pair_ids)
expect_error(run_border_bootstrap_power(list(misordered), config))
expect_error(run_border_bootstrap_power(list(input, input), config))
cat(sprintf("Scalar WCR11-FNW tests passed: 3 x 9999 package draws; max |t difference| %.3g.\n",
             maximum_t_error))
