# Numerical checks against full weighted fixed-effect estimation and refits.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "border_discontinuity", "power.R"))

near <- function(actual, expected, tolerance = 1e-9) {
  if (any(!is.finite(c(actual, expected))) ||
      max(abs(actual - expected)) > tolerance) {
    stop("Numerical mismatch: ", paste(actual, collapse = ", "), " vs ",
         paste(expected, collapse = ", "))
  }
}

set.seed(201103)
edges <- list(c("1", "2"), c("1", "3"), c("2", "3"),
              c("2", "4"), c("3", "5"), c("4", "5"))
rows <- list()
unit_counter <- 0L
for (pair in seq_along(edges)) {
  for (region in edges[[pair]]) {
    for (county in seq_len(3L)) {
      unit_counter <- unit_counter + 1L
      county_id <- sprintf("%05d", unit_counter)
      years <- 2008:2016
      region_position <- match(region, as.character(1:5))
      regional_wages <- 10 + region_position / 4 +
        sin((years - 2008) * (0.3 + region_position / 8)) +
        (years - 2008) * region_position / 20
      baseline <- stats::rnorm(1)
      rows[[length(rows) + 1L]] <- data.frame(
        county_fips = county_id, aewr_region_id = region,
        region_pair_id = paste0("pair_", pair), stratum_id = paste0("stratum_", pair),
        unit_fe_id = county_id,
        stratum_year_id = paste0("stratum_", pair, "_", years),
        year = years, estimation_weight = county,
        treatment = regional_wages, baseline_trend = baseline * (years - 2011),
        outcome = 0.2 * regional_wages + 0.1 * baseline * (years - 2011) +
          stats::rnorm(1) + cos(years + pair) + stats::rnorm(length(years), sd = 0.1)
      )
    }
  }
}
data <- do.call(rbind, rows)
x <- as.matrix(data[c("treatment", "baseline_trend")])
prepared <- border_power_projection(data, x, data$outcome)
formula <- outcome ~ treatment + baseline_trend | unit_fe_id + stratum_year_id
full <- fixest::feols(formula, data = data, weights = ~estimation_weight,
                      vcov = ~region_pair_id, nthreads = 1L, notes = FALSE)
near(prepared$fit, unname(stats::coef(full)))
near(prepared$residual_ss,
     sum(data$estimation_weight / mean(data$estimation_weight) * residuals(full)^2))
statistics <- border_power_draw_statistics(prepared, matrix(data$outcome, ncol = 1L))
near(statistics$coefficient_error, stats::coef(full)[["treatment"]])
near(statistics$variance["pair_CRV1", ], stats::vcov(full)["treatment", "treatment"])
two_way <- fixest::vcov_cluster(full, cluster = ~region_pair_id + aewr_region_id,
                                vcov_fix = FALSE)
near(statistics$variance["pair_region_CRV1", ], two_way["treatment", "treatment"])

# Direct observation-pair covariance enumeration checks the dyadic mapping:
# include a product whenever either endpoint is shared, count each product once.
score <- prepared$partial_influence * prepared$residual
dependent <- outer(data$region_pair_id, data$region_pair_id, Vectorize(function(a, b) {
  length(intersect(prepared$pair_regions[[a]], prepared$pair_regions[[b]])) > 0L
}))
expected_dyadic <- sum(outer(score, score) * dependent) *
  length(prepared$regions) / (length(prepared$regions) - 1)
near(statistics$variance["shared_endpoint_dyadic", ], expected_dyadic)

leave_out <- vapply(sort(unique(data$region_pair_id)), function(pair) {
  subset <- data[data$region_pair_id != pair, ]
  fit <- fixest::feols(formula, data = subset, weights = ~estimation_weight,
                       nthreads = 1L, notes = FALSE)
  stats::coef(fit)[["treatment"]]
}, numeric(1))
expected_cv3j <- (length(leave_out) - 1) / length(leave_out) *
  sum((leave_out - mean(leave_out))^2)
near(statistics$variance["pair_CV3J", ], expected_cv3j)

# A constant injected slope must change only the treatment coefficient.
shifted <- border_power_draw_statistics(
  prepared, matrix(data$outcome + 0.07 * data$treatment, ncol = 1L)
)
near(shifted$coefficient_error - statistics$coefficient_error, 0.07)
near(shifted$variance, statistics$variance)

# Missing and varying-weight panels must fail, never change sample silently.
bad <- data[-1L, ]
stopifnot(inherits(try(border_power_projection(
  bad, as.matrix(bad[c("treatment", "baseline_trend")]), bad$outcome
), silent = TRUE), "try-error"))
bad <- data
bad$estimation_weight[1L] <- 99
stopifnot(inherits(try(border_power_projection(
  bad, as.matrix(bad[c("treatment", "baseline_trend")]), bad$outcome
), silent = TRUE), "try-error"))

make_period <- function(years) {
  period <- data[data$year %in% years, ]
  list(data = period, x = as.matrix(period[c("treatment", "baseline_trend")]),
       y = period$outcome)
}
models <- list(list(model_id = "synthetic_weighted_controlled",
                    pre = make_period(2008:2010), policy = make_period(2011:2016)))
graph <- border_power_region_graph(models, 0.4)
near(rowSums(graph$loading^2), rep(1, 5L))
scenario <- list(region = 1, pair = 0, county = 0, spatial = FALSE)
policy <- border_power_projection(models[[1L]]$policy$data,
                                   models[[1L]]$policy$x,
                                   models[[1L]]$policy$y)
raw <- border_power_unit_errors(policy, 199L, scenario, 0.5, graph)
# Every same-region/year unit receives exactly the same regional disturbance,
# including units attached to different border pairs.
groups <- split(seq_len(nrow(policy$data)),
                interaction(policy$data$aewr_region_id, policy$data$year))
for (indices in groups) {
  if (length(indices) > 1L) {
    near(raw[indices, , drop = FALSE],
         matrix(raw[indices[1L], ], nrow = length(indices), ncol = ncol(raw), byrow = TRUE))
  }
}

config <- list(redesign = list(
  precision_assumptions = list(alpha = 0.05, target_power = 0.8, size_maximum = 0.075),
  power = list(
    seed = 201103L, repetitions = 199L, calibration_repetitions = 199L,
    chunk_size = 50L, ar1 = 0.5, nthreads = 1L, spatial_graph_coefficient = 0.4,
    dgp_ids = c("county_serial", "shared_region_and_pair_serial", "spatial_region_serial"),
    variance_fractions = list(
      county_serial = list(region = 0, pair = 0, county = 1),
      shared_region_and_pair_serial = list(region = 0.25, pair = 0.25, county = 0.5),
      spatial_region_serial = list(region = 0.5, pair = 0.25, county = 0.25)
    ),
    effect_grid = c(0, -0.01, 0.01, -0.2, 0.2), mde_grid = c(0.01, 0.05, 0.2)
  )
))
result <- run_border_power(models, config)
repeat_result <- run_border_power(models, config)
stopifnot(identical(result, repeat_result))
changed_pre_x <- models
changed_pre_x[[1L]]$pre$x[,] <- NA_real_
stopifnot(identical(result, run_border_power(changed_pre_x, config)))
stepped <- config
stepped$redesign$power$mde_grid <- NULL
stepped$redesign$power$mde_grid_step <- 0.001
stepped$redesign$power$mde_grid_max <- 0.02
stepped_result <- run_border_power(models, stepped)
stopifnot(all(stepped_result$mde$maximum_grid_effect == 0.02))
effect_key <- c("model_id", "dgp_id", "scale_information", "inference_method", "effect")
stopifnot(!anyDuplicated(stepped_result$power[effect_key]))
roundtrip_path <- tempfile("border-power-grid-", tmpdir = "/tmp", fileext = ".csv")
readr::write_csv(stepped_result$power, roundtrip_path)
serialized <- readr::read_csv(roundtrip_path, show_col_types = FALSE)
unlink(roundtrip_path)
stopifnot(!anyDuplicated(serialized[effect_key]))
central <- serialized[abs(serialized$effect) == 0.01, ]
stopifnot(nrow(central) == 3L * 2L * 4L * 2L,
          all(central$declared_meaningful_effect))
stopifnot(nrow(result$calibration) == 6L, all(result$calibration$sigma > 0))
stopifnot(all(result$calibration$calibration_projection[
  result$calibration$scale_information == "preperiod_only"
] == "unit_and_stratum_year_FE_only_no_wage_regressor"))
stopifnot(all(result$power$rejection_rate >= 0), all(result$power$rejection_rate <= 1))
stopifnot(all(result$power$mc_confidence_low <= result$power$rejection_rate),
          all(result$power$mc_confidence_high >= result$power$rejection_rate))
stopifnot(all(is.na(result$mde$credible_grid_mde[!result$mde$size_gate_pass])))
large <- subset(result$power, abs(effect) == 0.2 & inference_method == "pair_CRV1" &
                  dgp_id == "county_serial" & scale_information == "preperiod_only")
stopifnot(all(large$rejection_rate > 0.8))

# Invalid covariance draws are reported and cannot pass the precision gate.
invalid <- border_power_grid(c(-1, 0, 1), c(1, -0.1, NA_real_), 5,
                              c(0, -1, 1), 1, 0.05)
stopifnot(all(invalid$invalid_repetitions == 2L), all(invalid$valid_repetitions == 1L))

# Exported scalar crossproducts reuse the simulation draw, without assuming
# that a scalar bootstrap also applies to the controlled model above.
scalar_models <- models
scalar_models[[1L]]$model_id <- "synthetic_scalar"
scalar_models[[1L]]$pre$x <- scalar_models[[1L]]$pre$x[, 1L, drop = FALSE]
scalar_models[[1L]]$policy$x <- scalar_models[[1L]]$policy$x[, 1L, drop = FALSE]
scalar_config <- config
scalar_config$redesign$power$dgp_ids <- "county_serial"
scalar_config$redesign$power$bootstrap_model_ids <- "synthetic_scalar"
scalar_config$redesign$power$bootstrap_simulation_repetitions <- 99L
scalar_result <- run_border_power(scalar_models, scalar_config)
exported <- scalar_result$bootstrap_inputs[[1L]]
stopifnot(length(scalar_result$bootstrap_inputs) == 1L,
          ncol(exported$pair_null_scores) == 99L,
          nrow(exported$pair_null_scores) == length(edges),
          length(exported$pair_treatment_ss) == length(edges),
          identical(names(exported$scales),
                    c("preperiod_only", "policy_residual_sensitivity")))
scalar_prepared <- border_power_projection(
  scalar_models[[1L]]$policy$data, scalar_models[[1L]]$policy$x,
  scalar_models[[1L]]$policy$y
)
scalar_statistics <- border_power_draw_statistics(scalar_prepared, raw)
near(colSums(scalar_statistics$pair_null_scores) / sum(scalar_prepared$xw^2),
     scalar_statistics$coefficient_error)
cat("Border power numerical tests passed.\n")
