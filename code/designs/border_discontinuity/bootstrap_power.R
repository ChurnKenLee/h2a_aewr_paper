# Scalar WCR11-FNW conditional size/power supplement; no artifact I/O.
# This is the no-intercept, weighted FWL regression used by border_wild().
# Pair-constant Webb multipliers commute with absorption only because every
# fixed effect is nested in pair. Additional controls are not supported here.

border_bootstrap_webb_weights <- function(n_pairs, repetitions, seed) {
  if (length(n_pairs) != 1L || !is.finite(n_pairs) || n_pairs < 3L ||
      n_pairs != as.integer(n_pairs) || length(repetitions) != 1L ||
      !is.finite(repetitions) || repetitions < 1L ||
      repetitions != as.integer(repetitions) || length(seed) != 1L ||
      !is.finite(seed)) stop("Invalid scalar Webb bootstrap dimensions or seed.")
  # Match locked fwildclusterboot::get_weights(sampling="dqrng") exactly,
  # including drawing and then overwriting the observed first column.
  dqrng::dqset.seed(as.integer(seed))
  weights <- matrix(dqrng::dqsample(
    c(-sqrt((3:1) / 2), sqrt((1:3) / 2)),
    size = n_pairs * (repetitions + 1L), replace = TRUE
  ), nrow = n_pairs)
  weights[, 1L] <- 1
  t(weights)
}

border_bootstrap_scalar_t <- function(pair_null_scores, pair_treatment_ss, weights) {
  q <- as.matrix(pair_null_scores)
  h_total <- as.numeric(pair_treatment_ss)
  if (length(h_total) < 3L || nrow(q) != length(h_total) ||
      ncol(weights) != length(h_total) || nrow(weights) < 2L ||
      any(!is.finite(c(q, h_total, weights))) || any(h_total < 0) ||
      sum(h_total) <= 0 || any(weights[1L, ] != 1)) {
    stop("Scalar bootstrap scores and treatment support must be finite and aligned.")
  }
  h <- h_total / sum(h_total)
  numerator <- weights %*% q
  weighted_hq <- weights %*% (h * q)
  meat <- (weights^2) %*% (q^2) - 2 * numerator * weighted_hq +
    numerator^2 * sum(h^2)
  # CRV1 adjustment of the package's scalar no-intercept FWL regression:
  # G/(G-1) * (N-1)/(N-k), with k=1. No FE/intercept is counted again.
  adjustment <- length(h) / (length(h) - 1L)
  suppressWarnings(numerator / sqrt(adjustment * meat))
}

border_bootstrap_pvalues <- function(t_statistics) {
  t_statistics <- as.matrix(t_statistics)
  if (nrow(t_statistics) < 2L) stop("Observed and bootstrap t statistics required.")
  observed <- t_statistics[1L, ]
  boot <- t_statistics[-1L, , drop = FALSE]
  invalid_draws <- colSums(is.na(boot))
  valid_draws <- nrow(boot) - invalid_draws
  # Locked package convention is strict '<', with no +1 correction. It drops
  # NA/NaN bootstrap draws, retains infinite draws, and returns NaN if none
  # remain. Preserve that p-value while exposing failures to the size gate.
  exceedances <- colSums(sweep(abs(boot), 2L, abs(observed), ">"), na.rm = TRUE)
  p_value <- exceedances / valid_draws
  p_value[is.na(observed)] <- NA_real_
  list(
    p_value = p_value, observed_t = observed,
    invalid_bootstrap_draws = invalid_draws,
    infinite_bootstrap_draws = colSums(is.infinite(boot)),
    valid = is.finite(p_value) & is.finite(observed) & invalid_draws == 0L
  )
}

run_border_bootstrap_power <- function(bootstrap_inputs, config) {
  settings <- config$redesign$power
  bootstrap <- config$bootstrap
  assumptions <- config$redesign$precision_assumptions
  effects <- as.numeric(unlist(settings$bootstrap_effect_grid))
  model_ids <- as.character(unlist(settings$bootstrap_model_ids))
  dgp_ids <- as.character(unlist(settings$dgp_ids))
  repetitions <- as.integer(settings$bootstrap_simulation_repetitions)
  if (!identical(as.character(utils::packageVersion("fwildclusterboot")), "0.14.3")) {
    stop("Scalar shortcut was equivalence-tested only against fwildclusterboot 0.14.3.")
  }
  if (!identical(bootstrap$bootstrap_type, "fnw11") ||
      !identical(bootstrap$weights, "webb") ||
      !identical(bootstrap$cluster, "region_pair_id") ||
      length(repetitions) != 1L || !is.finite(repetitions) || repetitions < 99L ||
      repetitions > settings$repetitions || !length(effects) ||
      any(!is.finite(effects)) || anyDuplicated(effects) ||
      !0 %in% effects || !all(-effects %in% effects) ||
      !length(model_ids) || anyDuplicated(model_ids) ||
      !length(dgp_ids) || anyDuplicated(dgp_ids) ||
      !is.finite(assumptions$alpha) || assumptions$alpha <= 0 ||
      assumptions$alpha >= 1 || !is.finite(assumptions$size_maximum) ||
      !is.finite(settings$chunk_size) || settings$chunk_size < 1L) {
    stop("Invalid declared scalar bootstrap simulation settings.")
  }
  expected_keys <- as.vector(outer(model_ids, dgp_ids, paste, sep = "::"))
  keys <- vapply(bootstrap_inputs, function(z) paste(z$model_id, z$dgp_id, sep = "::"),
                 character(1))
  if (anyDuplicated(keys) || !setequal(keys, expected_keys)) {
    stop("Bootstrap inputs must contain every declared model/DGP exactly once.")
  }
  rows <- list()
  for (input in bootstrap_inputs) {
    message("Bootstrap power: ", input$model_id, " / ", input$dgp_id)
    q0 <- as.matrix(input$pair_null_scores)
    h_total <- as.numeric(input$pair_treatment_ss)
    if (!isTRUE(input$fixed_effects_nested_in_pair) ||
        !identical(as.integer(input$n_regressors), 1L) ||
        ncol(q0) != repetitions || nrow(q0) != input$n_pairs ||
        length(h_total) != input$n_pairs || input$n_obs <= input$n_pairs ||
        !is.character(input$pair_ids) || anyNA(input$pair_ids) ||
        any(!nzchar(input$pair_ids)) || anyDuplicated(input$pair_ids) ||
        length(input$pair_ids) != input$n_pairs ||
        !identical(input$pair_ids, sort(input$pair_ids)) ||
        !setequal(names(input$scales), c("preperiod_only", "policy_residual_sensitivity"))) {
      stop("Bootstrap shortcut requires aligned scalar scores and FE nested in pair.")
    }
    weights <- border_bootstrap_webb_weights(input$n_pairs, bootstrap$repetitions,
                                             bootstrap$seed)
    for (scale_name in names(input$scales)) {
      sigma <- input$scales[[scale_name]]$sigma
      if (length(sigma) != 1L || !is.finite(sigma) || sigma <= 0) {
        stop("Bootstrap error calibration must be positive and finite.")
      }
      curve <- list()
      for (effect in effects) {
        p_values <- rep(NA_real_, repetitions)
        valid <- rep(FALSE, repetitions)
        invalid_draws <- infinite_draws <- integer(repetitions)
        chunks <- split(seq_len(repetitions),
                        ceiling(seq_len(repetitions) / settings$chunk_size))
        for (indices in chunks) {
          # Overall sigma cancels from t. Scores under an injected alternative
          # are Q0 + (effect/sigma)*H, with the null still imposed at beta=0.
          q <- q0[, indices, drop = FALSE] + (effect / sigma) * h_total
          test <- border_bootstrap_pvalues(
            border_bootstrap_scalar_t(q, h_total, weights)
          )
          p_values[indices] <- test$p_value
          valid[indices] <- test$valid
          invalid_draws[indices] <- test$invalid_bootstrap_draws
          infinite_draws[indices] <- test$infinite_bootstrap_draws
        }
        reject <- valid & !is.na(p_values) & p_values <= assumptions$alpha
        successes <- sum(reject)
        interval <- border_power_wilson(successes, repetitions)
        curve[[length(curve) + 1L]] <- data.frame(
          model_id = input$model_id, dgp_id = input$dgp_id,
          scale_information = scale_name, inference_method = "pair_WCR11_FNW",
          effect = effect, repetitions = repetitions,
          valid_repetitions = sum(valid), invalid_repetitions = sum(!valid),
          rejections = successes, rejection_rate = successes / repetitions,
          rejection_rate_valid_only = if (any(valid)) successes / sum(valid) else NA_real_,
          rejection_rate_mc_se = sqrt(successes / repetitions *
                                       (1 - successes / repetitions) / repetitions),
          mc_confidence_low = interval[1L, "low"],
          mc_confidence_high = interval[1L, "high"],
          invalid_bootstrap_draws = sum(invalid_draws),
          repetitions_with_invalid_bootstrap_draws = sum(invalid_draws > 0),
          infinite_bootstrap_draws = sum(infinite_draws),
          alpha = assumptions$alpha, sigma = sigma,
          bootstrap_repetitions = bootstrap$repetitions,
          bootstrap_seed = bootstrap$seed, simulation_seed = settings$seed,
          simulation_draw_selection = "first_declared_repetitions_of_same_raw_power_draws",
          bootstrap_weights = "webb", bootstrap_type = "fnw11",
          bootstrap_pvalue_convention = "strict_absolute_exceedance_no_plus_one",
          bootstrap_package_version = "0.14.3", n_obs = input$n_obs,
          n_pairs = input$n_pairs, n_regions = input$n_regions,
          fixed_effects_nested_in_pair = TRUE, n_regressors = 1L
        )
      }
      curve <- do.call(rbind, curve)
      null <- curve[curve$effect == 0, , drop = FALSE]
      curve$size_gate_pass <- null$invalid_repetitions == 0L &&
        null$mc_confidence_high <= assumptions$size_maximum
      rows[[length(rows) + 1L]] <- curve
    }
  }
  do.call(rbind, rows)
}
