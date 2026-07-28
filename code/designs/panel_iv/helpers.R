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

webb_cluster_multipliers <- function(
  bootstrap_reps,
  cluster_count,
  seed
) {
  stopifnot(
    length(bootstrap_reps) == 1L,
    bootstrap_reps >= 1L,
    length(cluster_count) == 1L,
    cluster_count >= 2L
  )
  support <- c(
    -sqrt(3 / 2),
    -1,
    -sqrt(1 / 2),
    sqrt(1 / 2),
    1,
    sqrt(3 / 2)
  )
  set.seed(seed)
  matrix(
    sample(
      support,
      size = bootstrap_reps * cluster_count,
      replace = TRUE
    ),
    nrow = bootstrap_reps,
    ncol = cluster_count
  )
}

cluster_score_components <- function(y, z, cluster) {
  stopifnot(
    length(y) == length(z),
    length(y) == length(cluster)
  )
  keep <- is.finite(y) & is.finite(z) & !is.na(cluster)
  y <- as.numeric(y[keep])
  z <- as.numeric(z[keep])
  cluster <- as.character(cluster[keep])
  cluster_levels <- sort(unique(cluster))
  cluster_id <- match(cluster, cluster_levels)

  stopifnot(
    length(cluster_levels) >= 2L,
    length(y) > length(cluster_levels)
  )

  cross_zy <- rowsum(
    z * y,
    cluster_id,
    reorder = FALSE
  )[, 1]
  cross_zz <- rowsum(
    z^2,
    cluster_id,
    reorder = FALSE
  )[, 1]
  total_zz <- sum(cross_zz)
  if (!is.finite(total_zz) || total_zz <= .Machine$double.eps) {
    stop("The residualized instrument has no variation.", call. = FALSE)
  }

  list(
    cross_zy = cross_zy,
    cross_zz = cross_zz,
    total_zz = total_zz,
    cluster_levels = cluster_levels,
    observation_count = length(y)
  )
}

cluster_score_statistic <- function(components) {
  cluster_count <- length(components$cluster_levels)
  estimate <- sum(components$cross_zy) / components$total_zz
  score <- components$cross_zy - estimate * components$cross_zz
  correction <- cluster_count / (cluster_count - 1)
  standard_error <- sqrt(correction * sum(score^2)) /
    components$total_zz
  statistic <- if (
    is.finite(standard_error) &&
      standard_error > .Machine$double.eps
  ) {
    estimate / standard_error
  } else {
    NA_real_
  }
  list(
    estimate = estimate,
    standard_error = standard_error,
    statistic = statistic,
    cluster_count = cluster_count,
    observation_count = components$observation_count
  )
}

wild_cluster_score_test <- function(
  y,
  z,
  cluster,
  bootstrap_reps = 999L,
  seed = 1L,
  multipliers = NULL
) {
  components <- cluster_score_components(y, z, cluster)
  observed <- cluster_score_statistic(components)
  cluster_count <- observed$cluster_count

  if (is.null(multipliers)) {
    multipliers <- webb_cluster_multipliers(
      bootstrap_reps,
      cluster_count,
      seed
    )
  }
  stopifnot(
    ncol(multipliers) == cluster_count,
    nrow(multipliers) >= 1L
  )

  bootstrap_numerator <- as.vector(
    multipliers %*% components$cross_zy
  )
  bootstrap_estimate <- bootstrap_numerator / components$total_zz
  bootstrap_score <- multipliers *
    matrix(
      components$cross_zy,
      nrow = nrow(multipliers),
      ncol = cluster_count,
      byrow = TRUE
    ) -
    bootstrap_estimate *
      matrix(
        components$cross_zz,
        nrow = nrow(multipliers),
        ncol = cluster_count,
        byrow = TRUE
      )

  correction <- cluster_count / (cluster_count - 1)
  bootstrap_se <- sqrt(
    correction * rowSums(bootstrap_score^2)
  ) / components$total_zz
  bootstrap_t <- bootstrap_estimate / bootstrap_se
  finite_bootstrap <- is.finite(bootstrap_t)
  bootstrap_p <- if (
    !is.finite(observed$statistic) ||
      !any(finite_bootstrap)
  ) {
    NA_real_
  } else {
    (
      1 +
        sum(
          abs(bootstrap_t[finite_bootstrap]) >=
            abs(observed$statistic)
        )
    ) / (1 + sum(finite_bootstrap))
  }

  c(
    observed,
    list(
      bootstrap_p_value = bootstrap_p,
      bootstrap_reps = sum(finite_bootstrap),
      bootstrap_distribution = "Webb six-point"
    )
  )
}

make_ar_beta_grid <- function(
  y,
  endogenous,
  center = 0,
  points = 401L,
  radius_multiplier = 10
) {
  stopifnot(points >= 3L, points %% 2L == 1L)
  scale_ratio <- stats::sd(y, na.rm = TRUE) /
    stats::sd(endogenous, na.rm = TRUE)
  if (!is.finite(scale_ratio) || scale_ratio <= 0) {
    scale_ratio <- max(1, abs(center))
  }
  radius <- radius_multiplier * max(
    scale_ratio,
    abs(center),
    .Machine$double.eps
  )
  seq(center - radius, center + radius, length.out = points)
}

anderson_rubin_grid <- function(
  y,
  endogenous,
  instrument,
  cluster,
  beta_grid,
  bootstrap_reps = 999L,
  seed = 1L
) {
  stopifnot(
    length(y) == length(endogenous),
    length(y) == length(instrument),
    length(y) == length(cluster)
  )
  keep <- is.finite(y) &
    is.finite(endogenous) &
    is.finite(instrument) &
    !is.na(cluster)
  y <- y[keep]
  endogenous <- endogenous[keep]
  instrument <- instrument[keep]
  cluster <- cluster[keep]
  cluster_levels <- sort(unique(as.character(cluster)))
  multipliers <- webb_cluster_multipliers(
    bootstrap_reps,
    length(cluster_levels),
    seed
  )

  rows <- lapply(
    seq_along(beta_grid),
    function(index) {
      beta_null <- beta_grid[[index]]
      test <- wild_cluster_score_test(
        y = y - beta_null * endogenous,
        z = instrument,
        cluster = cluster,
        multipliers = multipliers
      )
      data.frame(
        beta_null = beta_null,
        ar_estimate = test$estimate,
        ar_standard_error = test$standard_error,
        ar_t_statistic = test$statistic,
        ar_bootstrap_p_value = test$bootstrap_p_value,
        stringsAsFactors = FALSE
      )
    }
  )
  do.call(rbind, rows)
}

accepted_grid_intervals <- function(
  beta_grid,
  p_value,
  level = 0.95
) {
  stopifnot(length(beta_grid) == length(p_value))
  alpha <- 1 - level
  accepted <- is.finite(p_value) & p_value >= alpha
  if (!any(accepted)) {
    return(data.frame(
      interval_id = integer(),
      lower = numeric(),
      upper = numeric(),
      lower_hits_grid = logical(),
      upper_hits_grid = logical()
    ))
  }

  run <- rle(accepted)
  run_end <- cumsum(run$lengths)
  run_start <- run_end - run$lengths + 1L
  keep <- which(run$values)
  data.frame(
    interval_id = seq_along(keep),
    lower = beta_grid[run_start[keep]],
    upper = beta_grid[run_end[keep]],
    lower_hits_grid = run_start[keep] == 1L,
    upper_hits_grid = run_end[keep] == length(beta_grid)
  )
}
