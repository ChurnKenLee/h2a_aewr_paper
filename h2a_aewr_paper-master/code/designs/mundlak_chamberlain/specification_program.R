# Specification-program helpers for the version-3 Mundlak-Chamberlain design.
#
# This file is sourced after design.R and helpers.R.  The functions are kept
# free of file-system side effects: driver scripts own caching, checkpoints,
# and reporting.

mc_sp_calendar_registry <- function() {
  rows <- list()
  for (horizon_count in 1:3) {
    maximum_lag <- horizon_count - 1L
    earliest_analysis <- min(MC_SPEC_HISTORY_YEARS) + maximum_lag
    analysis_starts <- earliest_analysis +
      MC_SPEC_ANALYSIS_START_DELAYS

    for (analysis_start in analysis_starts) {
      for (preperiod_length in MC_SPEC_PREPERIOD_LENGTHS) {
        for (preperiod_start in 2008:2012) {
          preperiod_end <- preperiod_start + preperiod_length - 1L
          admissible <- (
            preperiod_end <= 2012L &&
              preperiod_end < analysis_start - maximum_lag
          )
          if (!admissible) {
            next
          }
          rows[[length(rows) + 1L]] <- data.frame(
            horizon_count = as.integer(horizon_count),
            maximum_lag = as.integer(maximum_lag),
            preperiod_start = as.integer(preperiod_start),
            preperiod_end = as.integer(preperiod_end),
            preperiod_length = as.integer(preperiod_length),
            analysis_start = as.integer(analysis_start),
            analysis_end = MC_SPEC_ANALYSIS_END,
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }

  registry <- unique(do.call(rbind, rows))
  registry <- registry[
    order(
      registry$horizon_count,
      registry$analysis_start,
      registry$preperiod_start,
      registry$preperiod_end
    ),
    ,
    drop = FALSE
  ]
  rownames(registry) <- NULL
  registry$calendar_id <- sprintf(
    "h%d_a%d_p%d_%d",
    registry$horizon_count,
    registry$analysis_start,
    registry$preperiod_start,
    registry$preperiod_end
  )
  registry <- registry[
    c(
      "calendar_id",
      "horizon_count",
      "maximum_lag",
      "preperiod_start",
      "preperiod_end",
      "preperiod_length",
      "analysis_start",
      "analysis_end"
    )
  ]

  counts <- table(registry$horizon_count)
  if (
    nrow(registry) != 54L ||
      !identical(unname(as.integer(counts)), rep(18L, 3L))
  ) {
    stop(
      "The calendar compiler must emit 54 records, 18 per horizon count.",
      call. = FALSE
    )
  }
  registry
}

mc_specification_registry <- function() {
  calendars <- mc_sp_calendar_registry()
  rows <- list()
  for (calendar_index in seq_len(nrow(calendars))) {
    calendar <- calendars[calendar_index, , drop = FALSE]
    for (degree in MC_SPEC_POLYNOMIAL_DEGREES) {
      for (richness_tier in MC_SPEC_RICHNESS_TIERS) {
        rows[[length(rows) + 1L]] <- data.frame(
          calendar,
          polynomial_degree = as.integer(degree),
          moderated_polynomial_degree = as.integer(degree),
          richness_tier = as.integer(richness_tier),
          richness_label =
            MC_SPEC_RICHNESS_LABELS[[richness_tier + 1L]],
          heredity = "strong",
          history_rule = "maximal_common_basis",
          covariance_method = MC_CCV_METHOD,
          df_adjustment = MC_SPEC_DF_ADJUSTMENT,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  registry <- do.call(rbind, rows)
  rownames(registry) <- NULL
  registry$spec_id <- sprintf(
    "%s_d%d_r%d",
    registry$calendar_id,
    registry$polynomial_degree,
    registry$richness_tier
  )
  registry$primary_target <- with(
    registry,
    horizon_count == 3L &
      preperiod_start == min(MC_BASELINE_YEARS) &
      preperiod_end == max(MC_BASELINE_YEARS) &
      analysis_start == min(MC_ANALYSIS_YEARS) &
      analysis_end == max(MC_ANALYSIS_YEARS) &
      polynomial_degree == 2L &
      richness_tier == max(MC_SPEC_RICHNESS_TIERS)
  )
  registry$primary_family <- with(
    registry,
    horizon_count == 3L &
      preperiod_start == min(MC_BASELINE_YEARS) &
      preperiod_end == max(MC_BASELINE_YEARS) &
      analysis_start == min(MC_ANALYSIS_YEARS) &
      analysis_end == max(MC_ANALYSIS_YEARS) &
      polynomial_degree == 2L
  )
  registry$compact_lag_basis <- with(
    registry,
    preperiod_start == min(MC_BASELINE_YEARS) &
      preperiod_end == max(MC_BASELINE_YEARS) &
      analysis_start == min(MC_ANALYSIS_YEARS) &
      analysis_end == max(MC_ANALYSIS_YEARS) &
      richness_tier == 1L
  )
  registry$compact_calendar <- with(
    registry,
    horizon_count == 3L &
      polynomial_degree == 2L &
      richness_tier == 1L &
      (
        (
          analysis_start == min(MC_ANALYSIS_YEARS) &
            paste(preperiod_start, preperiod_end) %in%
              c("2008 2009", "2009 2010")
        ) |
          (
            preperiod_start == min(MC_BASELINE_YEARS) &
              preperiod_end == max(MC_BASELINE_YEARS) &
              analysis_start %in% c(2014L, 2015L)
          )
      )
  )
  registry$default_execution <- with(
    registry,
    primary_family | compact_lag_basis | compact_calendar
  )
  registry$candidate_status <- "candidate"

  if (
    nrow(registry) != 648L ||
      anyDuplicated(registry$spec_id) > 0L ||
      sum(registry$primary_target) != 1L
  ) {
    stop(
      "The specification compiler must emit 648 unique candidates and one target.",
      call. = FALSE
    )
  }
  registry
}

mc_sp_execution_registry <- function(
  registry,
  stage = MC_SPEC_DEFAULT_STAGE
) {
  supported <- c("primary", "compact", "exhaustive")
  if (!stage %in% supported) {
    stop(
      "MC_SPEC_STAGE must be one of: ",
      paste(supported, collapse = ", "),
      call. = FALSE
    )
  }
  selected <- switch(
    stage,
    primary = registry$primary_family,
    compact = registry$default_execution,
    exhaustive = rep(TRUE, nrow(registry))
  )
  result <- registry[selected, , drop = FALSE]
  result$execution_stage <- stage
  result$execution_reason <- ifelse(
    result$primary_family,
    "primary_family",
    ifelse(
      result$compact_lag_basis,
      "lag_basis_sensitivity",
      ifelse(
        result$compact_calendar,
        "calendar_sensitivity",
        "exhaustive_grid"
      )
    )
  )
  result$execution_priority <- match(
    result$execution_reason,
    c(
      "primary_family",
      "lag_basis_sensitivity",
      "calendar_sensitivity",
      "exhaustive_grid"
    )
  )
  result <- result[
    order(
      result$execution_priority,
      -result$richness_tier,
      result$horizon_count,
      result$polynomial_degree,
      result$calendar_id
    ),
    ,
    drop = FALSE
  ]
  rownames(result) <- NULL

  expected <- switch(
    stage,
    primary = 4L,
    compact = 16L,
    exhaustive = 648L
  )
  if (nrow(result) != expected || anyDuplicated(result$spec_id) > 0L) {
    stop(
      "Execution-stage compiler emitted ",
      nrow(result),
      " specifications; expected ",
      expected,
      ".",
      call. = FALSE
    )
  }
  result
}

mc_sp_specification <- function(registry_row) {
  if (!is.data.frame(registry_row) || nrow(registry_row) != 1L) {
    stop("A specification must be one registry row.", call. = FALSE)
  }
  list(
    spec_id = registry_row$spec_id[[1]],
    calendar_id = registry_row$calendar_id[[1]],
    preperiod_years = seq.int(
      registry_row$preperiod_start[[1]],
      registry_row$preperiod_end[[1]]
    ),
    analysis_years = seq.int(
      registry_row$analysis_start[[1]],
      registry_row$analysis_end[[1]]
    ),
    horizon_count = registry_row$horizon_count[[1]],
    treatment_columns = unname(MC_DYNAMIC_HORIZONS)[
      seq_len(registry_row$horizon_count[[1]])
    ],
    lag_orders = 0:registry_row$maximum_lag[[1]],
    polynomial_degrees =
      seq_len(registry_row$polynomial_degree[[1]]),
    moderated_polynomial_degrees =
      seq_len(registry_row$moderated_polynomial_degree[[1]]),
    richness_tier = registry_row$richness_tier[[1]],
    richness_label = registry_row$richness_label[[1]],
    heredity = registry_row$heredity[[1]],
    history_years = MC_SPEC_HISTORY_YEARS,
    region_budget = MC_SPEC_REGION_BUDGET,
    primary_target = registry_row$primary_target[[1]]
  )
}

mc_sp_treatment_columns <- function(specification) {
  if (!is.null(specification$treatment_columns)) {
    return(unname(specification$treatment_columns))
  }
  unname(MC_DYNAMIC_HORIZONS)[
    seq_len(specification$horizon_count)
  ]
}

mc_sp_treatment_lag_orders <- function(specification) {
  columns <- mc_sp_treatment_columns(specification)
  if (length(specification$lag_orders) != length(columns)) {
    stop("Every treatment column must have one lag order.", call. = FALSE)
  }
  as.integer(specification$lag_orders)
}

mc_sp_horizon_name <- function(treatment_column) {
  if (identical(treatment_column, "mc_dose_lead1")) {
    return("one_year_lead")
  }
  name <- names(MC_DYNAMIC_HORIZONS)[
    match(treatment_column, unname(MC_DYNAMIC_HORIZONS))
  ]
  if (length(name) != 1L || is.na(name)) {
    stop("Unknown treatment column: ", treatment_column, call. = FALSE)
  }
  name
}

mc_sp_placebo_specification <- function(specification) {
  placebo <- specification
  placebo$spec_id <- paste0(specification$spec_id, "_lead1")
  placebo$analysis_years <- specification$analysis_years[
    specification$analysis_years < max(specification$analysis_years)
  ]
  placebo$treatment_columns <- c(
    mc_sp_treatment_columns(specification),
    "mc_dose_lead1"
  )
  placebo$lag_orders <- c(
    mc_sp_treatment_lag_orders(specification),
    -1L
  )
  placebo$horizon_count <- length(placebo$treatment_columns)
  placebo$primary_target <- FALSE
  placebo$placebo <- TRUE
  placebo
}

mc_sp_history_selection <- function(specification) {
  rows <- list()
  forced_count <- 1L +
    length(mc_sp_treatment_columns(specification)) *
      length(specification$polynomial_degrees)
  per_year_capacity <- specification$region_budget - forced_count
  if (per_year_capacity < 0L) {
    return(data.frame(
      outcome_year = integer(),
      history_year = integer(),
      kept = logical(),
      priority = integer(),
      reason = character(),
      stringsAsFactors = FALSE
    ))
  }

  for (outcome_year in specification$analysis_years) {
    focal_years <- outcome_year - specification$lag_orders
    candidate_years <- setdiff(
      specification$history_years,
      focal_years
    )
    ordering <- order(
      abs(candidate_years - outcome_year),
      candidate_years > outcome_year,
      candidate_years
    )
    ranked <- candidate_years[ordering]
    kept_years <- head(ranked, per_year_capacity)
    rows[[length(rows) + 1L]] <- data.frame(
      outcome_year = as.integer(outcome_year),
      history_year = as.integer(ranked),
      kept = ranked %in% kept_years,
      priority = seq_along(ranked),
      distance = abs(ranked - outcome_year),
      future = ranked > outcome_year,
      reason = ifelse(
        ranked %in% kept_years,
        "kept_by_fixed_priority",
        "trimmed_for_region_budget"
      ),
      stringsAsFactors = FALSE
    )
  }
  selection <- do.call(rbind, rows)

  # Year and region fixed effects already occupy T + R - 1 dimensions of the
  # R x T assignment-cell space.  The remaining region-by-time capacity is
  # (R - 1)(T - 1), so a per-year count alone is not sufficient.  Apply the
  # same outcome-free priority globally and trim the least local coordinates
  # until the complete block fits that space.
  year_count <- length(specification$analysis_years)
  interaction_capacity <- (MC_CCV_REFERENCE_STATES - 1L) *
    (year_count - 1L)
  causal_count <- length(mc_sp_treatment_columns(specification)) *
    length(specification$polynomial_degrees) *
    year_count
  global_history_capacity <- max(
    interaction_capacity -
      causal_count -
      MC_SPEC_GLOBAL_REGION_RESERVE -
      if (is.null(specification$additional_history_reserve)) {
        0L
      } else {
        as.integer(specification$additional_history_reserve)
      },
    0L
  )
  retained <- which(selection$kept)
  if (length(retained) > global_history_capacity) {
    removal_order <- retained[
      order(
        -selection$distance[retained],
        -selection$future[retained],
        -selection$outcome_year[retained],
        -selection$history_year[retained]
      )
    ]
    remove_count <- length(retained) - global_history_capacity
    removed <- head(removal_order, remove_count)
    selection$kept[removed] <- FALSE
    selection$reason[removed] <-
      "trimmed_for_global_region_time_budget"
  }
  selection
}

mc_sp_region_budget_audit <- function(registry) {
  rows <- list()
  for (index in seq_len(nrow(registry))) {
    specification <- mc_sp_specification(
      registry[index, , drop = FALSE]
    )
    history <- mc_sp_history_selection(specification)
    for (outcome_year in specification$analysis_years) {
      history_count <- sum(
        history$outcome_year == outcome_year & history$kept
      )
      forced_count <- 1L +
        length(mc_sp_treatment_columns(specification)) *
          length(specification$polynomial_degrees)
      total <- forced_count + history_count
      rows[[length(rows) + 1L]] <- data.frame(
        spec_id = specification$spec_id,
        outcome_year = as.integer(outcome_year),
        year_coordinate = 1L,
        causal_region_coordinates =
          length(mc_sp_treatment_columns(specification)) *
            length(specification$polynomial_degrees),
        candidate_history_coordinates = sum(
          history$outcome_year == outcome_year
        ),
        retained_history_coordinates = history_count,
        region_coordinates = total,
        region_budget = specification$region_budget,
        status = ifelse(
          total <= specification$region_budget,
          "budget_admissible",
          "region_budget_exceeded"
        ),
        stringsAsFactors = FALSE
      )
    }
  }
  audit <- do.call(rbind, rows)
  if (any(audit$region_coordinates > audit$region_budget)) {
    stop("The history compiler emitted an over-budget record.", call. = FALSE)
  }
  audit
}

mc_sp_resolve_history_selection <- function(
  specification,
  panel,
  metadata
) {
  selection <- mc_sp_history_selection(specification)
  selection$kept_arithmetic <- selection$kept
  selection$kept <- FALSE
  selection$reason <- ifelse(
    selection$kept_arithmetic,
    "pending_all_state_rank",
    selection$reason
  )

  assignment_columns <- unique(c(
    unname(MC_DYNAMIC_HORIZONS),
    "mc_dose_lead1",
    metadata$region_treatment_history_map$constructed_column
  ))
  region_cells <- unique(
    panel[
      c(
        "aewr_region_id",
        "year",
        assignment_columns
      )
    ]
  )
  region_cells <- region_cells[
    order(region_cells$aewr_region_id, region_cells$year),
    ,
    drop = FALSE
  ]
  state_cells <- lapply(
    0:(MC_CCV_REFERENCE_STATES - 1L),
    function(state_index) {
      if (state_index == 0L) {
        mc_refresh_treatment_basis(region_cells)
      } else {
        mc_ccv_reference_state(
          data = region_cells,
          metadata = metadata,
          state_index = state_index
        )
      }
    }
  )

  fixed_matrices <- lapply(
    state_cells,
    function(state_data) {
      stats::model.matrix(
        ~factor(year) + factor(aewr_region_id),
        data = state_data
      )
    }
  )
  causal_matrices <- lapply(
    state_cells,
    function(state_data) {
      columns <- list()
      for (outcome_year in specification$analysis_years) {
        for (
          treatment_column in mc_sp_treatment_columns(specification)
        ) {
          for (degree in specification$polynomial_degrees) {
            basis_column <- mc_polynomial_basis_name(
              treatment_column,
              degree
            )
            column_name <- paste(
              outcome_year,
              basis_column,
              sep = ":"
            )
            columns[[column_name]] <-
              as.integer(state_data$year == outcome_year) *
                state_data[[basis_column]]
          }
        }
      }
      do.call(cbind, columns)
    }
  )
  active_matrices <- Map(
    cbind,
    fixed_matrices,
    causal_matrices
  )
  active_ranks <- vapply(
    active_matrices,
    function(matrix) qr(matrix, tol = 1e-9)$rank,
    integer(1)
  )
  expected_ranks <- vapply(
    fixed_matrices,
    function(matrix) qr(matrix, tol = 1e-9)$rank,
    integer(1)
  ) + ncol(causal_matrices[[1L]])
  if (any(active_ranks != expected_ranks)) {
    stop(
      "The forced causal region block is not full rank in every state.",
      call. = FALSE
    )
  }

  arithmetic_candidates <- which(selection$kept_arithmetic)
  candidate_order <- arithmetic_candidates[
    order(
      selection$priority[arithmetic_candidates],
      selection$distance[arithmetic_candidates],
      selection$future[arithmetic_candidates],
      selection$outcome_year[arithmetic_candidates],
      selection$history_year[arithmetic_candidates]
    )
  ]
  for (selection_index in candidate_order) {
    outcome_year <- selection$outcome_year[[selection_index]]
    history_year <- selection$history_year[[selection_index]]
    history_column <-
      metadata$region_treatment_history_map$constructed_column[
        metadata$region_treatment_history_map$history_year ==
          history_year
      ]
    candidate_columns <- lapply(
      state_cells,
      function(state_data) {
        as.integer(state_data$year == outcome_year) *
          state_data[[history_column]]
      }
    )
    candidate_matrices <- Map(
      function(matrix, column) cbind(matrix, column),
      active_matrices,
      candidate_columns
    )
    candidate_ranks <- vapply(
      candidate_matrices,
      function(matrix) qr(matrix, tol = 1e-9)$rank,
      integer(1)
    )
    if (all(candidate_ranks == active_ranks + 1L)) {
      selection$kept[[selection_index]] <- TRUE
      selection$reason[[selection_index]] <-
        "kept_by_all_state_rank"
      active_matrices <- candidate_matrices
      active_ranks <- candidate_ranks
    } else {
      selection$reason[[selection_index]] <-
        "trimmed_for_common_basis"
    }
  }
  selection
}

mc_sp_trajectory_weights <- function(years) {
  years <- as.integer(years)
  year_count <- length(years)
  if (year_count < 2L) {
    stop("Trajectory windows require at least two years.", call. = FALSE)
  }
  weights <- cbind(
    mean = rep(1 / year_count, year_count),
    stats::contr.helmert(year_count)
  )
  if (ncol(weights) > 1L) {
    for (column in 2:ncol(weights)) {
      weights[, column] <- weights[, column] /
        sqrt(sum(weights[, column]^2))
    }
    colnames(weights)[-1L] <- paste0(
      "trajectory_",
      seq_len(year_count - 1L)
    )
  }
  rownames(weights) <- as.character(years)
  weights
}

mc_sp_year_contrasts <- function(years) {
  years <- as.integer(years)
  contrasts <- stats::contr.helmert(length(years))
  for (column in seq_len(ncol(contrasts))) {
    contrasts[, column] <- contrasts[, column] /
      sqrt(sum(contrasts[, column]^2))
  }
  rownames(contrasts) <- as.character(years)
  colnames(contrasts) <- paste0(
    "mc_sp_year_c",
    seq_len(ncol(contrasts))
  )
  contrasts
}

mc_sp_prepare_shared_panel <- function(shared_panel) {
  if (
    !"h2a_prediction_cutoff_year" %in% names(shared_panel) ||
      any(
        !is.na(shared_panel$h2a_prediction_cutoff_year) &
          shared_panel$h2a_prediction_cutoff_year !=
            H2A_PREDICTION_CUTOFF_YEAR
      )
  ) {
    stop("The MC panel has an unexpected H-2A prediction cutoff.", call. = FALSE)
  }
  if (
    !"h2a_prediction_model_spec" %in% names(shared_panel) ||
      !identical(
        unique(shared_panel$h2a_prediction_model_spec[
          !is.na(shared_panel$h2a_prediction_model_spec)
        ]),
        H2A_PREDICTION_MODEL_SPEC
      )
  ) {
    stop("The MC panel has an unexpected H-2A model spec.", call. = FALSE)
  }
  prediction_contract <- shared_panel %>%
    filter(!is.na(h2a_predicted_share_2011)) %>%
    distinct(county_fips, h2a_predicted_share_2011)
  if (anyDuplicated(prediction_contract$county_fips) > 0L) {
    stop("The MC panel must use one static H-2A propensity per county.",
      call. = FALSE)
  }
  shared_panel$year <- as.integer(shared_panel$year)
  shared_panel$state_fips <- as.integer(shared_panel$state_fips)
  shared_panel$aewr_region_id <- as.integer(
    shared_panel$aewr_region_id
  )
  shared_panel$mc_market_id <- mc_make_market_id(
    shared_panel$aewr_region_id,
    shared_panel$state_fips,
    shared_panel$cz_id
  )
  shared_panel <- shared_panel[
    shared_panel$year >= 2008L &
      shared_panel$year <= MC_SPEC_ANALYSIS_END,
    ,
    drop = FALSE
  ]
  if (anyDuplicated(shared_panel[c("county_fips", "year")]) > 0L) {
    stop("The shared MC panel must have unique county-year keys.", call. = FALSE)
  }
  shared_panel
}

mc_sp_eligible_counties <- function(shared_panel) {
  eligible <- shared_panel[
    shared_panel$year == MC_REFERENCE_YEAR &
      is.finite(shared_panel$emp_farm_2011) &
      shared_panel$emp_farm_2011 > 0 &
      !is.na(shared_panel$mc_market_id),
    c(
      "county_fips",
      "state_fips",
      "aewr_region_id",
      "mc_market_id",
      "emp_farm_2011"
    ),
    drop = FALSE
  ]
  unique(eligible)
}

mc_sp_add_standardized_source <- function(
  county_design,
  values,
  column,
  source_column,
  statistic,
  source_variable,
  inventory_rows,
  scaling_rows
) {
  county_design[[column]] <- values
  county_design <- mc_impute_by_region(county_design, column)
  standardized <- mc_safe_standardize(county_design[[column]])
  standardized_column <- paste0(column, "_z")
  county_design[[standardized_column]] <- standardized$value
  inventory_rows[[length(inventory_rows) + 1L]] <- data.frame(
    constructed_column = standardized_column,
    source_variable = source_variable,
    source_column = source_column,
    statistic = statistic,
    hierarchy_level = "county_input",
    role = "specification_source",
    stringsAsFactors = FALSE
  )
  scaling_rows[[length(scaling_rows) + 1L]] <- data.frame(
    constructed_column = standardized_column,
    source_column = source_column,
    hierarchy_level = "county_input",
    center = standardized$center,
    scale = standardized$scale,
    stringsAsFactors = FALSE
  )
  list(
    data = county_design,
    standardized_column = standardized_column,
    inventory_rows = inventory_rows,
    scaling_rows = scaling_rows
  )
}

mc_sp_add_components <- function(
  county_design,
  standardized_column,
  stub,
  source_variable,
  statistic,
  inventory_rows,
  scaling_rows
) {
  result <- mc_hierarchical_components(
    county_design,
    standardized_column,
    stub
  )
  for (level in names(result$columns)) {
    inventory_rows[[length(inventory_rows) + 1L]] <- data.frame(
      constructed_column = result$columns[[level]],
      source_variable = source_variable,
      source_column = standardized_column,
      statistic = statistic,
      hierarchy_level = level,
      role = "specification_component",
      stringsAsFactors = FALSE
    )
  }
  scaling_rows[[length(scaling_rows) + 1L]] <- result$scaling
  list(
    data = result$data,
    columns = result$columns,
    inventory_rows = inventory_rows,
    scaling_rows = scaling_rows
  )
}

mc_sp_build_calendar_panel <- function(shared_panel, calendar_row) {
  if (!is.data.frame(calendar_row) || nrow(calendar_row) != 1L) {
    stop("Calendar construction requires one calendar row.", call. = FALSE)
  }
  shared_panel <- mc_sp_prepare_shared_panel(shared_panel)
  eligible_counties <- mc_sp_eligible_counties(shared_panel)
  preperiod_years <- seq.int(
    calendar_row$preperiod_start[[1]],
    calendar_row$preperiod_end[[1]]
  )
  analysis_years <- seq.int(
    calendar_row$analysis_start[[1]],
    calendar_row$analysis_end[[1]]
  )

  baseline_long <- shared_panel |>
    dplyr::semi_join(eligible_counties, by = "county_fips") |>
    dplyr::filter(.data$year %in% preperiod_years)
  for (variable_name in names(MC_BASELINE_VARIABLES)) {
    source_column <- MC_BASELINE_VARIABLES[[variable_name]]
    raw_column <- paste0("mc_sp_raw_", variable_name)
    baseline_long[[raw_column]] <- mc_transform_baseline(
      baseline_long[[source_column]],
      variable_name,
      baseline_long$emp_farm
    )
  }

  county_design <- eligible_counties
  inventory_rows <- list()
  scaling_rows <- list()
  summary_sources <- list()
  summary_components <- list()
  period_sources <- list()
  period_components <- list()
  trajectory_sources <- list()
  trajectory_components <- list()

  trajectory_weights <- mc_sp_trajectory_weights(preperiod_years)
  time_varying_variables <- names(MC_BASELINE_VARIABLES)

  for (variable_name in names(MC_BASELINE_VARIABLES)) {
    source_column <- MC_BASELINE_VARIABLES[[variable_name]]
    raw_column <- paste0("mc_sp_raw_", variable_name)
    summary_table <- baseline_long |>
      dplyr::group_by(.data$county_fips) |>
      dplyr::summarise(
        value_mean = mc_finite_mean(.data[[raw_column]]),
        value_trend = mc_linear_slope(
          .data[[raw_column]],
          .data$year
        ),
        .groups = "drop"
      )

    summary_sources[[variable_name]] <- list()
    summary_components[[variable_name]] <- list()
    for (statistic in c("mean", "trend")) {
      raw_summary_column <- paste0(
        "mc_b_",
        variable_name,
        "_",
        statistic
      )
      value_column <- paste0("value_", statistic)
      piece <- summary_table |>
        dplyr::transmute(
          county_fips = .data$county_fips,
          value = .data[[value_column]]
        )
      values <- piece$value[
        match(county_design$county_fips, piece$county_fips)
      ]
      added <- mc_sp_add_standardized_source(
        county_design = county_design,
        values = values,
        column = raw_summary_column,
        source_column = source_column,
        statistic = statistic,
        source_variable = variable_name,
        inventory_rows = inventory_rows,
        scaling_rows = scaling_rows
      )
      county_design <- added$data
      inventory_rows <- added$inventory_rows
      scaling_rows <- added$scaling_rows
      summary_sources[[variable_name]][[statistic]] <-
        added$standardized_column

      component <- mc_sp_add_components(
        county_design = county_design,
        standardized_column = added$standardized_column,
        stub = paste0(variable_name, "_", statistic),
        source_variable = variable_name,
        statistic = statistic,
        inventory_rows = inventory_rows,
        scaling_rows = scaling_rows
      )
      county_design <- component$data
      inventory_rows <- component$inventory_rows
      scaling_rows <- component$scaling_rows
      summary_components[[variable_name]][[statistic]] <-
        component$columns
    }

    if (!variable_name %in% time_varying_variables) {
      next
    }

    period_sources[[variable_name]] <- list()
    period_components[[variable_name]] <- list()
    trajectory_sources[[variable_name]] <- list()
    trajectory_components[[variable_name]] <- list()
    period_table <- baseline_long |>
      dplyr::select(
        .data$county_fips,
        .data$year,
        dplyr::all_of(raw_column)
      ) |>
      tidyr::pivot_wider(
        names_from = .data$year,
        values_from = dplyr::all_of(raw_column),
        names_prefix = "value_"
      )
    imputed_period_columns <- character()

    for (preperiod_year in preperiod_years) {
      value_column <- paste0("value_", preperiod_year)
      raw_period_column <- paste0(
        "mc_b_",
        variable_name,
        "_",
        preperiod_year
      )
      values <- period_table[[value_column]][
        match(county_design$county_fips, period_table$county_fips)
      ]
      added <- mc_sp_add_standardized_source(
        county_design = county_design,
        values = values,
        column = raw_period_column,
        source_column = source_column,
        statistic = paste0("period_", preperiod_year),
        source_variable = variable_name,
        inventory_rows = inventory_rows,
        scaling_rows = scaling_rows
      )
      county_design <- added$data
      inventory_rows <- added$inventory_rows
      scaling_rows <- added$scaling_rows
      period_sources[[variable_name]][[as.character(preperiod_year)]] <-
        added$standardized_column
      imputed_period_columns <- c(
        imputed_period_columns,
        raw_period_column
      )

      component <- mc_sp_add_components(
        county_design = county_design,
        standardized_column = added$standardized_column,
        stub = paste0(variable_name, "_", preperiod_year),
        source_variable = variable_name,
        statistic = paste0("period_", preperiod_year),
        inventory_rows = inventory_rows,
        scaling_rows = scaling_rows
      )
      county_design <- component$data
      inventory_rows <- component$inventory_rows
      scaling_rows <- component$scaling_rows
      period_components[[variable_name]][[as.character(preperiod_year)]] <-
        component$columns
    }

    imputed_period_values <- as.matrix(
      county_design[, imputed_period_columns, drop = FALSE]
    )
    for (
      contrast_index in seq_len(ncol(trajectory_weights) - 1L)
    ) {
      contrast_name <- paste0("trajectory_", contrast_index)
      raw_trajectory_column <- paste0(
        "mc_b_",
        variable_name,
        "_",
        contrast_name
      )
      contrast_values <- drop(
        imputed_period_values %*%
          trajectory_weights[, contrast_index + 1L]
      )
      added <- mc_sp_add_standardized_source(
        county_design = county_design,
        values = contrast_values,
        column = raw_trajectory_column,
        source_column = source_column,
        statistic = contrast_name,
        source_variable = variable_name,
        inventory_rows = inventory_rows,
        scaling_rows = scaling_rows
      )
      county_design <- added$data
      inventory_rows <- added$inventory_rows
      scaling_rows <- added$scaling_rows
      trajectory_sources[[variable_name]][[contrast_name]] <-
        added$standardized_column

      if (!identical(variable_name, MC_Z_VARIABLE)) {
        component <- mc_sp_add_components(
          county_design = county_design,
          standardized_column = added$standardized_column,
          stub = paste0(variable_name, "_", contrast_name),
          source_variable = variable_name,
          statistic = contrast_name,
          inventory_rows = inventory_rows,
          scaling_rows = scaling_rows
        )
        county_design <- component$data
        inventory_rows <- component$inventory_rows
        scaling_rows <- component$scaling_rows
        trajectory_components[[variable_name]][[contrast_name]] <-
          component$columns
      }
    }
  }

  bite_mean_column <- summary_sources[[MC_Z_VARIABLE]][["mean"]]
  county_design$mc_z <- county_design[[bite_mean_column]]

  nonbite_variables <- setdiff(
    names(MC_BASELINE_VARIABLES),
    MC_Z_VARIABLE
  )
  mean_nuisance_columns <- unlist(
    lapply(
      nonbite_variables,
      function(variable_name) {
        summary_components[[variable_name]][["mean"]][
          c("county", "market", "state")
        ]
      }
    ),
    use.names = FALSE
  )
  mean_causal_columns <- unlist(
    lapply(
      nonbite_variables,
      function(variable_name) {
        summary_components[[variable_name]][["mean"]][
          c("county", "market")
        ]
      }
    ),
    use.names = FALSE
  )

  trajectory_causal_columns <- mean_causal_columns
  trajectory_nuisance_columns <- character()
  for (variable_name in nonbite_variables) {
    for (
      contrast_name in names(
        trajectory_components[[variable_name]]
      )
    ) {
      component_columns <-
        trajectory_components[[variable_name]][[contrast_name]]
      trajectory_causal_columns <- c(
        trajectory_causal_columns,
        component_columns[c("county", "market")]
      )
      trajectory_nuisance_columns <- c(
        trajectory_nuisance_columns,
        component_columns[c("county", "market")]
      )
    }
  }
  bite_trajectory_columns <- c(
    "mc_z",
    unname(unlist(
      trajectory_sources[[MC_Z_VARIABLE]],
      use.names = FALSE
    ))
  )

  legacy_intercept_columns <- character()
  for (variable_name in names(MC_BASELINE_VARIABLES)) {
    if (variable_name %in% MC_CHAMBERLAIN_VARIABLES) {
      for (preperiod_year in preperiod_years) {
        legacy_intercept_columns <- c(
          legacy_intercept_columns,
          period_components[[variable_name]][[
            as.character(preperiod_year)
          ]][c("county", "market", "state")]
        )
      }
    } else {
      for (statistic in c("mean", "trend")) {
        legacy_intercept_columns <- c(
          legacy_intercept_columns,
          summary_components[[variable_name]][[statistic]][
            c("county", "market", "state")
          ]
        )
      }
    }
  }

  bite_nuisance_product_columns <- character()
  for (index in seq_along(mean_nuisance_columns)) {
    product_column <- sprintf(
      "mc_sp_bite_nuisance_%03d",
      index
    )
    county_design[[product_column]] <-
      county_design$mc_z *
        county_design[[mean_nuisance_columns[[index]]]]
    bite_nuisance_product_columns <- c(
      bite_nuisance_product_columns,
      product_column
    )
  }

  bite_three_way_columns <- character()
  for (index in seq_along(trajectory_causal_columns)) {
    product_column <- sprintf(
      "mc_sp_bite_causal_%03d",
      index
    )
    county_design[[product_column]] <-
      county_design$mc_z *
        county_design[[trajectory_causal_columns[[index]]]]
    bite_three_way_columns <- c(
      bite_three_way_columns,
      product_column
    )
  }

  quadratic_base_columns <- c(
    "mc_z",
    unlist(
      lapply(
        nonbite_variables,
        function(variable_name) {
          summary_components[[variable_name]][["mean"]][["county"]]
        }
      ),
      use.names = FALSE
    )
  )
  quadratic_columns <- character()
  quadratic_lookup_rows <- list()
  pair_index <- 0L
  for (left in seq_along(quadratic_base_columns)) {
    for (right in left:length(quadratic_base_columns)) {
      pair_index <- pair_index + 1L
      product_column <- sprintf(
        "mc_sp_county_pair_%03d",
        pair_index
      )
      county_design[[product_column]] <-
        county_design[[quadratic_base_columns[[left]]]] *
          county_design[[quadratic_base_columns[[right]]]]
      quadratic_columns <- c(quadratic_columns, product_column)
      quadratic_lookup_rows[[length(quadratic_lookup_rows) + 1L]] <-
        data.frame(
          constructed_column = product_column,
          left_column = quadratic_base_columns[[left]],
          right_column = quadratic_base_columns[[right]],
          stringsAsFactors = FALSE
        )
    }
  }
  if (length(quadratic_columns) != 78L) {
    stop("The county second-order block must contain 78 columns.", call. = FALSE)
  }

  region_treatment <- shared_panel |>
    dplyr::filter(.data$year >= 2010L) |>
    dplyr::distinct(
      .data$aewr_region_id,
      .data$year,
      .data$aewr
    ) |>
    dplyr::arrange(.data$aewr_region_id, .data$year) |>
    dplyr::group_by(.data$aewr_region_id) |>
    dplyr::mutate(
      mc_dose_current =
        100 * (log(.data$aewr) - dplyr::lag(log(.data$aewr))),
      mc_dose_lag1 = dplyr::lag(.data$mc_dose_current),
      mc_dose_lag2 = dplyr::lag(.data$mc_dose_current, n = 2L),
      mc_dose_lead1 = dplyr::lead(.data$mc_dose_current)
    ) |>
    dplyr::ungroup()
  region_history <- region_treatment |>
    dplyr::filter(.data$year %in% MC_SPEC_HISTORY_YEARS) |>
    dplyr::select(
      .data$aewr_region_id,
      .data$year,
      .data$mc_dose_current
    ) |>
    tidyr::pivot_wider(
      names_from = .data$year,
      values_from = .data$mc_dose_current,
      names_prefix = "mc_r_dose_history_"
    )
  region_history_columns <- setdiff(
    names(region_history),
    "aewr_region_id"
  )
  for (column in region_history_columns) {
    standardized <- mc_safe_standardize(region_history[[column]])
    region_history[[column]] <- standardized$value
  }

  analysis_panel <- shared_panel |>
    dplyr::semi_join(eligible_counties, by = "county_fips") |>
    dplyr::filter(.data$year %in% analysis_years) |>
    dplyr::select(-.data$mc_market_id) |>
    dplyr::left_join(
      county_design,
      by = c(
        "county_fips",
        "state_fips",
        "aewr_region_id",
        "emp_farm_2011"
      ),
      relationship = "many-to-one"
    ) |>
    dplyr::left_join(
      region_treatment |>
        dplyr::select(
          .data$aewr_region_id,
          .data$year,
          .data$mc_dose_current,
          .data$mc_dose_lag1,
          .data$mc_dose_lag2,
          .data$mc_dose_lead1
        ),
      by = c("aewr_region_id", "year"),
      relationship = "many-to-one"
    ) |>
    dplyr::left_join(
      region_history,
      by = "aewr_region_id",
      relationship = "many-to-one"
    ) |>
    dplyr::mutate(
      mc_y_applications = .data$nbr_applications_start_year,
      mc_y_employers = .data$nbr_employers_balanced_start_year,
      mc_y_requested_positions =
        .data$nbr_workers_requested_start_year,
      mc_y_certified_positions =
        .data$nbr_workers_certified_start_year,
      mc_y_certified_hours_thousands =
        .data$man_hours_certified_start_year / 1000,
      mc_y_applications_per_1000 =
        1000 * .data$nbr_applications_start_year /
          .data$emp_farm_2011,
      mc_y_requested_positions_per_1000 =
        1000 * .data$nbr_workers_requested_start_year /
          .data$emp_farm_2011,
      mc_y_certified_positions_per_1000 =
        1000 * .data$nbr_workers_certified_start_year /
          .data$emp_farm_2011,
      mc_y_certified_hours_per_worker =
        .data$man_hours_certified_start_year /
          .data$emp_farm_2011,
      mc_y_any_application =
        as.integer(.data$nbr_applications_start_year > 0),
      mc_y_positions_per_application = dplyr::if_else(
        .data$nbr_applications_start_year > 0,
        .data$nbr_workers_certified_start_year /
          .data$nbr_applications_start_year,
        NA_real_
      ),
      mc_y_hours_per_position = dplyr::if_else(
        .data$nbr_workers_certified_start_year > 0,
        .data$man_hours_certified_start_year /
          .data$nbr_workers_certified_start_year,
        NA_real_
      )
    )
  analysis_panel <- mc_refresh_treatment_basis(analysis_panel)

  year_contrasts <- mc_sp_year_contrasts(analysis_years)
  year_rows <- match(
    as.character(analysis_panel$year),
    rownames(year_contrasts)
  )
  for (column in colnames(year_contrasts)) {
    analysis_panel[[column]] <- year_contrasts[year_rows, column]
  }
  analysis_panel <- analysis_panel |>
    dplyr::group_by(.data$year) |>
    dplyr::mutate(
      mc_binding_quartile = dplyr::ntile(.data$mc_z, 4L),
      mc_baseline_h2a_quartile = dplyr::ntile(
        .data[["mc_b_h2a_cert_intensity_mean_z"]],
        4L
      )
    ) |>
    dplyr::ungroup() |>
    dplyr::arrange(.data$county_fips, .data$year) |>
    dplyr::mutate(mc_row_id = dplyr::row_number())

  expected_rows <- nrow(eligible_counties) * length(analysis_years)
  if (
    nrow(analysis_panel) != expected_rows ||
      anyDuplicated(
        analysis_panel[c("county_fips", "year")]
      ) > 0L
  ) {
    stop("A specification panel must be balanced.", call. = FALSE)
  }

  active_column <- function(columns) {
    columns <- unique(columns)
    columns[
      vapply(
        columns,
        function(column) {
          value <- analysis_panel[[column]]
          any(is.finite(value)) &&
            stats::sd(value[is.finite(value)]) >
              sqrt(.Machine$double.eps)
        },
        logical(1)
      )
    ]
  }

  metadata <- list(
    design_version = MC_SPEC_PROGRAM_VERSION,
    h2a_prediction_cutoff_year = H2A_PREDICTION_CUTOFF_YEAR,
    h2a_prediction_model_spec = H2A_PREDICTION_MODEL_SPEC,
    calendar_id = calendar_row$calendar_id[[1]],
    preperiod_years = preperiod_years,
    analysis_years = analysis_years,
    history_years = MC_SPEC_HISTORY_YEARS,
    outcomes = MC_OUTCOMES,
    ccv_method = MC_CCV_METHOD,
    ccv_reference_design = MC_CCV_REFERENCE_DESIGN,
    ccv_reference_states = MC_CCV_REFERENCE_STATES,
    cluster_df = MC_CCV_DF,
    region_treatment_history_map = data.frame(
      constructed_column = region_history_columns,
      history_year = as.integer(
        sub("^.*_", "", region_history_columns)
      ),
      stringsAsFactors = FALSE
    ),
    year_contrast_columns = colnames(year_contrasts),
    bite_mean_column = "mc_z",
    bite_trajectory_columns =
      active_column(bite_trajectory_columns),
    mean_causal_columns = active_column(mean_causal_columns),
    trajectory_causal_columns =
      active_column(trajectory_causal_columns),
    mean_nuisance_columns = active_column(mean_nuisance_columns),
    trajectory_nuisance_columns =
      active_column(trajectory_nuisance_columns),
    bite_nuisance_product_columns =
      active_column(bite_nuisance_product_columns),
    bite_three_way_columns =
      active_column(bite_three_way_columns),
    quadratic_columns = active_column(quadratic_columns),
    legacy_intercept_columns =
      active_column(legacy_intercept_columns),
    variable_inventory = dplyr::bind_rows(inventory_rows),
    scaling = dplyr::bind_rows(scaling_rows),
    trajectory_weights = trajectory_weights,
    quadratic_lookup =
      dplyr::bind_rows(quadratic_lookup_rows)
  )
  list(panel = as.data.frame(analysis_panel), metadata = metadata)
}

mc_sp_time_varying_terms <- function(
  columns,
  year_contrast_columns
) {
  columns <- unique(columns[nzchar(columns)])
  if (length(columns) == 0L) {
    return(character())
  }
  interaction_terms <- unlist(
    lapply(
      year_contrast_columns,
      function(year_column) {
        paste0(
          "I(",
          year_column,
          " * ",
          columns,
          ")"
        )
      }
    ),
    use.names = FALSE
  )
  c(columns, interaction_terms)
}

mc_sp_balanced_sum <- function(terms) {
  nodes <- lapply(unique(terms), str2lang)
  if (length(nodes) == 0L) {
    return(1)
  }
  while (length(nodes) > 1L) {
    pair_count <- length(nodes) %/% 2L
    next_level <- lapply(seq_len(pair_count), function(index) {
      call(
        "+",
        nodes[[2L * index - 1L]],
        nodes[[2L * index]]
      )
    })
    if (length(nodes) %% 2L == 1L) {
      next_level[[length(next_level) + 1L]] <-
        nodes[[length(nodes)]]
    }
    nodes <- next_level
  }
  nodes[[1L]]
}

mc_sp_causal_dictionary <- function(
  specification,
  metadata
) {
  treatment_columns <- mc_sp_treatment_columns(specification)
  treatment_lag_orders <- mc_sp_treatment_lag_orders(specification)
  moderator_columns <- switch(
    as.character(specification$richness_tier),
    "0" = metadata$bite_mean_column,
    "1" = c(
      metadata$bite_mean_column,
      metadata$mean_causal_columns
    ),
    "2" = c(
      metadata$bite_trajectory_columns,
      metadata$trajectory_causal_columns
    ),
    "3" = c(
      metadata$bite_trajectory_columns,
      metadata$trajectory_causal_columns,
      metadata$bite_three_way_columns
    ),
    stop("Unknown specification richness tier.", call. = FALSE)
  )
  moderator_columns <- unique(moderator_columns)

  rows <- list()
  for (horizon_index in seq_along(treatment_columns)) {
    treatment_column <- treatment_columns[[horizon_index]]
    for (degree in specification$polynomial_degrees) {
      basis_column <- mc_polynomial_basis_name(
        treatment_column,
        degree
      )
      base_formula_term <- paste0(
        "i(year, ",
        basis_column,
        ")"
      )
      for (outcome_year in specification$analysis_years) {
        rows[[length(rows) + 1L]] <- data.frame(
          outcome_year = as.integer(outcome_year),
          horizon_index = as.integer(horizon_index),
          lag_order = treatment_lag_orders[[horizon_index]],
          treatment_column = treatment_column,
          degree = as.integer(degree),
          basis_column = basis_column,
          moderator_column = NA_character_,
          causal_role = "unmoderated",
          formula_term = base_formula_term,
          coefficient_name = paste0(
            "year::",
            outcome_year,
            ":",
            basis_column
          ),
          stringsAsFactors = FALSE
        )
        for (moderator_column in moderator_columns) {
          role <- if (
            moderator_column %in%
              metadata$bite_three_way_columns
          ) {
            "bite_three_way"
          } else {
            "moderated"
          }
          rows[[length(rows) + 1L]] <- data.frame(
            outcome_year = as.integer(outcome_year),
            horizon_index = as.integer(horizon_index),
            lag_order = treatment_lag_orders[[horizon_index]],
            treatment_column = treatment_column,
            degree = as.integer(degree),
            basis_column = basis_column,
            moderator_column = moderator_column,
            causal_role = role,
            formula_term = paste0(
              "i(year, ",
              basis_column,
              " * ",
              moderator_column,
              ")"
            ),
            coefficient_name = paste0(
              "year::",
              outcome_year,
              ":",
              basis_column,
              " * ",
              moderator_column
            ),
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }
  dictionary <- do.call(rbind, rows)
  if (anyDuplicated(dictionary$coefficient_name) > 0L) {
    stop("The causal dictionary emitted duplicate coefficient names.", call. = FALSE)
  }
  dictionary
}

mc_sp_history_dictionary <- function(
  specification,
  metadata,
  history_selection = NULL
) {
  selection <- if (is.null(history_selection)) {
    mc_sp_history_selection(specification)
  } else {
    history_selection
  }
  selection <- selection[selection$kept, , drop = FALSE]
  rows <- list()
  for (index in seq_len(nrow(selection))) {
    selected <- selection[index, , drop = FALSE]
    history_column <-
      metadata$region_treatment_history_map$constructed_column[
        metadata$region_treatment_history_map$history_year ==
          selected$history_year[[1]]
      ]
    if (length(history_column) != 1L) {
      stop("A retained history year lacks one constructed column.", call. = FALSE)
    }
    rows[[length(rows) + 1L]] <- data.frame(
      outcome_year = selected$outcome_year[[1]],
      history_year = selected$history_year[[1]],
      history_column = history_column,
      moderator_column = NA_character_,
      formula_term = mc_explicit_year_term(
        selected$outcome_year[[1]],
        history_column
      ),
      stringsAsFactors = FALSE
    )
    rows[[length(rows) + 1L]] <- data.frame(
      outcome_year = selected$outcome_year[[1]],
      history_year = selected$history_year[[1]],
      history_column = history_column,
      moderator_column = metadata$bite_mean_column,
      formula_term = mc_explicit_year_term(
        selected$outcome_year[[1]],
        history_column,
        metadata$bite_mean_column
      ),
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0L) {
    return(data.frame(
      outcome_year = integer(),
      history_year = integer(),
      history_column = character(),
      moderator_column = character(),
      formula_term = character(),
      stringsAsFactors = FALSE
    ))
  }
  do.call(rbind, rows)
}

mc_sp_formula_bundle <- function(
  outcome,
  specification,
  metadata,
  history_selection = NULL,
  excluded_terms = character()
) {
  causal_dictionary <- mc_sp_causal_dictionary(
    specification,
    metadata
  )
  history_dictionary <- mc_sp_history_dictionary(
    specification,
    metadata,
    history_selection = history_selection
  )

  nuisance_varying <- c(
    metadata$bite_mean_column,
    metadata$mean_nuisance_columns,
    metadata$bite_nuisance_product_columns
  )
  if (specification$richness_tier >= 2L) {
    nuisance_varying <- c(
      nuisance_varying,
      setdiff(
        metadata$bite_trajectory_columns,
        metadata$bite_mean_column
      ),
      metadata$trajectory_nuisance_columns
    )
  }
  if (specification$richness_tier >= 3L) {
    nuisance_varying <- c(
      nuisance_varying,
      metadata$quadratic_columns
    )
  }

  nuisance_terms <- c(
    metadata$legacy_intercept_columns,
    mc_sp_time_varying_terms(
      nuisance_varying,
      metadata$year_contrast_columns
    )
  )
  terms <- unique(c(
    "0 + factor(year)",
    nuisance_terms,
    history_dictionary$formula_term,
    unique(causal_dictionary$formula_term)
  ))
  terms <- setdiff(terms, excluded_terms)
  # `as.formula(paste(...))` represents a long sum as a left-deep call tree.
  # At R2/R3 that overflows fixest's recursive formula walk before estimation
  # begins.  A balanced call tree spans the identical formula with logarithmic
  # expression depth and remains reusable in every dcCCV state.
  formula <- structure(
    call(
      "~",
      as.name(outcome),
      call(
        "|",
        mc_sp_balanced_sum(terms),
        as.name("aewr_region_id")
      )
    ),
    class = "formula",
    .Environment = environment()
  )
  planned_effective_parameters <-
    length(specification$analysis_years) - 1L +
    MC_CCV_REFERENCE_STATES - 1L +
    length(unique(metadata$legacy_intercept_columns)) +
    length(unique(nuisance_varying)) *
      length(specification$analysis_years) +
    nrow(history_dictionary) +
    nrow(causal_dictionary)
  list(
    formula = formula,
    causal_dictionary = causal_dictionary,
    history_dictionary = history_dictionary,
    nuisance_terms = nuisance_terms,
    formula_terms = terms,
    expanded_formula_characters = sum(nchar(terms)) +
      max(length(terms) - 1L, 0L) * 3L,
    planned_effective_parameters =
      planned_effective_parameters
  )
}

mc_sp_apply_sample_rule <- function(data, sample_rule) {
  switch(
    sample_rule,
    all = data,
    positive_applications = data[
      is.finite(data$mc_y_applications) &
        data$mc_y_applications > 0,
      ,
      drop = FALSE
    ],
    positive_certified_positions = data[
      is.finite(data$mc_y_certified_positions) &
        data$mc_y_certified_positions > 0,
      ,
      drop = FALSE
    ],
    stop("Unknown specification-program sample rule.", call. = FALSE)
  )
}

mc_sp_numeric_environment <- function(name, default) {
  value <- suppressWarnings(as.numeric(Sys.getenv(
    name,
    unset = as.character(default)
  )))
  if (length(value) != 1L || !is.finite(value) || value <= 0) {
    default
  } else {
    value
  }
}

mc_sp_configure_fixest_threads <- function() {
  threads <- suppressWarnings(as.integer(Sys.getenv(
    "MC_FIXEST_THREADS",
    unset = as.character(MC_SPEC_DEFAULT_FIXEST_THREADS)
  )))
  if (length(threads) != 1L || !is.finite(threads) || threads < 1L) {
    threads <- MC_SPEC_DEFAULT_FIXEST_THREADS
  }
  fixest::setFixest_nthreads(threads)
  as.integer(threads)
}

mc_sp_resource_budget <- function(observations, parameters) {
  dense_gib <- observations * parameters * 8 / 1024^3
  gram_gib <- parameters^2 * 8 / 1024^3
  estimated_peak_gib <-
    MC_SPEC_DENSE_PEAK_COPIES * dense_gib +
      MC_SPEC_GRAM_PEAK_COPIES * gram_gib
  list(
    dense_matrix_gib = dense_gib,
    gram_matrix_gib = gram_gib,
    estimated_peak_gib = estimated_peak_gib,
    dense_matrix_guard_gib = mc_sp_numeric_environment(
      "MC_SPEC_MAX_DENSE_GIB",
      MC_SPEC_MAX_DENSE_MATRIX_GIB
    ),
    estimated_peak_guard_gib = mc_sp_numeric_environment(
      "MC_SPEC_MAX_PEAK_GIB",
      MC_SPEC_MAX_ESTIMATED_PEAK_GIB
    )
  )
}

mc_sp_fit_ols <- function(formula, data) {
  fixest::feols(
    fml = formula,
    data = data,
    vcov = ~aewr_region_id,
    ssc = fixest::ssc(
      adj = TRUE,
      cluster.adj = TRUE,
      t.df = "min"
    ),
    warn = TRUE,
    notes = FALSE,
    data.save = FALSE,
    mem.clean = TRUE
  )
}

mc_sp_state_errors <- function(
  model,
  data,
  formula,
  metadata
) {
  coefficient_names <- names(stats::coef(model))
  residual <- stats::residuals(model)
  if (
    length(residual) != nrow(data) ||
      any(!is.finite(residual))
  ) {
    stop("Every fitted row must have one finite residual.", call. = FALSE)
  }
  state_count <- metadata$ccv_reference_states
  state_errors <- matrix(
    NA_real_,
    nrow = state_count,
    ncol = length(coefficient_names),
    dimnames = list(
      paste0("state_", 0:(state_count - 1L)),
      coefficient_names
    )
  )
  state_errors[1L, ] <- 0
  residual_formula <- mc_ccv_residual_formula(formula)
  basis_rows <- list()

  for (state_index in 1:(state_count - 1L)) {
    state_data <- mc_ccv_reference_state(
      data = data,
      metadata = metadata,
      state_index = state_index
    )
    state_data$.mc_ccv_residual <- residual
    state_coefficient <- fixest::feols(
      fml = residual_formula,
      data = state_data,
      warn = FALSE,
      notes = FALSE,
      only.coef = TRUE,
      mem.clean = TRUE
    )
    missing_coefficient <- setdiff(
      coefficient_names,
      names(state_coefficient)
    )
    finite_basis <- (
      length(missing_coefficient) == 0L &&
        all(is.finite(
          state_coefficient[coefficient_names]
        ))
    )
    basis_rows[[length(basis_rows) + 1L]] <- data.frame(
      state_index = as.integer(state_index),
      coefficient_count = length(state_coefficient),
      observed_coefficient_count = length(coefficient_names),
      missing_coefficient_count = length(missing_coefficient),
      common_basis = finite_basis,
      missing_coefficients = paste(
        missing_coefficient,
        collapse = " | "
      ),
      stringsAsFactors = FALSE
    )
    if (!finite_basis) {
      stop(
        "CCV state ",
        state_index,
        " does not retain the observed coefficient basis.",
        call. = FALSE
      )
    }
    state_errors[state_index + 1L, ] <-
      state_coefficient[coefficient_names]
  }

  centered_errors <- scale(
    state_errors,
    center = TRUE,
    scale = FALSE
  )
  small_kernel <- tcrossprod(centered_errors) / state_count
  eigenvalues <- eigen(
    (small_kernel + t(small_kernel)) / 2,
    symmetric = TRUE,
    only.values = TRUE
  )$values
  tolerance <- max(abs(eigenvalues), 1) * 1e-10
  list(
    state_errors = state_errors,
    basis_audit = dplyr::bind_rows(basis_rows),
    diagnostics = list(
      reference_states = state_count,
      covariance_rank = sum(eigenvalues > tolerance),
      minimum_kernel_eigenvalue = min(eigenvalues),
      maximum_observed_state_error =
        max(abs(state_errors[1L, ]))
    )
  )
}

mc_sp_model_matrix <- function(model, newdata) {
  matrix <- stats::model.matrix(model, data = newdata)
  coefficient_names <- names(stats::coef(model))
  absent <- setdiff(coefficient_names, colnames(matrix))
  if (length(absent) > 0L) {
    stop(
      "A counterfactual matrix lacks estimated columns: ",
      paste(absent, collapse = ", "),
      call. = FALSE
    )
  }
  matrix[, coefficient_names, drop = FALSE]
}

mc_sp_gradient <- function(
  model,
  data,
  treatment_column,
  dose_change = NULL,
  derivative = FALSE,
  subset = NULL,
  weights = NULL,
  normalize = TRUE,
  causal_dictionary = NULL,
  epsilon = 1e-5
) {
  if (
    (isTRUE(derivative) && !is.null(dose_change)) ||
      (!isTRUE(derivative) && is.null(dose_change))
  ) {
    stop("Choose either a derivative or a finite dose change.", call. = FALSE)
  }
  observations <- nrow(data)
  keep <- if (is.null(subset)) {
    rep(TRUE, observations)
  } else {
    as.logical(subset)
  }
  if (length(keep) != observations) {
    stop("Effect subsets must match the estimation sample.", call. = FALSE)
  }
  supplied_weights <- if (is.null(weights)) {
    rep(1, observations)
  } else {
    weights
  }
  if (
    length(supplied_weights) != observations ||
      !any(keep) ||
      any(!is.finite(supplied_weights[keep])) ||
      any(supplied_weights[keep] < 0)
  ) {
    stop("Invalid effect weights or subset.", call. = FALSE)
  }
  aggregation_weights <- ifelse(keep, supplied_weights, 0)
  denominator <- if (normalize) {
    sum(aggregation_weights)
  } else {
    1
  }
  if (!is.finite(denominator) || denominator <= 0) {
    stop("Effect weights have no positive mass.", call. = FALSE)
  }

  # Only the generated causal block changes under the intervention.  Building
  # the complete factual and counterfactual N x K matrices used to hold two
  # inputs and their difference simultaneously.  For rich specifications that
  # meant several GiB per worker.  Aggregate each named causal column directly
  # instead; nuisance and history coefficients have an exact zero gradient.
  if (!is.null(causal_dictionary)) {
    coefficient_names <- names(stats::coef(model))
    dictionary <- causal_dictionary[
      causal_dictionary$treatment_column == treatment_column,
      ,
      drop = FALSE
    ]
    absent <- setdiff(
      dictionary$coefficient_name,
      coefficient_names
    )
    if (length(absent) > 0L) {
      stop(
        "The direct delta dictionary contains unresolved coefficients: ",
        paste(absent, collapse = ", "),
        call. = FALSE
      )
    }
    gradient <- stats::setNames(
      numeric(length(coefficient_names)),
      coefficient_names
    )
    dose <- data[[treatment_column]]
    if (any(!is.finite(dose))) {
      stop("The effect sample contains a non-finite dose.", call. = FALSE)
    }
    for (degree in sort(unique(dictionary$degree))) {
      basis_contrast <- switch(
        as.character(degree),
        "1" = if (derivative) rep(1, observations) else
          rep(dose_change, observations),
        "2" = if (derivative) {
          2 * dose / 25
        } else {
          ((dose + dose_change)^2 - dose^2) / 25
        },
        "3" = if (derivative) {
          3 * dose^2 / 125
        } else {
          ((dose + dose_change)^3 - dose^3) / 125
        },
        stop("Unsupported causal polynomial degree.", call. = FALSE)
      )
      degree_rows <- dictionary[
        dictionary$degree == degree,
        ,
        drop = FALSE
      ]
      for (outcome_year in unique(degree_rows$outcome_year)) {
        year_rows <- degree_rows[
          degree_rows$outcome_year == outcome_year,
          ,
          drop = FALSE
        ]
        row_index <- data$year == outcome_year
        weighted_contrast <-
          aggregation_weights[row_index] *
            basis_contrast[row_index]
        unmoderated <- is.na(year_rows$moderator_column)
        if (any(unmoderated)) {
          gradient[
            year_rows$coefficient_name[unmoderated]
          ] <- sum(weighted_contrast) / denominator
        }
        moderator_columns <- year_rows$moderator_column[!unmoderated]
        if (length(moderator_columns) > 0L) {
          moderator_matrix <- as.matrix(
            data[
              row_index,
              moderator_columns,
              drop = FALSE
            ]
          )
          if (any(!is.finite(moderator_matrix))) {
            stop(
              "A causal moderator is non-finite in the effect sample.",
              call. = FALSE
            )
          }
          gradient[
            year_rows$coefficient_name[!unmoderated]
          ] <- drop(
            crossprod(weighted_contrast, moderator_matrix)
          ) / denominator
        }
      }
    }
    return(gradient)
  }

  if (derivative) {
    upper <- mc_counterfactual_data(
      data,
      treatment_column,
      epsilon
    )
    lower <- mc_counterfactual_data(
      data,
      treatment_column,
      -epsilon
    )
    difference <- (
      mc_sp_model_matrix(model, upper) -
        mc_sp_model_matrix(model, lower)
    ) / (2 * epsilon)
  } else {
    counterfactual <- mc_counterfactual_data(
      data,
      treatment_column,
      dose_change
    )
    difference <- mc_sp_model_matrix(model, counterfactual) -
      mc_sp_model_matrix(model, data)
  }
  drop(
    crossprod(aggregation_weights, difference)
  ) / denominator
}

mc_sp_project_effect <- function(
  gradient,
  coefficients,
  state_errors,
  observations,
  effective_parameters,
  inference_df = MC_CCV_DF
) {
  coefficient_names <- names(coefficients)
  gradient <- gradient[coefficient_names]
  if (any(!is.finite(gradient))) {
    stop("The delta gradient is not finite on the coefficient basis.", call. = FALSE)
  }
  state_projection <- drop(
    state_errors[, coefficient_names, drop = FALSE] %*%
      gradient
  )
  raw_variance <- mean(
    (state_projection - mean(state_projection))^2
  )
  df_multiplier <- observations /
    (observations - effective_parameters)
  adjusted_variance <- raw_variance * df_multiplier
  adjusted_standard_error <- sqrt(max(adjusted_variance, 0))
  raw_standard_error <- sqrt(max(raw_variance, 0))
  critical <- stats::qt(0.975, df = inference_df)

  leave_one_out <- lapply(
    seq_along(state_projection),
    function(state_position) {
      retained <- state_projection[-state_position]
      variance <- mean((retained - mean(retained))^2)
      adjusted <- variance * df_multiplier
      standard_error <- sqrt(max(adjusted, 0))
      loo_critical <- stats::qt(
        0.975,
        df = inference_df - 1L
      )
      data.frame(
        omitted_state = state_position - 1L,
        raw_variance = variance,
        adjusted_variance = adjusted,
        standard_error = standard_error,
        critical_value = loo_critical,
        stringsAsFactors = FALSE
      )
    }
  )

  list(
    estimate = sum(gradient * coefficients),
    raw_variance = raw_variance,
    raw_standard_error = raw_standard_error,
    adjusted_variance = adjusted_variance,
    standard_error = adjusted_standard_error,
    df_multiplier = df_multiplier,
    critical_value = critical,
    conf_low = sum(gradient * coefficients) -
      critical * adjusted_standard_error,
    conf_high = sum(gradient * coefficients) +
      critical * adjusted_standard_error,
    state_projection = state_projection,
    leave_one_out = dplyr::bind_rows(leave_one_out)
  )
}

mc_sp_effect_row <- function(
  projected,
  specification,
  outcome_specification,
  treatment_column,
  estimand,
  dose_change,
  observations
) {
  multiplier <- if (
    identical(
      outcome_specification$effect_unit[[1]],
      "probability"
    )
  ) {
    100
  } else {
    1
  }
  data.frame(
    spec_id = specification$spec_id,
    calendar_id = specification$calendar_id,
    outcome_id = outcome_specification$outcome_id[[1]],
    outcome_label = outcome_specification$outcome_label[[1]],
    treatment_column = treatment_column,
    horizon = mc_sp_horizon_name(treatment_column),
    estimand = estimand,
    dose_change = dose_change,
    standardization = "sample_average",
    estimate = projected$estimate * multiplier,
    raw_standard_error =
      projected$raw_standard_error * multiplier,
    standard_error =
      projected$standard_error * multiplier,
    conf_low = projected$conf_low * multiplier,
    conf_high = projected$conf_high * multiplier,
    reported_unit = if (multiplier == 100) {
      "percentage_points"
    } else {
      outcome_specification$effect_unit[[1]]
    },
    observations = observations,
    inference_df = MC_CCV_DF,
    df_multiplier = projected$df_multiplier,
    variance_method = MC_CCV_METHOD,
    stringsAsFactors = FALSE
  )
}

mc_sp_grid_effects <- function(
  model,
  data,
  coefficients,
  state_errors,
  specification,
  outcome_specification,
  effective_parameters,
  causal_dictionary
) {
  effect_rows <- list()
  influence_rows <- list()
  treatment_columns <- mc_sp_treatment_columns(specification)

  add_effect <- function(
    gradient,
    treatment_column,
    estimand,
    dose_change
  ) {
    projected <- mc_sp_project_effect(
      gradient = gradient,
      coefficients = coefficients,
      state_errors = state_errors,
      observations = nrow(data),
      effective_parameters = effective_parameters
    )
    effect <- mc_sp_effect_row(
      projected = projected,
      specification = specification,
      outcome_specification = outcome_specification,
      treatment_column = treatment_column,
      estimand = estimand,
      dose_change = dose_change,
      observations = nrow(data)
    )
    effect_rows[[length(effect_rows) + 1L]] <<- effect

    influence <- projected$leave_one_out
    multiplier <- if (
      identical(
        outcome_specification$effect_unit[[1]],
        "probability"
      )
    ) {
      100
    } else {
      1
    }
    influence$spec_id <- specification$spec_id
    influence$outcome_id <-
      outcome_specification$outcome_id[[1]]
    influence$treatment_column <- treatment_column
    influence$estimand <- estimand
    influence$dose_change <- dose_change
    influence$estimate <- effect$estimate[[1]]
    influence$standard_error <-
      influence$standard_error * multiplier
    influence$conf_low <- influence$estimate -
      influence$critical_value * influence$standard_error
    influence$conf_high <- influence$estimate +
      influence$critical_value * influence$standard_error
    influence_rows[[length(influence_rows) + 1L]] <<-
      influence
  }

  for (treatment_column in treatment_columns) {
    for (dose_change in MC_COUNTERFACTUAL_DOSES) {
      gradient <- mc_sp_gradient(
        model = model,
        data = data,
        treatment_column = treatment_column,
        dose_change = dose_change,
        derivative = FALSE,
        causal_dictionary = causal_dictionary
      )
      add_effect(
        gradient,
        treatment_column,
        "finite_dose_change",
        dose_change
      )
    }
    gradient <- mc_sp_gradient(
      model = model,
      data = data,
      treatment_column = treatment_column,
      derivative = TRUE,
      causal_dictionary = causal_dictionary
    )
    add_effect(
      gradient,
      treatment_column,
      "average_marginal_effect",
      NA_real_
    )
  }
  list(
    effects = dplyr::bind_rows(effect_rows),
    influence = dplyr::bind_rows(influence_rows)
  )
}

mc_sp_estimate_outcome <- function(
  panel,
  metadata,
  specification,
  outcome_specification
) {
  if (!identical(outcome_specification$family[[1]], "gaussian")) {
    stop(
      "The specification program currently implements the linear OLS design.",
      call. = FALSE
    )
  }
  outcome_id <- outcome_specification$outcome_id[[1]]
  outcome_column <- outcome_specification$outcome_column[[1]]
  panel <- panel[
    panel$year %in% specification$analysis_years,
    ,
    drop = FALSE
  ]
  treatment_columns <- mc_sp_treatment_columns(specification)
  finite_treatment <- Reduce(
    `&`,
    lapply(
      treatment_columns,
      function(column) is.finite(panel[[column]])
    )
  )
  panel <- panel[finite_treatment, , drop = FALSE]
  model_data <- mc_sp_apply_sample_rule(
    panel,
    outcome_specification$sample_rule[[1]]
  )
  model_data <- model_data[
    is.finite(model_data[[outcome_column]]),
    ,
    drop = FALSE
  ]
  history_selection <- mc_sp_resolve_history_selection(
    specification = specification,
    panel = panel,
    metadata = metadata
  )
  row_guard <- if (
    identical(
      outcome_specification$sample_rule[[1]],
      "all"
    )
  ) {
    MC_SPEC_FULL_SAMPLE_PARAMETER_ROW_MAX
  } else {
    MC_SPEC_RESTRICTED_SAMPLE_PARAMETER_ROW_MAX
  }
  preflight_bundle <- mc_sp_formula_bundle(
    outcome = outcome_column,
    specification = specification,
    metadata = metadata,
    history_selection = history_selection
  )
  planned_effective_parameters <-
    preflight_bundle$planned_effective_parameters
  planned_parameter_row_ratio <-
    planned_effective_parameters / nrow(model_data)
  preflight_resource <- mc_sp_resource_budget(
    nrow(model_data),
    planned_effective_parameters
  )
  preflight_rejections <- character()
  if (
    !is.finite(planned_parameter_row_ratio) ||
      planned_parameter_row_ratio > row_guard ||
      planned_effective_parameters >= nrow(model_data)
  ) {
    preflight_rejections <- c(
      preflight_rejections,
      "preflight_parameter_row_guard"
    )
  }
  if (
    preflight_resource$dense_matrix_gib >
      preflight_resource$dense_matrix_guard_gib
  ) {
    preflight_rejections <- c(
      preflight_rejections,
      "preflight_dense_matrix_guard"
    )
  }
  if (
    preflight_resource$estimated_peak_gib >
      preflight_resource$estimated_peak_guard_gib
  ) {
    preflight_rejections <- c(
      preflight_rejections,
      "preflight_peak_memory_guard"
    )
  }
  if (length(preflight_rejections) > 0L) {
    diagnostics <- data.frame(
      spec_id = specification$spec_id,
      calendar_id = specification$calendar_id,
      outcome_id = outcome_id,
      richness_tier = specification$richness_tier,
      horizon_count = specification$horizon_count,
      polynomial_degree =
        max(specification$polynomial_degrees),
      observations = nrow(model_data),
      counties = length(unique(model_data$county_fips)),
      estimated_parameters = NA_integer_,
      effective_parameters = planned_effective_parameters,
      planned_effective_parameters =
        planned_effective_parameters,
      parameter_row_ratio = planned_parameter_row_ratio,
      parameter_row_guard = row_guard,
      collinear_terms = NA_integer_,
      compiler_pruned_terms = 0L,
      dropped_causal_terms = NA_integer_,
      warning_count = 0L,
      warning_messages = "",
      formula_characters =
        preflight_bundle$expanded_formula_characters,
      dense_matrix_gib =
        preflight_resource$dense_matrix_gib,
      gram_matrix_gib = preflight_resource$gram_matrix_gib,
      estimated_peak_gib =
        preflight_resource$estimated_peak_gib,
      dense_matrix_guard_gib =
        preflight_resource$dense_matrix_guard_gib,
      estimated_peak_guard_gib =
        preflight_resource$estimated_peak_guard_gib,
      status = "rejected",
      rejection_reason = paste(
        preflight_rejections,
        collapse = " | "
      ),
      stringsAsFactors = FALSE
    )
    return(list(
      status = "rejected",
      design_version = MC_SPEC_PROGRAM_VERSION,
      spec_id = specification$spec_id,
      calendar_id = specification$calendar_id,
      outcome_id = outcome_id,
      diagnostics = diagnostics,
      dropped_terms = character(),
      dropped_causal_terms = character(),
      compiler_pruned_terms = character(),
      formula = deparse1(preflight_bundle$formula),
      causal_dictionary =
        preflight_bundle$causal_dictionary,
      history_dictionary =
        preflight_bundle$history_dictionary,
      history_selection = history_selection,
      effects = data.frame(),
      influence = data.frame()
    ))
  }
  warning_messages <- character()
  compiler_pruned_terms <- character()
  for (resolution_pass in seq_len(10L)) {
    formula_bundle <- mc_sp_formula_bundle(
      outcome = outcome_column,
      specification = specification,
      metadata = metadata,
      history_selection = history_selection,
      excluded_terms = compiler_pruned_terms
    )
    model <- withCallingHandlers(
      mc_sp_fit_ols(formula_bundle$formula, model_data),
      warning = function(condition) {
        warning_messages <<- c(
          warning_messages,
          conditionMessage(condition)
        )
        invokeRestart("muffleWarning")
      }
    )
    pass_dropped_terms <- model$collin.var
    if (is.null(pass_dropped_terms)) {
      pass_dropped_terms <- character()
    }
    pass_dropped_causal <- intersect(
      pass_dropped_terms,
      formula_bundle$causal_dictionary$coefficient_name
    )
    removable_terms <- intersect(
      setdiff(pass_dropped_terms, pass_dropped_causal),
      formula_bundle$formula_terms
    )
    if (
      length(pass_dropped_causal) > 0L ||
        length(removable_terms) == 0L
    ) {
      break
    }
    compiler_pruned_terms <- unique(c(
      compiler_pruned_terms,
      removable_terms
    ))
  }
  observation_index <- fixest::obs(model)
  estimation_data <- model_data[
    observation_index,
    ,
    drop = FALSE
  ]
  coefficients <- stats::coef(model)
  dropped_terms <- model$collin.var
  if (is.null(dropped_terms)) {
    dropped_terms <- character()
  }
  dropped_causal_terms <- intersect(
    dropped_terms,
    formula_bundle$causal_dictionary$coefficient_name
  )
  effective_parameters <- as.integer(
    fixest::degrees_freedom(model, "k")
  )
  observations <- nrow(estimation_data)
  parameter_row_ratio <- effective_parameters / observations
  actual_resource <- mc_sp_resource_budget(
    observations,
    effective_parameters
  )
  base_diagnostics <- data.frame(
    spec_id = specification$spec_id,
    calendar_id = specification$calendar_id,
    outcome_id = outcome_id,
    richness_tier = specification$richness_tier,
    horizon_count = specification$horizon_count,
    polynomial_degree =
      max(specification$polynomial_degrees),
    observations = observations,
    counties = length(unique(estimation_data$county_fips)),
    estimated_parameters = length(coefficients),
    effective_parameters = effective_parameters,
    planned_effective_parameters =
      planned_effective_parameters,
    parameter_row_ratio = parameter_row_ratio,
    parameter_row_guard = row_guard,
    collinear_terms = length(dropped_terms),
    compiler_pruned_terms = length(compiler_pruned_terms),
    dropped_causal_terms = length(dropped_causal_terms),
    warning_count = length(unique(warning_messages)),
    warning_messages = paste(
      unique(warning_messages),
      collapse = " | "
    ),
    formula_characters =
      formula_bundle$expanded_formula_characters,
    dense_matrix_gib = actual_resource$dense_matrix_gib,
    gram_matrix_gib = actual_resource$gram_matrix_gib,
    estimated_peak_gib = actual_resource$estimated_peak_gib,
    dense_matrix_guard_gib =
      actual_resource$dense_matrix_guard_gib,
    estimated_peak_guard_gib =
      actual_resource$estimated_peak_guard_gib,
    stringsAsFactors = FALSE
  )

  rejected_reason <- character()
  if (length(dropped_causal_terms) > 0L) {
    rejected_reason <- c(
      rejected_reason,
      "dropped_causal_terms"
    )
  }
  if (
    !is.finite(parameter_row_ratio) ||
      parameter_row_ratio > row_guard ||
      effective_parameters >= observations
  ) {
    rejected_reason <- c(
      rejected_reason,
      "parameter_row_guard"
    )
  }
  if (
    actual_resource$dense_matrix_gib >
      actual_resource$dense_matrix_guard_gib ||
      actual_resource$estimated_peak_gib >
        actual_resource$estimated_peak_guard_gib
  ) {
    rejected_reason <- c(
      rejected_reason,
      "postfit_memory_guard"
    )
  }
  if (length(rejected_reason) > 0L) {
    base_diagnostics$status <- "rejected"
    base_diagnostics$rejection_reason <- paste(
      unique(rejected_reason),
      collapse = " | "
    )
    return(list(
      status = "rejected",
      design_version = MC_SPEC_PROGRAM_VERSION,
      spec_id = specification$spec_id,
      calendar_id = specification$calendar_id,
      outcome_id = outcome_id,
      diagnostics = base_diagnostics,
      dropped_terms = dropped_terms,
      dropped_causal_terms = dropped_causal_terms,
      compiler_pruned_terms = compiler_pruned_terms,
      formula = deparse1(formula_bundle$formula),
      causal_dictionary =
        formula_bundle$causal_dictionary,
      history_dictionary =
        formula_bundle$history_dictionary,
      history_selection = history_selection,
      effects = data.frame(),
      influence = data.frame()
    ))
  }

  ccv <- mc_sp_state_errors(
    model = model,
    data = estimation_data,
    formula = formula_bundle$formula,
    metadata = metadata
  )
  if (ccv$diagnostics$covariance_rank > MC_CCV_DF) {
    stop("Projected dcCCV rank exceeds 16.", call. = FALSE)
  }
  grid_effects <- mc_sp_grid_effects(
    model = model,
    data = estimation_data,
    coefficients = coefficients,
    state_errors = ccv$state_errors,
    specification = specification,
    outcome_specification = outcome_specification,
    effective_parameters = effective_parameters,
    causal_dictionary = formula_bundle$causal_dictionary
  )
  centered_errors <- scale(
    ccv$state_errors,
    center = TRUE,
    scale = FALSE
  )
  raw_parameter_variance <- colMeans(centered_errors^2)
  df_multiplier <- observations /
    (observations - effective_parameters)
  adjusted_parameter_variance <-
    raw_parameter_variance * df_multiplier

  base_diagnostics$status <- "estimated"
  base_diagnostics$rejection_reason <- ""
  base_diagnostics$ccv_covariance_rank <-
    ccv$diagnostics$covariance_rank
  base_diagnostics$minimum_kernel_eigenvalue <-
    ccv$diagnostics$minimum_kernel_eigenvalue
  base_diagnostics$maximum_observed_state_error <-
    ccv$diagnostics$maximum_observed_state_error
  base_diagnostics$df_multiplier <- df_multiplier

  list(
    status = "estimated",
    design_version = MC_SPEC_PROGRAM_VERSION,
    spec_id = specification$spec_id,
    calendar_id = specification$calendar_id,
    outcome_id = outcome_id,
    specification = specification,
    diagnostics = base_diagnostics,
    coefficients = coefficients,
    raw_parameter_standard_error = sqrt(
      pmax(raw_parameter_variance, 0)
    ),
    parameter_standard_error = sqrt(
      pmax(adjusted_parameter_variance, 0)
    ),
    conventional_cluster_standard_error =
      unname(fixest::se(model)),
    coefficient_names = names(coefficients),
    state_errors = ccv$state_errors,
    common_basis_audit = ccv$basis_audit,
    dropped_terms = dropped_terms,
    dropped_causal_terms = dropped_causal_terms,
    compiler_pruned_terms = compiler_pruned_terms,
    formula = deparse1(formula_bundle$formula),
    causal_dictionary = formula_bundle$causal_dictionary,
    history_dictionary = formula_bundle$history_dictionary,
    history_selection = history_selection,
    effects = grid_effects$effects,
    influence = grid_effects$influence,
    estimation_keys = estimation_data[
      c("county_fips", "year")
    ]
  )
}

mc_sp_primary_selection <- function(
  registry,
  diagnostics,
  outcomes = MC_OUTCOMES
) {
  target_calendar <- registry[
    registry$horizon_count == 3L &
      registry$preperiod_start == min(MC_BASELINE_YEARS) &
      registry$preperiod_end == max(MC_BASELINE_YEARS) &
      registry$analysis_start == min(MC_ANALYSIS_YEARS) &
      registry$analysis_end == max(MC_ANALYSIS_YEARS) &
      registry$polynomial_degree == 2L,
    ,
    drop = FALSE
  ]
  candidates <- merge(
    target_calendar,
    diagnostics,
    by = c("spec_id", "calendar_id"),
    all.x = TRUE,
    sort = FALSE
  )
  rows <- list()
  for (outcome_id in outcomes$outcome_id) {
    outcome_candidates <- candidates[
      candidates$outcome_id == outcome_id &
        candidates$status == "estimated",
      ,
      drop = FALSE
    ]
    outcome_candidates <- outcome_candidates[
      order(
        -outcome_candidates$richness_tier,
        outcome_candidates$parameter_row_ratio
      ),
      ,
      drop = FALSE
    ]
    if (nrow(outcome_candidates) == 0L) {
      stop(
        "No admissible primary specification for ",
        outcome_id,
        ".",
        call. = FALSE
      )
    }
    selected <- outcome_candidates[1L, , drop = FALSE]
    rows[[length(rows) + 1L]] <- data.frame(
      outcome_id = outcome_id,
      model_id = MC_PRIMARY_MODEL_ID,
      spec_id = selected$spec_id[[1]],
      calendar_id = selected$calendar_id[[1]],
      richness_tier = selected$richness_tier[[1]],
      selection_rule =
        "highest_richness_current_calendar_h3_d2_passing_guards",
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}
