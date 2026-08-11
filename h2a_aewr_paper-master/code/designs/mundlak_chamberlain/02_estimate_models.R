# Purpose: Estimate the complete model ladder for the standalone multilevel
# Mundlak-Chamberlain-Wooldridge design.
# Inputs:
#   data/processed/mundlak_chamberlain_county_year.parquet
#   data/intermediate/mundlak_chamberlain_metadata.rds
# Outputs:
#   data/intermediate/mundlak_chamberlain_models.rds
#   outputs/tables/mc_model_diagnostics.csv
#   outputs/tables/mc_collinear_terms.csv
#   outputs/tables/mc_model_warnings.csv
#   outputs/tables/mc_parameter_estimates.csv
#   outputs/tables/mc_ccv_diagnostics.csv

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "mundlak_chamberlain", "design.R"))
source(path_code("designs", "mundlak_chamberlain", "helpers.R"))
library(arrow)
library(dplyr)
library(fixest)
library(purrr)
library(readr)
library(tibble)

legacy_fixest_threads <- suppressWarnings(as.integer(Sys.getenv(
  "MC_FIXEST_THREADS",
  unset = as.character(MC_SPEC_DEFAULT_FIXEST_THREADS)
)))
if (
  length(legacy_fixest_threads) != 1L ||
    !is.finite(legacy_fixest_threads) ||
    legacy_fixest_threads < 1L
) {
  legacy_fixest_threads <- MC_SPEC_DEFAULT_FIXEST_THREADS
}
fixest::setFixest_nthreads(legacy_fixest_threads)

estimation_model_ids <- if (
  identical(Sys.getenv("MC_BENCHMARK_ONLY", unset = "0"), "1")
) {
  c("twfe_benchmark", "mundlak_multilevel")
} else {
  MC_MODEL_IDS
}

dir.create(path_int(), recursive = TRUE, showWarnings = FALSE)
dir.create(path_tables(), recursive = TRUE, showWarnings = FALSE)

panel <- read_parquet(
  path_processed("mundlak_chamberlain_county_year.parquet")
) %>%
  as.data.frame()
metadata <- readRDS(
  path_int("mundlak_chamberlain_metadata.rds")
)

if (!identical(metadata$design_version, MC_LEGACY_DESIGN_VERSION)) {
  stop("Unexpected MC metadata version.", call. = FALSE)
}
if (
  anyDuplicated(panel$mc_row_id) > 0L ||
    any(!is.finite(panel$mc_row_id))
) {
  stop("mc_row_id must be complete and unique.", call. = FALSE)
}

mc_apply_sample_rule <- function(data, sample_rule) {
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
    stop("Unknown MC sample rule: ", sample_rule, call. = FALSE)
  )
}

mc_fit_model <- function(
  formula,
  data,
  family,
  offset_column = NA_character_
) {
  common_arguments <- list(
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
    data.save = FALSE
  )

  if (identical(family, "gaussian")) {
    return(do.call(fixest::feols, common_arguments))
  }

  if (
    length(offset_column) == 1L &&
      !is.na(offset_column) &&
      nzchar(offset_column)
  ) {
    common_arguments$offset <- stats::as.formula(
      paste0("~", offset_column)
    )
  }
  common_arguments$glm.iter <- 100L
  common_arguments$glm.tol <- 1e-8

  if (identical(family, "poisson")) {
    return(do.call(fixest::fepois, common_arguments))
  }
  if (identical(family, "binomial")) {
    common_arguments$family <- stats::binomial("logit")
    return(do.call(fixest::feglm, common_arguments))
  }
  stop("Unsupported MC estimation family: ", family, call. = FALSE)
}

model_store <- setNames(
  vector("list", nrow(metadata$outcomes)),
  metadata$outcomes$outcome_id
)
sample_store <- model_store
diagnostic_rows <- list()
collinear_rows <- list()
parameter_rows <- list()
warning_rows <- list()
ccv_diagnostic_rows <- list()

for (outcome_index in seq_len(nrow(metadata$outcomes))) {
  outcome_specification <- metadata$outcomes[outcome_index, ]
  outcome_id <- outcome_specification$outcome_id[[1]]
  outcome_column <- outcome_specification$outcome_column[[1]]
  family <- outcome_specification$family[[1]]
  offset_column <- outcome_specification$offset_column[[1]]

  model_data <- mc_apply_sample_rule(
    panel,
    outcome_specification$sample_rule[[1]]
  )
  model_data <- model_data[
    is.finite(model_data[[outcome_column]]),
    ,
    drop = FALSE
  ]
  if (
    length(offset_column) == 1L &&
      !is.na(offset_column) &&
      nzchar(offset_column)
  ) {
    model_data <- model_data[
      is.finite(model_data[[offset_column]]),
      ,
      drop = FALSE
    ]
  }

  model_store[[outcome_id]] <- list()
  sample_store[[outcome_id]] <- list()

  for (model_id in estimation_model_ids) {
    message("Estimating ", outcome_id, ": ", model_id)
    formula <- mc_build_formula(
      outcome = outcome_column,
      model_id = model_id,
      metadata = metadata,
      reference_year = metadata$reference_year
    )
    warning_messages <- character()
    capture_warnings <- function(expression) {
      withCallingHandlers(
        expression,
        warning = function(condition) {
          warning_messages <<- c(
            warning_messages,
            conditionMessage(condition)
          )
          invokeRestart("muffleWarning")
        }
      )
    }
    started <- proc.time()[["elapsed"]]
    model <- capture_warnings(
      mc_fit_model(
        formula = formula,
        data = model_data,
        family = family,
        offset_column = offset_column
      )
    )
    elapsed_seconds <- proc.time()[["elapsed"]] - started

    observation_index <- fixest::obs(model)
    estimation_data <- model_data[
      observation_index,
      ,
      drop = FALSE
    ]
    sample_ids <- estimation_data$mc_row_id
    dropped_terms <- model$collin.var
    if (is.null(dropped_terms)) {
      dropped_terms <- character()
    }
    dropped_causal_terms <- grep(
      "mc_dose_(current|lag1|lag2)",
      dropped_terms,
      value = TRUE
    )
    if (
      identical(model_id, MC_PRIMARY_MODEL_ID) &&
        length(dropped_causal_terms) > 0L
    ) {
      stop(
        "Primary master specification dropped causal columns for ",
        outcome_id,
        ": ",
        paste(dropped_causal_terms, collapse = ", "),
        call. = FALSE
      )
    }

    sample_store[[outcome_id]][[model_id]] <- sample_ids

    conventional_cluster_covariance <- capture_warnings(stats::vcov(model))

    # Replace the reporting covariance with the continuous-treatment
    # design-covariance CCV.  `mc_design_covariance_ccv()` holds the fitted
    # residual fixed, reassigns all 17 complete AEWR paths under the balanced
    # finite reference law, re-solves the OLS Gram matrix in each state, and
    # takes the probability covariance of the resulting coefficient errors.
    # The conventional AEWR-region clustered covariance is retained alongside
    # it solely as a transparent comparison diagnostic.
    ccv_result <- mc_design_covariance_ccv(
      model = model,
      data = estimation_data,
      formula = formula,
      metadata = metadata
    )
    covariance <- ccv_result$covariance
    ccv_diagnostic <- ccv_result$diagnostics
    warning_messages <- unique(warning_messages)
    diagnostic_rows[[length(diagnostic_rows) + 1L]] <- tibble(
      outcome_id = outcome_id,
      outcome_label = outcome_specification$outcome_label[[1]],
      model_id = model_id,
      family = family,
      observations = nobs(model),
      counties = n_distinct(estimation_data$county_fips),
      market_cells = n_distinct(estimation_data$mc_market_id),
      states = n_distinct(estimation_data$state_fips),
      aewr_regions = n_distinct(estimation_data$aewr_region_id),
      minimum_year = min(estimation_data$year),
      maximum_year = max(estimation_data$year),
      estimated_parameters = length(stats::coef(model)),
      collinear_terms = length(dropped_terms),
      dropped_causal_terms = length(dropped_causal_terms),
      covariance_rank_upper_bound = n_distinct(estimation_data$aewr_region_id) -
        1L,
      covariance_method = ccv_diagnostic$method,
      ccv_reference_states = ccv_diagnostic$reference_states,
      ccv_covariance_rank = ccv_diagnostic$covariance_rank,
      covariance_minimum_diagonal = min(diag(covariance)),
      conventional_cluster_minimum_diagonal = min(
        diag(conventional_cluster_covariance)
      ),
      warning_count = length(warning_messages),
      warning_messages = paste(warning_messages, collapse = " | "),
      elapsed_seconds = elapsed_seconds,
      formula_characters = nchar(deparse1(formula))
    )

    ccv_diagnostic_rows[[length(ccv_diagnostic_rows) + 1L]] <- tibble(
      outcome_id = outcome_id,
      model_id = model_id,
      method = ccv_diagnostic$method,
      reference_design = ccv_diagnostic$reference_design,
      reference_states = ccv_diagnostic$reference_states,
      design_df = ccv_diagnostic$design_df,
      covariance_rank = ccv_diagnostic$covariance_rank,
      minimum_kernel_eigenvalue =
        ccv_diagnostic$minimum_kernel_eigenvalue,
      minimum_variance = ccv_diagnostic$minimum_variance,
      maximum_observed_state_error =
        ccv_diagnostic$maximum_observed_state_error,
      mean_state_error_norm = ccv_diagnostic$mean_state_error_norm
    )

    if (length(dropped_terms) > 0L) {
      collinear_rows[[length(collinear_rows) + 1L]] <- tibble(
        outcome_id = outcome_id,
        model_id = model_id,
        term = dropped_terms,
        causal_term = grepl(
          "mc_dose_(current|lag1|lag2|lead1)",
          dropped_terms
        )
      )
    }
    if (length(warning_messages) > 0L) {
      warning_rows[[length(warning_rows) + 1L]] <- tibble(
        outcome_id = outcome_id,
        model_id = model_id,
        warning = warning_messages
      )
    }

    coefficient_estimate <- stats::coef(model)
    standard_error <- sqrt(pmax(diag(covariance), 0))
    statistic <- coefficient_estimate / standard_error
    parameter_rows[[length(parameter_rows) + 1L]] <- tibble(
      outcome_id = outcome_id,
      model_id = model_id,
      variance_method = metadata$ccv_method,
      term = names(coefficient_estimate),
      estimate = unname(coefficient_estimate),
      standard_error = unname(standard_error),
      statistic = unname(statistic),
      p_value = 2 * stats::pt(
        -abs(statistic),
        df = metadata$cluster_df
      ),
      conventional_cluster_standard_error = sqrt(pmax(
        diag(conventional_cluster_covariance),
        0
      ))
    )

    compact_model <- structure(
      list(
        outcome_id = outcome_id,
        model_id = model_id,
        family = family,
        coefficients = stats::coef(model),
        covariance = covariance,
        covariance_method = metadata$ccv_method,
        conventional_cluster_covariance = conventional_cluster_covariance,
        ccv_diagnostics = ccv_diagnostic,
        observations = nobs(model),
        formula = deparse1(formula),
        collinear_terms = dropped_terms,
        warnings = warning_messages
      ),
      class = "mc_compact_model"
    )
    model_store[[outcome_id]][[model_id]] <- compact_model
  }

  checkpoint <- list(
    design_version = metadata$design_version,
    estimated_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
    metadata = metadata,
    models = model_store,
    sample_row_ids = sample_store,
    diagnostics = bind_rows(diagnostic_rows),
    ccv_diagnostics = bind_rows(ccv_diagnostic_rows)
  )
  saveRDS(
    checkpoint,
    path_int("mundlak_chamberlain_models.rds")
  )
}

diagnostics <- bind_rows(diagnostic_rows)
collinear_terms <- bind_rows(collinear_rows)
parameter_estimates <- bind_rows(parameter_rows)
ccv_diagnostics <- bind_rows(ccv_diagnostic_rows)
model_warnings <- bind_rows(warning_rows)
if (ncol(model_warnings) == 0L) {
  model_warnings <- tibble(
    outcome_id = character(),
    model_id = character(),
    warning = character()
  )
}

write_csv(
  diagnostics,
  path_tables("mc_model_diagnostics.csv")
)
write_csv(
  collinear_terms,
  path_tables("mc_collinear_terms.csv")
)
write_csv(
  model_warnings,
  path_tables("mc_model_warnings.csv")
)
write_csv(
  parameter_estimates,
  path_tables("mc_parameter_estimates.csv")
)
write_csv(
  ccv_diagnostics,
  path_tables("mc_ccv_diagnostics.csv")
)

model_bundle <- list(
  design_version = metadata$design_version,
  estimated_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  metadata = metadata,
  models = model_store,
  sample_row_ids = sample_store,
  diagnostics = diagnostics,
  ccv_diagnostics = ccv_diagnostics
)
saveRDS(
  model_bundle,
  path_int("mundlak_chamberlain_models.rds")
)

message(
  "Estimated ",
  nrow(diagnostics),
  " MC models across ",
  n_distinct(diagnostics$outcome_id),
  " outcomes."
)
