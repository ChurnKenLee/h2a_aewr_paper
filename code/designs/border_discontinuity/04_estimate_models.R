# Purpose: Estimate the frozen AEWR-border ring registry and influence audit.
# Outputs: Ring estimates, inference comparisons, sample audits, and diagnostics.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)
library(fixest)
library(jsonlite)
library(purrr)
library(readr)
library(tibble)

design_path <- path_code("designs", "border_discontinuity", "design.json")
design <- jsonlite::fromJSON(design_path, simplifyVector = TRUE)

if (!requireNamespace("fwildclusterboot", quietly = TRUE)) {
  stop(
    paste(
      "The preferred border inference requires fwildclusterboot.",
      "Restore the updated renv lock before running this stage."
    ),
    call. = FALSE
  )
}
if (!requireNamespace("dqrng", quietly = TRUE)) {
  stop("fwildclusterboot inference requires dqrng.", call. = FALSE)
}

panel <- read_parquet(path_processed("border_discontinuity_panel.parquet")) %>%
  as_tibble()
# Stage04 reproduces the contaminated v5 historical audit explicitly. Canonical
# treatment in the persistent panel and stage07 remains missing in 2009.
panel[[design$treatment]] <- panel$v5_aewr_ppi_comparator
legacy_baseline <- panel %>%
  filter(year >= design$feature_start_year, year <= design$feature_end_year) %>%
  group_by(county_fips) %>%
  summarise(value = mean(.data[[design$treatment]], na.rm = TRUE), .groups = "drop")
legacy_baseline$value <- as.numeric(scale(legacy_baseline$value))
panel[[paste0("baseline_", design$treatment, "_2008_2010_z")]] <-
  legacy_baseline$value[match(panel$county_fips, legacy_baseline$county_fips)]
memberships <- read_parquet(
  path_int("border_discontinuity_geography_memberships.parquet")
) %>%
  as_tibble()

geography_registry <- as_tibble(design$geography_specs) %>%
  filter(spec_id %in% design$benchmark_geography_ids) %>%
  transmute(
    spec_id,
    geography_family = family,
    membership_mode,
    role
  )

# docs-ground:start border-discontinuity-estimation-contract
base_registry <- geography_registry %>%
  transmute(
    model_id = paste0(spec_id, "__base"),
    spec_id,
    geography_family,
    membership_mode,
    role,
    period = "policy",
    weight_spec = "equal_county",
    control_spec = "none"
  )

primary_registry <- geography_registry %>%
  filter(spec_id == design$diagnostic_audit$primary_spec_id)
if (nrow(primary_registry) != 1L) {
  stop("Exactly one ring geography must be declared primary.", call. = FALSE)
}

model_registry <- bind_rows(
  base_registry,
  primary_registry %>%
    transmute(
      model_id = paste0(spec_id, "__equal_stratum_region"),
      spec_id,
      geography_family,
      membership_mode,
      role = "weight_sensitivity",
      period = "policy",
      weight_spec = "equal_stratum_region",
      control_spec = "none"
    ),
  primary_registry %>%
    transmute(
      model_id = paste0(spec_id, "__baseline_trends"),
      spec_id,
      geography_family,
      membership_mode,
      role = "control_sensitivity",
      period = "policy",
      weight_spec = "equal_county",
      control_spec = "baseline_trends"
    ),
  primary_registry %>%
    transmute(
      model_id = paste0(spec_id, "__preperiod_placebo"),
      spec_id,
      geography_family,
      membership_mode,
      role = "placebo",
      period = "placebo",
      weight_spec = "equal_county",
      control_spec = "none"
    )
)

outcome <- design$outcome
treatment <- design$treatment
baseline_control_columns <- paste0(
  "baseline_",
  design$baseline_control_columns,
  "_2008_2010_z"
)
bootstrap_method <- design$bootstrap$method
bootstrap_type <- design$bootstrap$bootstrap_type
bootstrap_weights <- design$bootstrap$weights
bootstrap_repetitions <- as.integer(design$bootstrap$repetitions)
bootstrap_seed <- as.integer(design$bootstrap$seed)
inference_cluster <- design$bootstrap$cluster
confidence_level <- as.numeric(design$bootstrap$confidence_level)
two_way_crv1_method <- design$two_way_cluster_sensitivity$method
two_way_cluster_dimensions <- unname(unlist(
  design$two_way_cluster_sensitivity$clusters,
  use.names = FALSE
))
two_way_confidence_level <- as.numeric(
  design$two_way_cluster_sensitivity$confidence_level
)
if (
  two_way_crv1_method != "CRV1" ||
    !identical(
      two_way_cluster_dimensions,
      c("region_pair_id", "aewr_region_id")
    )
) {
  stop(
    "The two-way sensitivity must use CRV1 by border pair and AEWR region.",
    call. = FALSE
  )
}
two_way_cluster_formula <- stats::as.formula(paste(
  "~",
  paste(two_way_cluster_dimensions, collapse = " + ")
))
diagnostic_margin_columns <- unname(unlist(
  design$diagnostic_margin_columns,
  use.names = TRUE
))
cluster_jackknife_method <- design$diagnostic_audit$cluster_jackknife_method
cluster_jackknife_confidence_level <- as.numeric(
  design$diagnostic_audit$cluster_jackknife_confidence_level
)
# docs-ground:end border-discontinuity-estimation-contract

drop_singletons <- function(data) {
  result <- data
  repeat {
    unit_counts <- table(result$unit_fe_id)
    stratum_year_counts <- table(result$stratum_year_id)
    keep <- unit_counts[result$unit_fe_id] > 1L &
      stratum_year_counts[result$stratum_year_id] > 1L
    if (all(keep)) {
      break
    }
    result <- result[keep, , drop = FALSE]
    if (nrow(result) == 0L) {
      stop("Singleton removal emptied an estimation sample.", call. = FALSE)
    }
  }
  result
}

model_data <- function(registry_row) {
  specification <- registry_row$spec_id[[1L]]
  member <- memberships %>%
    filter(spec_id == specification) %>%
    select(
      county_fips,
      membership_id,
      stratum_id,
      region_pair_id,
      membership_mode,
      member_region_id = aewr_region_id,
      analysis_weight
    )
  if (nrow(member) == 0L) {
    stop(paste("No memberships for", specification), call. = FALSE)
  }

  cell_weights <- member %>%
    distinct(stratum_id, member_region_id, county_fips) %>%
    count(stratum_id, member_region_id, name = "stratum_region_counties")
  member <- member %>%
    left_join(
      cell_weights,
      by = c("stratum_id", "member_region_id"),
      relationship = "many-to-one"
    )

  joined <- panel %>%
    inner_join(member, by = "county_fips", relationship = "many-to-many") %>%
    filter(aewr_region_id == member_region_id)
  period <- registry_row$period[[1L]]
  if (period == "policy") {
    joined <- joined %>%
      filter(year >= design$policy_start_year, year <= design$policy_end_year)
  } else if (period == "placebo") {
    joined <- joined %>%
      filter(year >= design$placebo_start_year, year <= design$placebo_end_year)
  } else if (period == "full") {
    joined <- joined %>%
      filter(
        year >= design$diagnostic_audit$annual_start_year,
        year <= design$diagnostic_audit$annual_end_year
      )
  } else {
    stop(paste("Unknown period", period), call. = FALSE)
  }

  joined <- joined %>%
    mutate(
      estimation_weight = if (
        registry_row$weight_spec[[1L]] == "equal_stratum_region"
      ) {
        analysis_weight / stratum_region_counties
      } else {
        analysis_weight
      },
      unit_fe_id = if_else(
        membership_mode == "stacked",
        membership_id,
        county_fips
      ),
      stratum_year_id = paste(stratum_id, year, sep = "__")
    )

  required <- c(
    outcome,
    treatment,
    "county_fips",
    "aewr_region_id",
    "region_pair_id",
    "stratum_id",
    "unit_fe_id",
    "stratum_year_id",
    "estimation_weight"
  )
  if (registry_row$control_spec[[1L]] == "baseline_trends") {
    required <- c(required, baseline_control_columns, "year_centered")
  }
  complete <- stats::complete.cases(joined[, required])
  complete_data <- joined[complete, , drop = FALSE]
  singleton_free <- drop_singletons(complete_data)
  singleton_free$estimation_weight <- singleton_free$estimation_weight /
    mean(singleton_free$estimation_weight)
  singleton_free$unit_fe_id <- factor(singleton_free$unit_fe_id)
  singleton_free$stratum_year_id <- factor(singleton_free$stratum_year_id)
  singleton_free$aewr_region_id <- factor(singleton_free$aewr_region_id)
  singleton_free$region_pair_id <- factor(singleton_free$region_pair_id)

  list(
    data = singleton_free,
    audit = tibble(
      model_id = registry_row$model_id[[1L]],
      joined_rows = nrow(joined),
      complete_case_rows = nrow(complete_data),
      estimation_rows = nrow(singleton_free),
      dropped_missing_rows = nrow(joined) - nrow(complete_data),
      dropped_singleton_rows = nrow(complete_data) - nrow(singleton_free),
      n_counties = n_distinct(singleton_free$county_fips),
      n_strata = n_distinct(singleton_free$stratum_id),
      n_regions = n_distinct(singleton_free$aewr_region_id),
      n_inference_clusters = n_distinct(
        singleton_free[[inference_cluster]]
      ),
      minimum_year = min(singleton_free$year),
      maximum_year = max(singleton_free$year)
    )
  )
}

prepare_bootstrap_regression <- function(model, data) {
  within_matrix <- fixest::demean(
    model,
    as.matrix = TRUE,
    tol = 1e-10,
    nthreads = 1L
  )
  coefficient_names <- names(stats::coef(model))
  if (
    nrow(within_matrix) != nrow(data) ||
      !all(coefficient_names %in% colnames(within_matrix))
  ) {
    stop(
      "Fixed-effect absorption did not preserve rows or regressors.",
      call. = FALSE
    )
  }

  transformed <- within_matrix[, c(1L, match(
    coefficient_names,
    colnames(within_matrix)
  )), drop = FALSE]
  transformed <- transformed * sqrt(data$estimation_weight)
  bootstrap_terms <- sprintf("bootstrap_x_%03d", seq_along(coefficient_names))
  bootstrap_data <- as.data.frame(transformed, check.names = FALSE)
  names(bootstrap_data) <- c("bootstrap_y", bootstrap_terms)
  bootstrap_data[[inference_cluster]] <- data[[inference_cluster]]
  bootstrap_formula <- stats::reformulate(
    bootstrap_terms,
    response = "bootstrap_y",
    intercept = FALSE
  )
  bootstrap_model <- fixest::feols(
    bootstrap_formula,
    data = bootstrap_data,
    notes = FALSE
  )
  coefficient_difference <- max(abs(
    unname(stats::coef(bootstrap_model)[bootstrap_terms]) -
      unname(stats::coef(model)[coefficient_names])
  ))
  if (!is.finite(coefficient_difference) || coefficient_difference > 1e-8) {
    stop(
      "Weighted FWL regression did not reproduce the fixed-effect estimates.",
      call. = FALSE
    )
  }

  list(
    model = bootstrap_model,
    parameter_map = stats::setNames(bootstrap_terms, coefficient_names)
  )
}

wild_bootstrap_test <- function(bootstrap_regression, parameter) {
  set.seed(bootstrap_seed)
  dqrng::dqset.seed(bootstrap_seed)
  result <- fwildclusterboot::boottest(
    bootstrap_regression$model,
    param = unname(bootstrap_regression$parameter_map[[parameter]]),
    B = bootstrap_repetitions,
    clustid = inference_cluster,
    bootcluster = inference_cluster,
    type = bootstrap_weights,
    impose_null = TRUE,
    bootstrap_type = bootstrap_type,
    conf_int = TRUE,
    engine = "R",
    nthreads = 1L
  ) %>%
    generics::tidy()
  if (nrow(result) != 1L) {
    stop("Wild-bootstrap output must contain exactly one test.", call. = FALSE)
  }
  result
}

fit_border_model <- function(model_formula, data) {
  weighted_model <- !all(
    data$estimation_weight == data$estimation_weight[[1L]]
  )
  if (weighted_model) {
    fixest::feols(
      model_formula,
      data = data,
      weights = ~ estimation_weight,
      vcov = stats::as.formula(paste("~", inference_cluster)),
      notes = FALSE
    )
  } else {
    fixest::feols(
      model_formula,
      data = data,
      vcov = stats::as.formula(paste("~", inference_cluster)),
      notes = FALSE
    )
  }
}

two_way_crv1_inference <- function(model, data, parameters) {
  missing_dimensions <- setdiff(two_way_cluster_dimensions, names(data))
  if (length(missing_dimensions) > 0L) {
    stop(
      "Two-way clustering fields are missing: ",
      paste(missing_dimensions, collapse = ", "),
      call. = FALSE
    )
  }
  raw_covariance <- fixest::vcov_cluster(
    model,
    cluster = two_way_cluster_formula,
    vcov_fix = FALSE
  )
  if (any(!is.finite(raw_covariance))) {
    stop("Raw two-way covariance contains nonfinite values.", call. = FALSE)
  }
  raw_minimum_eigenvalue <- min(eigen(
    raw_covariance,
    symmetric = TRUE,
    only.values = TRUE
  )$values)
  two_way_model <- summary(
    model,
    vcov = two_way_cluster_formula,
    vcov_fix = TRUE
  )
  adjusted_covariance <- stats::vcov(two_way_model)
  maximum_psd_adjustment <- max(abs(
    adjusted_covariance - raw_covariance
  ))
  intervals <- stats::confint(
    two_way_model,
    parm = parameters,
    level = two_way_confidence_level
  )
  if (is.null(dim(intervals))) {
    intervals <- matrix(intervals, nrow = 1L)
  }
  tibble(
    parameter = parameters,
    two_way_crv1_standard_error = unname(
      fixest::se(two_way_model)[parameters]
    ),
    two_way_crv1_p_value = unname(
      fixest::pvalue(two_way_model)[parameters]
    ),
    two_way_crv1_confidence_low = unname(intervals[, 1L]),
    two_way_crv1_confidence_high = unname(intervals[, 2L]),
    two_way_crv1_raw_minimum_eigenvalue = raw_minimum_eigenvalue,
    two_way_crv1_maximum_psd_adjustment = maximum_psd_adjustment,
    two_way_crv1_psd_adjusted = maximum_psd_adjustment > 1e-12,
    two_way_crv1_method = two_way_crv1_method,
    two_way_cluster_dimensions = paste(
      two_way_cluster_dimensions,
      collapse = "+"
    ),
    two_way_confidence_level = two_way_confidence_level,
    n_two_way_pair_clusters = n_distinct(data$region_pair_id),
    n_two_way_region_clusters = n_distinct(data$aewr_region_id),
    n_two_way_intersection_clusters = n_distinct(interaction(
      data$region_pair_id,
      data$aewr_region_id,
      drop = TRUE
    ))
  )
}

cluster_jackknife_inference <- function(
  model_formula,
  data,
  parameters,
  full_estimates
) {
  cluster_values <- sort(unique(as.character(data[[inference_cluster]])))
  leave_one_out <- purrr::map_dfr(cluster_values, function(excluded_cluster) {
    jackknife_data <- data %>%
      filter(
        as.character(.data[[inference_cluster]]) != excluded_cluster
      ) %>%
      droplevels() %>%
      drop_singletons()
    jackknife_data$estimation_weight <- jackknife_data$estimation_weight /
      mean(jackknife_data$estimation_weight)
    jackknife_model <- fit_border_model(model_formula, jackknife_data)
    missing_parameters <- setdiff(
      parameters,
      names(stats::coef(jackknife_model))
    )
    if (length(missing_parameters) > 0L) {
      stop(
        "Cluster jackknife left parameters unidentified after excluding ",
        excluded_cluster,
        ": ",
        paste(missing_parameters, collapse = ", "),
        call. = FALSE
      )
    }
    tibble(
      excluded_cluster_id = excluded_cluster,
      parameter = parameters,
      leave_one_out_estimate = unname(
        stats::coef(jackknife_model)[parameters]
      )
    )
  })

  cluster_count <- length(cluster_values)
  critical_value <- stats::qt(
    1 - (1 - cluster_jackknife_confidence_level) / 2,
    df = cluster_count - 1L
  )
  summary <- leave_one_out %>%
    group_by(parameter) %>%
    summarise(
      jackknife_mean_leave_one_out_estimate = mean(
        leave_one_out_estimate
      ),
      crv3_jackknife_standard_error = sqrt(
        (cluster_count - 1) / cluster_count *
          sum(
            (
              leave_one_out_estimate -
                jackknife_mean_leave_one_out_estimate
            )^2
          )
      ),
      .groups = "drop"
    ) %>%
    mutate(
      estimate = unname(full_estimates[parameter]),
      crv3_jackknife_p_value = 2 * stats::pt(
        -abs(estimate / crv3_jackknife_standard_error),
        df = cluster_count - 1L
      ),
      crv3_jackknife_confidence_low =
        estimate - critical_value * crv3_jackknife_standard_error,
      crv3_jackknife_confidence_high =
        estimate + critical_value * crv3_jackknife_standard_error,
      crv3_jackknife_method = cluster_jackknife_method,
      crv3_jackknife_cluster = inference_cluster,
      crv3_jackknife_confidence_level =
        cluster_jackknife_confidence_level,
      crv3_jackknife_clusters = cluster_count
    )

  list(summary = summary, leave_one_out = leave_one_out)
}

estimate_one <- function(registry_row) {
  message("Estimating ", registry_row$model_id[[1L]], ".")
  prepared <- model_data(registry_row)
  data <- prepared$data
  control_terms <- if (
    registry_row$control_spec[[1L]] == "baseline_trends"
  ) {
    paste0(baseline_control_columns, ":year_centered")
  } else {
    character()
  }
  model_formula <- stats::as.formula(paste0(
    outcome,
    " ~ ",
    paste(c(treatment, control_terms), collapse = " + "),
    " | unit_fe_id + stratum_year_id"
  ))
  model <- fit_border_model(model_formula, data)
  if (!treatment %in% names(stats::coef(model))) {
    stop(
      paste("Treatment is unidentified in", registry_row$model_id[[1L]]),
      call. = FALSE
    )
  }

  conventional_interval <- stats::confint(
    model,
    parm = treatment,
    level = confidence_level
  )
  two_way_result <- two_way_crv1_inference(model, data, treatment)
  bootstrap_result <- wild_bootstrap_test(
    prepare_bootstrap_regression(model, data),
    treatment
  )
  jackknife_result <- cluster_jackknife_inference(
    model_formula,
    data,
    treatment,
    stats::coef(model)
  )
  jackknife_summary <- jackknife_result$summary
  estimate <- tibble(
    model_id = registry_row$model_id[[1L]],
    spec_id = registry_row$spec_id[[1L]],
    geography_family = registry_row$geography_family[[1L]],
    membership_mode = registry_row$membership_mode[[1L]],
    role = registry_row$role[[1L]],
    period = registry_row$period[[1L]],
    weight_spec = registry_row$weight_spec[[1L]],
    control_spec = registry_row$control_spec[[1L]],
    outcome = outcome,
    treatment = treatment,
    treatment_unit = "one_2012_dollar_per_hour",
    estimate = unname(stats::coef(model)[[treatment]]),
    crv1_standard_error = unname(fixest::se(model)[[treatment]]),
    crv1_p_value = unname(fixest::pvalue(model)[[treatment]]),
    crv1_confidence_low = unname(as.numeric(conventional_interval[1L, 1L])),
    crv1_confidence_high = unname(as.numeric(conventional_interval[1L, 2L])),
    two_way_crv1_standard_error =
      two_way_result$two_way_crv1_standard_error[[1L]],
    two_way_crv1_p_value = two_way_result$two_way_crv1_p_value[[1L]],
    two_way_crv1_confidence_low =
      two_way_result$two_way_crv1_confidence_low[[1L]],
    two_way_crv1_confidence_high =
      two_way_result$two_way_crv1_confidence_high[[1L]],
    two_way_crv1_raw_minimum_eigenvalue =
      two_way_result$two_way_crv1_raw_minimum_eigenvalue[[1L]],
    two_way_crv1_maximum_psd_adjustment =
      two_way_result$two_way_crv1_maximum_psd_adjustment[[1L]],
    two_way_crv1_psd_adjusted =
      two_way_result$two_way_crv1_psd_adjusted[[1L]],
    two_way_crv1_method = two_way_result$two_way_crv1_method[[1L]],
    two_way_cluster_dimensions =
      two_way_result$two_way_cluster_dimensions[[1L]],
    two_way_confidence_level =
      two_way_result$two_way_confidence_level[[1L]],
    n_two_way_pair_clusters =
      two_way_result$n_two_way_pair_clusters[[1L]],
    n_two_way_region_clusters =
      two_way_result$n_two_way_region_clusters[[1L]],
    n_two_way_intersection_clusters =
      two_way_result$n_two_way_intersection_clusters[[1L]],
    crv3_jackknife_standard_error =
      jackknife_summary$crv3_jackknife_standard_error[[1L]],
    crv3_jackknife_p_value =
      jackknife_summary$crv3_jackknife_p_value[[1L]],
    crv3_jackknife_confidence_low =
      jackknife_summary$crv3_jackknife_confidence_low[[1L]],
    crv3_jackknife_confidence_high =
      jackknife_summary$crv3_jackknife_confidence_high[[1L]],
    crv3_jackknife_method =
      jackknife_summary$crv3_jackknife_method[[1L]],
    crv3_jackknife_cluster =
      jackknife_summary$crv3_jackknife_cluster[[1L]],
    crv3_jackknife_confidence_level =
      jackknife_summary$crv3_jackknife_confidence_level[[1L]],
    crv3_jackknife_clusters =
      jackknife_summary$crv3_jackknife_clusters[[1L]],
    wcb_p_value = bootstrap_result$p.value[[1L]],
    wcb_confidence_low = bootstrap_result$conf.low[[1L]],
    wcb_confidence_high = bootstrap_result$conf.high[[1L]],
    wcb_method = bootstrap_method,
    wcb_bootstrap_type = bootstrap_type,
    wcb_weights = bootstrap_weights,
    wcb_repetitions = bootstrap_repetitions,
    wcb_seed = bootstrap_seed,
    wcb_fixed_effect_handling = "weighted_fwl_absorption",
    inference_cluster = inference_cluster,
    confidence_level = confidence_level,
    n_observations = stats::nobs(model),
    n_counties = n_distinct(data$county_fips),
    n_strata = n_distinct(data$stratum_id),
    n_inference_clusters = n_distinct(data[[inference_cluster]])
  )
  list(
    estimate = estimate,
    audit = prepared$audit,
    model = model,
    data = data,
    jackknife = jackknife_result
  )
}

estimated <- purrr::map(
  seq_len(nrow(model_registry)),
  ~ estimate_one(model_registry[.x, , drop = FALSE])
)
estimates <- bind_rows(purrr::map(estimated, "estimate"))
sample_audit <- bind_rows(purrr::map(estimated, "audit")) %>%
  left_join(model_registry, by = "model_id", relationship = "one-to-one")

balance_variables <- c(
  paste0("baseline_", outcome, "_2008_2010_z"),
  baseline_control_columns
)
balance_treatment <- paste0("baseline_", treatment, "_2008_2010_z")
balance_results <- list()
for (specification in primary_registry$spec_id) {
  member <- memberships %>%
    filter(spec_id == specification) %>%
    distinct(
      county_fips,
      stratum_id,
      region_pair_id,
      aewr_region_id,
      analysis_weight
    )
  baseline <- panel %>%
    filter(year == design$feature_end_year) %>%
    select(county_fips, all_of(c(balance_variables, balance_treatment))) %>%
    inner_join(member, by = "county_fips", relationship = "one-to-many")
  for (variable in balance_variables) {
    data <- baseline %>%
      filter(stats::complete.cases(across(all_of(c(variable, balance_treatment)))))
    formula <- stats::as.formula(paste0(
      variable,
      " ~ ",
      balance_treatment,
      " | stratum_id"
    ))
    model <- fixest::feols(
      formula,
      data = data,
      weights = ~ analysis_weight,
      vcov = stats::as.formula(paste("~", inference_cluster)),
      notes = FALSE
    )
    balance_results[[length(balance_results) + 1L]] <- tibble(
      spec_id = specification,
      balance_variable = variable,
      treatment_variable = balance_treatment,
      estimate = unname(stats::coef(model)[[balance_treatment]]),
      crv1_standard_error = unname(fixest::se(model)[[balance_treatment]]),
      crv1_p_value = unname(fixest::pvalue(model)[[balance_treatment]]),
      n_counties = n_distinct(data$county_fips),
      n_strata = n_distinct(data$stratum_id),
      n_regions = n_distinct(data$aewr_region_id),
      n_inference_clusters = n_distinct(data[[inference_cluster]])
    )
  }
}
balance <- bind_rows(balance_results)

primary_spec_id <- design$diagnostic_audit$primary_spec_id
primary_model_id <- paste0(primary_spec_id, "__base")
primary_position <- match(primary_model_id, estimates$model_id)
if (is.na(primary_position)) {
  stop("The diagnostic audit could not locate the primary base model.", call. = FALSE)
}
primary_model <- estimated[[primary_position]]$model
primary_data <- estimated[[primary_position]]$data
primary_estimate <- estimates$estimate[[primary_position]]

placebo <- estimates %>%
  filter(role == "placebo")
inference_comparison <- estimates %>%
  transmute(
    model_id,
    spec_id,
    role,
    period,
    estimate,
    crv1_p_value,
    crv3_jackknife_p_value,
    wcb_p_value,
    two_way_crv1_p_value,
    p_value_difference_crv3_minus_crv1 =
      crv3_jackknife_p_value - crv1_p_value,
    p_value_difference_wcb_minus_crv1 = wcb_p_value - crv1_p_value,
    p_value_difference_two_way_minus_pair_crv1 =
      two_way_crv1_p_value - crv1_p_value,
    crv3_p_value_larger = crv3_jackknife_p_value > crv1_p_value,
    wcb_p_value_smaller = wcb_p_value < crv1_p_value,
    two_way_p_value_smaller = two_way_crv1_p_value < crv1_p_value,
    crv1_rejects_at_5_percent = crv1_p_value < 0.05,
    crv3_rejects_at_5_percent = crv3_jackknife_p_value < 0.05,
    wcb_rejects_at_5_percent = wcb_p_value < 0.05,
    two_way_rejects_at_5_percent = two_way_crv1_p_value < 0.05,
    rejection_conclusions_differ =
      crv1_rejects_at_5_percent != crv3_rejects_at_5_percent |
        crv1_rejects_at_5_percent != wcb_rejects_at_5_percent |
        crv1_rejects_at_5_percent != two_way_rejects_at_5_percent,
    n_inference_clusters
  )

annual_registry <- primary_registry %>%
  transmute(
    model_id = paste0(spec_id, "__annual_path"),
    spec_id,
    geography_family,
    membership_mode,
    role = "diagnostic",
    period = "full",
    weight_spec = "equal_county",
    control_spec = "none"
  )
annual_data <- model_data(annual_registry)$data
annual_years <- seq.int(
  as.integer(design$diagnostic_audit$annual_start_year),
  as.integer(design$diagnostic_audit$annual_end_year)
)
annual_terms <- paste0("aewr_year_", annual_years)
for (index in seq_along(annual_years)) {
  annual_data[[annual_terms[[index]]]] <- ifelse(
    annual_data$year == annual_years[[index]],
    annual_data[[treatment]],
    0
  )
}
annual_formula <- stats::as.formula(paste0(
  outcome,
  " ~ ",
  paste(annual_terms, collapse = " + "),
  " | unit_fe_id + stratum_year_id"
))
annual_model <- fit_border_model(annual_formula, annual_data)
if (!all(annual_terms %in% names(stats::coef(annual_model)))) {
  stop("One or more annual treatment coefficients are unidentified.", call. = FALSE)
}
annual_jackknife <- cluster_jackknife_inference(
  annual_formula,
  annual_data,
  annual_terms,
  stats::coef(annual_model)
)$summary
annual_bootstrap_regression <- prepare_bootstrap_regression(
  annual_model,
  annual_data
)
annual_coefficients <- purrr::map_dfr(seq_along(annual_terms), function(index) {
  parameter <- annual_terms[[index]]
  conventional_interval <- stats::confint(
    annual_model,
    parm = parameter,
    level = confidence_level
  )
  bootstrap_result <- wild_bootstrap_test(
    annual_bootstrap_regression,
    parameter
  )
  jackknife_result <- annual_jackknife %>%
    filter(.data$parameter == .env$parameter)
  if (nrow(jackknife_result) != 1L) {
    stop("Annual CV3J output must contain one row per year.", call. = FALSE)
  }
  tibble(
    model_id = annual_registry$model_id[[1L]],
    spec_id = primary_spec_id,
    year = annual_years[[index]],
    period = if_else(
      annual_years[[index]] < design$policy_start_year,
      "placebo",
      "policy"
    ),
    estimate = unname(stats::coef(annual_model)[[parameter]]),
    crv1_standard_error = unname(fixest::se(annual_model)[[parameter]]),
    crv1_p_value = unname(fixest::pvalue(annual_model)[[parameter]]),
    crv1_confidence_low = unname(as.numeric(conventional_interval[1L, 1L])),
    crv1_confidence_high = unname(as.numeric(conventional_interval[1L, 2L])),
    crv3_jackknife_standard_error =
      jackknife_result$crv3_jackknife_standard_error[[1L]],
    crv3_jackknife_p_value =
      jackknife_result$crv3_jackknife_p_value[[1L]],
    crv3_jackknife_confidence_low =
      jackknife_result$crv3_jackknife_confidence_low[[1L]],
    crv3_jackknife_confidence_high =
      jackknife_result$crv3_jackknife_confidence_high[[1L]],
    crv3_jackknife_method =
      jackknife_result$crv3_jackknife_method[[1L]],
    crv3_jackknife_cluster =
      jackknife_result$crv3_jackknife_cluster[[1L]],
    crv3_jackknife_confidence_level =
      jackknife_result$crv3_jackknife_confidence_level[[1L]],
    crv3_jackknife_clusters =
      jackknife_result$crv3_jackknife_clusters[[1L]],
    wcb_p_value = bootstrap_result$p.value[[1L]],
    wcb_confidence_low = bootstrap_result$conf.low[[1L]],
    wcb_confidence_high = bootstrap_result$conf.high[[1L]],
    wcb_method = bootstrap_method,
    wcb_bootstrap_type = bootstrap_type,
    wcb_weights = bootstrap_weights,
    wcb_repetitions = bootstrap_repetitions,
    wcb_seed = bootstrap_seed,
    inference_cluster = inference_cluster,
    n_observations = stats::nobs(annual_model),
    n_counties = n_distinct(annual_data$county_fips),
    n_strata = n_distinct(annual_data$stratum_id),
    n_inference_clusters = n_distinct(annual_data[[inference_cluster]])
  )
})

primary_formula <- stats::formula(primary_model)
cluster_values <- sort(unique(as.character(primary_data[[inference_cluster]])))
leave_one_out <- purrr::map_dfr(cluster_values, function(excluded_border_pair) {
  data <- primary_data %>%
    filter(
      as.character(.data[[inference_cluster]]) != excluded_border_pair
    ) %>%
    droplevels() %>%
    drop_singletons()
  data$estimation_weight <- data$estimation_weight /
    mean(data$estimation_weight)
  model <- fit_border_model(primary_formula, data)
  interval <- stats::confint(
    model,
    parm = treatment,
    level = confidence_level
  )
  estimate <- unname(stats::coef(model)[[treatment]])
  tibble(
    model_id = primary_model_id,
    spec_id = primary_spec_id,
    excluded_border_pair_id = excluded_border_pair,
    estimate = estimate,
    difference_from_full_estimate = estimate - primary_estimate,
    crv1_standard_error = unname(fixest::se(model)[[treatment]]),
    crv1_p_value = unname(fixest::pvalue(model)[[treatment]]),
    crv1_confidence_low = unname(as.numeric(interval[1L, 1L])),
    crv1_confidence_high = unname(as.numeric(interval[1L, 2L])),
    n_observations = stats::nobs(model),
    n_counties = n_distinct(data$county_fips),
    n_strata = n_distinct(data$stratum_id),
    n_inference_clusters = n_distinct(data[[inference_cluster]])
  )
})

jackknife_confidence_level <- as.numeric(
  design$diagnostic_audit$cluster_jackknife_confidence_level
)
jackknife_cluster_count <- nrow(leave_one_out)
jackknife_mean <- mean(leave_one_out$estimate)
jackknife_standard_error <- sqrt(
  (jackknife_cluster_count - 1) / jackknife_cluster_count *
    sum((leave_one_out$estimate - jackknife_mean)^2)
)
jackknife_critical_value <- stats::qt(
  1 - (1 - jackknife_confidence_level) / 2,
  df = jackknife_cluster_count - 1
)
cluster_jackknife <- tibble(
  model_id = primary_model_id,
  spec_id = primary_spec_id,
  estimate = primary_estimate,
  jackknife_mean_leave_one_out_estimate = jackknife_mean,
  crv3_jackknife_method = cluster_jackknife_method,
  crv3_jackknife_cluster = inference_cluster,
  crv3_jackknife_standard_error = jackknife_standard_error,
  crv3_jackknife_p_value = 2 * stats::pt(
    -abs(primary_estimate / jackknife_standard_error),
    df = jackknife_cluster_count - 1
  ),
  crv3_jackknife_confidence_low =
    primary_estimate - jackknife_critical_value * jackknife_standard_error,
  crv3_jackknife_confidence_high =
    primary_estimate + jackknife_critical_value * jackknife_standard_error,
  cluster_jackknife_standard_error = jackknife_standard_error,
  cluster_jackknife_p_value = 2 * stats::pt(
    -abs(primary_estimate / jackknife_standard_error),
    df = jackknife_cluster_count - 1
  ),
  cluster_jackknife_confidence_low =
    primary_estimate - jackknife_critical_value * jackknife_standard_error,
  cluster_jackknife_confidence_high =
    primary_estimate + jackknife_critical_value * jackknife_standard_error,
  confidence_level = jackknife_confidence_level,
  n_inference_clusters = jackknife_cluster_count
)

within_matrix <- fixest::demean(
  primary_model,
  as.matrix = TRUE,
  tol = 1e-10,
  nthreads = 1L
)
within_outcome <- within_matrix[, 1L]
within_treatment <- within_matrix[, treatment]
within_residual <- within_outcome - within_treatment * primary_estimate
influence_work <- tibble(
  region_pair_id = as.character(primary_data[[inference_cluster]]),
  within_treatment = within_treatment,
  within_residual = within_residual,
  estimation_weight = primary_data$estimation_weight
) %>%
  group_by(region_pair_id) %>%
  summarise(
    cluster_score = sum(
      estimation_weight * within_treatment * within_residual
    ),
    treatment_sum_squares = sum(
      estimation_weight * within_treatment^2
    ),
    .groups = "drop"
  )
total_treatment_sum_squares <- sum(influence_work$treatment_sum_squares)
total_absolute_score <- sum(abs(influence_work$cluster_score))
cluster_influence <- influence_work %>%
  mutate(
    treatment_leverage =
      treatment_sum_squares / total_treatment_sum_squares,
    absolute_score_share = abs(cluster_score) / total_absolute_score,
    one_step_coefficient_contribution =
      cluster_score / total_treatment_sum_squares,
    effective_inference_cluster_count =
      1 / sum(treatment_leverage^2)
  ) %>%
  left_join(
    leave_one_out %>%
      select(
        region_pair_id = excluded_border_pair_id,
        leave_one_out_estimate = estimate,
        difference_from_full_estimate
      ),
    by = "region_pair_id",
    relationship = "one-to-one"
  ) %>%
  mutate(
    model_id = primary_model_id,
    spec_id = primary_spec_id,
    .before = 1L
  )

missing_diagnostic_margins <- setdiff(
  diagnostic_margin_columns,
  names(primary_data)
)
if (length(missing_diagnostic_margins) > 0L) {
  stop(
    "Primary-model data are missing diagnostic margins: ",
    paste(missing_diagnostic_margins, collapse = ", "),
    call. = FALSE
  )
}

primary_decomposition <- primary_data %>%
  transmute(
    model_id = primary_model_id,
    spec_id = primary_spec_id,
    county_fips,
    countyname,
    state_abbrev,
    aewr_region_id = as.character(aewr_region_id),
    region_pair_id = as.character(region_pair_id),
    stratum_id,
    year,
    observed_outcome = .data[[outcome]],
    observed_treatment = .data[[treatment]],
    estimation_weight,
    within_outcome,
    within_treatment,
    model_residual = within_residual
  )
diagnostic_margin_map <- unlist(
  design$diagnostic_margin_columns,
  use.names = TRUE
)
for (margin_name in names(diagnostic_margin_map)) {
  source_column <- diagnostic_margin_map[[margin_name]]
  primary_decomposition[[margin_name]] <- primary_data[[source_column]]
}

primary_decomposition <- primary_decomposition %>%
  mutate(
    coefficient_numerator =
      estimation_weight * within_treatment * within_outcome,
    model_score =
      estimation_weight * within_treatment * model_residual,
    treatment_sum_squares =
      estimation_weight * within_treatment^2
  )
decomposition_treatment_sum_squares <- sum(
  primary_decomposition$treatment_sum_squares
)
decomposition_coefficient_numerator <- sum(
  primary_decomposition$coefficient_numerator
)
decomposition_absolute_observation_score <- sum(
  abs(primary_decomposition$model_score)
)
if (
  !is.finite(decomposition_treatment_sum_squares) ||
    decomposition_treatment_sum_squares <= 0 ||
    !is.finite(decomposition_absolute_observation_score) ||
    decomposition_absolute_observation_score <= 0 ||
    abs(
      decomposition_coefficient_numerator /
        decomposition_treatment_sum_squares - primary_estimate
    ) > 1e-10 ||
    abs(sum(primary_decomposition$model_score)) > 1e-8
) {
  stop(
    "Primary score decomposition does not reproduce the fitted model.",
    call. = FALSE
  )
}
primary_decomposition <- primary_decomposition %>%
  mutate(
    coefficient_contribution =
      coefficient_numerator / decomposition_treatment_sum_squares,
    treatment_leverage =
      treatment_sum_squares / decomposition_treatment_sum_squares,
    absolute_observation_score_share =
      abs(model_score) / decomposition_absolute_observation_score
  ) %>%
  arrange(county_fips, year)

annual_change_columns <- c(
  "observed_treatment",
  setdiff(names(diagnostic_margin_map), "baseline_farm_employment")
)
for (column in annual_change_columns) {
  change_column <- paste0(column, "_annual_change")
  primary_decomposition <- primary_decomposition %>%
    group_by(county_fips) %>%
    arrange(year, .by_group = TRUE) %>%
    mutate(
      !!change_column := .data[[column]] - dplyr::lag(.data[[column]])
    ) %>%
    ungroup()
}

county_decomposition <- primary_decomposition %>%
  group_by(
    model_id,
    spec_id,
    county_fips,
    countyname,
    state_abbrev,
    aewr_region_id,
    region_pair_id,
    stratum_id
  ) %>%
  summarise(
    n_years = n(),
    mean_outcome = mean(observed_outcome),
    total_requested_positions = sum(requested_positions),
    total_certified_positions = sum(certified_positions),
    total_applications = sum(applications),
    mean_balanced_linked_worksite_businesses = mean(
      balanced_linked_worksite_businesses
    ),
    maximum_balanced_linked_worksite_businesses = max(
      balanced_linked_worksite_businesses
    ),
    baseline_farm_employment = first(baseline_farm_employment),
    coefficient_numerator = sum(coefficient_numerator),
    coefficient_contribution = sum(coefficient_contribution),
    model_score = sum(model_score),
    treatment_sum_squares = sum(treatment_sum_squares),
    treatment_leverage = sum(treatment_leverage),
    absolute_observation_score_share =
      sum(absolute_observation_score_share),
    .groups = "drop"
  ) %>%
  mutate(
    absolute_aggregated_score_share =
      abs(model_score) / sum(abs(model_score))
  ) %>%
  arrange(desc(absolute_aggregated_score_share), county_fips)

stratum_region_year_decomposition <- primary_decomposition %>%
  group_by(
    model_id,
    spec_id,
    region_pair_id,
    stratum_id,
    aewr_region_id,
    year
  ) %>%
  summarise(
    state_abbreviations = paste(sort(unique(state_abbrev)), collapse = ", "),
    n_counties = n_distinct(county_fips),
    observed_treatment = first(observed_treatment),
    treatment_annual_change = first(observed_treatment_annual_change),
    mean_outcome = mean(observed_outcome),
    total_requested_positions = sum(requested_positions),
    total_requested_positions_annual_change = if (
      all(is.na(requested_positions_annual_change))
    ) NA_real_ else sum(requested_positions_annual_change, na.rm = TRUE),
    total_certified_positions = sum(certified_positions),
    total_certified_positions_annual_change = if (
      all(is.na(certified_positions_annual_change))
    ) NA_real_ else sum(certified_positions_annual_change, na.rm = TRUE),
    total_applications = sum(applications),
    total_applications_annual_change = if (
      all(is.na(applications_annual_change))
    ) NA_real_ else sum(applications_annual_change, na.rm = TRUE),
    summed_balanced_linked_worksite_businesses = sum(
      balanced_linked_worksite_businesses
    ),
    summed_balanced_linked_worksite_businesses_annual_change = if (
      all(is.na(balanced_linked_worksite_businesses_annual_change))
    ) NA_real_ else sum(
      balanced_linked_worksite_businesses_annual_change,
      na.rm = TRUE
    ),
    total_baseline_farm_employment = sum(baseline_farm_employment),
    coefficient_numerator = sum(coefficient_numerator),
    coefficient_contribution = sum(coefficient_contribution),
    model_score = sum(model_score),
    treatment_sum_squares = sum(treatment_sum_squares),
    treatment_leverage = sum(treatment_leverage),
    .groups = "drop"
  ) %>%
  mutate(
    absolute_aggregated_score_share =
      abs(model_score) / sum(abs(model_score))
  ) %>%
  arrange(stratum_id, year, aewr_region_id)

region_decomposition <- primary_decomposition %>%
  group_by(model_id, spec_id, aewr_region_id) %>%
  summarise(
    state_abbreviations = paste(sort(unique(state_abbrev)), collapse = ", "),
    n_counties = n_distinct(county_fips),
    n_observations = n(),
    total_requested_positions = sum(requested_positions),
    total_certified_positions = sum(certified_positions),
    total_applications = sum(applications),
    mean_balanced_linked_worksite_businesses = mean(
      balanced_linked_worksite_businesses
    ),
    active_county_year_share = mean(certified_positions > 0),
    mean_outcome = mean(observed_outcome),
    residual_standard_deviation = stats::sd(model_residual),
    coefficient_numerator = sum(coefficient_numerator),
    coefficient_contribution = sum(coefficient_contribution),
    model_score = sum(model_score),
    treatment_sum_squares = sum(treatment_sum_squares),
    treatment_leverage = sum(treatment_leverage),
    .groups = "drop"
  ) %>%
  mutate(
    certified_position_share =
      total_certified_positions / sum(total_certified_positions),
    absolute_aggregated_score_share =
      abs(model_score) / sum(abs(model_score)),
    certified_position_rank = min_rank(desc(total_certified_positions)),
    mean_outcome_rank = min_rank(desc(mean_outcome)),
    absolute_score_rank = min_rank(desc(absolute_aggregated_score_share))
  ) %>%
  arrange(as.integer(aewr_region_id))

border_pair_decomposition <- primary_decomposition %>%
  group_by(model_id, spec_id, region_pair_id) %>%
  summarise(
    aewr_region_ids = paste(
      sort(unique(as.integer(aewr_region_id))),
      collapse = ", "
    ),
    state_abbreviations = paste(sort(unique(state_abbrev)), collapse = ", "),
    n_counties = n_distinct(county_fips),
    n_observations = n(),
    total_requested_positions = sum(requested_positions),
    total_certified_positions = sum(certified_positions),
    total_applications = sum(applications),
    mean_balanced_linked_worksite_businesses = mean(
      balanced_linked_worksite_businesses
    ),
    active_county_year_share = mean(certified_positions > 0),
    mean_outcome = mean(observed_outcome),
    residual_standard_deviation = stats::sd(model_residual),
    coefficient_numerator = sum(coefficient_numerator),
    coefficient_contribution = sum(coefficient_contribution),
    model_score = sum(model_score),
    treatment_sum_squares = sum(treatment_sum_squares),
    treatment_leverage = sum(treatment_leverage),
    .groups = "drop"
  ) %>%
  mutate(
    certified_position_share =
      total_certified_positions / sum(total_certified_positions),
    absolute_aggregated_score_share =
      abs(model_score) / sum(abs(model_score)),
    certified_position_rank = min_rank(desc(total_certified_positions)),
    mean_outcome_rank = min_rank(desc(mean_outcome)),
    absolute_score_rank = min_rank(desc(absolute_aggregated_score_share))
  ) %>%
  arrange(region_pair_id)

score_comparison <- cluster_influence %>%
  select(region_pair_id, cluster_score, absolute_score_share) %>%
  left_join(
    border_pair_decomposition %>%
      select(
        region_pair_id,
        decomposition_score = model_score,
        decomposition_absolute_score_share = absolute_aggregated_score_share
      ),
    by = "region_pair_id",
    relationship = "one-to-one"
  )
if (
  any(abs(score_comparison$cluster_score -
    score_comparison$decomposition_score) > 1e-10) ||
    any(abs(score_comparison$absolute_score_share -
      score_comparison$decomposition_absolute_score_share) > 1e-10)
) {
  stop(
    "Border-pair decomposition does not reproduce cluster influence.",
    call. = FALSE
  )
}

treatment_support <- annual_data %>%
  group_by(region_pair_id, stratum_id, year) %>%
  summarise(
    n_counties = n_distinct(county_fips),
    n_regions = n_distinct(aewr_region_id),
    treatment_minimum = min(.data[[treatment]]),
    treatment_maximum = max(.data[[treatment]]),
    treatment_range = treatment_maximum - treatment_minimum,
    informative_treatment_contrast = treatment_range > 1e-10,
    .groups = "drop"
  ) %>%
  mutate(
    model_id = primary_model_id,
    spec_id = primary_spec_id,
    .before = 1L
  )

write_csv(estimates, path_tables("border_discontinuity_estimates.csv"))
write_csv(sample_audit, path_tables("border_discontinuity_sample_audit.csv"))
write_csv(balance, path_tables("border_discontinuity_balance.csv"))
write_csv(placebo, path_tables("border_discontinuity_placebo.csv"))
write_csv(
  inference_comparison,
  path_tables("border_discontinuity_ring_inference_comparison.csv")
)
write_csv(
  annual_coefficients,
  path_tables("border_discontinuity_ring_annual_coefficients.csv")
)
write_csv(
  leave_one_out,
  path_tables("border_discontinuity_ring_leave_one_border_pair_out.csv")
)
write_csv(
  cluster_jackknife,
  path_tables("border_discontinuity_ring_border_pair_jackknife.csv")
)
write_csv(
  cluster_influence,
  path_tables("border_discontinuity_ring_border_pair_influence.csv")
)
write_csv(
  primary_decomposition,
  path_tables("border_discontinuity_ring_county_year_decomposition.csv")
)
write_csv(
  county_decomposition,
  path_tables("border_discontinuity_ring_county_decomposition.csv")
)
write_csv(
  stratum_region_year_decomposition,
  path_tables("border_discontinuity_ring_stratum_region_year_decomposition.csv")
)
write_csv(
  region_decomposition,
  path_tables("border_discontinuity_ring_region_decomposition.csv")
)
write_csv(
  border_pair_decomposition,
  path_tables("border_discontinuity_ring_border_pair_decomposition.csv")
)
write_csv(
  treatment_support,
  path_tables("border_discontinuity_ring_treatment_support.csv")
)

message(
  "Estimated ",
  nrow(estimates),
  " AEWR-border ring models and the bounded primary-specification audit with ",
  bootstrap_method,
  " inference."
)
