# Purpose: Declared-design power for the statewide-AEWR state-border comparison.
# Inputs: statewide_design.json, design.json power settings, state-border
#   memberships, AEWR schedules, h2a_data_year (pre-period fiscal-year counts
#   only) and the shared panel's fixed 2011 farm employment.
# Outputs:
#   data/intermediate/border_statewide_power_panel.parquet
#   outputs/tables/border_statewide_{treatment_support,calibration,power,mde,decision}.csv
#   outputs/reports/border_statewide_power.md
#
# No post-policy outcome is read. Errors are simulated on the actual state-border
# geography and the actual FY2026/FY2027 statewide rates; their scale comes
# only from pre-period FE-only outcome residuals.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "border_discontinuity", "power.R"))
library(arrow)
library(dplyr)
library(readr)

registry <- jsonlite::read_json(
  path_code("designs", "border_discontinuity", "statewide_design.json"),
  simplifyVector = TRUE
)
v6 <- jsonlite::read_json(
  path_code("designs", "border_discontinuity", "design.json"),
  simplifyVector = TRUE
)
settings <- v6$redesign$power
assumptions <- v6$redesign$precision_assumptions
if (settings$nthreads != 1L) stop("Power must run on one thread.", call. = FALSE)

memberships <- read_parquet(path_int("border_statewide_memberships.parquet"))
iv_ids <- c("county_fips", "state_fips", "region_pair_id", "stratum_id")
if (!all(vapply(memberships[iv_ids], is.character, logical(1)))) {
  stop("Membership identifiers must be strings.", call. = FALSE)
}

# Treatment schedule by state and fiscal year ---------------------------------

treatment_date <- function(fiscal_year) as.Date(sprintf("%d-04-01", fiscal_year))
all_years <- seq.int(
  min(registry$calibration$sensitivity_years),
  max(unlist(registry$scenarios$policy_years))
)
post_years <- as.integer(registry$period$post_fiscal_years)

regional <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>%
  filter(
    occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"),
    worker_category == "covered_workers"
  ) %>%
  transmute(
    state_fips,
    aewr = as.numeric(aewr),
    start = as.Date(effective_start),
    end = as.Date(effective_end_exclusive)
  )
statewide <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  # The parsed source keeps the workbook's trailing whitespace in sheet names.
  filter(trimws(source_sheet) == registry$policy_event$source_sheet) %>%
  transmute(
    state_fips,
    worker_category,
    skill_level = as.character(skill_level),
    aewr = as.numeric(hourly_aewr),
    start = as.Date(effective_start),
    end = as.Date(effective_end_exclusive)
  )

rate_on <- function(schedule, date) {
  selected <- schedule %>%
    filter(start <= date, is.na(end) | end > date)
  if (anyDuplicated(selected$state_fips)) {
    stop("Overlapping AEWR schedules on ", date, call. = FALSE)
  }
  selected %>% select(state_fips, aewr)
}

treatment_variants <- bind_rows(
  as_tibble(registry$treatment$primary) %>% mutate(role = "primary"),
  as_tibble(registry$treatment$sensitivities) %>% mutate(role = "sensitivity")
)
states <- sort(unique(memberships$state_fips))
treatment <- bind_rows(lapply(seq_len(nrow(treatment_variants)), function(index) {
  variant <- treatment_variants[index, ]
  bind_rows(lapply(all_years, function(year) {
    date <- treatment_date(year)
    rates <- if (year %in% post_years) {
      rate_on(
        statewide %>% filter(
          worker_category == variant$worker_category,
          skill_level == variant$skill_level
        ),
        date
      )
    } else {
      rate_on(regional, date)
    }
    tibble(state_fips = states, year = year) %>%
      left_join(rates, by = "state_fips") %>%
      mutate(treatment_label = variant$label)
  }))
}))
if (any(!is.finite(treatment$aewr))) {
  missing <- treatment %>% filter(!is.finite(aewr)) %>% distinct(treatment_label, year)
  stop("Missing AEWR for retained states: ", paste(missing$year, collapse = ","), call. = FALSE)
}

# Pre-period outcomes ------------------------------------------------------------

denominator <- read_parquet(
  path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "emp_farm_2011")
) %>%
  distinct()
if (anyDuplicated(denominator$county_fips)) {
  stop("Fixed 2011 farm employment must be unique by county.", call. = FALSE)
}
outcome_variants <- as_tibble(registry$outcome_variants)
wage_variant <- registry$outcome_variants[registry$outcome_variants$outcome_id ==
  "offered_wage_start_year", ]
counts <- read_parquet(
  path_int("h2a_data_year.parquet"),
  col_select = c(
    "county_fips", "year", registry$outcome$numerator,
    wage_variant$numerator, wage_variant$denominator
  )
) %>%
  mutate(year = as.integer(year))
pre_years_available <- sort(unique(counts$year))
if (any(post_years %in% pre_years_available)) {
  stop("A post-policy fiscal or start year is present; this power stage must not read it.",
    call. = FALSE)
}

# Simulation inputs ---------------------------------------------------------------

power_frame <- function(spec_id, years, keep_counties = NULL) {
  members <- memberships %>%
    filter(spec_id == !!spec_id) %>%
    inner_join(denominator, by = "county_fips") %>%
    filter(is.finite(emp_farm_2011), emp_farm_2011 > 0)
  if (!is.null(keep_counties)) {
    members <- members %>% filter(county_fips %in% keep_counties)
  }
  # Dropping a nonpositive denominator can strip a side; keep two-sided pairs.
  members <- members %>%
    group_by(region_pair_id) %>%
    filter(n_distinct(state_fips) == 2L) %>%
    ungroup()
  tidyr::crossing(members, year = years) %>%
    mutate(
      aewr_region_id = state_fips,
      unit_fe_id = county_fips,
      stratum_year_id = paste(stratum_id, year, sep = "__"),
      estimation_weight = 1
    ) %>%
    arrange(stratum_id, county_fips, year) %>%
    as.data.frame()
}

with_outcome <- function(frame, outcome_id) {
  joined <- frame %>% left_join(counts, by = c("county_fips", "year"))
  if (outcome_id == "certifications_fiscal_year") {
    joined <- joined %>% mutate(
      numerator = coalesce(.data[[registry$outcome$numerator]], 0),
      border_y = numerator / emp_farm_2011
    )
  } else {
    hours <- joined[[wage_variant$denominator]]
    joined$border_y <- ifelse(
      is.finite(hours) & hours > 0,
      joined[[wage_variant$numerator]] / hours,
      NA_real_
    )
  }
  as.data.frame(joined)
}

# Wage offers exist only where certified hourly offers exist. Keep counties
# observed in every calibration year, then pairs that retain both sides.
wage_counties <- function(spec_id, years) {
  observed <- with_outcome(power_frame(spec_id, years), "offered_wage_start_year") %>%
    group_by(county_fips) %>%
    summarise(complete = all(is.finite(border_y)) && n() == length(years), .groups = "drop")
  observed$county_fips[observed$complete]
}

precision_for <- function(outcome_id) {
  if (outcome_id == "certifications_fiscal_year") {
    list(
      effects = unique(round(as.numeric(settings$effect_grid), 12L)),
      mde_grid = round(seq(settings$mde_grid_step, settings$mde_grid_max,
        by = settings$mde_grid_step), 12L),
      threshold = assumptions$primary_threshold_beta
    )
  } else {
    declared <- wage_variant$precision
    list(
      effects = unique(round(as.numeric(unlist(declared$effect_grid)), 12L)),
      mde_grid = round(seq(declared$mde_grid_step, declared$mde_grid_max,
        by = declared$mde_grid_step), 12L),
      threshold = declared$primary_threshold_beta
    )
  }
}

with_treatment <- function(frame, label) {
  frame %>%
    left_join(
      treatment %>% filter(treatment_label == label) %>% select(state_fips, year, aewr),
      by = c("state_fips", "year")
    ) %>%
    mutate(border_x = aewr) %>%
    as.data.frame()
}

pair_support <- function(policy) {
  data <- policy$data
  data$score <- policy$xw[, 1L]^2
  pair_ss <- tapply(data$score, data$region_pair_id, sum)
  signed <- policy$incidence
  for (pair in colnames(signed)) signed[policy$pair_regions[[pair]][2L], pair] <- -1
  list(
    partial_treatment_ss = sum(pair_ss),
    effective_pair_leverage_count = sum(pair_ss)^2 / sum(pair_ss^2),
    maximum_pair_leverage_share = max(pair_ss) / sum(pair_ss),
    signed_incidence_rank = qr(signed)$rank
  )
}

crossing_effect <- function(grid, values, target) {
  positions <- which(values >= target)
  if (length(positions)) grid[positions[1L]] else NA_real_
}

dgps <- unlist(settings$dgp_ids)
specs <- as_tibble(registry$geography$specs)
scenarios <- as_tibble(registry$scenarios)
calibrations <- list(
  primary = registry$calibration$primary_years,
  sensitivity = registry$calibration$sensitivity_years
)

combinations <- bind_rows(
  tidyr::crossing(
    outcome_id = outcome_variants$outcome_id,
    spec_id = specs$spec_id[specs$role == "primary"],
    treatment_label = treatment_variants$label,
    scenario_id = scenarios$scenario_id,
    calibration = names(calibrations)
  ),
  tidyr::crossing(
    outcome_id = outcome_variants$outcome_id,
    spec_id = specs$spec_id[specs$role != "primary"],
    treatment_label = treatment_variants$label[treatment_variants$role == "primary"],
    scenario_id = scenarios$scenario_id,
    calibration = "primary"
  )
)

power_rows <- list()
mde_rows <- list()
calibration_rows <- list()
support_rows <- list()
panel_rows <- list()

for (index in seq_len(nrow(combinations))) {
  combo <- combinations[index, ]
  message("Statewide power: ", paste(combo, collapse = " / "))
  scenario <- scenarios[scenarios$scenario_id == combo$scenario_id, ]
  policy_range <- range(unlist(scenario$policy_years))
  policy_years <- seq.int(policy_range[1], policy_range[2])
  pre_years <- seq.int(calibrations[[combo$calibration]][1], calibrations[[combo$calibration]][2])
  if (!all(pre_years %in% pre_years_available)) {
    stop("Calibration years lack outcome data: ", combo$calibration, call. = FALSE)
  }

  precision <- precision_for(combo$outcome_id)
  effects <- precision$effects
  mde_grid <- precision$mde_grid
  keep <- if (combo$outcome_id == "offered_wage_start_year") {
    wage_counties(combo$spec_id, pre_years)
  } else NULL
  pre_data <- with_outcome(power_frame(combo$spec_id, pre_years, keep), combo$outcome_id)
  # The policy panel carries treatment only. Its outcome slot is a declared
  # zero placeholder because no post-period outcome may enter this stage.
  policy_data <- with_treatment(power_frame(combo$spec_id, policy_years, keep), combo$treatment_label)
  policy_data$border_y <- 0
  pre <- border_power_projection(pre_data, NULL, pre_data$border_y, fe_only = TRUE)
  policy <- border_power_projection(
    policy_data, cbind(border_x = policy_data$border_x), policy_data$border_y
  )
  if (!identical(pre$units, policy$units)) {
    stop("Calibration and policy panels must share units.", call. = FALSE)
  }
  graph <- border_power_region_graph(
    list(list(policy = list(data = policy_data))),
    settings$spatial_graph_coefficient
  )
  support <- pair_support(policy)
  gaps <- policy_data %>%
    filter(year %in% post_years) %>%
    group_by(region_pair_id, year) %>%
    summarise(gap = diff(range(border_x)), .groups = "drop")
  support_rows[[index]] <- tibble(
    !!!combo,
    threshold = precision$threshold,
    counties = length(policy$units),
    pairs = length(policy$pairs),
    states = length(policy$regions),
    policy_years = paste(range(policy_years), collapse = "-"),
    median_post_pair_gap = median(gaps$gap),
    maximum_post_pair_gap = max(gaps$gap),
    zero_gap_post_pair_years = sum(gaps$gap < 1e-9),
    pre_mean_outcome = mean(pre_data$border_y),
    pre_sd_outcome = stats::sd(pre_data$border_y),
    !!!support
  )
  if (combo$calibration == "primary" && combo$scenario_id == scenarios$scenario_id[1]) {
    panel_rows[[length(panel_rows) + 1L]] <- policy_data %>%
      select(county_fips, state_fips, former_aewr_region_id, region_pair_id, year, border_x) %>%
      mutate(outcome_id = combo$outcome_id, spec_id = combo$spec_id,
        treatment_label = combo$treatment_label) %>%
      left_join(
        pre_data %>% select(county_fips, year, pre_outcome = border_y),
        by = c("county_fips", "year")
      )
  }

  for (dgp in dgps) {
    set.seed(as.integer(settings$seed))
    fractions <- settings$variance_fractions[[dgp]]
    process <- c(
      as.list(fractions[c("region", "pair", "county")]),
      list(spatial = identical(dgp, "spatial_region_serial"))
    )
    scale <- border_power_calibrate(pre, process, settings, graph)
    draws <- list()
    chunks <- split(seq_len(settings$repetitions),
      ceiling(seq_len(settings$repetitions) / settings$chunk_size))
    for (chunk in chunks) {
      errors <- border_power_unit_errors(policy, length(chunk), process, settings$ar1, graph)
      draws[[length(draws) + 1L]] <- border_power_draw_statistics(policy, errors)
    }
    coefficient_errors <- unlist(lapply(draws, `[[`, "coefficient_error"))
    variances <- do.call(cbind, lapply(draws, `[[`, "variance"))
    reference_df <- draws[[1L]]$df
    calibration_rows[[length(calibration_rows) + 1L]] <- tibble(
      !!!combo, dgp_id = dgp, sigma = scale$sigma,
      observed_preperiod_residual_ss = scale$observed_residual_ss,
      expected_unit_residual_ss = scale$expected_unit_residual_ss,
      expected_unit_residual_ss_mc_se = scale$expected_unit_residual_ss_mc_se
    )
    for (method in registry$inference_methods) {
      curve <- border_power_grid(
        coefficient_errors, variances[method, ], reference_df[[method]],
        unique(c(effects, mde_grid, -mde_grid)), scale$sigma, assumptions$alpha
      )
      null <- curve[curve$effect == 0, , drop = FALSE]
      size_gate <- null$invalid_repetitions == 0L &&
        null$mc_confidence_high <= assumptions$size_maximum
      power_rows[[length(power_rows) + 1L]] <- tibble(
        !!!combo, dgp_id = dgp, inference_method = method, size_gate_pass = size_gate,
        curve[curve$effect %in% effects, , drop = FALSE]
      )
      for (sign in c(-1, 1)) {
        directional <- curve[match(sign * mde_grid, curve$effect), , drop = FALSE]
        descriptive <- crossing_effect(mde_grid, directional$rejection_rate, assumptions$target_power)
        mde_rows[[length(mde_rows) + 1L]] <- tibble(
          !!!combo, dgp_id = dgp, inference_method = method, effect_sign = sign,
          grid_mde = descriptive,
          size_valid_mde = if (size_gate) descriptive else NA_real_,
          size_gate_pass = size_gate,
          null_rejection_rate = null$rejection_rate,
          null_mc_confidence_high = null$mc_confidence_high,
          invalid_repetitions = null$invalid_repetitions
        )
      }
    }
  }
}

support_table <- bind_rows(support_rows)
calibration_table <- bind_rows(calibration_rows)
power_table <- bind_rows(power_rows)
mde_table <- bind_rows(mde_rows)

shared_dgps <- setdiff(dgps, "county_serial")
thresholds <- tibble(
  outcome_id = outcome_variants$outcome_id,
  threshold = vapply(outcome_variants$outcome_id, function(id) precision_for(id)$threshold, numeric(1))
)
decision_keys <- c("outcome_id", "spec_id", "treatment_label", "scenario_id", "calibration",
  "inference_method")
decision <- power_table %>%
  inner_join(thresholds, by = "outcome_id") %>%
  filter(dgp_id %in% shared_dgps, abs(abs(effect) - threshold) < 1e-12) %>%
  group_by(across(all_of(c(decision_keys, "threshold")))) %>%
  summarise(minimum_power_both_signs = min(rejection_rate), .groups = "drop") %>%
  left_join(
    mde_table %>%
      filter(dgp_id %in% shared_dgps) %>%
      group_by(across(all_of(decision_keys))) %>%
      summarise(
        all_shared_size_gates = all(size_gate_pass),
        maximum_size_valid_mde = if (all(is.finite(size_valid_mde))) max(size_valid_mde) else NA_real_,
        maximum_null_rejection = max(null_rejection_rate),
        .groups = "drop"
      ),
    by = decision_keys
  ) %>%
  mutate(
    conditional_precision_pass = all_shared_size_gates &
      minimum_power_both_signs >= assumptions$target_power &
      is.finite(maximum_size_valid_mde) & maximum_size_valid_mde <= threshold
  )

iv_dir <- function(path) dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
panel_path <- path_int("border_statewide_power_panel.parquet")
write_parquet(bind_rows(panel_rows), panel_path)
for (name in c("treatment_support", "calibration", "power", "mde", "decision")) {
  table <- switch(name, treatment_support = support_table, calibration = calibration_table,
    power = power_table, mde = mde_table, decision = decision)
  path <- path_tables(paste0("border_statewide_", name, ".csv"))
  iv_dir(path)
  write_csv(table, path)
}

# Report ---------------------------------------------------------------------------

markdown_table <- function(data, digits = 3L) {
  formatted <- data %>%
    mutate(across(where(is.numeric), ~ ifelse(is.na(.x), "unavailable",
      formatC(.x, format = "fg", digits = digits))))
  c(
    paste("|", paste(names(formatted), collapse = " | "), "|"),
    paste("|", paste(rep("---", ncol(formatted)), collapse = " | "), "|"),
    apply(formatted, 1, function(row) paste("|", paste(row, collapse = " | "), "|"))
  )
}
primary <- function(data, outcome) {
  data %>% filter(
    outcome_id == outcome,
    spec_id == specs$spec_id[specs$role == "primary"],
    treatment_label == treatment_variants$label[treatment_variants$role == "primary"],
    calibration == "primary"
  )
}
outcome_units <- c(
  certifications_fiscal_year = paste(
    "certified positions per fixed-2011 farm worker per nominal $1/hour",
    "(0.01 equals 10 per 1,000); fiscal year of decision"
  ),
  offered_wage_start_year = paste(
    "dollars of hours-weighted hourly offered wage per statutory dollar",
    "(pass-through); calendar year of contract start, counties with offers in",
    "every calibration year"
  )
)
outcome_sections <- unlist(lapply(outcome_variants$outcome_id, function(outcome) {
  threshold <- precision_for(outcome)$threshold
  c(
    paste0("## Outcome `", outcome, "`"),
    "",
    paste0("Units: ", outcome_units[[outcome]], ". Declared primary threshold: ",
      threshold, "."),
    "",
    "### Treatment support",
    "",
    markdown_table(support_table %>%
      filter(outcome_id == outcome, calibration == "primary") %>%
      select(spec_id, treatment_label, scenario_id, counties, pairs, states,
        median_post_pair_gap, effective_pair_leverage_count,
        maximum_pair_leverage_share, signed_incidence_rank, pre_mean_outcome,
        pre_sd_outcome)),
    "",
    "### Primary specification: power at the declared threshold",
    "",
    markdown_table(primary(power_table, outcome) %>%
      filter(abs(abs(effect) - threshold) < 1e-12) %>%
      select(scenario_id, dgp_id, inference_method, effect, rejection_rate, size_gate_pass)),
    "",
    "### Primary specification: size and minimum detectable effects",
    "",
    markdown_table(primary(mde_table, outcome) %>%
      select(scenario_id, dgp_id, inference_method, effect_sign, grid_mde, size_valid_mde,
        null_rejection_rate, null_mc_confidence_high)),
    "",
    "### Declared precision standard",
    "",
    markdown_table(decision %>% filter(outcome_id == outcome) %>% select(-outcome_id)),
    "",
    paste0("Conditional precision passes for ",
      sum(decision$conditional_precision_pass[decision$outcome_id == outcome]), " of ",
      sum(decision$outcome_id == outcome), " specification/inference combinations."),
    ""
  )
}))

report <- c(
  "# Statewide-AEWR border design: declared power",
  "",
  paste0(
    "Generated by `13_statewide_power.R`; registry `", registry$design_id,
    "` (frozen ", registry$registry_frozen, "); seed ", settings$seed, "; ",
    settings$repetitions, " repetitions per DGP."
  ),
  "",
  "## What this measures",
  "",
  paste(
    "The 2025-10-02 rule replaced regional FLS AEWRs with statewide rates. Within a",
    "former FLS region, adjacent states had identical AEWRs before 2026 and",
    "different ones afterwards. The simulation fixes the actual state-border",
    "geography and the actual 2026/2027 rates, injects signed effects, and",
    "draws errors from the v6 county/state/pair/spatial processes, scaled by",
    "pre-period outcome residuals after county and pair-year fixed effects. No",
    "post-policy outcome is read; this is design-stage power, not post-hoc power.",
    "A combination passes when every shared-dependence DGP (state, state-plus-pair,",
    "spatial) passes the size gate, power at both signs of the threshold is at",
    "least 0.80, and the largest size-valid MDE is at most the threshold."
  ),
  "",
  outcome_sections,
  "## Limits",
  "",
  paste(
    "Simulated size and power are conditional on assumed error processes; they do",
    "not establish that state AEWR gaps are exogenous. The rule also changed rates",
    "nationally, added a state housing adjustment and coincided with other 2025",
    "policy changes; state-specific concurrent shocks would bias the design and",
    "are not represented here. Skill-level assignment is an employer choice, so",
    "the primary treatment is an intent-to-treat statutory floor. FY2026 includes",
    "cases decided under prior rates. The wage outcome requires post-period offers",
    "in the same counties. Estimation requires the upstream producers to parse the",
    "2025 and later disclosure files."
  )
)
report_path <- here::here("outputs", "reports", "border_statewide_power.md")
iv_dir(report_path)
writeLines(report, report_path)
print(decision, n = Inf, width = Inf)
