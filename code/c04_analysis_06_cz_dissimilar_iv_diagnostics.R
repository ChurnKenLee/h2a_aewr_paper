rm(list = ls())

if (file.exists("paths.R")) {
  source("paths.R")
} else {
  source(file.path("code", "paths.R"))
}

ensure_project_dirs()

library(arrow)
library(tidyverse)
library(fixest)

## Diagnostics for leave-agro-cluster-out donor wage instruments ---------------

primary_iv <- "dissimilar_donor_oews_ag_wage_ppi_l1"
primary_iv_post <- "dissimilar_donor_oews_ag_wage_ppi_l1_x_post"
placebo_iv <- "dissimilar_donor_oews_ag_wage_ppi_l1_f1"

baseline_years <- 2008:2011
post_start_year <- 2012
weather_normal_years <- 2000:2007

write_outputs <- !tolower(Sys.getenv("CZD_WRITE_OUTPUTS", "true")) %in%
  c("0", "false", "no")

wtd_mean <- function(x, w) {
  keep <- !is.na(x) & is.finite(x) & !is.na(w) & is.finite(w) & w > 0

  if (!any(keep)) {
    return(NA_real_)
  }

  weighted.mean(x[keep], w[keep])
}

safe_log1p <- function(x) {
  if_else(!is.na(x) & x >= 0, log1p(x), NA_real_)
}

std_var <- function(x) {
  x_sd <- sd(x, na.rm = TRUE)

  if (!is.finite(x_sd) || x_sd == 0) {
    return(rep(NA_real_, length(x)))
  }

  (x - mean(x, na.rm = TRUE)) / x_sd
}

safe_first <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) {
    return(NA)
  }
  x[[1]]
}

tidy_fixest_iv <- function(model, term, diagnostic, spec, outcome, error = NA_character_) {
  if (is.null(model)) {
    return(tibble(
      diagnostic = diagnostic,
      specification = spec,
      outcome = outcome,
      term = term,
      estimate = NA_real_,
      std_error = NA_real_,
      t_stat = NA_real_,
      f_stat = NA_real_,
      p_value = NA_real_,
      nobs = NA_integer_,
      r2_within = NA_real_,
      error = error
    ))
  }

  ct <- coeftable(model)

  if (!term %in% rownames(ct)) {
    return(tibble(
      diagnostic = diagnostic,
      specification = spec,
      outcome = outcome,
      term = term,
      estimate = NA_real_,
      std_error = NA_real_,
      t_stat = NA_real_,
      f_stat = NA_real_,
      p_value = NA_real_,
      nobs = nobs(model),
      r2_within = fitstat(model, "wr2")[[1]],
      error = NA_character_
    ))
  }

  t_stat <- unname(ct[term, "t value"])

  tibble(
    diagnostic = diagnostic,
    specification = spec,
    outcome = outcome,
    term = term,
    estimate = unname(ct[term, "Estimate"]),
    std_error = unname(ct[term, "Std. Error"]),
    t_stat = t_stat,
    f_stat = t_stat^2,
    p_value = unname(ct[term, "Pr(>|t|)"]),
    nobs = nobs(model),
    r2_within = fitstat(model, "wr2")[[1]],
    error = NA_character_
  )
}

estimate_diag <- function(data, outcome, iv, spec_name, fe_rhs, diagnostic) {
  model_data <- data %>%
    transmute(
      outcome_z = std_var(.data[[outcome]]),
      iv_z = std_var(.data[[iv]]),
      cz_out10,
      year,
      aewr_region_num,
      aewr_region_year_fe,
      cz_aewr_region_fe
    ) %>%
    filter(!is.na(outcome_z), !is.na(iv_z))

  if (nrow(model_data) < 100 || n_distinct(model_data$cz_out10) < 10) {
    return(tibble(
      diagnostic = diagnostic,
      specification = spec_name,
      outcome = outcome,
      term = "iv_z",
      estimate = NA_real_,
      std_error = NA_real_,
      t_stat = NA_real_,
      f_stat = NA_real_,
      p_value = NA_real_,
      nobs = nrow(model_data),
      r2_within = NA_real_,
      error = "Insufficient complete observations"
    ))
  }

  fit <- tryCatch(
    feols(
      as.formula(str_c("outcome_z ~ iv_z | ", fe_rhs)),
      data = model_data,
      vcov = ~cz_out10,
      notes = FALSE
    ),
    error = function(e) e
  )

  if (inherits(fit, "error")) {
    return(tidy_fixest_iv(
      NULL,
      "iv_z",
      diagnostic,
      spec_name,
      outcome,
      error = conditionMessage(fit)
    ) %>%
      mutate(nobs = nrow(model_data)))
  }

  tidy_fixest_iv(fit, "iv_z", diagnostic, spec_name, outcome)
}

climate_file_index <- function(raw_dir, years) {
  files <- list.files(raw_dir, pattern = "^[0-9]{6}[.]parquet$", full.names = TRUE)

  tibble(path = files) %>%
    mutate(
      ym = str_remove(basename(path), "[.]parquet$"),
      year = as.integer(str_sub(ym, 1, 4)),
      month = as.integer(str_sub(ym, 5, 6))
    ) %>%
    filter(year %in% years) %>%
    arrange(year, month)
}

read_climate_month_summary <- function(path, year, month) {
  growing_month <- month %in% 4:9

  read_parquet(
    path,
    col_select = c("fips", "tmax", "tmin", "tavg", "prcp")
  ) %>%
    transmute(
      countyfips = as.character(fips),
      year = year,
      tmax = suppressWarnings(as.numeric(tmax)),
      tmin = suppressWarnings(as.numeric(tmin)),
      tavg = suppressWarnings(as.numeric(tavg)),
      prcp = suppressWarnings(as.numeric(prcp)),
      growing = growing_month
    ) %>%
    group_by(countyfips, year) %>%
    summarise(
      n_days = sum(!is.na(tavg)),
      tavg_sum = sum(tavg, na.rm = TRUE),
      tavg_sq_sum = sum(tavg^2, na.rm = TRUE),
      growing_n_days = sum(growing & !is.na(tavg)),
      growing_tavg_sum = sum(if_else(growing, tavg, NA_real_), na.rm = TRUE),
      gdd10 = sum(pmax(tavg - 10, 0), na.rm = TRUE),
      growing_gdd10 = sum(if_else(growing, pmax(tavg - 10, 0), NA_real_), na.rm = TRUE),
      heat_dd29 = sum(pmax(tavg - 29, 0), na.rm = TRUE),
      heat_days32 = sum(tmax >= 32.2, na.rm = TRUE),
      frost_days = sum(tmin < 0, na.rm = TRUE),
      prcp_sum = sum(pmax(prcp, 0), na.rm = TRUE),
      growing_prcp_sum = sum(if_else(growing, pmax(prcp, 0), NA_real_), na.rm = TRUE),
      wet_days = sum(prcp >= 1, na.rm = TRUE),
      growing_wet_days = sum(growing & prcp >= 1, na.rm = TRUE),
      .groups = "drop"
    )
}

build_county_weather_annual <- function(raw_dir, years) {
  file_index <- climate_file_index(raw_dir, years)

  if (nrow(file_index) == 0) {
    stop(
      "No raw EpiNOAA monthly parquets found for weather diagnostics in ",
      raw_dir,
      ". Run code/a09_01_h2a_prediction_pull_noaa.py first.",
      call. = FALSE
    )
  }

  month_summaries <- pmap_dfr(
    file_index,
    function(path, ym, year, month) {
      read_climate_month_summary(path, year, month)
    }
  )

  month_summaries %>%
    group_by(countyfips, year) %>%
    summarise(
      total_days = sum(n_days, na.rm = TRUE),
      growing_days = sum(growing_n_days, na.rm = TRUE),
      weather_tavg_mean = sum(tavg_sum, na.rm = TRUE) / total_days,
      weather_tavg_growing_mean =
        sum(growing_tavg_sum, na.rm = TRUE) / growing_days,
      weather_tavg_sd = sqrt(pmax(
        sum(tavg_sq_sum, na.rm = TRUE) / total_days - weather_tavg_mean^2,
        0
      )),
      weather_gdd10 = sum(gdd10, na.rm = TRUE),
      weather_growing_gdd10 = sum(growing_gdd10, na.rm = TRUE),
      weather_heat_dd29 = sum(heat_dd29, na.rm = TRUE),
      weather_heat_days32 = sum(heat_days32, na.rm = TRUE),
      weather_frost_days = sum(frost_days, na.rm = TRUE),
      weather_prcp = sum(prcp_sum, na.rm = TRUE),
      weather_growing_prcp = sum(growing_prcp_sum, na.rm = TRUE),
      weather_wet_days = sum(wet_days, na.rm = TRUE),
      weather_growing_wet_days = sum(growing_wet_days, na.rm = TRUE),
      .groups = "drop"
    )
}

## Target CZ-year panel --------------------------------------------------------

county_panel <- read_parquet(path_processed("county_df_analysis_year.parquet")) %>%
  mutate(
    countyfips = as.character(countyfips),
    cz_out10 = as.character(cz_out10),
    aewr_region_num = as.integer(aewr_region_num),
    year = as.integer(year)
  ) %>%
  distinct(countyfips, year, cz_out10, aewr_region_num, .keep_all = TRUE) %>%
  filter(!is.na(cz_out10), !is.na(aewr_region_num))

cz_year_panel <- county_panel %>%
  mutate(
    cropland_weight = if_else(
      !is.na(cropland_acr_2007) & cropland_acr_2007 > 0,
      cropland_acr_2007,
      1
    )
  ) %>%
  group_by(aewr_region_num, cz_out10, year) %>%
  summarise(
    cz_aewr_region_fe = first(cz_aewr_region_fe),
    aewr_ppi = first(aewr_ppi),
    aewr_ppi_l1 = first(aewr_ppi_l1),
    ln_aewr_ppi = first(ln_aewr_ppi),
    ln_aewr_ppi_l1 = first(ln_aewr_ppi_l1),
    aewr_cz_p25_l1 = wtd_mean(aewr_cz_p25_l1, cropland_weight),
    aewr_cz_p50_l1 = wtd_mean(aewr_cz_p50_l1, cropland_weight),
    wage_p10 = wtd_mean(wage_p10, cropland_weight),
    wage_p25 = wtd_mean(wage_p25, cropland_weight),
    wage_p50 = wtd_mean(wage_p50, cropland_weight),
    fisher_index_ppi = wtd_mean(fisher_index_ppi, cropland_weight),
    emp_tot = sum(emp_tot, na.rm = TRUE),
    emp_farm = sum(emp_farm, na.rm = TRUE),
    emp_nonfarm = sum(emp_nonfarm, na.rm = TRUE),
    emp_privatenonfarm = sum(emp_privatenonfarm, na.rm = TRUE),
    pop_census = sum(pop_census, na.rm = TRUE),
    farm_cashcrops_ppi = sum(farm_cashcrops_ppi, na.rm = TRUE),
    farm_cashanimal_ppi = sum(farm_cashanimal_ppi, na.rm = TRUE),
    farm_prodexp_ppi = sum(farm_prodexp_ppi, na.rm = TRUE),
    farm_laborexpense_ppi = sum(farm_laborexpense_ppi, na.rm = TRUE),
    farm_cashandinc_ppi = sum(farm_cashandinc_ppi, na.rm = TRUE),
    nbr_workers_certified_start_year =
      sum(nbr_workers_certified_start_year, na.rm = TRUE),
    nbr_workers_requested_start_year =
      sum(nbr_workers_requested_start_year, na.rm = TRUE),
    man_hours_certified_start_year =
      sum(man_hours_certified_start_year, na.rm = TRUE),
    nbr_applications_start_year =
      sum(nbr_applications_start_year, na.rm = TRUE),
    h2a_cert_share_farm_workers_2011_start_year =
      wtd_mean(h2a_cert_share_farm_workers_2011_start_year, cropland_weight),
    h2a_req_share_farm_workers_2011_start_year =
      wtd_mean(h2a_req_share_farm_workers_2011_start_year, cropland_weight),
    h2a_predicted_share_2011 =
      wtd_mean(h2a_predicted_share_2011, cropland_weight),
    .groups = "drop"
  ) %>%
  mutate(
    postdummy = as.integer(year >= post_start_year),
    aewr_cz_p25_l1_x_post = aewr_cz_p25_l1 * postdummy,
    aewr_cz_p50_l1_x_post = aewr_cz_p50_l1 * postdummy,
    ln_emp_farm = safe_log1p(emp_farm),
    ln_emp_nonfarm = safe_log1p(emp_nonfarm),
    ln_emp_privatenonfarm = safe_log1p(emp_privatenonfarm),
    ln_pop_census = safe_log1p(pop_census),
    emp_pop_ratio = if_else(pop_census > 0, emp_tot / pop_census, NA_real_),
    ln_farm_cashcrops_ppi = safe_log1p(farm_cashcrops_ppi),
    ln_farm_cashanimal_ppi = safe_log1p(farm_cashanimal_ppi),
    ln_farm_prodexp_ppi = safe_log1p(farm_prodexp_ppi),
    ln_farm_laborexpense_ppi = safe_log1p(farm_laborexpense_ppi),
    ln_farm_cashandinc_ppi = safe_log1p(farm_cashandinc_ppi),
    share_farm_laborexpense_prodexp = if_else(
      farm_prodexp_ppi > 0,
      farm_laborexpense_ppi / farm_prodexp_ppi,
      NA_real_
    ),
    share_farm_crop_cashandinc = if_else(
      farm_cashandinc_ppi > 0,
      farm_cashcrops_ppi / farm_cashandinc_ppi,
      NA_real_
    ),
    ln1_h2a_cert_workers = safe_log1p(nbr_workers_certified_start_year),
    ln1_h2a_req_workers = safe_log1p(nbr_workers_requested_start_year),
    ln1_h2a_cert_hours = safe_log1p(man_hours_certified_start_year),
    ln1_h2a_applications = safe_log1p(nbr_applications_start_year),
    aewr_region_year_fe = str_c(aewr_region_num, "_", year)
  )

## Interpretable realized weather shocks --------------------------------------

weather_county_year <- build_county_weather_annual(
  path_raw("epinoaa_nclimgrid"),
  c(weather_normal_years, sort(unique(county_panel$year)))
)

weather_cols <- names(weather_county_year) %>%
  keep(~ str_starts(.x, "weather_"))

weather_normals <- weather_county_year %>%
  filter(year %in% weather_normal_years) %>%
  group_by(countyfips) %>%
  summarise(
    across(all_of(weather_cols), ~ mean(.x, na.rm = TRUE), .names = "normal_{.col}"),
    .groups = "drop"
  )

weather_shock_county <- weather_county_year %>%
  filter(year %in% unique(county_panel$year)) %>%
  left_join(weather_normals, by = "countyfips")

for (weather_col in weather_cols) {
  weather_shock_county[[paste0(weather_col, "_shock")]] <-
    weather_shock_county[[weather_col]] -
    weather_shock_county[[paste0("normal_", weather_col)]]
}

weather_shock_cols <- paste0(weather_cols, "_shock")

weather_cz_year <- weather_shock_county %>%
  select(countyfips, year, all_of(weather_shock_cols)) %>%
  inner_join(
    county_panel %>%
      transmute(
        countyfips,
        year,
        aewr_region_num,
        cz_out10,
        weather_weight = if_else(
          !is.na(cropland_acr_2007) & cropland_acr_2007 > 0,
          cropland_acr_2007,
          1
        )
      ),
    by = c("countyfips", "year")
  ) %>%
  group_by(aewr_region_num, cz_out10, year) %>%
  summarise(
    across(all_of(weather_shock_cols), ~ wtd_mean(.x, weather_weight)),
    .groups = "drop"
  )

## Instrument panel and diagnostics -------------------------------------------

iv_panel <- read_parquet(
  path_processed("cz_dissimilar_oews_instruments.parquet")
) %>%
  {
    if (!"donor_variant" %in% names(.)) {
      mutate(., donor_variant = "cluster_baseline")
    } else {
      .
    }
  } %>%
  {
    if (!"selection_rule" %in% names(.)) {
      mutate(., selection_rule = "leave_cluster_out_ward")
    } else {
      .
    }
  } %>%
  mutate(
    donor_variant = as.character(donor_variant),
    selection_rule = as.character(selection_rule),
    cz_out10 = as.character(cz_out10),
    aewr_region_num = as.integer(aewr_region_num),
    year = as.integer(year)
  ) %>%
  arrange(donor_variant, aewr_region_num, cz_out10, year) %>%
  group_by(donor_variant, aewr_region_num, cz_out10) %>%
  mutate(
    dissimilar_donor_oews_ag_wage_ppi_l1_f1 =
      lead(dissimilar_donor_oews_ag_wage_ppi_l1),
    dissimilar_donor_oews_ag_wage_ppi_l1_x_post =
      dissimilar_donor_oews_ag_wage_ppi_l1 * as.integer(year >= post_start_year)
  ) %>%
  ungroup()

donor_quality_summary <- iv_panel %>%
  group_by(donor_variant, selection_rule) %>%
  summarise(
    n_cz_years = n(),
    mean_selected_donors = mean(n_selected_donors, na.rm = TRUE),
    median_selected_donors = median(n_selected_donors, na.rm = TRUE),
    mean_effective_donors = mean(effective_n_donors, na.rm = TRUE),
    median_effective_donors = median(effective_n_donors, na.rm = TRUE),
    mean_valid_real_donor_weight = mean(valid_real_donor_weight, na.rm = TRUE),
    p10_valid_real_donor_weight =
      quantile(valid_real_donor_weight, 0.10, na.rm = TRUE),
    mean_donor_cluster_count = mean(donor_cluster_count, na.rm = TRUE),
    median_donor_cluster_count = median(donor_cluster_count, na.rm = TRUE),
    .groups = "drop"
  )

diagnostic_panel <- cz_year_panel %>%
  left_join(weather_cz_year, by = c("aewr_region_num", "cz_out10", "year")) %>%
  left_join(iv_panel, by = c("aewr_region_num", "cz_out10", "year")) %>%
  arrange(donor_variant, aewr_region_num, cz_out10, year) %>%
  group_by(donor_variant, aewr_region_num, cz_out10) %>%
  mutate(
    across(
      c(
        wage_p10,
        wage_p25,
        wage_p50,
        fisher_index_ppi,
        ln_emp_farm,
        ln_emp_nonfarm,
        ln_emp_privatenonfarm,
        emp_pop_ratio,
        ln_farm_cashcrops_ppi,
        ln_farm_cashanimal_ppi,
        ln_farm_prodexp_ppi,
        ln_farm_laborexpense_ppi,
        ln_farm_cashandinc_ppi,
        share_farm_laborexpense_prodexp,
        share_farm_crop_cashandinc,
        h2a_cert_share_farm_workers_2011_start_year,
        h2a_req_share_farm_workers_2011_start_year,
        ln1_h2a_cert_workers,
        ln1_h2a_req_workers,
        ln1_h2a_cert_hours,
        ln1_h2a_applications
      ),
      list(d1 = ~ .x - lag(.x)),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  ungroup()

if (write_outputs) {
  write_parquet(
    diagnostic_panel,
    path_processed("cz_dissimilar_iv_diagnostic_panel.parquet")
  )
}

first_stage_outcomes <- tibble::tribble(
  ~diagnostic, ~outcome, ~description,
  "First stage", "aewr_cz_p25_l1", "Lagged AEWR minus broad ACS p25 wage",
  "First stage", "aewr_cz_p25_l1_x_post", "Post interaction: lagged AEWR minus broad ACS p25 wage",
  "First stage", "aewr_cz_p50_l1", "Lagged AEWR minus broad ACS p50 wage",
  "First stage", "aewr_cz_p50_l1_x_post", "Post interaction: lagged AEWR minus broad ACS p50 wage"
)

shock_balance_outcomes <- tibble::tribble(
  ~diagnostic, ~outcome, ~description,
  "Weather shocks", "weather_tavg_mean_shock", "Target annual mean temperature shock",
  "Weather shocks", "weather_tavg_growing_mean_shock", "Target growing-season mean temperature shock",
  "Weather shocks", "weather_gdd10_shock", "Target annual GDD10 shock",
  "Weather shocks", "weather_growing_gdd10_shock", "Target growing-season GDD10 shock",
  "Weather shocks", "weather_heat_dd29_shock", "Target extreme heat degree-day shock",
  "Weather shocks", "weather_heat_days32_shock", "Target extreme heat days shock",
  "Weather shocks", "weather_frost_days_shock", "Target frost days shock",
  "Weather shocks", "weather_prcp_shock", "Target annual precipitation shock",
  "Weather shocks", "weather_growing_prcp_shock", "Target growing-season precipitation shock",
  "Weather shocks", "weather_wet_days_shock", "Target annual wet-day shock",
  "Weather shocks", "weather_growing_wet_days_shock", "Target growing-season wet-day shock",
  "Crop-price shocks", "fisher_index_ppi_d1", "Change in target crop-price index"
)

nonag_balance_outcomes <- tibble::tribble(
  ~diagnostic, ~outcome, ~description,
  "Non-ag labor-market balance", "wage_p10_d1", "Change in target broad ACS p10 wage",
  "Non-ag labor-market balance", "wage_p25_d1", "Change in target broad ACS p25 wage",
  "Non-ag labor-market balance", "wage_p50_d1", "Change in target broad ACS p50 wage",
  "Non-ag labor-market balance", "ln_emp_nonfarm_d1", "Change in target nonfarm employment",
  "Non-ag labor-market balance", "ln_emp_privatenonfarm_d1", "Change in target private nonfarm employment",
  "Non-ag labor-market balance", "emp_pop_ratio_d1", "Change in target employment-population ratio"
)

ag_balance_outcomes <- tibble::tribble(
  ~diagnostic, ~outcome, ~description,
  "Agricultural-demand balance", "ln_emp_farm_d1", "Change in target farm employment",
  "Agricultural-demand balance", "ln_farm_cashcrops_ppi_d1", "Change in target crop cash receipts",
  "Agricultural-demand balance", "ln_farm_cashanimal_ppi_d1", "Change in target animal cash receipts",
  "Agricultural-demand balance", "ln_farm_prodexp_ppi_d1", "Change in target farm production expenses",
  "Agricultural-demand balance", "ln_farm_laborexpense_ppi_d1", "Change in target farm labor expenses",
  "Agricultural-demand balance", "ln_farm_cashandinc_ppi_d1", "Change in target farm cash receipts and income",
  "Agricultural-demand balance", "share_farm_laborexpense_prodexp_d1", "Change in target farm labor expense share",
  "Agricultural-demand balance", "share_farm_crop_cashandinc_d1", "Change in target crop share of farm income"
)

h2a_pretrend_outcomes <- tibble::tribble(
  ~diagnostic, ~outcome, ~description,
  "H-2A outcome pretrend", "h2a_cert_share_farm_workers_2011_start_year_d1", "Change in target certified H-2A share",
  "H-2A outcome pretrend", "h2a_req_share_farm_workers_2011_start_year_d1", "Change in target requested H-2A share",
  "H-2A outcome pretrend", "ln1_h2a_cert_workers_d1", "Change in log certified H-2A workers",
  "H-2A outcome pretrend", "ln1_h2a_req_workers_d1", "Change in log requested H-2A workers",
  "H-2A outcome pretrend", "ln1_h2a_cert_hours_d1", "Change in log certified H-2A hours",
  "H-2A outcome pretrend", "ln1_h2a_applications_d1", "Change in log H-2A applications"
)

specs <- tibble::tribble(
  ~specification, ~fe_rhs,
  "CZ FE + year FE", "cz_out10 + year",
  "CZ FE + AEWR-region-year FE", "cz_out10 + aewr_region_year_fe"
)

diagnostic_variants <- iv_panel %>%
  distinct(donor_variant, selection_rule) %>%
  arrange(donor_variant)

run_diag_grid <- function(outcomes, data, iv, suffix, specs_tbl = specs) {
  crossing(diagnostic_variants, outcomes, specs_tbl) %>%
    pmap_dfr(function(
        donor_variant,
        selection_rule,
        diagnostic,
        outcome,
        description,
        specification,
        fe_rhs) {
      estimate_diag(
        data = data %>% filter(.data$donor_variant == !!donor_variant),
        outcome = outcome,
        iv = iv,
        spec_name = specification,
        fe_rhs = fe_rhs,
        diagnostic = str_c(diagnostic, suffix)
      ) %>%
        mutate(
          donor_variant = donor_variant,
          selection_rule = selection_rule,
          description = description,
          iv = iv,
          .before = 1
        )
    })
}

first_stage_level_results <- run_diag_grid(
  first_stage_outcomes %>% filter(!str_detect(outcome, "_x_post$")),
  diagnostic_panel,
  primary_iv,
  ""
)

first_stage_post_results <- run_diag_grid(
  first_stage_outcomes %>% filter(str_detect(outcome, "_x_post$")),
  diagnostic_panel,
  primary_iv_post,
  ""
)

primary_balance_results <- run_diag_grid(
  bind_rows(shock_balance_outcomes, nonag_balance_outcomes, ag_balance_outcomes),
  diagnostic_panel,
  primary_iv,
  ""
)

placebo_results <- run_diag_grid(
  bind_rows(
    shock_balance_outcomes,
    nonag_balance_outcomes,
    ag_balance_outcomes,
    h2a_pretrend_outcomes
  ),
  diagnostic_panel,
  placebo_iv,
  " placebo"
)

pre_period_results <- run_diag_grid(
  bind_rows(
    shock_balance_outcomes,
    nonag_balance_outcomes,
    ag_balance_outcomes,
    h2a_pretrend_outcomes
  ),
  diagnostic_panel %>% filter(year %in% baseline_years),
  primary_iv,
  " pre-period",
  specs_tbl = specs %>% filter(specification == "CZ FE + year FE")
)

diagnostic_results <- bind_rows(
  first_stage_level_results,
  first_stage_post_results,
  primary_balance_results,
  placebo_results,
  pre_period_results
) %>%
  mutate(
    test_family = case_when(
      diagnostic == "First stage" ~ "first_stage",
      str_detect(diagnostic, " placebo$") ~ "lead_placebo",
      str_detect(diagnostic, " pre-period$") ~ "pre_period",
      TRUE ~ "primary_balance"
    )
  ) %>%
  group_by(donor_variant, test_family, specification) %>%
  mutate(p_value_fdr = p.adjust(p_value, method = "BH")) %>%
  ungroup() %>%
  relocate(description, .after = outcome) %>%
  relocate(test_family, .before = diagnostic) %>%
  arrange(donor_variant, test_family, diagnostic, specification, outcome)

if (write_outputs) {
  write_csv(
    diagnostic_results,
    path_tables("cz_dissimilar_iv_exclusion_diagnostics.csv")
  )
  write_csv(
    donor_quality_summary,
    path_tables("cz_dissimilar_iv_donor_quality_summary.csv")
  )
}

diagnostic_table <- diagnostic_results %>%
  mutate(
    estimate_se = str_c(
      sprintf("%.3f", estimate),
      " (",
      sprintf("%.3f", std_error),
      ")"
    ),
    p_value = sprintf("%.3f", p_value),
    p_value_fdr = sprintf("%.3f", p_value_fdr),
    f_stat = sprintf("%.2f", f_stat)
  ) %>%
  select(
    donor_variant,
    selection_rule,
    test_family,
    diagnostic,
    specification,
    description,
    estimate_se,
    f_stat,
    p_value,
    p_value_fdr,
    nobs
  )

if (write_outputs) {
  write_csv(
    diagnostic_table,
    path_tables("table_cz_dissimilar_iv_exclusion_diagnostics.csv")
  )
}

tex_lines <- c(
  "\\begin{tabular}{llllllrrrr}",
  "\\toprule",
  "Variant & Rule & Family & Diagnostic & Specification & Outcome & Estimate (SE) & F/t$^2$ & p-value & FDR p-value \\\\",
  "\\midrule",
  diagnostic_table %>%
    mutate(
      line = str_c(
        donor_variant,
        " & ",
        selection_rule,
        " & ",
        test_family,
        " & ",
        diagnostic,
        " & ",
        specification,
        " & ",
        description,
        " & ",
        estimate_se,
        " & ",
        f_stat,
        " & ",
        p_value,
        " & ",
        p_value_fdr,
        " \\\\"
      )
    ) %>%
    pull(line),
  "\\bottomrule",
  "\\end{tabular}"
)

if (write_outputs) {
  write_lines(
    tex_lines,
    path_tables("table_cz_dissimilar_iv_exclusion_diagnostics.tex")
  )
}

red_flags <- diagnostic_results %>%
  filter(
    test_family != "first_stage",
    !is.na(p_value),
    p_value < 0.05
  ) %>%
  arrange(p_value) %>%
  select(
    donor_variant,
    selection_rule,
    test_family,
    diagnostic,
    specification,
    outcome,
    estimate,
    std_error,
    p_value,
    p_value_fdr,
    nobs
  )

if (write_outputs) {
  cat("\nAlternative donor IV diagnostics written to:\n")
  cat("  ", path_processed("cz_dissimilar_iv_diagnostic_panel.parquet"), "\n")
  cat("  ", path_tables("cz_dissimilar_iv_exclusion_diagnostics.csv"), "\n")
  cat("  ", path_tables("cz_dissimilar_iv_donor_quality_summary.csv"), "\n")
  cat("  ", path_tables("table_cz_dissimilar_iv_exclusion_diagnostics.tex"), "\n")
} else {
  cat("\nCZD_WRITE_OUTPUTS=false; diagnostics were run without writing files.\n")
}

cat("\nDonor quality summary:\n")
print(donor_quality_summary)

cat("\nPrimary first-stage diagnostics:\n")
print(
  diagnostic_results %>%
    filter(test_family == "first_stage") %>%
    select(
      donor_variant,
      selection_rule,
      specification,
      outcome,
      iv,
      estimate,
      std_error,
      f_stat,
      p_value,
      nobs
    )
)

cat("\nExclusion-diagnostic coefficients with p < 0.05:\n")
print(red_flags, n = Inf)
