# Phase 0 §4a: put earlier headline estimates on one scale.
# Refits the plain panel, panel IV and bite 1(a)/1(b) models exactly as in
# explorations/bite/cross_design_panel.R (same samples, FE, controls) so the
# estimation-sample mean price is available, then adds converted rows for the
# border, statewide-border power and employer design 4 from their retained
# scratch or report outputs. Elasticity = beta * mean(price) / mean(outcome).
# Argument 1: scratch output dir. Argument 2 (optional): the bite scratch dir
# holding exposures.parquet, average_effect_conversion.csv, border_estimates.csv.
source("explorations/identification/id_helpers.R")
out_dir <- id_output_dir()
bite_dir <- commandArgs(trailingOnly = TRUE)[2]
if (is.na(bite_dir)) bite_dir <- "/tmp/h2a-bite-cross-design"
ex <- read_parquet(file.path(bite_dir, "exposures.parquet"))
p <- panel_iv_prepare_outcomes(read_parquet(path_processed("panel_iv_county_year.parquet")), bite_spec)
bite_check_keys(p)
p <- inner_join(p, ex %>% select(county_fips, E_gap), by = "county_fips", relationship = "many-to-one")

rows <- wild <- list()
for (cs in c("bea", "continuing")) {
  ctl <- panel_iv_control_set(bite_spec, cs); controls <- paste(ctl$control_terms, collapse = " + ")
  for (nm in c("cert", "hours")) {
    y <- bite_outcomes[[nm]]
    d <- p %>% filter(year <= ctl$policy_end_year) %>%
      mutate(A = aewr_ppi, Z = z_wage_seasonal_composition_real, E = E_gap, X = A * E)
    d <- bite_complete(d, c(y, "A", "Z", "z_wage_only_real", "E", ctl$control_columns, "aewr_iv_cluster_id"))
    menu <- list(
      baseline_ols = list(rhs = paste("A +", controls), fe = "county_fips + year", term = "A"),
      baseline_iv = list(rhs = controls, fe = "county_fips + year", iv = "A ~ Z", term = "fit_A"),
      bite_1b = list(rhs = "X", fe = "county_fips + aewr_region_id^year", term = "X"),
      bite_1a = list(rhs = "X", fe = "county_fips + aewr_region_id^year + year[E]", term = "X"))
    for (id in names(menu)) {
      cfg <- menu[[id]]
      m <- bite_fit(y, cfg$rhs, cfg$fe, d, cfg$iv)
      meta <- tibble(design = paste0("panel_", id), control_set = cs, outcome = nm)
      rows[[paste(cs, nm, id)]] <- id_rows(m, d, cfg$term, meta,
        c("aewr_region_id", "state_fips", "aewr_iv_cluster_id"), "A")
      if (cs == "bea") wild[[paste(cs, nm, id)]] <- bind_cols(meta, id_wild(m, d, cfg$term, "aewr_region_id"))
    }
  }
}
panel <- bind_rows(rows) %>% left_join(bind_rows(wild) %>% select(design, control_set, outcome, wild_p),
  by = c("design", "control_set", "outcome"))
panel <- panel %>% mutate(estimand = case_when(
  grepl("bite", design) ~ "differential slope per 1 SD frozen gap (heterogeneity, not average)",
  grepl("iv", design) ~ "average effect, IV (LATE on regional AEWR deviations)",
  TRUE ~ "average effect, two-way FE OLS"))

# Conditional average-effect conversion of 1(b) (bite report §6 restriction).
avg <- read_csv(file.path(bite_dir, "average_effect_conversion.csv"), show_col_types = FALSE) %>%
  filter(outcome %in% c("cert", "hours"))
price_1b <- panel %>% filter(design == "panel_bite_1b", cluster == "aewr_region_id") %>%
  select(control_set, outcome, mean_price, mean_y_panel = mean_y)
avg <- avg %>% inner_join(price_1b, by = c("control_set", "outcome")) %>%
  transmute(design = "panel_bite_1b_average_conversion", control_set, outcome, term = "beta_bar",
    cluster = "aewr_region_id", clusters, estimate = average_effect,
    se = (average_high - average_low) / (2 * qt(.975, clusters - 1)), n, mean_y, mean_price, price = "A",
    mde = average_mde, mde_pct_mean = 100 * average_mde / mean_y,
    elasticity = id_elasticity(average_effect, mean_price, mean_y),
    elasticity_low = id_elasticity(average_low, mean_price, mean_y),
    elasticity_high = id_elasticity(average_high, mean_price, mean_y),
    mde_elasticity = id_elasticity(average_mde, mean_price, mean_y), wild_p,
    estimand = "average effect under beta_c proportional to frozen gap (model-based)")

# Border v6/v7 rows: pair CRV1, mean real AEWR on the ring membership sample.
bp <- read_parquet(path_processed("border_discontinuity_panel.parquet"),
  col_select = c("county_fips", "year", "aewr_ppi", "stratum__ring_h1_disjoint", "stratum__ring_h1_screened_disjoint"))
border_price <- bind_rows(lapply(c("ring_h1_disjoint", "ring_h1_screened_disjoint"), function(g) {
  s <- bp[[paste0("stratum__", g)]]
  tibble(geography = g, mean_price = mean(bp$aewr_ppi[!is.na(s) & nzchar(s) & bp$year %in% 2011:2025], na.rm = TRUE))
}))
border <- read_csv(file.path(bite_dir, "border_estimates.csv"), show_col_types = FALSE) %>%
  filter(exposure == "E_gap", outcome %in% c("cert", "hours"), cluster == "region_pair_id",
    (model == "baseline" & term == "main_a") | (model == "heterogeneity" & term == "X") | model == "lead") %>%
  inner_join(border_price, by = "geography") %>%
  transmute(design = paste0("border_", geography, "_", model), control_set = "none", outcome, term, cluster,
    clusters, estimate, se, p, n, mean_y, mean_price, price = "aewr_ppi (ring sample mean)", mde, mde_pct_mean,
    elasticity = id_elasticity(estimate, mean_price, mean_y),
    elasticity_low = id_elasticity(low, mean_price, mean_y), elasticity_high = id_elasticity(high, mean_price, mean_y),
    mde_elasticity = id_elasticity(mde, mean_price, mean_y),
    estimand = ifelse(term == "X", "differential border slope per 1 SD pair exposure", "average border effect"))

# Statewide-border design-stage power (retained tables; nominal dollars).
sw_mde <- read_csv(path_tables("border_statewide_mde.csv"), show_col_types = FALSE) %>%
  filter(outcome_id == "certifications_fiscal_year", calibration == "primary",
    treatment_label == "us_skill_1_total_compensation_floor", dgp_id != "county_serial")
sw_support <- read_csv(path_tables("border_statewide_treatment_support.csv"), show_col_types = FALSE) %>%
  filter(outcome_id == "certifications_fiscal_year", calibration == "primary",
    treatment_label == "us_skill_1_total_compensation_floor")
sw_panel <- read_parquet(path_int("border_statewide_power_panel.parquet")) %>%
  filter(outcome_id == "certifications_fiscal_year", treatment_label == "us_skill_1_total_compensation_floor", year <= 2025)
sw_price <- sw_panel %>% group_by(spec_id) %>% summarise(mean_price = mean(border_x), .groups = "drop")
statewide <- sw_mde %>% group_by(spec_id, scenario_id) %>%
  summarise(mde = max(grid_mde, na.rm = TRUE), size_gate_all = all(size_gate_pass), .groups = "drop") %>%
  inner_join(sw_support %>% select(spec_id, scenario_id, mean_y = pre_mean_outcome), by = c("spec_id", "scenario_id")) %>%
  inner_join(sw_price, by = "spec_id") %>%
  transmute(design = paste0("statewide_border_power_", spec_id, "_", scenario_id), control_set = "none",
    outcome = "cert_fiscal_year", term = "border_x", cluster = "simulated (worst shared DGP)", mean_y, mean_price,
    price = "nominal AEWR 2020-2025 on membership", mde, mde_pct_mean = 100 * mde / mean_y,
    mde_elasticity = id_elasticity(mde, mean_price, mean_y), size_gate_all,
    estimand = "design-stage power only (no estimate)")

# Employer design 4 (bite report §4, 1(a)-analog 2011-2019, state CRV1).
# Mean real AEWR over the region-years of 2011-2019 (unweighted region mean).
ra <- read_parquet(path_processed("county_year_panel.parquet"), col_select = c("aewr_region_id", "year", "aewr_ppi")) %>%
  filter(year %in% 2011:2019, is.finite(aewr_ppi)) %>% distinct(aewr_region_id, year, aewr_ppi)
emp_price <- mean(ra$aewr_ppi)
employer <- tibble(design = "employer_design4_1a_2011_2019", control_set = "none",
  outcome = c("log1p_workers", "participation"), term = "bound x A", cluster = "state_fips",
  estimate = c(-0.130, -0.055), se = c(0.051, 0.016), mean_price = emp_price,
  mean_y = c(NA, 0.385), price = "real regional AEWR 2011-2019 region-year mean") %>%
  mutate(mde = 2.8 * se,
    elasticity = ifelse(outcome == "log1p_workers", estimate * mean_price, estimate * mean_price / mean_y),
    mde_elasticity = ifelse(outcome == "log1p_workers", mde * mean_price, mde * mean_price / mean_y),
    mde_pct_mean = ifelse(outcome == "log1p_workers", 100 * mde, 100 * mde / mean_y),
    estimand = "differential bound-vs-unbound effect; fails cohort replication",
    note = "inputs from markdowns/bite_designs_exploration.md §4; participation mean = midpoint of reported 0.27-0.50")

common <- bind_rows(panel, avg, border, statewide, employer) %>%
  mutate(gate_elasticity_mde_le_1 = mde_elasticity <= 1, gate_mde_lt_25pct = mde_pct_mean < 25)
id_write(common, out_dir, "common_unit_all.csv")
headline <- common %>% filter(cluster %in% c("aewr_region_id", "region_pair_id", "state_fips",
  "simulated (worst shared DGP)") & (control_set %in% c("bea", "none"))) %>%
  filter(!(grepl("^panel", design) & cluster != "aewr_region_id")) %>%
  select(design, outcome, term, cluster, estimate, se, wild_p, mean_price, mean_y, mde_pct_mean,
    elasticity, elasticity_low, elasticity_high, mde_elasticity, estimand)
id_write(headline, out_dir, "common_unit_headline.csv")
print(as.data.frame(headline), digits = 3)
