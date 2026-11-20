# Phase 0 §4b: more signal per region-year? Treatment (real regional AEWR)
# and identifying variation (county + year FE; region-year AEWR deviations)
# stay fixed; only the outcome, weighting, aggregation and estimator change.
# Samples: the panel-IV BEA-control sample (2011-2023) for county specs.
source("explorations/identification/id_helpers.R")
out_dir <- id_output_dir()
set.seed(20261102)

p <- panel_iv_prepare_outcomes(read_parquet(path_processed("panel_iv_county_year.parquet")), bite_spec)
bite_check_keys(p)
ctl <- panel_iv_control_set(bite_spec, "bea"); controls <- paste(ctl$control_terms, collapse = " + ")
d <- p %>% filter(year <= ctl$policy_end_year) %>%
  mutate(A = aewr_ppi, Z = z_wage_seasonal_composition_real,
    cert = .data[[bite_outcomes[["cert"]]]], hours = .data[[bite_outcomes[["hours"]]]])
d <- bite_complete(d, c("cert", "hours", "A", "Z", "z_wage_only_real", ctl$control_columns, "aewr_iv_cluster_id", "emp_farm_2011"))
# Counts implied by the retained ratio and its fixed denominator: identical sample.
d <- d %>% mutate(cert_n = cert * emp_farm_2011, hours_n = hours * emp_farm_2011)
CL <- c("aewr_region_id", "state_fips", "aewr_iv_cluster_id")
rows <- wild <- share <- list()
add <- function(key, m, dd, term, meta, price = "A", wild_ok = TRUE) {
  rows[[key]] <<- id_rows(m, dd, term, meta, CL, price)
  if (wild_ok) wild[[key]] <<- bind_cols(meta, id_wild(m, dd, term, "aewr_region_id"))
}

for (nm in c("cert", "hours")) {
  cnt <- paste0(nm, "_n")
  # 1. Baseline ratio, equal county weights (OLS and IV column 4).
  m <- feols(as.formula(paste(nm, "~ A +", controls, "| county_fips + year")), d, vcov = ~aewr_region_id, demeaned = TRUE)
  add(paste(nm, 1, "ols"), m, d, "A", tibble(spec = "1_baseline_ratio", estimator = "OLS", outcome = nm))
  m <- feols(as.formula(paste(nm, "~", controls, "| county_fips + year | A ~ Z")), d, vcov = ~aewr_region_id, demeaned = TRUE)
  add(paste(nm, 1, "iv"), m, d, "fit_A", tibble(spec = "1_baseline_ratio", estimator = "IV", outcome = nm))
  share[[paste(nm, 1)]] <- tibble(spec = "1_baseline_ratio", outcome = nm,
    region_year_share = id_region_year_share(d, nm), share_definition = "R2 of county+year-FE residual on region-year cells, equal weights")

  # 2. PPML on counts (semi-elasticity; size-weighted by construction).
  m <- fepois(as.formula(paste(cnt, "~ A +", controls, "| county_fips + year")), d, vcov = ~aewr_region_id)
  dd <- d[obs(m), ]
  r <- id_rows(m, d, "A", tibble(spec = "2_ppml_counts", estimator = "PPML", outcome = nm), CL, "A")
  # For PPML the coefficient is a semi-elasticity: elasticity = beta * mean(A).
  rows[[paste(nm, 2, "ppml")]] <- r %>% mutate(elasticity = estimate * mean_price,
    elasticity_low = (estimate - qt(.975, clusters - 1) * se) * mean_price,
    elasticity_high = (estimate + qt(.975, clusters - 1) * se) * mean_price,
    mde_elasticity = mde * mean_price, mde_pct_mean = 100 * mde)
  # Control-function PPML-IV: first-stage residual from the column-4 first stage.
  fs <- feols(as.formula(paste("A ~ Z +", controls, "| county_fips + year")), dd)
  dd$v <- resid(fs)
  mcf <- fepois(as.formula(paste(cnt, "~ A + v +", controls, "| county_fips + year")), dd, vcov = ~aewr_region_id)
  # Region pairs bootstrap (199 draws) redoes both stages; naive SE kept for reference.
  regions <- unique(dd$aewr_region_id)
  bs <- vapply(seq_len(199), function(b) {
    take <- sample(regions, length(regions), replace = TRUE)
    bd <- bind_rows(lapply(seq_along(take), function(k) dd[dd$aewr_region_id == take[k], ] %>%
      mutate(county_fips = paste0(county_fips, "_", k))))
    out <- tryCatch({
      bd$v <- resid(feols(as.formula(paste("A ~ Z +", controls, "| county_fips + year")), bd))
      coef(fepois(as.formula(paste(cnt, "~ A + v +", controls, "| county_fips + year")), bd))[["A"]]
    }, error = function(e) NA_real_)
    out
  }, numeric(1))
  bse <- sd(bs, na.rm = TRUE); mp <- mean(dd$A)
  rows[[paste(nm, 2, "ppml_cf")]] <- tibble(spec = "2_ppml_counts", estimator = "PPML control-function IV",
    outcome = nm, term = "A", cluster = "aewr_region_id pairs bootstrap (199)", clusters = length(regions),
    estimate = coef(mcf)[["A"]], se = bse, p = 2 * pnorm(-abs(coef(mcf)[["A"]] / bse)), n = nrow(dd),
    counties = n_distinct(dd$county_fips), first_year = min(dd$year), last_year = max(dd$year),
    mean_y = mean(dd[[cnt]]), mean_price = mp, price = "A", mde = 2.8 * bse, mde_pct_mean = 100 * 2.8 * bse,
    elasticity = coef(mcf)[["A"]] * mp, elasticity_low = (coef(mcf)[["A"]] - qt(.975, length(regions) - 1) * bse) * mp,
    elasticity_high = (coef(mcf)[["A"]] + qt(.975, length(regions) - 1) * bse) * mp, mde_elasticity = 2.8 * bse * mp,
    naive_se = se(mcf)[["A"]], bootstrap_failures = sum(is.na(bs)))
  m0 <- fepois(as.formula(paste(cnt, "~ 1 | county_fips + year")), dd)
  e <- dd[[cnt]] - fitted(m0); cell <- paste(dd$aewr_region_id, dd$year)
  share[[paste(nm, 2)]] <- tibble(spec = "2_ppml_counts", outcome = nm,
    region_year_share = sum(ave(e, cell)^2) / sum((e - mean(e))^2),
    share_definition = "R2 of raw count residual from FE-only PPML on region-year cells")

  # 3. Ratio weighted by 2011 farm employment (changes the estimand).
  m <- feols(as.formula(paste(nm, "~ A +", controls, "| county_fips + year")), d, weights = ~emp_farm_2011, vcov = ~aewr_region_id)
  rows[[paste(nm, 3, "ols")]] <- id_rows(m, d, "A", tibble(spec = "3_ratio_farm_emp_weighted", estimator = "WLS", outcome = nm), CL, "A")
  m <- feols(as.formula(paste(nm, "~", controls, "| county_fips + year | A ~ Z")), d, weights = ~emp_farm_2011, vcov = ~aewr_region_id)
  rows[[paste(nm, 3, "iv")]] <- id_rows(m, d, "fit_A", tibble(spec = "3_ratio_farm_emp_weighted", estimator = "WIV", outcome = nm), CL, "A")
  share[[paste(nm, 3)]] <- tibble(spec = "3_ratio_farm_emp_weighted", outcome = nm,
    region_year_share = id_region_year_share(d, nm, weights = ~emp_farm_2011), share_definition = "as spec 1, 2011 farm-employment weights")
}

# 4. Region-year aggregate: log total positions / hours, region + year FE.
agg <- d %>% group_by(aewr_region_id, year) %>%
  summarise(A = first(A), cert_n = sum(cert_n), hours_n = sum(hours_n),
    Z = weighted.mean(Z, emp_farm_2011), emp = sum(emp_farm_2011), counties = n(), .groups = "drop") %>%
  mutate(state_fips = aewr_region_id, aewr_iv_cluster_id = aewr_region_id,
    log_cert = log(cert_n), log_hours = log(hours_n), county_fips = aewr_region_id)
stopifnot(!anyDuplicated(agg[c("aewr_region_id", "year")]), all(is.finite(agg$log_cert)))
for (nm in c("cert", "hours")) {
  y <- paste0("log_", nm)
  m <- feols(as.formula(paste(y, "~ A | aewr_region_id + year")), agg, vcov = ~aewr_region_id, demeaned = TRUE)
  r <- id_rows(m, agg, "A", tibble(spec = "4_region_year_aggregate", estimator = "OLS", outcome = nm), "aewr_region_id", "A")
  rows[[paste(nm, 4, "ols")]] <- r %>% mutate(elasticity = estimate * mean_price,
    elasticity_low = (estimate - qt(.975, clusters - 1) * se) * mean_price,
    elasticity_high = (estimate + qt(.975, clusters - 1) * se) * mean_price, mde_elasticity = mde * mean_price, mde_pct_mean = 100 * mde)
  wild[[paste(nm, 4, "ols")]] <- bind_cols(tibble(spec = "4_region_year_aggregate", estimator = "OLS", outcome = nm),
    id_wild(m, agg, "A", "aewr_region_id"))
  m <- feols(as.formula(paste(y, "~ 1 | aewr_region_id + year | A ~ Z")), agg, vcov = ~aewr_region_id, demeaned = TRUE)
  r <- id_rows(m, agg, "fit_A", tibble(spec = "4_region_year_aggregate", estimator = "IV", outcome = nm), "aewr_region_id", "A")
  rows[[paste(nm, 4, "iv")]] <- r %>% mutate(elasticity = estimate * mean_price,
    elasticity_low = (estimate - qt(.975, clusters - 1) * se) * mean_price,
    elasticity_high = (estimate + qt(.975, clusters - 1) * se) * mean_price, mde_elasticity = mde * mean_price, mde_pct_mean = 100 * mde,
    first_stage_F = fitstat(m, "ivwald1")[[1]]$stat)
  share[[paste(nm, 4)]] <- tibble(spec = "4_region_year_aggregate", outcome = nm, region_year_share = 1,
    share_definition = "all residual variation is region-year by construction; 17 x years cells")
}

# 5. Employer panel: v2 balanced worksite businesses present at baseline.
# Cohorts: start years 2011-2012 (panel 2011-2019) and 2020-2021 (panel 2020-2025),
# respecting the 2020 linkage break. Single baseline AEWR region required.
sr <- read_parquet(path_processed("county_year_panel.parquet"), col_select = c("state_fips", "aewr_region_id")) %>% distinct()
stopifnot(!anyDuplicated(sr$state_fips))
ra <- read_parquet(path_processed("county_year_panel.parquet"), col_select = c("aewr_region_id", "year", "aewr_ppi")) %>%
  filter(is.finite(aewr_ppi)) %>% distinct(aewr_region_id, year, aewr_ppi) %>% rename(region = aewr_region_id, A = aewr_ppi)
stopifnot(!anyDuplicated(ra[c("region", "year")]))
w <- id_read_worksites(c("employer_id_balanced", "employer_is_worksite_business", "adjusted_nbr_workers_certified",
  "period_begin_date", "period_end_date", "number_of_hours", "county_fips_list")) %>%
  filter(employer_is_worksite_business, !is.na(employer_id_balanced), employer_id_balanced != "") %>%
  mutate(begin = as.Date(period_begin_date), end = as.Date(period_end_date), year = as.integer(format(begin, "%Y")),
    workers = coalesce(as.numeric(adjusted_nbr_workers_certified), 0),
    hours = pmax(as.numeric(end - begin), 0) / 7 * suppressWarnings(as.numeric(number_of_hours)) * workers,
    state_fips = substr(trimws(sub(",.*", "", county_fips_list)), 1, 2)) %>%
  filter(!is.na(year), year >= 2011, year <= 2025) %>% left_join(sr, by = "state_fips")
emp_panel <- function(base_years, years) {
  cohort <- w %>% filter(year %in% base_years, workers > 0, !is.na(aewr_region_id)) %>%
    group_by(employer_id_balanced) %>% summarise(regions = n_distinct(aewr_region_id),
      region = first(aewr_region_id), state = names(which.max(tapply(workers, state_fips, sum))), .groups = "drop") %>%
    filter(regions == 1)
  ey <- w %>% semi_join(cohort, by = "employer_id_balanced") %>% filter(year %in% years) %>%
    group_by(employer_id_balanced, year) %>% summarise(workers = sum(workers), hours = sum(hours, na.rm = TRUE), .groups = "drop")
  cohort %>% select(employer_id_balanced, region, state) %>% tidyr::crossing(year = years) %>%
    left_join(ey, by = c("employer_id_balanced", "year")) %>%
    mutate(workers = coalesce(workers, 0), hours = coalesce(hours, 0), l_cert = log1p(workers), l_hours = log1p(hours)) %>%
    left_join(ra, by = c("region", "year")) %>% filter(is.finite(A)) %>%
    mutate(aewr_region_id = region, state_fips = state, aewr_iv_cluster_id = region, county_fips = employer_id_balanced)
}
for (win in list(list(b = 2011:2012, y = 2011:2019, lab = "2011-2019"), list(b = 2020:2021, y = 2020:2025, lab = "2020-2025"))) {
  ep <- emp_panel(win$b, win$y)
  for (nm in c("cert", "hours")) {
    y <- paste0("l_", nm)
    m <- feols(as.formula(paste(y, "~ A | employer_id_balanced + year")), ep, vcov = ~aewr_region_id, demeaned = TRUE)
    meta <- tibble(spec = paste0("5_employer_panel_", win$lab), estimator = "OLS", outcome = nm)
    r <- id_rows(m, ep, "A", meta, c("aewr_region_id", "state_fips"), "A")
    rows[[paste(nm, 5, win$lab)]] <- r %>% mutate(elasticity = estimate * mean_price,
      elasticity_low = (estimate - qt(.975, clusters - 1) * se) * mean_price,
      elasticity_high = (estimate + qt(.975, clusters - 1) * se) * mean_price, mde_elasticity = mde * mean_price,
      mde_pct_mean = 100 * mde, employers = n_distinct(ep$employer_id_balanced))
    wild[[paste(nm, 5, win$lab)]] <- bind_cols(meta, id_wild(m, ep, "A", "aewr_region_id"))
    share[[paste(nm, 5, win$lab)]] <- tibble(spec = meta$spec, outcome = nm,
      region_year_share = id_region_year_share(ep, y, unit = "employer_id_balanced", cell = "aewr_region_id"),
      share_definition = "R2 of employer+year-FE residual on baseline-region-year cells")
  }
}

est <- bind_rows(rows) %>% left_join(bind_rows(wild) %>% select(spec, estimator, outcome, wild_p),
  by = c("spec", "estimator", "outcome"))
est <- est %>% mutate(gate_elasticity_mde_le_1 = mde_elasticity <= 1)
id_write(est, out_dir, "precision_estimates.csv")
id_write(bind_rows(share), out_dir, "precision_region_year_share.csv")
print(as.data.frame(est %>% filter(cluster %in% c("aewr_region_id", "aewr_region_id pairs bootstrap (199)")) %>%
  select(spec, estimator, outcome, estimate, se, wild_p, n, mean_y, mean_price, elasticity, mde_elasticity, mde_pct_mean)), digits = 3)
print(as.data.frame(bind_rows(share)), digits = 3)
