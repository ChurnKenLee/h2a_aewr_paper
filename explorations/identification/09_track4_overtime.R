# Track 4: state farmworker overtime thresholds as an hourly-cost shifter
# (report section 6). User approved acquiring the schedule on 2026-11-02; the
# sourced schedule is explorations/identification/sources/state_ag_overtime_schedule.csv
# (notes alongside). Support check first, then the specifications declared in
# section 6: (i) county + former-region x year FE, (ii) county + state x year FE,
# (iii) PPML counts with (i)'s FE; plus a one-year lead check (added here, labeled).
#
# Coding rules, fixed before any outcome regression:
#   - threshold in effect on 1 July of calendar year t (outcomes are start-year);
#   - rows for crop/general agriculture only: WA dairy-only, MA post-harvest
#     packing and ME egg-operation rows are ignored;
#   - employer class: CA 26 or more employees; CO 2024 "other_employer" (54 h)
#     [sensitivity: "highly_seasonal_employer" (48 h)]; HI 20 or more employees;
#     MD more than 500 farm-worker-days in a quarter (large employers);
#   - daily thresholds converted with a 6-day week: weekly overtime hours =
#     max(h - weekly threshold, 6 * max(0, h/6 - daily threshold), 0);
#   - dose for weekly hours h: 0.5 * overtime hours / h (proportional increase in
#     hourly cost at a constant straight-time wage);
#   - frozen exposure D_ct: certified-hours-weighted mean dose over the county's
#     FY2015-FY2018 job orders at that county's state threshold in year t;
#     counties without baseline job orders use their state's baseline mix;
#   - sample: counties with emp_farm_2011 > 0, excluding AK, HI, DC; 2011-2025.
# Elasticity of use with respect to hourly cost: beta / mean(y) (beta for PPML).
source("explorations/identification/id_helpers.R")
out_dir <- id_output_dir()
SEED <- 20261102L; RI_DRAWS <- 999L
ot <- read_csv("explorations/identification/sources/state_ag_overtime_schedule.csv", show_col_types = FALSE,
  col_types = cols(.default = "c")) %>%
  mutate(start = as.Date(effective_start), end = as.Date(ifelse(effective_end == "" | is.na(effective_end), "2099-12-31", effective_end)),
    wk = as.numeric(weekly_threshold_hours), dy = as.numeric(daily_threshold_hours))
stopifnot(all(nchar(ot$state_fips) == 2), !anyNA(ot$start))
ignore_scope <- c("dairy_only", "post_harvest_cleaning_sorting_packing_on_farm", "large_egg_production_only")
pick_class <- function(st, cls, co2024) dplyr::case_when(
  st == "CA" ~ cls == "26_or_more_employees",
  st == "CO" ~ cls %in% c("all", co2024),
  st == "HI" ~ cls == "20_or_more_employees",
  st == "MD" ~ grepl("^employers_using_more_than_500", cls),
  TRUE ~ TRUE)
threshold_on <- function(co2024 = "other_employer") {
  bind_rows(lapply(2008:2026, function(y) {
    date <- as.Date(sprintf("%d-07-01", y))
    ot %>% filter(!sector_scope %in% ignore_scope, pick_class(state_abbrev, employer_size_class, co2024),
      start <= date, end >= date) %>%
      group_by(state_fips) %>% summarise(n = n(), wk = first(wk), dy = first(dy), .groups = "drop") %>% mutate(year = y)
  }))
}
thr <- threshold_on(); thr_hs <- threshold_on("highly_seasonal_employer")
stopifnot(all(thr$n == 1), all(thr_hs$n == 1))
ot_dose <- function(h, wk, dy) {
  w <- ifelse(is.na(wk), 0, pmax(h - wk, 0)); d <- ifelse(is.na(dy), 0, 6 * pmax(h / 6 - dy, 0))
  0.5 * pmax(w, d) / h
}

# Baseline weekly-hours mix, FY2015-FY2018 certified hours (b01 hours rule) ----
bl <- id_read_worksites(c("case_number", "county_fips_list", "adjusted_nbr_workers_certified",
  "period_begin_date", "period_end_date", "number_of_hours")) %>%
  filter(fiscal_year %in% 2015:2018) %>%
  mutate(begin = as.Date(period_begin_date), end = as.Date(period_end_date), h = suppressWarnings(as.numeric(number_of_hours)),
    workers = as.numeric(adjusted_nbr_workers_certified)) %>%
  filter(!is.na(begin), is.finite(h), h > 0, h <= 100, is.finite(workers), workers > 0) %>%
  mutate(hours = as.numeric(end - begin) / 7 * h * workers, n_counties = stringr::str_count(county_fips_list, ",") + 1L) %>%
  filter(is.finite(hours), hours > 0) %>%
  tidyr::separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), w = hours / n_counties, h = round(h, 1))
mix_c <- bl %>% group_by(county_fips, h) %>% summarise(w = sum(w), .groups = "drop")

p <- read_parquet(path_processed("county_year_panel.parquet"), col_select = c("county_fips", "state_fips", "state_abbrev",
  "aewr_region_id", "year", "emp_farm_2011", "h2a_cert_share_farm_workers_2011_start_year",
  "h2a_cert_hours_per_farm_worker_2011_start_year", "nbr_workers_certified_start_year", "man_hours_certified_start_year")) %>%
  filter(year >= 2010, year <= 2025, !state_fips %in% c("02", "15", "11"), is.finite(emp_farm_2011), emp_farm_2011 > 0)
bite_check_keys(p)
cty <- p %>% distinct(county_fips, state_fips)
mix_s <- mix_c %>% inner_join(cty, by = "county_fips") %>% group_by(state_fips, h) %>% summarise(w = sum(w), .groups = "drop")
mix <- bind_rows(mix_c %>% inner_join(cty, by = "county_fips") %>% mutate(src = "county"),
  cty %>% filter(!county_fips %in% mix_c$county_fips) %>% inner_join(mix_s, by = "state_fips", relationship = "many-to-many") %>% mutate(src = "state")) %>%
  group_by(county_fips) %>% mutate(w = w / sum(w)) %>% ungroup()
stopifnot(abs(tapply(mix$w, mix$county_fips, sum) - 1) < 1e-9)
exposure <- function(th, mix, label) {
  # th: state_fips, year, wk, dy. Returns county-year D.
  mix %>% inner_join(th %>% select(state_fips, year, wk, dy), by = "state_fips", relationship = "many-to-many") %>%
    group_by(county_fips, year) %>% summarise(!!label := sum(w * ot_dose(h, wk, dy)), .groups = "drop")
}
p <- p %>% left_join(exposure(thr, mix, "D"), by = c("county_fips", "year")) %>%
  left_join(exposure(thr_hs, mix, "D_co_hs"), by = c("county_fips", "year")) %>%
  mutate(D = coalesce(D, 0), D_co_hs = coalesce(D_co_hs, 0)) %>%
  group_by(county_fips) %>% arrange(year, .by_group = TRUE) %>% mutate(D_lead = lead(D)) %>% ungroup()
lead_2026 <- exposure(thr %>% filter(year == 2026), mix, "D26") %>% mutate(year = 2025L)
p <- p %>% left_join(lead_2026, by = c("county_fips", "year")) %>% mutate(D_lead = ifelse(year == 2025, coalesce(D26, 0), D_lead)) %>% select(-D26)
p <- p %>% filter(year >= 2011)

# Support check (section 6, run first) ---------------------------------------------
st_thr <- thr %>% filter(year >= 2011, year <= 2025, state_fips %in% p$state_fips) %>% arrange(state_fips, year) %>%
  group_by(state_fips) %>% mutate(changed = coalesce(wk != lag(wk), !is.na(wk) & is.na(lag(wk)), FALSE) |
    coalesce(dy != lag(dy), xor(is.na(dy), is.na(lag(dy))), FALSE)) %>% ungroup()
bh <- bl %>% inner_join(cty, by = "county_fips")
share_above <- bind_rows(lapply(c(40, 45, 48, 50, 52, 54, 55, 56, 60), function(k)
  tibble(threshold = k, share_baseline_hours_above = sum(bh$w[bh$h > k]) / sum(bh$w))))
treated_states <- unique(st_thr$state_fips[st_thr$changed & st_thr$year > 2011])
r_sy <- p %>% group_by(state_fips, year) %>% mutate(dev = D - mean(D)) %>% ungroup()
support <- tibble(
  states_with_threshold_change_2012_2025 = length(treated_states),
  treated_state_list = paste(sort(unique(p$state_abbrev[p$state_fips %in% treated_states])), collapse = ";"),
  state_years_with_change = sum(st_thr$changed & st_thr$year > 2011),
  counties = n_distinct(p$county_fips), counties_in_treated_states = n_distinct(p$county_fips[p$state_fips %in% treated_states]),
  counties_with_D_positive_ever = n_distinct(p$county_fips[p$D > 0]),
  counties_with_own_baseline_mix = n_distinct(mix$county_fips[mix$src == "county"]),
  within_state_year_sd_D = sd(r_sy$dev), within_state_year_sd_D_treated_post = sd(r_sy$dev[r_sy$state_fips %in% treated_states & r_sy$D > 0]),
  mean_D_treated_2025 = mean(p$D[p$state_fips %in% treated_states & p$year == 2025]),
  max_D = max(p$D))
by_state <- p %>% filter(state_fips %in% treated_states) %>% group_by(state_abbrev, year) %>%
  summarise(mean_D = mean(D), sd_D = sd(D), counties = n(), positions = sum(nbr_workers_certified_start_year, na.rm = TRUE), .groups = "drop")
id_write(support, out_dir, "track4_support.csv"); id_write(share_above, out_dir, "track4_baseline_hours_above.csv")
id_write(by_state, out_dir, "track4_exposure_by_state_year.csv")
id_write(thr %>% filter(!is.na(wk) | !is.na(dy)), out_dir, "track4_threshold_schedule_used.csv")
print(as.data.frame(support)); print(as.data.frame(share_above)); print(as.data.frame(by_state %>% filter(year %in% c(2018, 2020, 2022, 2025))), digits = 3)

# Specifications ------------------------------------------------------------------
Y <- c(cert = "h2a_cert_share_farm_workers_2011_start_year", hours = "h2a_cert_hours_per_farm_worker_2011_start_year")
YC <- c(cert = "nbr_workers_certified_start_year", hours = "man_hours_certified_start_year")
FE1 <- "county_fips + aewr_region_id^year"; FE2 <- "county_fips + state_fips^year"
ri_perm <- function(d, m, y, x, fe) {
  # Reassign whole state threshold schedules across states (each county keeps its
  # own baseline hours mix) and refit the within-FE slope.
  d <- d[obs(m), , drop = FALSE]
  fl <- lapply(strsplit(fe, " \\+ ")[[1]], function(f) if (grepl("\\^", f)) factor(do.call(paste, d[strsplit(f, "\\^")[[1]]])) else factor(d[[f]]))
  yw <- drop(fixest::demean(cbind(d[[y]]), f = fl, nthreads = 1L))
  sts <- sort(unique(d$state_fips)); set.seed(SEED)
  # Dose of every county's baseline mix at every distinct threshold pair.
  th <- thr %>% filter(state_fips %in% sts, year %in% unique(d$year)) %>% mutate(tid = paste(wk, dy))
  tp <- th %>% distinct(tid, wk, dy)
  dm <- mix %>% filter(county_fips %in% d$county_fips) %>% tidyr::crossing(tp) %>%
    group_by(county_fips, tid) %>% summarise(v = sum(w * ot_dose(h, wk, dy)), .groups = "drop")
  Dm <- matrix(0, n_distinct(d$county_fips), nrow(tp), dimnames = list(sort(unique(d$county_fips)), tp$tid))
  Dm[cbind(dm$county_fips, dm$tid)] <- dm$v
  ci <- match(d$county_fips, rownames(Dm))
  T <- matrix(NA_character_, length(sts), max(d$year)); T[cbind(match(th$state_fips, sts), th$year)] <- th$tid
  X <- vapply(seq_len(RI_DRAWS), function(i) {
    perm <- setNames(sample(sts), sts)  # county in state s receives the schedule of state perm[s]
    Dm[cbind(ci, match(T[cbind(match(perm[d$state_fips], sts), d$year)], colnames(Dm)))]
  }, numeric(nrow(d)))
  Xw <- fixest::demean(X, f = fl, nthreads = 1L)
  b <- colSums(Xw * yw) / colSums(Xw^2)
  stopifnot(abs(drop(Dm[cbind(ci, match(T[cbind(match(d$state_fips, sts), d$year)], colnames(Dm)))]) - d[[x]]) < 1e-12)
  (1 + sum(abs(b) >= abs(coef(m)[[x]]) - 1e-12)) / (RI_DRAWS + 1)
}
rows <- list()
for (nm in names(Y)) for (sp in list(list("i_region_year_FE", FE1, "D"), list("ii_state_year_FE", FE2, "D"),
  list("i_region_year_FE_CO_highly_seasonal", FE1, "D_co_hs"), list("lead_check_region_year_FE", FE1, "D + D_lead"))) {
  y <- Y[[nm]]; d <- bite_complete(p, c(y, "D"))
  m <- feols(as.formula(paste(y, "~", sp[[3]], "|", sp[[2]])), d, vcov = ~state_fips, demeaned = TRUE)
  x <- strsplit(sp[[3]], " \\+ ")[[1]]
  for (term in x) {
    s <- coeftable(m); dd <- d[obs(m), ]; my <- mean(dd[[y]]); G <- n_distinct(dd$state_fips)
    rows[[length(rows) + 1]] <- tibble(spec = sp[[1]], outcome = nm, estimator = "OLS", term = term, n = nobs(m), states = G,
      mean_y = my, mean_D = mean(dd[[term]]), estimate = s[term, 1], se_state_crv1 = s[term, 2], p_state_crv1 = s[term, 4],
      elasticity_cost = s[term, 1] / my, elasticity_low = (s[term, 1] - qt(.975, G - 1) * s[term, 2]) / my,
      elasticity_high = (s[term, 1] + qt(.975, G - 1) * s[term, 2]) / my, mde_elasticity = 2.8 * s[term, 2] / my,
      wild_p_state = id_wild(m, d, term, "state_fips", B = 9999L, seed = SEED)$wild_p,
      ri_p_state_schedules = if (term == "D" && sp[[3]] == "D") ri_perm(d, m, y, "D", sp[[2]]) else NA_real_)
    message(sp[[1]], " ", nm, " ", term, ": ", signif(s[term, 1], 4), " (", signif(s[term, 2], 3), ")")
  }
}
for (nm in names(YC)) {
  y <- YC[[nm]]; d <- bite_complete(p, c(y, "D"))
  m <- fepois(as.formula(paste(y, "~ D |", FE1)), d, vcov = ~state_fips)
  s <- coeftable(m); G <- n_distinct(d$state_fips[obs(m)])
  rows[[length(rows) + 1]] <- tibble(spec = "iii_ppml_region_year_FE", outcome = nm, estimator = "PPML", term = "D", n = nobs(m), states = G,
    mean_y = mean(d[[y]][obs(m)]), mean_D = mean(d$D[obs(m)]), estimate = s["D", 1], se_state_crv1 = s["D", 2], p_state_crv1 = s["D", 4],
    elasticity_cost = s["D", 1], elasticity_low = s["D", 1] - qt(.975, G - 1) * s["D", 2], elasticity_high = s["D", 1] + qt(.975, G - 1) * s["D", 2],
    mde_elasticity = 2.8 * s["D", 2], wild_p_state = NA_real_, ri_p_state_schedules = NA_real_)
}
res <- bind_rows(rows)
id_write(res, out_dir, "track4_estimates.csv")
print(as.data.frame(res %>% select(spec, outcome, estimator, term, estimate, se_state_crv1, p_state_crv1, wild_p_state,
  ri_p_state_schedules, elasticity_cost, elasticity_low, elasticity_high, mde_elasticity)), digits = 3)
