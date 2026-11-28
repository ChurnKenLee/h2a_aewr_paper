# Finalize Phase 1 strictly outcome-blind verification (250 exact nonlinear draws for
# borderline survivors R2a_direct_2020_2025 [1b & 1a] and R4a_bite_h2a_duration [C+RY & C+SY])
# and assemble the unified explorations/bite_power_rescue/phase1_triage.csv across all R1-R6.
# HARD RULE: Strictly outcome-blind! ZERO real post-policy outcome contrasts estimated here.

source("explorations/bite_power_rescue/helpers_phase1.R")

cat("=== 1. 250-Draw Exact Nonlinear CV3J Size Verification on Potential Survivors ===\n")
bp_full <- read_parquet(path_processed("bite_county_year_panel.parquet")) %>% mutate(year = as.integer(year))
bp_est  <- bp_full %>% filter(year >= 2011L, year <= 2025L)
cls_b <- sort(unique(as.character(bp_est$aewr_region_id))); Gb <- length(cls_b)

# Rebuild direct-grower 2020-2025 start-year panel (strictly <= 2025)
disc_2025 <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, case_number, h2a_labor_contractor, type_of_employer_application) %>%
  filter(fiscal_year >= 2019L, fiscal_year <= 2026L) %>% collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(
    lc_raw = first(na.omit(as.character(h2a_labor_contractor))),
    typ_raw = first(na.omit(as.character(type_of_employer_application))), .groups = "drop"
  ) %>%
  mutate(
    is_h2alc = (toupper(trimws(coalesce(lc_raw, ""))) %in% c("Y", "YES", "TRUE", "1")) |
      grepl("LABOR CONTRACTOR|JOB CONTRACTOR", toupper(coalesce(typ_raw, "")))
  ) %>% select(case_number, fiscal_year, is_h2alc)

nh_pre_mean <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, period_begin_date, number_of_hours) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2025L) %>% collect() %>%
  filter(!is.na(as.Date(period_begin_date))) %>%
  summarise(m = mean(suppressWarnings(as.numeric(number_of_hours)), na.rm = TRUE)) %>% pull(m)

ws_dg20 <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date, number_of_hours) %>%
  filter(fiscal_year >= 2019L, fiscal_year <= 2026L) %>% collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year), begin = as.Date(period_begin_date), end = as.Date(period_end_date),
         start_year = as.integer(format(begin, "%Y"))) %>%
  filter(!is.na(begin), !is.na(end), start_year >= 2020L, start_year <= 2025L) %>%
  left_join(disc_2025, by = c("case_number", "fiscal_year")) %>%
  mutate(
    is_h2alc = coalesce(is_h2alc, FALSE),
    nh = coalesce(suppressWarnings(as.numeric(number_of_hours)), nh_pre_mean),
    workers = as.numeric(adjusted_nbr_workers_certified),
    hours = pmax(as.numeric(end - begin) / 7, 0) * nh * workers,
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    nc = stringr::str_count(county_fips_list, ",") + 1L
  ) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), share = 1 / nc) %>%
  group_by(county_fips, year = start_year) %>%
  summarise(
    direct_hours_sy = sum(hours * share * (!is_h2alc), na.rm = TRUE),
    all_hours_sy    = sum(hours * share, na.rm = TRUE),
    .groups = "drop"
  )

bp_2025 <- bp_est %>% filter(year >= 2020L) %>%
  left_join(ws_dg20, by = c("county_fips", "year")) %>%
  mutate(direct_hours_sy = coalesce(direct_hours_sy, 0), all_hours_sy = coalesce(all_hours_sy, 0))

pos_dg20 <- bp_2025 %>% group_by(county_fips) %>% summarise(s = sum(direct_hours_sy), .groups = "drop") %>% filter(s > 0) %>% pull(county_fips)
d20_act <- bp_2025 %>% filter(county_fips %in% pos_dg20) %>%
  select(county_fips, aewr_region_id, state_fips, year, direct_hours_sy, E_p25) %>% as.data.frame()
reg_paths_2025 <- bp_2025 %>% distinct(aewr_region_id, year, ln_aewr) %>% arrange(aewr_region_id, year)

# 250 exact nonlinear fepois CV3J draws for Direct (2020-25) 1b and 1a!
set.seed(SEED_PHASE1 + 801L)
B_250 <- 250L
b_dg1b <- se_dg1b <- b_dg1a <- se_dg1a <- numeric(B_250)
for (b in seq_len(B_250)) {
  pm <- setNames(sample(cls_b), cls_b)
  sm <- setNames(sample(c(-1, 1), Gb, replace = TRUE), cls_b)
  rp <- reg_paths_2025 %>% mutate(tr = names(pm)[match(aewr_region_id, pm)]) %>%
    select(aewr_region_id = tr, year, ln_aewr_p = ln_aewr)
  db <- d20_act %>% left_join(rp, by = c("aewr_region_id", "year")) %>%
    mutate(Ep = E_p25 * unname(sm[as.character(aewr_region_id)]), xp = Ep * ln_aewr_p)
  f1b <- fit_cv3j_fixest(db, direct_hours_sy ~ xp | county_fips + aewr_region_id^year, "xp", "aewr_region_id", "fepois")
  f1a <- fit_cv3j_fixest(db, direct_hours_sy ~ xp + i(year, Ep, ref = 2020) | county_fips + aewr_region_id^year, "xp", "aewr_region_id", "fepois")
  b_dg1b[[b]] <- f1b$estimate; se_dg1b[[b]] <- f1b$se_cv3j
  b_dg1a[[b]] <- f1a$estimate; se_dg1a[[b]] <- f1a$se_cv3j
}
em_dg1b_250 <- compute_empirical_mde(b_dg1b, se_dg1b, 16L)
em_dg1a_250 <- compute_empirical_mde(b_dg1a, se_dg1a, 16L)
cat(sprintf("250-draw EXACT NONLINEAR R2a Direct (2020-25) 1b: size=%.3f [%.3f,%.3f], se_med=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  em_dg1b_250$emp_size, em_dg1b_250$wilson_lo, em_dg1b_250$wilson_hi, em_dg1b_250$se_median, em_dg1b_250$mde_upper, 2 * em_dg1b_250$mde_upper))
cat(sprintf("250-draw EXACT NONLINEAR R2a Direct (2020-25) 1a: size=%.3f [%.3f,%.3f], se_med=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  em_dg1a_250$emp_size, em_dg1a_250$wilson_lo, em_dg1a_250$wilson_hi, em_dg1a_250$se_median, em_dg1a_250$mde_upper, 2 * em_dg1a_250$mde_upper))

# Also run 250 exact nonlinear fepois CV3J draws for R4a Bite H-2A Duration (2011-2025, C+RY & C+SY)
FIXED_COST_F <- 2500.0
std_within <- function(x, grp) {
  out <- rep(NA_real_, length(x))
  for (g in unique(grp[!is.na(grp)])) {
    idx <- which(grp == g & is.finite(x))
    if (length(idx) >= 2L) { s <- stats::sd(x[idx]); if (is.finite(s) && s > 0) out[idx] <- (x[idx] - mean(x[idx])) / s }
  }
  out
}
ws_0810_dur <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date) %>%
  filter(fiscal_year %in% c(2008L, 2009L, 2010L)) %>% collect() %>%
  mutate(weeks = as.numeric(as.Date(period_end_date) - as.Date(period_begin_date)) / 7,
         workers = as.numeric(adjusted_nbr_workers_certified),
         county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
         nc = stringr::str_count(county_fips_list, ",") + 1L) %>%
  filter(is.finite(weeks), weeks > 0, weeks <= 60, is.finite(workers), workers > 0) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), w = workers / nc) %>%
  group_by(county_fips) %>% summarise(mean_weeks_0810 = sum(weeks * w) / sum(w), .groups = "drop")

cty_dur <- bp_full %>% filter(year %in% c(2008L, 2010L)) %>%
  group_by(county_fips, aewr_region_id) %>% summarise(aewr_0810 = mean(aewr, na.rm = TRUE), .groups = "drop") %>%
  left_join(ws_0810_dur, by = "county_fips") %>%
  mutate(s_wage = ifelse(is.finite(mean_weeks_0810), (aewr_0810 * 40 * mean_weeks_0810) / (aewr_0810 * 40 * mean_weeks_0810 + FIXED_COST_F), NA_real_),
         E_dur = std_within(s_wage, aewr_region_id)) %>%
  filter(is.finite(E_dur))

d_dur_act <- bp_est %>% inner_join(cty_dur %>% select(county_fips, E_dur), by = "county_fips") %>%
  select(county_fips, aewr_region_id, state_fips, year, man_hours_certified_start_year, E_dur) %>% as.data.frame()
reg_paths_1125 <- bp_est %>% distinct(aewr_region_id, year, ln_aewr) %>% arrange(aewr_region_id, year)

set.seed(SEED_PHASE1 + 802L)
b_dry <- se_dry <- b_dsy <- se_dsy <- numeric(B_250)
for (b in seq_len(B_250)) {
  pm <- setNames(sample(cls_b), cls_b)
  sm <- setNames(sample(c(-1, 1), Gb, replace = TRUE), cls_b)
  rp <- reg_paths_1125 %>% mutate(tr = names(pm)[match(aewr_region_id, pm)]) %>%
    select(aewr_region_id = tr, year, ln_aewr_p = ln_aewr)
  db <- d_dur_act %>% left_join(rp, by = c("aewr_region_id", "year")) %>%
    mutate(xp = E_dur * unname(sm[as.character(aewr_region_id)]) * ln_aewr_p)
  fry <- fit_cv3j_fixest(db, man_hours_certified_start_year ~ xp | county_fips + aewr_region_id^year, "xp", "aewr_region_id", "fepois")
  fsy <- fit_cv3j_fixest(db, man_hours_certified_start_year ~ xp | county_fips + state_fips^year, "xp", "aewr_region_id", "fepois")
  b_dry[[b]] <- fry$estimate; se_dry[[b]] <- fry$se_cv3j
  b_dsy[[b]] <- fsy$estimate; se_dsy[[b]] <- fsy$se_cv3j
}
em_dry_250 <- compute_empirical_mde(b_dry, se_dry, 16L)
em_dsy_250 <- compute_empirical_mde(b_dsy, se_dsy, 16L)
cat(sprintf("250-draw EXACT NONLINEAR R4a Dur (2011-25) C+RY : size=%.3f [%.3f,%.3f], se_med=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  em_dry_250$emp_size, em_dry_250$wilson_lo, em_dry_250$wilson_hi, em_dry_250$se_median, em_dry_250$mde_upper, 2 * em_dry_250$mde_upper))
cat(sprintf("250-draw EXACT NONLINEAR R4a Dur (2011-25) C+SY : size=%.3f [%.3f,%.3f], se_med=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  em_dsy_250$emp_size, em_dsy_250$wilson_lo, em_dsy_250$wilson_hi, em_dsy_250$se_median, em_dsy_250$mde_upper, 2 * em_dsy_250$mde_upper))

# Check baseline covariate balance of E_p25 on 2020-2025 direct-grower counties
cat("\nBaseline 2008-2010 Covariate Balance for E_p25 on 2020-2025 direct-grower counties:\n")
cty_ep25 <- bp_full %>% filter(year == 2010L, county_fips %in% pos_dg20) %>%
  select(county_fips, aewr_region_id, E_p25, ln_pop_2010, farm_emp_share_0810, ln_emp_farm_0810)
for (cov in c("ln_pop_2010", "farm_emp_share_0810", "ln_emp_farm_0810")) {
  sc <- cty_ep25 %>% filter(is.finite(.data[[cov]]))
  fb <- fit_cv3j_fixest(sc, stats::as.formula(paste0(cov, " ~ E_p25 | aewr_region_id")), "E_p25", "aewr_region_id", "feols")
  cat(sprintf("  %-22s: corr=%+.3f, within-reg slope=%+.4f [CV3J %.4f], p=%.3f\n", cov, cor(sc$E_p25, sc[[cov]]), fb$estimate, fb$se_cv3j, fb$p_cv3j))
}

# ==============================================================================
# 2. Assemble unified explorations/bite_power_rescue/phase1_triage.csv
# ==============================================================================
cat("\n=== 2. Compiling Unified phase1_triage.csv ===\n")

t_r1_r5 <- read_csv("explorations/bite_power_rescue/phase1_r1_r5_triage.csv", show_col_types = FALSE)
t_r3    <- read_csv("explorations/bite_power_rescue/phase1_r3_triage.csv", show_col_types = FALSE)
t_r4_r6 <- read_csv("explorations/bite_power_rescue/phase1_r4_r6_triage.csv", show_col_types = FALSE)

# Update R4a_bite_h2a_duration_ry and _sy with 250-draw exact nonlinear CV3J numbers!
t_r4_r6 <- t_r4_r6 %>%
  mutate(
    n_placebo_draws = case_when(
      candidate %in% c("R4a_bite_h2a_duration_ry", "R4a_bite_h2a_duration_sy") ~ 250L,
      TRUE ~ as.integer(n_placebo_draws)
    ),
    emp_size_alpha05 = case_when(
      candidate == "R4a_bite_h2a_duration_ry" ~ em_dry_250$emp_size,
      candidate == "R4a_bite_h2a_duration_sy" ~ em_dsy_250$emp_size,
      TRUE ~ emp_size_alpha05
    ),
    size_wilson_hi = case_when(
      candidate == "R4a_bite_h2a_duration_ry" ~ em_dry_250$wilson_hi,
      candidate == "R4a_bite_h2a_duration_sy" ~ em_dsy_250$wilson_hi,
      TRUE ~ size_wilson_hi
    ),
    mde_primary = case_when(
      candidate == "R4a_bite_h2a_duration_ry" ~ 2 * em_dry_250$mde_upper,
      candidate == "R4a_bite_h2a_duration_sy" ~ 2 * em_dsy_250$mde_upper,
      TRUE ~ mde_primary
    ),
    mde_eps_cost = case_when(
      candidate %in% c("R4a_bite_h2a_duration_ry", "R4a_bite_h2a_duration_sy") ~ 3.994,
      TRUE ~ mde_eps_cost
    ),
    placebo_p = case_when(
      candidate %in% c("R4a_bite_h2a_duration_ry", "R4a_bite_h2a_duration_sy") ~ 0.622,
      TRUE ~ placebo_p
    ),
    notes = case_when(
      candidate == "R4a_bite_h2a_duration_ry" ~ sprintf("250-draw exact nonlinear CV3J: 1SD=%.3f / 2SD=%.3f (999-draw lin 0.243/0.485); 2008-10 CV3J pretrend p=0.622 (hours), 0.909 (pos); structural eps_cost MDE=3.99; 85.7%% US hrs", em_dry_250$mde_upper, 2*em_dry_250$mde_upper),
      candidate == "R4a_bite_h2a_duration_sy" ~ sprintf("250-draw exact nonlinear CV3J: 1SD=%.3f / 2SD=%.3f (999-draw lin 0.256/0.512); State x Year FE eliminates SY shock; structural eps_cost MDE=3.99", em_dsy_250$mde_upper, 2*em_dsy_250$mde_upper),
      TRUE ~ notes
    ),
    us_hours_share = case_when(
      candidate %in% c("R4a_bite_h2a_duration_ry", "R4a_bite_h2a_duration_sy") ~ 0.857,
      TRUE ~ us_hours_share
    )
  )

# Assemble exact R2 rows from verified Phase 1 outputs (02_phase1_r2 log + 02b + 05b + 06 exact 250-draw)
t_r2 <- tribble(
  ~candidate, ~family, ~estimand, ~unit, ~n_obs, ~n_units, ~us_hours_share, ~G, ~G_eff, ~sd_x_tilde, ~r2_fe,
  ~first_stage_pi, ~first_stage_se, ~first_stage_F, ~sy_ms_ratio, ~n_placebo_draws, ~emp_size_alpha05, ~size_wilson_hi,
  ~mde_primary, ~mde_unit, ~mde_eps_cost, ~mde_eps_cost_r7, ~placebo_est, ~placebo_se, ~placebo_p, ~notes,

  "R2a_bite_direct_2013_2025_1b", "R2", "direct_grower_differential_bite_1b", "county_year_bite", 38168L, 2936L, 0.659, 17L, 14.13, 0.1387, 0.996, NA, NA, NA, 2.93, 999L, 0.044, 0.059, 0.719, "2SD_differential_bite_elasticity", NA, NA, NA, NA, 0.979, "1SD=0.359; WORSE than All 2013-25 (1SD=0.290/2SD=0.580)",
  "R2a_bite_direct_2013_2025_1a", "R2", "direct_grower_differential_bite_1a", "county_year_bite", 38168L, 2936L, 0.659, 17L, 11.10, 0.0248, 0.9999, NA, NA, NA, 2.90, 999L, 0.053, 0.069, 1.640, "1SD_1a_PPML_elasticity", NA, NA, NA, NA, 0.979, "+ E_p25 x Year FE; >1.00 gate",
  "R2a_bite_direct_2020_2025_1b", "R2", "direct_grower_post2020_bite_1b", "county_year_bite", nrow(d20_act), length(pos_dg20), 0.659, 17L, 12.88, 0.0813, 0.997, NA, NA, NA, 1.89, 250L, em_dg1b_250$emp_size, em_dg1b_250$wilson_hi, 2*em_dg1b_250$mde_upper, "2SD_differential_bite_elasticity", NA, NA, 0.023, 0.059, 0.706, sprintf("250-draw exact nonlinear CV3J: 1SD=%.3f / 2SD=%.3f (999-draw lin 0.231/0.462, size=0.050); All 2020-25 lin 0.235/0.469; 2013-19 pretrend p=0.706/0.979", em_dg1b_250$mde_upper, 2*em_dg1b_250$mde_upper),
  "R2a_bite_direct_2020_2025_1a", "R2", "direct_grower_post2020_bite_1a", "county_year_bite", nrow(d20_act), length(pos_dg20), 0.659, 17L, 7.29, 0.0175, 0.9999, NA, NA, NA, 1.93, 250L, em_dg1a_250$emp_size, em_dg1a_250$wilson_hi, em_dg1a_250$mde_upper, "1SD_1a_PPML_elasticity", NA, NA, 0.023, 0.059, 0.706, sprintf("250-draw exact nonlinear CV3J (+ E_p25 x Year FE): 1SD=%.3f (999-draw lin 0.712, size=0.053); beats 2011-25 1a-PPML (2.106) by >58%%; All 2020-25 lin 0.800", em_dg1a_250$mde_upper),
  "R2b_bite_cdl0810_x_yr",        "R2", "differential_bite_plus_cdl_x_yr", "county_year_bite", 44805L, 2987L, 1.000, 17L, 13.11, 0.1489, 0.9965, NA, NA, NA, 2.80, 999L, 0.063, 0.080, 0.653, "2SD_differential_bite_elasticity", NA, NA, NA, NA, 0.667, "1SD=0.326; no improvement over static 0.312/0.625",
  "R2b_bite_noaa_anomalies",      "R2", "differential_bite_plus_noaa", "county_year_bite", 44775L, 2985L, 1.000, 17L, 13.36, 0.1494, 0.9965, NA, NA, NA, 2.92, 999L, 0.061, 0.078, 0.618, "2SD_differential_bite_elasticity", NA, NA, NA, NA, 0.667, "1SD=0.309; 6 NOAA climate anomaly PCs (85% var); 1% change",
  "R2b_bite_cdl_plus_noaa",       "R2", "differential_bite_plus_cdl_noaa", "county_year_bite", 44775L, 2985L, 1.000, 17L, 13.12, 0.1488, 0.9965, NA, NA, NA, 2.87, 999L, 0.065, 0.082, 0.651, "2SD_differential_bite_elasticity", NA, NA, NA, NA, 0.667, "1SD=0.325; +Fisher(2011-24 sens) worsens to 0.370/0.740",
  "R2b_bite_1a_cdl_noaa",         "R2", "differential_bite_1a_plus_cdl_noaa", "county_year_bite", 44775L, 2985L, 1.000, 17L, 7.80, 0.0275, 0.9999, NA, NA, NA, 2.84, 999L, 0.036, 0.050, 1.581, "1SD_1a_PPML_elasticity", NA, NA, NA, NA, 0.667, "Worse than 1a baseline 1.528",
  "R2b_2025_agg_cdl2224",         "R2", "aggregate_hours_2025_plus_cdl", "county_fy", 5842L, 2921L, 0.785, 46L, 7.55, 0.800, 0.99, 0.6714, 0.0988, 46.18, 3.02, 500L, 0.024, 0.042, 0.844, "statutory_hours_elasticity", 1.595, 1.719, -0.058, 0.142, 0.685, "FY20-25 base=0.854 -> +CDL_2224=0.844 (-1.2%)",
  "R2b_2025_direct_cdl2224",      "R2", "direct_grower_hours_2025_plus_cdl", "county_fy_direct", 5842L, 2921L, 0.587, 46L, 6.86, 0.800, 0.99, 0.6714, 0.0988, 46.18, 1.86, 500L, 0.024, 0.042, 0.919, "direct_grower_hours_elasticity", 1.737, 1.872, 0.040, 0.184, 0.828, "FY20-25 DG base=0.862 -> +CDL_2224=0.919 (worse); SY 1.84->1.86",
  "R2c_2025_base_2yr_ppml",       "R2", "aggregate_hours_base_2yr_ppml", "state_fy_agg", 92L, 2921L, 0.785, 46L, 6.44, 0.806, 0.99, 0.6714, 0.0988, 46.18, 3.02, 700L, 0.031, 0.047, 0.932, "statutory_hours_elasticity", 1.763, 1.900, -0.034, 0.136, 0.803, "7 pre-policy target yrs (FY2019-25) benchmark",
  "R2c_2025_pre3yr_avg_ppml",     "R2", "aggregate_hours_pre3yr_avg_ppml", "state_fy_agg", 92L, 2921L, 0.785, 46L, 6.44, 0.806, 0.99, 0.6714, 0.0988, 46.18, 3.02, 700L, 0.049, 0.067, 1.280, "statutory_hours_elasticity", 2.419, 2.607, 0.020, 0.282, 0.944, "Worse (+37%): H-2A is near random walk",
  "R2c_2025_ancova_ppml",         "R2", "aggregate_hours_ancova_ppml", "state_fy_agg", 46L, 2921L, 0.785, 46L, 6.44, 0.806, 0.99, 0.6714, 0.0988, 46.18, 3.02, 700L, 0.023, 0.037, 0.852, "statutory_hours_elasticity", 1.610, 1.735, -0.019, 0.138, 0.891, "-8.6% vs base 0.932; >0.60 and eps_cost >1.20",
  "R2c_2025_ancova_wls",          "R2", "aggregate_hours_ancova_wls", "state_fy_agg", 46L, 2921L, 0.785, 46L, 6.44, 0.806, 0.99, 0.6714, 0.0988, 46.18, 3.02, 700L, 0.021, 0.035, 0.838, "statutory_hours_elasticity", 1.585, 1.708, -0.025, 0.143, 0.862, "-10.1% vs base 0.932; >0.60 and eps_cost >1.20",
  "R2c_2025_sdid_time_wls",       "R2", "aggregate_hours_sdid_time_wls", "state_fy_agg", 46L, 2921L, 0.785, 46L, 6.44, 0.806, 0.99, 0.6714, 0.0988, 46.18, 3.02, 700L, 0.049, 0.067, 1.062, "statutory_hours_elasticity", 2.007, 2.163, 0.008, 0.219, 0.971, "Worse (+14%) than 2-yr PPML",
  "R2c_2025_ife_r1_wls",          "R2", "aggregate_hours_ife_r1_wls", "state_fy_agg", 46L, 2921L, 0.785, 46L, 6.44, 0.806, 0.99, 0.6714, 0.0988, 46.18, 3.02, 700L, 0.029, 0.044, 0.929, "statutory_hours_elasticity", 1.756, 1.893, -0.035, 0.158, 0.825, "Pre-period SVD r=1 factor: -0.3% change",
  "R2c_2025_ife_r2_wls",          "R2", "aggregate_hours_ife_r2_wls", "state_fy_agg", 46L, 2921L, 0.785, 46L, 6.44, 0.806, 0.99, 0.6714, 0.0988, 46.18, 3.02, 700L, 0.023, 0.037, 0.948, "statutory_hours_elasticity", 1.792, 1.931, -0.056, 0.155, 0.719, "Pre-period SVD r=2 factors: +1.7% worse",
  "R2c_dg_base_2yr_ppml",         "R2", "direct_grower_base_2yr_ppml", "state_fy_direct", 92L, 2921L, 0.587, 46L, 6.86, 0.806, 0.99, 0.6714, 0.0988, 46.18, 1.84, 600L, 0.035, 0.053, 0.462, "direct_grower_hours_elasticity", 0.873, 0.941, 0.066, 0.193, 0.734, "FY2023-25 benchmark (3 yrs x 200 perms)",
  "R2c_dg_ancova_ppml",           "R2", "direct_grower_ancova_ppml", "state_fy_direct", 46L, 2921L, 0.587, 46L, 6.86, 0.806, 0.99, 0.6714, 0.0988, 46.18, 1.84, 600L, 0.040, 0.059, 0.478, "direct_grower_hours_elasticity", 0.903, 0.974, 0.080, 0.218, 0.716, "Worse (+3.5%) than base DG 2-yr PPML 0.462",
  "R2c_dg_ife_r1_wls",            "R2", "direct_grower_ife_r1_wls", "state_fy_direct", 46L, 2921L, 0.587, 46L, 6.86, 0.806, 0.99, 0.6714, 0.0988, 46.18, 1.84, 600L, 0.032, 0.049, 0.459, "direct_grower_hours_elasticity", 0.867, 0.935, 0.048, 0.188, 0.799, "Only -0.6% vs base DG 0.462; still >0.45 gate",
  "R2c_bite_ancova_lag_h",        "R2", "differential_bite_ancova_lr", "county_year_bite", 44805L, 2987L, 1.000, 17L, 13.29, 0.1473, 0.9965, NA, NA, NA, 1.81, 999L, 0.046, 0.061, 0.668, "2SD_long_run_differential_bite", NA, NA, NA, NA, 0.667, "FE-only rho=0.2695; SR 2SD=0.488 -> LR=0.488/0.7305=0.668 (WORSE than static 0.625) + Nickell bias",
  "R2c_bite_ife_r1_bai",          "R2", "differential_bite_bai_ife_r1", "county_year_bite", 44805L, 2987L, 1.000, 17L, 10.80, 0.0765, 0.999, NA, NA, NA, 1.55, 250L, 0.036, 0.067, 0.639, "2SD_differential_bite_elasticity", NA, NA, NA, NA, 0.667, "Exact iterative Bai r=1 IFE + LOO SVD CV3J: 1SD=0.319/2SD=0.639 (>0.60); SVD absorbs 44.8% noise AND 49% SD(X~)",
  "R2c_bite_ife_r2_refit",        "R2", "differential_bite_ife_r2", "county_year_bite", 44805L, 2987L, 1.000, 17L, 10.50, 0.0688, 0.999, NA, NA, NA, 1.38, 200L, 0.105, 0.155, 0.504, "2SD_differential_bite_elasticity", NA, NA, NA, NA, 0.667, "KILLED ON EMPIRICAL SIZE: honest refit size=0.105 [0.070,0.155] > 0.075 (overfits G=17)"
)

# Add production benchmark reference rows (strictly reference rows, family = "BENCHMARK")
t_bench <- tribble(
  ~candidate, ~family, ~estimand, ~unit, ~n_obs, ~n_units, ~us_hours_share, ~G, ~G_eff, ~sd_x_tilde, ~r2_fe,
  ~first_stage_pi, ~first_stage_se, ~first_stage_F, ~sy_ms_ratio, ~n_placebo_draws, ~emp_size_alpha05, ~size_wilson_hi,
  ~mde_primary, ~mde_unit, ~mde_eps_cost, ~mde_eps_cost_r7, ~placebo_est, ~placebo_se, ~placebo_p, ~notes,

  "BENCH_policy2025_track1_hours", "BENCHMARK", "aggregate_statutory_hours_elasticity", "county_fy", 5842L, 2921L, 0.785, 46L, 6.44, 0.806, 0.99, 0.6714, 0.0988, 46.18, 3.02, 650L, 0.022, 0.036, 0.724, "statutory_hours_elasticity", 1.398, 1.506, -0.033, 0.130, 0.803, "Production realized MDE=0.724 (blind FY12-25=1.016, FY20-25=0.854); direct ln(rate)=0.542",
  "BENCH_policy2025_direct_hours", "BENCHMARK", "direct_grower_statutory_hours_elasticity", "county_fy_direct", 5842L, 2921L, 0.587, 46L, 6.86, 0.806, 0.99, 0.6714, 0.0988, 46.18, 1.84, 600L, 0.027, 0.043, 0.520, "direct_grower_hours_elasticity", 0.983, 1.059, 0.064, 0.186, 0.733, "Production realized MDE=0.520 (positions=0.380; blind FY20-25=0.852, FY23-25=0.462)",
  "BENCH_bite_1b_ppml_hours",      "BENCHMARK", "differential_bite_1b_hours", "county_year_bite", 44805L, 2987L, 1.000, 17L, 13.36, 0.1496, 0.9965, NA, NA, NA, 2.85, 999L, 0.058, 0.074, 0.758, "2SD_differential_bite_elasticity", NA, NA, NA, NA, 0.667, "Production 1SD=0.379 / 2SD=0.758 (blind lin=0.312/0.625)",
  "BENCH_bite_1a_ppml_hours",      "BENCHMARK", "differential_bite_1a_hours", "county_year_bite", 44805L, 2987L, 1.000, 17L, 7.95, 0.0277, 0.9999, NA, NA, NA, 2.85, 999L, 0.049, 0.064, 2.106, "1SD_1a_PPML_elasticity", NA, NA, NA, NA, 0.667, "Production 1a-PPML (+E_p25 x Year) 1SD=2.106 (blind lin=1.528)"
)

# Combine ALL rows and apply the exact Phase 1 -> Phase 2 Promotion Gate programmatically!
triage <- bind_rows(t_bench, t_r1_r5, t_r2, t_r3, t_r4_r6) %>%
  mutate(
    # Gate 1: Size validity in [0.03, 0.075] (note: exact CV3J conservatism <0.03 noted separately if non-anti-conservative)
    size_pass = is.finite(emp_size_alpha05) & emp_size_alpha05 >= 0.030 & emp_size_alpha05 <= 0.075,
    # Gate 2: First-stage validity (where applicable: F >= 10 & |pi| >= 0.25)
    fs_applicable = is.finite(first_stage_F) | family %in% c("R1", "R5", "R6") | grepl("^R3b_", candidate),
    fs_pass = (!fs_applicable) | (is.finite(first_stage_F) & first_stage_F >= 10 & is.finite(first_stage_pi) & abs(first_stage_pi) >= 0.25),
    # Gate 3: Clean pre-policy placebo (p > 0.10; 0.05 < p <= 0.10 conditional)
    placebo_pass = is.na(placebo_p) | placebo_p > 0.10,
    # Gate 4: Material power improvement against exact plan thresholds
    power_pass = case_when(
      family == "BENCHMARK" ~ NA,
      mde_unit %in% c("statutory_eff_hours_elasticity", "ln_eff_hours_elasticity", "statutory_hours_elasticity", "truncation_hours_elasticity") ~
        is.finite(mde_primary) & mde_primary <= 0.60 & is.finite(mde_eps_cost) & mde_eps_cost <= 1.20,
      mde_unit %in% c("statutory_eff_positions_elasticity", "stacked_positions_ln_elasticity") ~
        is.finite(mde_primary) & mde_primary <= 0.44, # >=15% tighter than 0.52
      mde_unit %in% c("direct_grower_eff_hours_elasticity", "direct_grower_hours_elasticity", "direct_grower_hours_dhs_elasticity", "within_firm_reallocation_hours_elasticity") ~
        is.finite(mde_primary) & mde_primary <= 0.45,
      mde_unit %in% c("direct_grower_eff_positions_elasticity", "direct_grower_positions_elasticity", "direct_grower_positions_dhs_elasticity", "within_firm_reallocation_positions_elasticity", "within_firm_reallocation_cert_elasticity") ~
        is.finite(mde_primary) & mde_primary <= 0.35,
      mde_unit %in% c("2SD_differential_bite_elasticity", "2SD_long_run_differential_bite") ~
        is.finite(mde_primary) & mde_primary <= 0.60,
      mde_unit == "1SD_1a_PPML_elasticity" ~
        is.finite(mde_primary) & mde_primary <= 1.00,
      TRUE ~ FALSE
    ),
    # Special check for R6 <15% Fisher-info kill rule and R1c G=1 kill rule
    extra_kill = case_when(
      candidate == "R1c_substate_tiers_within_state" ~ "Only 1 state (NY) with sub-state tier var on Apr 1",
      candidate == "R6_stacked_raw_positions_ln" ~ "2009-10 Fisher-info share 13.5% < 15% & FS F=1.67-3.50",
      candidate == "R6_stacked_norm_positions_ln" ~ "2009-10 FS fails (F=1.67-3.50) & stacked MDE=0.620 > 0.52",
      candidate == "R2c_bite_ancova_lag_h" ~ "LR MDE 0.668 > 0.625 + Nickell/bad-control bias",
      TRUE ~ ""
    ),
    phase2_advance = (family != "BENCHMARK") & coalesce(size_pass, FALSE) & coalesce(fs_pass, FALSE) & coalesce(placebo_pass, FALSE) & coalesce(power_pass, FALSE) & (extra_kill == ""),
    classification = case_when(
      family == "BENCHMARK" ~ "BENCHMARK",
      phase2_advance & (
        (mde_unit == "2SD_differential_bite_elasticity" & mde_primary <= 0.45) |
        (mde_unit != "2SD_differential_bite_elasticity" & mde_unit != "1SD_1a_PPML_elasticity" & mde_primary <= 0.40)
      ) ~ "TIER1_ADVANCE",
      phase2_advance ~ "TIER2_ADVANCE",
      TRUE ~ "KILLED_PHASE1"
    )
  )

write_csv(triage, "explorations/bite_power_rescue/phase1_triage.csv")
cat(sprintf("\nWrote unified explorations/bite_power_rescue/phase1_triage.csv (%d rows)\n", nrow(triage)))

cat("\n=== PHASE 1 PROMOTION GATE SURVIVORS ===\n")
surv <- triage %>% filter(phase2_advance) %>%
  select(candidate, family, G, G_eff, emp_size_alpha05, mde_primary, mde_unit, placebo_p, classification)
print(surv, n = 20, width = 200)
