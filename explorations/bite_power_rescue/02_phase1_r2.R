# Phase 1 Strictly Outcome-Blind Audit: Candidate R2
# (R2a: Direct-Grower-Only 2011-2025 Bite;
#  R2b: Exogenous CDL Crop-Mix x Year & NOAA Weather Shock Absorption [Bite + 2025];
#  R2c: Lagged-Outcome ANCOVA, Multi-Year Pre-Avg, SDID Time-Weights, & IFE Factor Models [2025 + Bite])
# HARD RULE: NEVER reads or regresses FY2026 H-2A quantity outcomes!
# For 2011-2025 bite, uses FE-only residuals (strictly zero treatment slope inspection)
# plus pre-estimation 2008-2010/2011 pre-trend check only.

source("explorations/bite_power_rescue/helpers_phase1.R")

cat("======================================================================\n")
cat("CANDIDATE R2a: Direct-Grower-Only Bite (2013-2025 & 2020-2025)\n")
cat("======================================================================\n")

# Build direct-grower vs all-employer county x start-year hours & positions from cached worksites + disclosures.
# Strictly filter to start_year <= 2025 (ZERO FY2026 quantities!).
bp_full <- read_parquet(path_processed("bite_county_year_panel.parquet")) %>%
  mutate(year = as.integer(year))
bp_est <- bp_full %>% filter(year >= 2011L, year <= 2025L)

# Check H-2ALC indicator in disclosure records:
# - FY2020-FY2025: h2a_labor_contractor == "Y" (or type_of_employer_application == "H-2A Labor Contractor...")
# - FY2013-FY2019: organization_flag == "H-2A Labor Contractor or Job Contractor"
disc_h2alc <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, case_number, h2a_labor_contractor, organization_flag, type_of_employer_application) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2026L) %>%
  collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(
    lc_raw = first(na.omit(as.character(h2a_labor_contractor))),
    org_raw = first(na.omit(as.character(organization_flag))),
    typ_raw = first(na.omit(as.character(type_of_employer_application))),
    .groups = "drop"
  ) %>%
  mutate(
    lc_up = toupper(trimws(coalesce(lc_raw, ""))),
    org_up = toupper(trimws(coalesce(org_raw, ""))),
    typ_up = toupper(trimws(coalesce(typ_raw, ""))),
    is_h2alc = (lc_up %in% c("Y", "YES", "TRUE", "1")) |
      grepl("LABOR CONTRACTOR|JOB CONTRACTOR", org_up) |
      grepl("LABOR CONTRACTOR|JOB CONTRACTOR", typ_up)
  ) %>%
  select(case_number, fiscal_year, is_h2alc)

nh_pre_mean <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, period_begin_date, number_of_hours) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2025L) %>%
  collect() %>%
  filter(!is.na(as.Date(period_begin_date))) %>%
  summarise(m = mean(suppressWarnings(as.numeric(number_of_hours)), na.rm = TRUE)) %>%
  pull(m)

ws_sy <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date, number_of_hours) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2026L) %>%
  collect() %>%
  mutate(
    fiscal_year = as.integer(fiscal_year),
    begin = as.Date(period_begin_date),
    end = as.Date(period_end_date),
    start_year = as.integer(format(begin, "%Y"))
  ) %>%
  filter(!is.na(begin), !is.na(end), start_year >= 2013L, start_year <= 2025L) %>%
  left_join(disc_h2alc, by = c("case_number", "fiscal_year")) %>%
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
    all_hours_sy = sum(hours * share, na.rm = TRUE),
    direct_hours_sy = sum(hours * share * (!is_h2alc), na.rm = TRUE),
    h2alc_hours_sy = sum(hours * share * is_h2alc, na.rm = TRUE),
    all_workers_sy = sum(workers * share, na.rm = TRUE),
    direct_workers_sy = sum(workers * share * (!is_h2alc), na.rm = TRUE),
    .groups = "drop"
  )

bp_1325 <- bp_est %>%
  filter(year >= 2013L) %>%
  left_join(ws_sy, by = c("county_fips", "year")) %>%
  mutate(across(c(all_hours_sy, direct_hours_sy, h2alc_hours_sy, all_workers_sy, direct_workers_sy), ~ coalesce(.x, 0)))

# Print annual direct-grower share diagnostic (strictly pre-outcome descriptive check)
dg_share_yr <- bp_1325 %>%
  group_by(year) %>%
  summarise(
    tot_h = sum(all_hours_sy),
    dg_share = sum(direct_hours_sy) / sum(all_hours_sy),
    h2alc_share = sum(h2alc_hours_sy) / sum(all_hours_sy),
    .groups = "drop"
  )
cat("Start-year Direct-Grower vs H-2ALC hours shares (2013-2025):\n")
print(dg_share_yr, n = 15)

# Run strictly outcome-blind FE-only calibration for R2a (2013-2025 and 2020-2025)
cal_r2a_all13_1b <- calibrate_multiperiod_ppml_blind(bp_1325, "all_hours_sy", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year")
cal_r2a_dg13_1b  <- calibrate_multiperiod_ppml_blind(bp_1325, "direct_hours_sy", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year")
cal_r2a_dg13_w   <- calibrate_multiperiod_ppml_blind(bp_1325, "direct_workers_sy", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year")
cal_r2a_dg13_1a  <- calibrate_multiperiod_ppml_blind(bp_1325, "direct_hours_sy", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year", ctrl_rhs = "i(year, E_p25, ref = 2013)")
cal_r2a_dg20_1b  <- calibrate_multiperiod_ppml_blind(bp_1325 %>% filter(year >= 2020L), "direct_hours_sy", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year")
cal_r2a_all20_1b <- calibrate_multiperiod_ppml_blind(bp_1325 %>% filter(year >= 2020L), "all_hours_sy", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year")
cal_r2a_dg20_1a  <- calibrate_multiperiod_ppml_blind(bp_1325 %>% filter(year >= 2020L), "direct_hours_sy", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year", ctrl_rhs = "i(year, E_p25, ref = 2020)")

cat(sprintf("R2a Bench All (2013-25) 1b: G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r2a_all13_1b$g_eff, cal_r2a_all13_1b$sd_x_tilde_wt, cal_r2a_all13_1b$sy_ms_ratio, cal_r2a_all13_1b$emp_size, cal_r2a_all13_1b$mde_1sd, cal_r2a_all13_1b$mde_2sd))
cat(sprintf("R2a Direct    (2013-25) 1b: G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r2a_dg13_1b$g_eff, cal_r2a_dg13_1b$sd_x_tilde_wt, cal_r2a_dg13_1b$sy_ms_ratio, cal_r2a_dg13_1b$emp_size, cal_r2a_dg13_1b$mde_1sd, cal_r2a_dg13_1b$mde_2sd))
cat(sprintf("R2a DirectPos (2013-25) 1b: G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r2a_dg13_w$g_eff, cal_r2a_dg13_w$sd_x_tilde_wt, cal_r2a_dg13_w$sy_ms_ratio, cal_r2a_dg13_w$emp_size, cal_r2a_dg13_w$mde_1sd, cal_r2a_dg13_w$mde_2sd))
cat(sprintf("R2a Direct    (2013-25) 1a: G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r2a_dg13_1a$g_eff, cal_r2a_dg13_1a$sd_x_tilde_wt, cal_r2a_dg13_1a$sy_ms_ratio, cal_r2a_dg13_1a$emp_size, cal_r2a_dg13_1a$mde_1sd, cal_r2a_dg13_1a$mde_2sd))
cat(sprintf("R2a Bench All (2020-25) 1b: G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r2a_all20_1b$g_eff, cal_r2a_all20_1b$sd_x_tilde_wt, cal_r2a_all20_1b$sy_ms_ratio, cal_r2a_all20_1b$emp_size, cal_r2a_all20_1b$mde_1sd, cal_r2a_all20_1b$mde_2sd))
cat(sprintf("R2a Direct    (2020-25) 1b: G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r2a_dg20_1b$g_eff, cal_r2a_dg20_1b$sd_x_tilde_wt, cal_r2a_dg20_1b$sy_ms_ratio, cal_r2a_dg20_1b$emp_size, cal_r2a_dg20_1b$mde_1sd, cal_r2a_dg20_1b$mde_2sd))
cat(sprintf("R2a Direct    (2020-25) 1a: G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r2a_dg20_1a$g_eff, cal_r2a_dg20_1a$sd_x_tilde_wt, cal_r2a_dg20_1a$sy_ms_ratio, cal_r2a_dg20_1a$emp_size, cal_r2a_dg20_1a$mde_1sd, cal_r2a_dg20_1a$mde_2sd))

cat("\n======================================================================\n")
cat("CANDIDATE R2b: Exogenous CDL Crop-Mix x Year & NOAA Climate Anomalies\n")
cat("======================================================================\n")

# 1. Predetermined 2008-2010 CDL crop shares (for 2011-2025 bite) & 2022-2024 CDL crop shares (for 2025 rule)
cdl_raw <- read_parquet(path_int("croplandcros_county_crop_type_acres.parquet")) %>%
  mutate(
    county_fips = harmonize_county_fips_2010(county_fips),
    year = as.integer(year),
    crop_grp = tolower(trimws(as.character(crop_type_label))),
    acres = as.numeric(acres)
  ) %>%
  filter(crop_grp != "non-crop")

build_cdl_shares <- function(yrs, suffix) {
  cdl_raw %>%
    filter(year %in% yrs) %>%
    group_by(county_fips) %>%
    summarise(
      tot = sum(acres, na.rm = TRUE),
      fruit = sum(acres[crop_grp == "fruit & tree nuts"], na.rm = TRUE),
      veg = sum(acres[crop_grp == "vegetables"], na.rm = TRUE),
      field = sum(acres[crop_grp == "field crops"], na.rm = TRUE),
      hort = sum(acres[crop_grp == "horticulture"], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    transmute(
      county_fips,
      !!paste0("cdl_fruit_", suffix) := ifelse(tot > 0, fruit / tot, 0),
      !!paste0("cdl_veg_", suffix)   := ifelse(tot > 0, veg / tot, 0),
      !!paste0("cdl_field_", suffix) := ifelse(tot > 0, field / tot, 0),
      !!paste0("cdl_hort_", suffix)  := ifelse(tot > 0, hort / tot, 0)
    )
}

cdl_0810 <- build_cdl_shares(2008L:2010L, "0810")
cdl_2224 <- build_cdl_shares(2022L:2024L, "2224")

# 2. NOAA annual climate anomalies (2008-2025): 14 annual spline anomalies (cb - normal_cb)
#    summarized into top 6 orthogonal PCs (explaining >90% climate variance without collinearity)
noaa_cols_base <- c(paste0("cb_temp_tavg_b0", 0:7), paste0("cb_prcp_log1p_b0", 0:5))
noaa_cols_norm <- paste0("normal_", noaa_cols_base)
noaa_raw <- read_parquet(
  path_int("county_h2a_prediction_climate_basis_annual.parquet"),
  col_select = all_of(c("county_fips", "year", noaa_cols_base, noaa_cols_norm))
) %>%
  filter(year >= 2008L, year <= 2025L) %>%
  mutate(county_fips = harmonize_county_fips_2010(county_fips), year = as.integer(year))

anom_mat <- as.matrix(noaa_raw[, noaa_cols_base]) - as.matrix(noaa_raw[, noaa_cols_norm])
pca_noaa <- stats::prcomp(anom_mat, center = TRUE, scale. = TRUE)
cat(sprintf("NOAA 14-spline anomaly PCA: top 6 PCs explain %.1f%% variance\n",
  100 * sum(pca_noaa$sdev[1:6]^2) / sum(pca_noaa$sdev^2)))
noaa_pcs <- tibble(
  county_fips = noaa_raw$county_fips,
  year = noaa_raw$year,
  noaa_pc1 = pca_noaa$x[, 1],
  noaa_pc2 = pca_noaa$x[, 2],
  noaa_pc3 = pca_noaa$x[, 3],
  noaa_pc4 = pca_noaa$x[, 4],
  noaa_pc5 = pca_noaa$x[, 5],
  noaa_pc6 = pca_noaa$x[, 6]
)

bp_r2b <- bp_est %>%
  left_join(cdl_0810, by = "county_fips") %>%
  left_join(noaa_pcs, by = c("county_fips", "year")) %>%
  left_join(ws_sy %>% select(county_fips, year, direct_hours_sy), by = c("county_fips", "year")) %>%
  mutate(direct_hours_sy = coalesce(direct_hours_sy, 0))

ctrl_cdl_bite <- "i(year, cdl_fruit_0810, ref = 2011) + i(year, cdl_veg_0810, ref = 2011) + i(year, cdl_field_0810, ref = 2011) + i(year, cdl_hort_0810, ref = 2011)"
ctrl_noaa_bite <- "noaa_pc1 + noaa_pc2 + noaa_pc3 + noaa_pc4 + noaa_pc5 + noaa_pc6"
ctrl_cdl_noaa_bite <- paste(ctrl_cdl_bite, "+", ctrl_noaa_bite)
ctrl_cdl_noaa_fisher <- paste(ctrl_cdl_noaa_bite, "+ fisher_index_ppi + fisher_quantity_index")

# Blind calibration on 2011-2025 Bite (All Hours & Direct-Grower 2013-2025)
cal_r2b_cdl      <- calibrate_multiperiod_ppml_blind(bp_r2b, "man_hours_certified_start_year", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year", ctrl_rhs = ctrl_cdl_bite)
cal_r2b_noaa     <- calibrate_multiperiod_ppml_blind(bp_r2b, "man_hours_certified_start_year", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year", ctrl_rhs = ctrl_noaa_bite)
cal_r2b_cdl_noaa <- calibrate_multiperiod_ppml_blind(bp_r2b, "man_hours_certified_start_year", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year", ctrl_rhs = ctrl_cdl_noaa_bite)
cal_r2b_fisher   <- calibrate_multiperiod_ppml_blind(bp_r2b %>% filter(year <= 2024L), "man_hours_certified_start_year", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year", ctrl_rhs = ctrl_cdl_noaa_fisher)
cal_r2b_dg_cn    <- calibrate_multiperiod_ppml_blind(
  bp_r2b %>% filter(year >= 2013L), "direct_hours_sy", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year",
  ctrl_rhs = "i(year, cdl_fruit_0810, ref = 2013) + i(year, cdl_veg_0810, ref = 2013) + i(year, cdl_field_0810, ref = 2013) + i(year, cdl_hort_0810, ref = 2013) + noaa_pc1 + noaa_pc2 + noaa_pc3 + noaa_pc4 + noaa_pc5 + noaa_pc6"
)
cal_r2b_1a_cn    <- calibrate_multiperiod_ppml_blind(
  bp_r2b, "man_hours_certified_start_year", "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year",
  ctrl_rhs = paste("i(year, E_p25, ref = 2011) +", ctrl_cdl_noaa_bite)
)

cat(sprintf("R2b Bite + CDL0810xYr          : G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r2b_cdl$g_eff, cal_r2b_cdl$sd_x_tilde_wt, cal_r2b_cdl$sy_ms_ratio, cal_r2b_cdl$emp_size, cal_r2b_cdl$mde_1sd, cal_r2b_cdl$mde_2sd))
cat(sprintf("R2b Bite + NOAA PCs            : G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r2b_noaa$g_eff, cal_r2b_noaa$sd_x_tilde_wt, cal_r2b_noaa$sy_ms_ratio, cal_r2b_noaa$emp_size, cal_r2b_noaa$mde_1sd, cal_r2b_noaa$mde_2sd))
cat(sprintf("R2b Bite + CDL+NOAA            : G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r2b_cdl_noaa$g_eff, cal_r2b_cdl_noaa$sd_x_tilde_wt, cal_r2b_cdl_noaa$sy_ms_ratio, cal_r2b_cdl_noaa$emp_size, cal_r2b_cdl_noaa$mde_1sd, cal_r2b_cdl_noaa$mde_2sd))
cat(sprintf("R2b Bite + CDL+NOAA+Fisher(sens): G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r2b_fisher$g_eff, cal_r2b_fisher$sd_x_tilde_wt, cal_r2b_fisher$sy_ms_ratio, cal_r2b_fisher$emp_size, cal_r2b_fisher$mde_1sd, cal_r2b_fisher$mde_2sd))
cat(sprintf("R2b Direct(13-25)+CDL+NOAA     : G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r2b_dg_cn$g_eff, cal_r2b_dg_cn$sd_x_tilde_wt, cal_r2b_dg_cn$sy_ms_ratio, cal_r2b_dg_cn$emp_size, cal_r2b_dg_cn$mde_1sd, cal_r2b_dg_cn$mde_2sd))
cat(sprintf("R2b Bite 1a + CDL+NOAA         : G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f\n",
  cal_r2b_1a_cn$g_eff, cal_r2b_1a_cn$sd_x_tilde_wt, cal_r2b_1a_cn$sy_ms_ratio, cal_r2b_1a_cn$emp_size, cal_r2b_1a_cn$mde_1sd))

# --- R2b on 2025 Rule (strictly pre-policy FY2012-FY2025 & FY2020-FY2025!) ---
cat("\n--- R2b on 2025 Rule: CDL_2224 x Post (+ Direct-Grower) ---\n")
qp_pre <- read_parquet(path_processed("policy_2025", "quantity_county_fy.parquet")) %>%
  mutate(year = as.integer(year)) %>%
  filter(year <= 2025L) %>%
  left_join(cdl_2224, by = "county_fips")

st_sched <- read_parquet(
  path_processed("policy_2025", "quantity_county_fy.parquet"),
  col_select = c("state_fips", "aewr_region_id", "year", "x_us_skill_1", "mw_ag")
) %>%
  filter(year %in% c(2025L, 2026L)) %>%
  distinct() %>%
  mutate(x_eff = pmax(x_us_skill_1, mw_ag)) %>%
  pivot_wider(names_from = year, values_from = c(x_us_skill_1, mw_ag, x_eff)) %>%
  mutate(dose_raw = x_us_skill_1_2026 - x_us_skill_1_2025, dose_eff = x_eff_2026 - x_eff_2025) %>%
  arrange(state_fips)
mean_p_raw <- mean(c(st_sched$x_us_skill_1_2025, st_sched$x_us_skill_1_2026))
mean_p_eff <- mean(c(st_sched$x_eff_2025, st_sched$x_eff_2026))

# Also build county-level direct-grower FY2020-FY2025 panel (strictly <= 2025!)
disc_fy2025 <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, case_number, decision_date, h2a_labor_contractor) %>%
  filter(fiscal_year >= 2020L, fiscal_year <= 2025L) %>%
  collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(
    dec = as.Date(substr(first(na.omit(as.character(decision_date))), 1L, 10L)),
    lc = first(na.omit(as.character(h2a_labor_contractor))),
    .groups = "drop"
  ) %>%
  filter(!is.na(dec), dec >= as.Date(sprintf("%d-10-01", fiscal_year - 1L)), dec <= as.Date(sprintf("%d-09-30", fiscal_year))) %>%
  mutate(is_h2alc = toupper(trimws(coalesce(lc, ""))) %in% c("Y", "YES", "TRUE", "1"))

counties_t1 <- qp_pre %>% filter(year == 2025L) %>% select(county_fips, state_fips, aewr_region_id, cdl_fruit_2224, cdl_veg_2224, cdl_field_2224, cdl_hort_2224)

ws_dg_cty <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date, number_of_hours) %>%
  filter(fiscal_year >= 2020L, fiscal_year <= 2025L) %>%
  collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year), begin = as.Date(period_begin_date), end = as.Date(period_end_date)) %>%
  filter(!is.na(begin)) %>%
  inner_join(disc_fy2025 %>% select(case_number, fiscal_year, is_h2alc), by = c("case_number", "fiscal_year")) %>%
  filter(!is_h2alc) %>%
  mutate(
    nh = coalesce(suppressWarnings(as.numeric(number_of_hours)), nh_pre_mean),
    workers = as.numeric(adjusted_nbr_workers_certified),
    hours = as.numeric(end - begin) / 7 * nh * workers,
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    nc = stringr::str_count(county_fips_list, ",") + 1L
  ) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), share = 1 / nc) %>%
  semi_join(counties_t1, by = "county_fips") %>%
  group_by(county_fips, year = fiscal_year) %>%
  summarise(dg_hours = sum(hours * share, na.rm = TRUE), dg_cert = sum(workers * share, na.rm = TRUE), .groups = "drop")

dg_cty_panel <- tidyr::crossing(counties_t1, year = 2020L:2025L) %>%
  left_join(ws_dg_cty, by = c("county_fips", "year")) %>%
  mutate(dg_hours = coalesce(dg_hours, 0), dg_cert = coalesce(dg_cert, 0))

# Check SY shared-shock ratio on direct-grower FY2020-25 without vs with CDL x Year
m0_dg_base <- fixest::fepois(dg_hours ~ 1 | county_fips + aewr_region_id^year, data = dg_cty_panel, notes = FALSE)
u_dg_b <- dg_cty_panel[fixest::obs(m0_dg_base), ]
sy_dg_base <- compute_sy_noise_ratio_ppml(u_dg_b$dg_hours, fitted(m0_dg_base), u_dg_b$aewr_region_id, u_dg_b$state_fips, u_dg_b$year)

m0_dg_cdl <- fixest::fepois(dg_hours ~ i(year, cdl_fruit_2224, ref = 2020) + i(year, cdl_veg_2224, ref = 2020) +
  i(year, cdl_field_2224, ref = 2020) + i(year, cdl_hort_2224, ref = 2020) | county_fips + aewr_region_id^year,
  data = dg_cty_panel, notes = FALSE)
u_dg_c <- dg_cty_panel[fixest::obs(m0_dg_cdl), ]
sy_dg_cdl <- compute_sy_noise_ratio_ppml(u_dg_c$dg_hours, fitted(m0_dg_cdl), u_dg_c$aewr_region_id, u_dg_c$state_fips, u_dg_c$year)
cat(sprintf("Direct-grower SY ratio (FY2020-25): base=%.2f -> +CDL_2224xYr=%.2f\n", sy_dg_base$ms_ratio, sy_dg_cdl$ms_ratio))

# Calibrate R2b on 2025 rule (Agg hours FY2020-25 & FY2012-25 + Direct-grower FY2020-25) with CDL_2224 x Post (500 draws!)
run_2025_cdl_cal <- function(panel_df, y_col, pairs, B_per_yr, use_eff = FALSE) {
  set.seed(SEED_PHASE1 + 20L)
  n_d <- length(pairs) * B_per_yr
  b_base <- se_base <- b_cdl <- se_cdl <- numeric(n_d)
  mp <- if (use_eff) mean_p_eff else mean_p_raw
  idx <- 0L
  for (yr in pairs) {
    w <- panel_df %>%
      filter(year %in% c(yr - 1L, yr)) %>%
      select(county_fips, state_fips, aewr_region_id, cdl_fruit_2224, cdl_veg_2224, cdl_field_2224, cdl_hort_2224, year, y = all_of(y_col)) %>%
      pivot_wider(names_from = year, values_from = y, names_prefix = "y_")
    y0 <- w[[paste0("y_", yr - 1L)]]; y1 <- w[[paste0("y_", yr)]]
    Z_cdl <- as.matrix(w[, c("cdl_fruit_2224", "cdl_veg_2224", "cdl_field_2224", "cdl_hort_2224")])
    # Collapse to state for fast base comparison
    w_st <- w %>% group_by(state_fips, aewr_region_id) %>% summarise(y0 = sum(.data[[paste0("y_", yr - 1L)]]), y1 = sum(.data[[paste0("y_", yr)]]), .groups = "drop")
    ord_st <- match(st_sched$state_fips, w_st$state_fips)
    for (k in seq_len(B_per_yr)) {
      idx <- idx + 1L
      perm <- st_sched %>% group_by(aewr_region_id) %>% mutate(po = sample.int(n()), d = if (use_eff) dose_eff[po] else dose_raw[po]) %>% ungroup()
      dx_cty <- perm$d[match(w$state_fips, perm$state_fips)]
      rb <- fit_2period_ppml_cv3j(w_st$y0[ord_st], w_st$y1[ord_st], perm$d, st_sched$aewr_region_id, st_sched$state_fips)
      rc <- fit_2period_ppml_cv3j(y0, y1, dx_cty, w$aewr_region_id, w$state_fips, Z = Z_cdl)
      b_base[[idx]] <- rb$beta * mp; se_base[[idx]] <- rb$se_cv3j * mp
      b_cdl[[idx]]  <- rc$beta * mp; se_cdl[[idx]]  <- rc$se_cv3j * mp
    }
  }
  # Also matched FY2024-25 placebo + support
  w2425 <- panel_df %>%
    filter(year %in% c(2024L, 2025L)) %>%
    select(county_fips, state_fips, aewr_region_id, cdl_fruit_2224, cdl_veg_2224, cdl_field_2224, cdl_hort_2224, year, y = all_of(y_col)) %>%
    pivot_wider(names_from = year, values_from = y, names_prefix = "y_")
  dx2425 <- (if (use_eff) st_sched$dose_eff else st_sched$dose_raw)[match(w2425$state_fips, st_sched$state_fips)]
  Z2425 <- as.matrix(w2425[, c("cdl_fruit_2224", "cdl_veg_2224", "cdl_field_2224", "cdl_hort_2224")])
  sup <- compute_2period_blind_support(w2425$y_2024, w2425$y_2025, dx2425, w2425$aewr_region_id, w2425$state_fips, Z = Z2425)
  pl <- fit_2period_ppml_cv3j(w2425$y_2024, w2425$y_2025, dx2425, w2425$aewr_region_id, w2425$state_fips, Z = Z2425)
  list(
    base = compute_empirical_mde(b_base, se_base, 45L),
    cdl  = compute_empirical_mde(b_cdl, se_cdl, 45L),
    sup = sup,
    pl_est = pl$beta * mp, pl_se = pl$se_cv3j * mp, pl_p = 2 * pt(-abs(pl$beta / pl$se_cv3j), 45L)
  )
}

# 500 draws each (5 transitions FY2020-25 x 100 perms = 500 draws!)
r2b_25_agg <- run_2025_cdl_cal(qp_pre, "hours_full_fy", 2021L:2025L, 100L, use_eff = FALSE)
r2b_25_dg  <- run_2025_cdl_cal(dg_cty_panel, "dg_hours", 2021L:2025L, 100L, use_eff = FALSE)
r2b_25_dg_eff <- run_2025_cdl_cal(dg_cty_panel, "dg_hours", 2021L:2025L, 100L, use_eff = TRUE)

cat(sprintf("R2b 2025 Agg Hours (FY20-25, 500 draws): Base MDE=%.3f (size=%.3f) -> +CDL MDE=%.3f (size=%.3f), G_eff=%.2f, pl2425=%.3f (p=%.3f)\n",
  r2b_25_agg$base$mde_upper, r2b_25_agg$base$emp_size, r2b_25_agg$cdl$mde_upper, r2b_25_agg$cdl$emp_size,
  r2b_25_agg$sup$g_eff, r2b_25_agg$pl_est, r2b_25_agg$pl_p))
cat(sprintf("R2b 2025 DG Hours  (FY20-25, 500 draws): Base MDE=%.3f (size=%.3f) -> +CDL MDE=%.3f (size=%.3f), G_eff=%.2f, pl2425=%.3f (p=%.3f)\n",
  r2b_25_dg$base$mde_upper, r2b_25_dg$base$emp_size, r2b_25_dg$cdl$mde_upper, r2b_25_dg$cdl$emp_size,
  r2b_25_dg$sup$g_eff, r2b_25_dg$pl_est, r2b_25_dg$pl_p))
cat(sprintf("R2b 2025 DG EffFlr (FY20-25, 500 draws): Base MDE=%.3f (size=%.3f) -> +CDL MDE=%.3f (size=%.3f), G_eff=%.2f, pl2425=%.3f (p=%.3f)\n",
  r2b_25_dg_eff$base$mde_upper, r2b_25_dg_eff$base$emp_size, r2b_25_dg_eff$cdl$mde_upper, r2b_25_dg_eff$cdl$emp_size,
  r2b_25_dg_eff$sup$g_eff, r2b_25_dg_eff$pl_est, r2b_25_dg_eff$pl_p))

cat("\n======================================================================\n")
cat("CANDIDATE R2c: ANCOVA, Multi-Year Pre-Avg, SDID Time-Weights & IFE\n")
cat("======================================================================\n")

# Test on 46-state pre-policy panel (FY2012-FY2025) for target years t in 2019..2025 (7 target years x 100 perms = 700 draws!)
# Notice: by exact Poisson aggregation identity, state-level totals give exact PPML for state-level dose!
st_panel <- qp_pre %>%
  group_by(state_fips, aewr_region_id, year) %>%
  summarise(hours = sum(hours_full_fy), cert = sum(cert_full_fy), .groups = "drop") %>%
  arrange(state_fips, year)

# Helper: Arkhangelsky et al. (2021) SDID ridge-simplex time weights on pre-period log outcomes (S x T_pre)
# Minimizes || Y_pre[, 1:(T-1)] %*% lambda + lambda0 - Y_pre[, T] ||_2^2 + zeta^2 * ||lambda||_2^2
# subject to lambda >= 0, sum(lambda) = 1.
solve_sdid_time_weights <- function(Y_pre_mat, weights_s = NULL) {
  S <- nrow(Y_pre_mat); Tm1 <- ncol(Y_pre_mat) - 1L
  if (is.null(weights_s)) weights_s <- rep(1, S)
  sw <- sqrt(weights_s / sum(weights_s))
  A <- Y_pre_mat[, seq_len(Tm1), drop = FALSE]
  b <- Y_pre_mat[, Tm1 + 1L]
  # Center each column by weighted mean to profile out intercept lambda0 exactly
  w_norm <- weights_s / sum(weights_s)
  A_c <- sweep(A, 2L, colSums(A * w_norm), "-")
  b_c <- b - sum(b * w_norm)
  # Regularization scale zeta from pre-period first differences (Arkhangelsky et al.)
  diffs <- Y_pre_mat[, -1L, drop = FALSE] - Y_pre_mat[, -ncol(Y_pre_mat), drop = FALSE]
  zeta <- max(stats::sd(as.vector(diffs)), 1e-4) * (S^(1/4)) / sqrt(S)

  # Projected gradient descent onto simplex (exact convex QP, converges in <200 iters)
  proj_simplex <- function(v) {
    u <- sort(v, decreasing = TRUE)
    cssv <- cumsum(u) - 1
    ind <- seq_along(v)
    cond <- u - cssv / ind > 0
    rho <- max(which(cond))
    theta <- cssv[rho] / rho
    pmax(v - theta, 0)
  }
  AtA <- crossprod(A_c * sw) + diag(zeta^2, Tm1)
  Atb <- drop(crossprod(A_c * sw, b_c * sw))
  L <- max(eigen(AtA, symmetric = TRUE, only.values = TRUE)$values)
  lam <- rep(1 / Tm1, Tm1)
  for (it in seq_len(400L)) {
    grad <- as.vector(AtA %*% lam - Atb)
    lam_new <- proj_simplex(lam - grad / max(L, 1e-8))
    if (max(abs(lam_new - lam)) < 1e-9) { lam <- lam_new; break }
    lam <- lam_new
  }
  lam
}

# Helper: Ultra-fast vectorized state WLS + exact 46-state CV3J
fit_state_wls_cv3j <- function(y, dx, reg, st, w, Z = NULL) {
  fit_wls_fe_cv3j_fast(y = y, dx = dx, grp = reg, st = st, w = w, Z = Z)
}

# Evaluate 6 R2c estimators across target years 2019..2025 (7 target years x 100 perms = 700 draws!):
#   (1) Base 2-yr PPML (t-1 vs t) [reference]
#   (2) Multi-year Pre-Average PPML (mean(t-3..t-1) vs t)
#   (3) ANCOVA PPML (Y_t ~ dose + ln(Y_{t-1}) + ln(Y_{t-2}) | region)
#   (4) ANCOVA WLS (ln(Y_t) ~ dose + ln(Y_{t-1}) + ln(Y_{t-2}) | region, weights = Y_{t-1})
#   (5) SDID Time-Weighted WLS (ln(Y_t) - sum_{tau} lam_tau ln(Y_tau) ~ dose | region, weights = Y_{t-1})
#   (6) IFE r=1 & r=2 Factor-Loading WLS (ln(Y_t/Y_{t-1}) ~ dose + pre-SVD loadings | region, weights = Y_{t-1})
#       Note: To avoid any post-period leakage, SVD loadings are extracted STRICTLY from pre-periods [t-7..t-1]!
r2c_targets <- 2019L:2025L
B_r2c <- 100L
n_r2c <- length(r2c_targets) * B_r2c
methods_r2c <- c("base_2yr_ppml", "pre3yr_avg_ppml", "ancova_ppml", "ancova_wls", "sdid_time_wls", "ife_r1_wls", "ife_r2_wls")
B_mat <- matrix(NA_real_, nrow = n_r2c, ncol = length(methods_r2c), dimnames = list(NULL, methods_r2c))
S_mat <- matrix(NA_real_, nrow = n_r2c, ncol = length(methods_r2c), dimnames = list(NULL, methods_r2c))

# Also store matched FY2024-25 (t = 2025) placebos using actual 2025-rule dose!
pl2425_r2c <- list()

set.seed(SEED_PHASE1 + 30L)
idx <- 0L
for (yr in r2c_targets) {
  pre_yrs <- (yr - 7L):(yr - 1L)
  sub <- st_panel %>% filter(year %in% c(pre_yrs, yr))
  # Build S x 8 matrix aligned to st_sched$state_fips
  Y_raw <- matrix(0, nrow = nrow(st_sched), ncol = 8L)
  for (j in seq_len(8L)) {
    yy <- c(pre_yrs, yr)[[j]]
    tmp <- sub %>% filter(year == yy)
    Y_raw[, j] <- tmp$hours[match(st_sched$state_fips, tmp$state_fips)]
  }
  Y_log <- log(pmax(Y_raw, 1))
  w_base <- pmax(Y_raw[, 7L], 1) # t-1 hours weights (predetermined!)
  y_t   <- Y_raw[, 8L]
  y_tm1 <- Y_raw[, 7L]
  y_tm2 <- Y_raw[, 6L]
  y_pre3 <- rowMeans(Y_raw[, 5L:7L])

  # SDID time weights strictly on pre-periods 1..7 (matching cols 1..6 to col 7!)
  lam_sdid <- solve_sdid_time_weights(Y_log[, 1L:7L], weights_s = w_base)
  # Combine: SDID counterfactual for t using pre-periods 1..7 -> predict t-1 from 1..6, or shift to predict t from 2..7!
  # Standard Arkhangelsky et al.: solve lam on pre-periods 1..(T_pre-1) predicting T_pre?
  # Actually standard SDID solves min || sum_{t<=T0} lam_t Y_{it} - Y_{i,post} || for control units;
  # with continuous treatment across all 46 states, strictly pre-period time weights solve for T0 = t-1
  # using t-7..t-2 (applied to t-6..t-1 to forecast t!), OR use optimal AR weights on pre-period!
  y_sdid_diff <- Y_log[, 8L] - as.vector(Y_log[, 2L:7L] %*% lam_sdid)

  # Pre-period SVD interactive factors (strictly on pre_yrs 1..7, double-demeaned by state + region x year!)
  pre_df <- expand.grid(s_idx = seq_len(nrow(st_sched)), t_idx = 1L:7L)
  pre_df$st <- st_sched$state_fips[pre_df$s_idx]
  pre_df$reg <- st_sched$aewr_region_id[pre_df$s_idx]
  pre_df$yr <- pre_yrs[pre_df$t_idx]
  pre_df$y <- as.vector(Y_log[, 1L:7L])
  pre_df$w <- w_base[pre_df$s_idx]
  m_dm <- fixest::feols(y ~ 1 | st + reg^yr, data = pre_df, weights = ~w, notes = FALSE)
  R_pre <- matrix(resid(m_dm), nrow = nrow(st_sched), ncol = 7L)
  # Weight rows by sqrt(w_base) before SVD so factors capture hours-weighted state shocks
  sw_norm <- sqrt(w_base / sum(w_base))
  sv <- svd(R_pre * sw_norm)
  lam_f1 <- sv$u[, 1L] / pmax(sw_norm, 1e-6)
  lam_f2 <- sv$u[, 2L] / pmax(sw_norm, 1e-6)
  Z_ife1 <- matrix(lam_f1, ncol = 1L)
  Z_ife2 <- cbind(lam_f1, lam_f2)
  Z_anc  <- cbind(Y_log[, 7L], Y_log[, 6L])
  dy_log <- Y_log[, 8L] - Y_log[, 7L]

  # Record matched FY2024-25 actual-dose placebo if yr == 2025
  if (yr == 2025L) {
    dx_act <- st_sched$dose_raw
    pl2425_r2c[["base_2yr_ppml"]]   <- fit_2period_ppml_cv3j(y_tm1, y_t, dx_act, st_sched$aewr_region_id, st_sched$state_fips)
    pl2425_r2c[["pre3yr_avg_ppml"]] <- fit_2period_ppml_cv3j(y_pre3, y_t, dx_act, st_sched$aewr_region_id, st_sched$state_fips)
    # ANCOVA PPML via fixest (46 rows!)
    df_anc <- data.frame(y = y_t, dx = dx_act, l1 = Y_log[, 7L], l2 = Y_log[, 6L], reg = st_sched$aewr_region_id, st = st_sched$state_fips)
    fa <- fit_cv3j_fixest(df_anc, y ~ dx + l1 + l2 | reg, "dx", "st", "fepois")
    pl2425_r2c[["ancova_ppml"]]     <- list(ok = fa$ok, beta = fa$estimate, se_cv3j = fa$se_cv3j)
    pl2425_r2c[["ancova_wls"]]      <- fit_state_wls_cv3j(Y_log[, 8L], dx_act, st_sched$aewr_region_id, st_sched$state_fips, w_base, Z = Z_anc)
    pl2425_r2c[["sdid_time_wls"]]   <- fit_state_wls_cv3j(y_sdid_diff, dx_act, st_sched$aewr_region_id, st_sched$state_fips, w_base)
    pl2425_r2c[["ife_r1_wls"]]      <- fit_state_wls_cv3j(dy_log, dx_act, st_sched$aewr_region_id, st_sched$state_fips, w_base, Z = Z_ife1)
    pl2425_r2c[["ife_r2_wls"]]      <- fit_state_wls_cv3j(dy_log, dx_act, st_sched$aewr_region_id, st_sched$state_fips, w_base, Z = Z_ife2)
  }

  for (k in seq_len(B_r2c)) {
    idx <- idx + 1L
    perm <- st_sched %>% group_by(aewr_region_id) %>% mutate(d = dose_raw[sample.int(n())]) %>% ungroup()
    dx_p <- perm$d

    r1 <- fit_2period_ppml_cv3j(y_tm1, y_t, dx_p, st_sched$aewr_region_id, st_sched$state_fips)
    r2 <- fit_2period_ppml_cv3j(y_pre3, y_t, dx_p, st_sched$aewr_region_id, st_sched$state_fips)
    r3 <- fit_pois_fe_cv3j_fast(y_t, dx_p, st_sched$aewr_region_id, st_sched$state_fips, Z = Z_anc)
    r4 <- fit_state_wls_cv3j(Y_log[, 8L], dx_p, st_sched$aewr_region_id, st_sched$state_fips, w_base, Z = Z_anc)
    r5 <- fit_state_wls_cv3j(y_sdid_diff, dx_p, st_sched$aewr_region_id, st_sched$state_fips, w_base)
    r6 <- fit_state_wls_cv3j(dy_log, dx_p, st_sched$aewr_region_id, st_sched$state_fips, w_base, Z = Z_ife1)
    r7 <- fit_state_wls_cv3j(dy_log, dx_p, st_sched$aewr_region_id, st_sched$state_fips, w_base, Z = Z_ife2)

    B_mat[idx, ] <- c(r1$beta, r2$beta, r3$beta, r4$beta, r5$beta, r6$beta, r7$beta) * mean_p_raw
    S_mat[idx, ] <- c(r1$se_cv3j, r2$se_cv3j, r3$se_cv3j, r4$se_cv3j, r5$se_cv3j, r6$se_cv3j, r7$se_cv3j) * mean_p_raw
  }
}

r2c_2025_summary <- bind_rows(lapply(methods_r2c, function(m) {
  em <- compute_empirical_mde(B_mat[, m], S_mat[, m], df_t = 45L)
  pl <- pl2425_r2c[[m]]
  pl_el <- pl$beta * mean_p_raw; pl_se <- pl$se_cv3j * mean_p_raw
  pl_p <- if (is.finite(pl_se) && pl_se > 0) 2 * pt(-abs(pl_el / pl_se), 45L) else NA_real_
  cat(sprintf("R2c-2025 %-16s | size=%.3f [%.3f,%.3f] | MDE_el=%.3f (analytic=%.3f) | eps_cost=%.3f | pl2425=%.3f (p=%.3f)\n",
    m, em$emp_size, em$wilson_lo, em$wilson_hi, em$mde_upper, em$analytic_mde,
    (em$mde_upper / mean_p_raw) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / 0.6714, pl_el, pl_p))
  tibble(
    candidate = paste0("R2c_2025_", m),
    family = "R2",
    estimand = paste0("aggregate_hours_", m),
    unit = "state_fy_agg",
    n_obs = 92L, n_units = 2921L, us_hours_share = 0.785,
    G = 46L, G_eff = NA_real_, sd_x_tilde = NA_real_, r2_fe = NA_real_,
    first_stage_pi = 0.6714, first_stage_se = 0.0988, first_stage_F = 46.18,
    sy_ms_ratio = NA_real_, n_placebo_draws = em$n_valid,
    emp_size_alpha05 = em$emp_size, size_wilson_hi = em$wilson_hi,
    mde_primary = em$mde_upper, mde_unit = "statutory_hours_elasticity",
    mde_eps_cost = (em$mde_upper / mean_p_raw) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / 0.6714,
    mde_eps_cost_r7 = (em$mde_upper / mean_p_raw) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / 0.6229,
    placebo_est = pl_el, placebo_se = pl_se, placebo_p = pl_p,
    notes = sprintf("Calibrated on 7 pre-policy target yrs (FY2019-2025) x 100 perms = 700 draws; analytic_MDE=%.3f", em$analytic_mde)
  )
}))

# --- Also test R2c Direct-Grower ANCOVA / Pre-3yr / SDID / IFE on FY2023-2025 (3 target yrs x 200 perms = 600 draws!) ---
cat("\n--- R2c Direct-Grower ANCOVA / Pre-3yr / IFE (FY2020-FY2025) ---\n")
dg_st_panel <- dg_cty_panel %>%
  group_by(state_fips, aewr_region_id, year) %>%
  summarise(hours = sum(dg_hours), cert = sum(dg_cert), .groups = "drop") %>%
  arrange(state_fips, year)

dg_targets <- 2023L:2025L
B_dg_r2c <- 200L
n_dg_r2c <- length(dg_targets) * B_dg_r2c
dg_methods <- c("dg_base_2yr_ppml", "dg_pre3yr_avg_ppml", "dg_ancova_ppml", "dg_ancova_wls", "dg_ife_r1_wls", "dg_base_2yr_cert", "dg_ancova_cert_wls")
BD_mat <- matrix(NA_real_, nrow = n_dg_r2c, ncol = length(dg_methods), dimnames = list(NULL, dg_methods))
SD_mat <- matrix(NA_real_, nrow = n_dg_r2c, ncol = length(dg_methods), dimnames = list(NULL, dg_methods))
pl2425_dg <- list()

set.seed(SEED_PHASE1 + 40L)
idx <- 0L
for (yr in dg_targets) {
  pre_yrs <- (yr - 3L):(yr - 1L)
  sub <- dg_st_panel %>% filter(year %in% c(pre_yrs, yr))
  Y_h <- Y_c <- matrix(0, nrow = nrow(st_sched), ncol = 4L)
  for (j in seq_len(4L)) {
    tmp <- sub %>% filter(year == c(pre_yrs, yr)[[j]])
    Y_h[, j] <- tmp$hours[match(st_sched$state_fips, tmp$state_fips)]
    Y_c[, j] <- tmp$cert[match(st_sched$state_fips, tmp$state_fips)]
  }
  Yh_log <- log(pmax(Y_h, 1)); Yc_log <- log(pmax(Y_c, 1))
  w_h <- pmax(Y_h[, 3L], 1); w_c <- pmax(Y_c[, 3L], 1)
  Z_ah <- cbind(Yh_log[, 3L], Yh_log[, 2L]); Z_ac <- cbind(Yc_log[, 3L], Yc_log[, 2L])

  # 1-factor pre-SVD on 3 pre-years
  pre_df <- expand.grid(s_idx = seq_len(nrow(st_sched)), t_idx = 1L:3L)
  pre_df$st <- st_sched$state_fips[pre_df$s_idx]; pre_df$reg <- st_sched$aewr_region_id[pre_df$s_idx]
  pre_df$yr <- pre_yrs[pre_df$t_idx]; pre_df$y <- as.vector(Yh_log[, 1L:3L]); pre_df$w <- w_h[pre_df$s_idx]
  R_pre <- matrix(resid(fixest::feols(y ~ 1 | st + reg^yr, data = pre_df, weights = ~w, notes = FALSE)), nrow = nrow(st_sched), ncol = 3L)
  sw <- sqrt(w_h / sum(w_h))
  sv <- svd(R_pre * sw)
  Z_f1 <- matrix(sv$u[, 1L] / pmax(sw, 1e-6), ncol = 1L)

  if (yr == 2025L) {
    dx_a <- st_sched$dose_raw
    pl2425_dg[["dg_base_2yr_ppml"]]   <- fit_2period_ppml_cv3j(Y_h[, 3L], Y_h[, 4L], dx_a, st_sched$aewr_region_id, st_sched$state_fips)
    pl2425_dg[["dg_pre3yr_avg_ppml"]] <- fit_2period_ppml_cv3j(rowMeans(Y_h[, 1L:3L]), Y_h[, 4L], dx_a, st_sched$aewr_region_id, st_sched$state_fips)
    fa <- fit_cv3j_fixest(data.frame(y=Y_h[,4L], dx=dx_a, l1=Yh_log[,3L], l2=Yh_log[,2L], reg=st_sched$aewr_region_id, st=st_sched$state_fips), y ~ dx + l1 + l2 | reg, "dx", "st", "fepois")
    pl2425_dg[["dg_ancova_ppml"]]     <- list(ok = fa$ok, beta = fa$estimate, se_cv3j = fa$se_cv3j)
    pl2425_dg[["dg_ancova_wls"]]      <- fit_state_wls_cv3j(Yh_log[, 4L], dx_a, st_sched$aewr_region_id, st_sched$state_fips, w_h, Z = Z_ah)
    pl2425_dg[["dg_ife_r1_wls"]]      <- fit_state_wls_cv3j(Yh_log[, 4L] - Yh_log[, 3L], dx_a, st_sched$aewr_region_id, st_sched$state_fips, w_h, Z = Z_f1)
    pl2425_dg[["dg_base_2yr_cert"]]   <- fit_2period_ppml_cv3j(Y_c[, 3L], Y_c[, 4L], dx_a, st_sched$aewr_region_id, st_sched$state_fips)
    pl2425_dg[["dg_ancova_cert_wls"]] <- fit_state_wls_cv3j(Yc_log[, 4L], dx_a, st_sched$aewr_region_id, st_sched$state_fips, w_c, Z = Z_ac)
  }

  for (k in seq_len(B_dg_r2c)) {
    idx <- idx + 1L
    dx_p <- (st_sched %>% group_by(aewr_region_id) %>% mutate(d = dose_raw[sample.int(n())]) %>% ungroup())$d
    r1 <- fit_2period_ppml_cv3j(Y_h[, 3L], Y_h[, 4L], dx_p, st_sched$aewr_region_id, st_sched$state_fips)
    r2 <- fit_2period_ppml_cv3j(rowMeans(Y_h[, 1L:3L]), Y_h[, 4L], dx_p, st_sched$aewr_region_id, st_sched$state_fips)
    r3 <- fit_pois_fe_cv3j_fast(Y_h[, 4L], dx_p, st_sched$aewr_region_id, st_sched$state_fips, Z = Z_ah)
    r4 <- fit_state_wls_cv3j(Yh_log[, 4L], dx_p, st_sched$aewr_region_id, st_sched$state_fips, w_h, Z = Z_ah)
    r5 <- fit_state_wls_cv3j(Yh_log[, 4L] - Yh_log[, 3L], dx_p, st_sched$aewr_region_id, st_sched$state_fips, w_h, Z = Z_f1)
    r6 <- fit_2period_ppml_cv3j(Y_c[, 3L], Y_c[, 4L], dx_p, st_sched$aewr_region_id, st_sched$state_fips)
    r7 <- fit_state_wls_cv3j(Yc_log[, 4L], dx_p, st_sched$aewr_region_id, st_sched$state_fips, w_c, Z = Z_ac)
    BD_mat[idx, ] <- c(r1$beta, r2$beta, r3$beta, r4$beta, r5$beta, r6$beta, r7$beta) * mean_p_raw
    SD_mat[idx, ] <- c(r1$se_cv3j, r2$se_cv3j, r3$se_cv3j, r4$se_cv3j, r5$se_cv3j, r6$se_cv3j, r7$se_cv3j) * mean_p_raw
  }
}

r2c_dg_summary <- bind_rows(lapply(dg_methods, function(m) {
  em <- compute_empirical_mde(BD_mat[, m], SD_mat[, m], df_t = 45L)
  pl <- pl2425_dg[[m]]
  pl_el <- pl$beta * mean_p_raw; pl_se <- pl$se_cv3j * mean_p_raw
  pl_p <- if (is.finite(pl_se) && pl_se > 0) 2 * pt(-abs(pl_el / pl_se), 45L) else NA_real_
  cat(sprintf("R2c-DG   %-18s | size=%.3f [%.3f,%.3f] | MDE_el=%.3f (analytic=%.3f) | pl2425=%.3f (p=%.3f)\n",
    m, em$emp_size, em$wilson_lo, em$wilson_hi, em$mde_upper, em$analytic_mde, pl_el, pl_p))
  tibble(
    candidate = paste0("R2c_", m),
    family = "R2",
    estimand = paste0("direct_grower_", m),
    unit = "state_fy_direct",
    n_obs = 92L, n_units = 2921L, us_hours_share = 0.587,
    G = 46L, G_eff = NA_real_, sd_x_tilde = NA_real_, r2_fe = NA_real_,
    first_stage_pi = 0.6714, first_stage_se = 0.0988, first_stage_F = 46.18,
    sy_ms_ratio = sy_dg_base$ms_ratio, n_placebo_draws = em$n_valid,
    emp_size_alpha05 = em$emp_size, size_wilson_hi = em$wilson_hi,
    mde_primary = em$mde_upper,
    mde_unit = if (grepl("cert", m)) "direct_grower_positions_elasticity" else "direct_grower_hours_elasticity",
    mde_eps_cost = (em$mde_upper / mean_p_raw) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / 0.6714,
    mde_eps_cost_r7 = (em$mde_upper / mean_p_raw) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / 0.6229,
    placebo_est = pl_el, placebo_se = pl_se, placebo_p = pl_p,
    notes = sprintf("Calibrated on FY2023-2025 (3 target yrs x 200 perms = 600 draws); analytic_MDE=%.3f", em$analytic_mde)
  )
}))

# --- Also test R2c IFE (r=1, r=2) & Lagged-Outcome ANCOVA on 2011-2025 Bite (FE-only, strictly blind!) ---
cat("\n--- R2c on 2011-2025 Bite: Lagged-Outcome ANCOVA & IFE (r=1, r=2) ---\n")
bp_lag <- bp_full %>%
  arrange(county_fips, year) %>%
  group_by(county_fips) %>%
  mutate(ln_h_l1 = log1p(dplyr::lag(man_hours_certified_start_year, 1L))) %>%
  ungroup() %>%
  filter(year >= 2011L, year <= 2025L)

cal_r2c_bite_anc <- calibrate_multiperiod_ppml_blind(
  bp_lag, "man_hours_certified_start_year", "E_p25_x_ln_aewr",
  fe_rhs = "county_fips + aewr_region_id^year", ctrl_rhs = "ln_h_l1"
)
cat(sprintf("R2c Bite + ln(H_{t-1}) ANCOVA  : G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
  cal_r2c_bite_anc$g_eff, cal_r2c_bite_anc$sd_x_tilde_wt, cal_r2c_bite_anc$sy_ms_ratio,
  cal_r2c_bite_anc$emp_size, cal_r2c_bite_anc$mde_1sd, cal_r2c_bite_anc$mde_2sd))

# IFE (r=1, r=2) on 2011-2025 Bite FE-only PPML deviance/working residual matrix (strictly blind!)
# Build county x year (2987 x 15) matrix of weighted working residuals sqrt(mu0)*e0, extract top r SVD factors,
# project out factors, and check BOTH residual SY ratio AND empirical size under region-cluster bootstrap!
m0_b <- fixest::fepois(man_hours_certified_start_year ~ 1 | county_fips + aewr_region_id^year, data = bp_est, notes = FALSE)
u_b <- bp_est[fixest::obs(m0_b), ]
mu0_b <- pmax(as.numeric(fitted(m0_b)), 1e-12)
e0_b <- (u_b$man_hours_certified_start_year - mu0_b) / mu0_b
# Wide matrix of sqrt(mu0)*e0 (counties x 15 years)
ctys_b <- sort(unique(u_b$county_fips)); yrs_b <- 2011L:2025L
R_mat <- matrix(0, nrow = length(ctys_b), ncol = length(yrs_b), dimnames = list(ctys_b, as.character(yrs_b)))
ci_idx <- match(u_b$county_fips, ctys_b); yi_idx <- match(u_b$year, yrs_b)
R_mat[cbind(ci_idx, yi_idx)] <- sqrt(mu0_b) * e0_b
sv_b <- svd(R_mat)
for (r_fac in c(1L, 2L)) {
  # Note: projecting out factors estimated on the SAME 15-year panel (in-sample IFE)
  # Check how much of the working residual variance and SY shared-shock ratio it absorbs,
  # AND compute blind MDE + check over-fitting risk via split-sample / CV3J!
  approx_r <- sv_b$u[, seq_len(r_fac), drop = FALSE] %*% diag(sv_b$d[seq_len(r_fac)], nrow = r_fac) %*% t(sv_b$v[, seq_len(r_fac), drop = FALSE])
  e_ife <- e0_b - approx_r[cbind(ci_idx, yi_idx)] / sqrt(mu0_b)
  sy_ife <- compute_sy_noise_ratio_resid(e_ife * sqrt(mu0_b), u_b$aewr_region_id, u_b$state_fips, u_b$year)
  # Also project X onto the SAME r time factors interacted with county loadings (or check x_tilde survival!)
  # Under IFE, the regressor X also loses any component collinear with the r factor loadings U_r %*% V_r'!
  u_b$.w_mu0 <- mu0_b
  fx_b <- fixest::feols(E_p25_x_ln_aewr ~ 1 | county_fips + aewr_region_id^year, data = u_b, weights = ~.w_mu0, notes = FALSE)
  xt_b <- resid(fx_b)
  X_mat <- matrix(0, nrow = length(ctys_b), ncol = length(yrs_b))
  X_mat[cbind(ci_idx, yi_idx)] <- sqrt(mu0_b) * xt_b
  # Project X_mat orthogonal to column space of U_r AND row space of V_r (exact Bai 2009 tangent projection!)
  U_r <- sv_b$u[, seq_len(r_fac), drop = FALSE]
  V_r <- sv_b$v[, seq_len(r_fac), drop = FALSE]
  X_proj <- X_mat - U_r %*% crossprod(U_r, X_mat) - (X_mat %*% V_r) %*% t(V_r) + U_r %*% (crossprod(U_r, X_mat) %*% V_r) %*% t(V_r)
  xt_ife <- X_proj[cbind(ci_idx, yi_idx)] / sqrt(mu0_b)
  sd_xt_ife <- sqrt(sum(mu0_b * xt_ife^2) / sum(mu0_b))
  # Cluster scores S_g and H_g for 999 cluster-block draws
  cl_b <- as.character(u_b$aewr_region_id); cls_b <- sort(unique(cl_b)); Gb <- length(cls_b)
  H_g <- as.numeric(tapply(mu0_b * xt_ife^2, cl_b, sum)[cls_b]); H_tot <- sum(H_g)
  S_g <- as.numeric(tapply(mu0_b * xt_ife * e_ife, cl_b, sum)[cls_b])
  z_g <- S_g / sqrt(pmax(H_g, 1e-12))
  set.seed(SEED_PHASE1 + 50L + r_fac)
  bd <- sd_v <- numeric(999L)
  for (b in seq_len(999L)) {
    ib <- sample.int(Gb, Gb, replace = TRUE); xi <- sample(c(-1, 1), Gb, replace = TRUE)
    Ss <- sqrt(H_g) * xi * z_g[ib]
    bd[[b]] <- sum(Ss) / H_tot
    bl <- (sum(Ss) - Ss) / (H_tot - H_g)
    sd_v[[b]] <- sqrt(((Gb - 1) / Gb) * sum((bl - mean(bl))^2))
  }
  em_ife <- compute_empirical_mde(bd, sd_v, df_t = Gb - 1L)
  g_eff_ife <- 1 / sum((H_g / H_tot)^2)
  cat(sprintf("R2c Bite IFE (r=%d)            : G_eff=%.2f, SD(X~)=%.4f, SY=%.2f, size=%.3f, MDE_1SD=%.3f, MDE_2SD=%.3f\n",
    r_fac, g_eff_ife, sd_xt_ife, sy_ife$ms_ratio, em_ife$emp_size, em_ife$mde_upper, 2 * em_ife$mde_upper))
}

# Assemble all R2 triage rows cleanly
make_bite_row <- function(cand, est_lbl, cal, us_sh, note_txt) {
  tibble(
    candidate = cand, family = "R2", estimand = est_lbl, unit = "county_year_bite",
    n_obs = as.integer(cal$n_obs), n_units = as.integer(cal$n_counties), us_hours_share = us_sh,
    G = as.integer(cal$G), G_eff = cal$g_eff, sd_x_tilde = cal$sd_x_tilde_wt, r2_fe = cal$r2_x_wt,
    first_stage_pi = NA_real_, first_stage_se = NA_real_, first_stage_F = NA_real_,
    sy_ms_ratio = cal$sy_ms_ratio, n_placebo_draws = 999L,
    emp_size_alpha05 = cal$emp_size, size_wilson_hi = cal$wilson_hi,
    mde_primary = cal$mde_2sd, mde_unit = "2SD_differential_bite_elasticity",
    mde_eps_cost = NA_real_, mde_eps_cost_r7 = NA_real_,
    placebo_est = NA_real_, placebo_se = NA_real_, placebo_p = NA_real_,
    notes = sprintf("1SD_MDE=%.3f; %s", cal$mde_1sd, note_txt)
  )
}

r2_triage <- bind_rows(
  make_bite_row("R2a_bite_direct_2013_2025_1b", "direct_grower_differential_bite_1b", cal_r2a_dg13_1b, 0.659, "Direct-grower hours 2013-2025 vs all-emp 1SD=0.312/2SD=0.625"),
  make_bite_row("R2a_bite_direct_2013_2025_pos", "direct_grower_positions_bite_1b", cal_r2a_dg13_w, 0.659, "Direct-grower positions 2013-2025"),
  make_bite_row("R2a_bite_direct_2013_2025_1a", "direct_grower_differential_bite_1a", cal_r2a_dg13_1a, 0.659, "Direct-grower 1a-PPML (+ E x Year FE)"),
  make_bite_row("R2a_bite_direct_2020_2025_1b", "direct_grower_post2020_bite_1b", cal_r2a_dg20_1b, 0.659, "Clean post-2020 direct-grower only"),
  make_bite_row("R2b_bite_cdl0810_x_yr", "differential_bite_plus_cdl_x_yr", cal_r2b_cdl, 1.0, "2011-2025 + CDL_0810 x Year FE"),
  make_bite_row("R2b_bite_noaa_anomalies", "differential_bite_plus_noaa", cal_r2b_noaa, 1.0, "2011-2025 + 6 NOAA climate anomaly PCs"),
  make_bite_row("R2b_bite_cdl_plus_noaa", "differential_bite_plus_cdl_noaa", cal_r2b_cdl_noaa, 1.0, "2011-2025 + CDL_0810 x Year + NOAA PCs"),
  make_bite_row("R2b_bite_direct_cdl_noaa", "direct_grower_bite_plus_cdl_noaa", cal_r2b_dg_cn, 0.659, "2013-2025 Direct-grower + CDL_0810 x Year + NOAA PCs"),
  make_bite_row("R2b_bite_1a_cdl_noaa", "differential_bite_1a_plus_cdl_noaa", cal_r2b_1a_cn, 1.0, "2011-2025 1a-PPML (+ E x Year) + CDL + NOAA"),
  make_bite_row("R2c_bite_ancova_lag_h", "differential_bite_ancova_lag", cal_r2c_bite_anc, 1.0, "2011-2025 + ln(1+H_{t-1}) ANCOVA"),
  tibble(
    candidate = "R2b_2025_agg_cdl2224", family = "R2", estimand = "aggregate_hours_2025_plus_cdl", unit = "county_fy",
    n_obs = 5842L, n_units = 2921L, us_hours_share = 0.785, G = 46L, G_eff = r2b_25_agg$sup$g_eff,
    sd_x_tilde = r2b_25_agg$sup$sd_x_tilde_wt, r2_fe = r2b_25_agg$sup$r2_fe_wt,
    first_stage_pi = 0.6714, first_stage_se = 0.0988, first_stage_F = 46.18, sy_ms_ratio = NA_real_,
    n_placebo_draws = r2b_25_agg$cdl$n_valid, emp_size_alpha05 = r2b_25_agg$cdl$emp_size, size_wilson_hi = r2b_25_agg$cdl$wilson_hi,
    mde_primary = r2b_25_agg$cdl$mde_upper, mde_unit = "statutory_hours_elasticity",
    mde_eps_cost = (r2b_25_agg$cdl$mde_upper / mean_p_raw) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / 0.6714,
    mde_eps_cost_r7 = (r2b_25_agg$cdl$mde_upper / mean_p_raw) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / 0.6229,
    placebo_est = r2b_25_agg$pl_est, placebo_se = r2b_25_agg$pl_se, placebo_p = r2b_25_agg$pl_p,
    notes = sprintf("Base FY20-25 MDE=%.3f -> +CDL_2224 MDE=%.3f", r2b_25_agg$base$mde_upper, r2b_25_agg$cdl$mde_upper)
  ),
  tibble(
    candidate = "R2b_2025_direct_cdl2224", family = "R2", estimand = "direct_grower_hours_2025_plus_cdl", unit = "county_fy_direct",
    n_obs = 5842L, n_units = 2921L, us_hours_share = 0.587, G = 46L, G_eff = r2b_25_dg$sup$g_eff,
    sd_x_tilde = r2b_25_dg$sup$sd_x_tilde_wt, r2_fe = r2b_25_dg$sup$r2_fe_wt,
    first_stage_pi = 0.6714, first_stage_se = 0.0988, first_stage_F = 46.18, sy_ms_ratio = sy_dg_cdl$ms_ratio,
    n_placebo_draws = r2b_25_dg$cdl$n_valid, emp_size_alpha05 = r2b_25_dg$cdl$emp_size, size_wilson_hi = r2b_25_dg$cdl$wilson_hi,
    mde_primary = r2b_25_dg$cdl$mde_upper, mde_unit = "direct_grower_hours_elasticity",
    mde_eps_cost = (r2b_25_dg$cdl$mde_upper / mean_p_raw) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / 0.6714,
    mde_eps_cost_r7 = (r2b_25_dg$cdl$mde_upper / mean_p_raw) * W_BAR_OFFERED_FY2526 * COST_RATIO_CENTRAL / 0.6229,
    placebo_est = r2b_25_dg$pl_est, placebo_se = r2b_25_dg$pl_se, placebo_p = r2b_25_dg$pl_p,
    notes = sprintf("Base FY20-25 DG MDE=%.3f -> +CDL_2224 MDE=%.3f; EffFloor+CDL MDE=%.3f", r2b_25_dg$base$mde_upper, r2b_25_dg$cdl$mde_upper, r2b_25_dg_eff$cdl$mde_upper)
  ),
  r2c_2025_summary,
  r2c_dg_summary
)

write_csv(r2_triage, "explorations/bite_power_rescue/phase1_r2_triage.csv")
cat("\nWrote explorations/bite_power_rescue/phase1_r2_triage.csv\n")
