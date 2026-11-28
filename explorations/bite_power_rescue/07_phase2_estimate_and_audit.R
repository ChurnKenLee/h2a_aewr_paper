# Phase 2 Estimation, Matched Placebos, Holm Multiplicity, LOO Influence,
# and Adversarial Referee Audit for the 4 Locked Tier 2 Survivors ONLY.
# Hard rules:
#   - Verifies phase2_registry.sha256 before running any estimation (fail-closed).
#   - ZERO FY2026 outcome regressions (zero 2025-rule candidates cleared Phase 1).
#   - Exact delete-one-region CV3J (G=17, df=16) throughout; never downgrades clustering.
#   - Writes outputs strictly under explorations/bite_power_rescue/.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "bite", "helpers.R"))

suppressPackageStartupMessages({
  library(arrow)
  library(digest)
  library(dplyr)
  library(fixest)
  library(jsonlite)
  library(readr)
  library(stringr)
  library(tibble)
  library(tidyr)
})

# ==============================================================================
# 0. Fail-Closed Registry Lock Verification
# ==============================================================================
reg_path <- "explorations/bite_power_rescue/phase2_registry.json"
sha_path <- "explorations/bite_power_rescue/phase2_registry.sha256"
expected_sha <- strsplit(trimws(readLines(sha_path, n = 1L)), "\\s+")[[1]][1]
actual_sha   <- digest::digest(file = reg_path, algo = "sha256")
if (!identical(actual_sha, expected_sha)) {
  stop(sprintf("Registry SHA-256 mismatch! expected=%s actual=%s", expected_sha, actual_sha))
}
cat(sprintf("Verified locked Phase 2 registry SHA-256: %s\n", actual_sha))
reg <- jsonlite::fromJSON(reg_path)
stopifnot(isTRUE(reg$locked_before_phase2_estimation), isTRUE(reg$outcome_blind_fy2026_preserved))

# ==============================================================================
# 1. Load Canonical Cached Inputs (GEO-001 & AEWR-001 enforced)
# ==============================================================================
bp_full <- read_parquet(path_processed("bite_county_year_panel.parquet")) %>%
  mutate(
    year = as.integer(year),
    ln_aewr_real = ifelse(is.finite(aewr_ppi) & aewr_ppi > 0, log(aewr_ppi), NA_real_)
  )
assert_geo_columns(bp_full, c("county_fips", "state_fips", "cz_id", "aewr_region_id"))

# Verify AEWR-001 (2009 aewr strictly NULL) and ln_aewr == log(aewr) [nominal statutory AEWR]
stopifnot(all(is.na(bp_full$aewr[bp_full$year == 2009L])))
aewr_nom_diff <- max(abs(bp_full$ln_aewr - log(bp_full$aewr)), na.rm = TRUE)
cat(sprintf("AEWR-001 verified (2009 aewr all NA); max|ln_aewr - log(aewr)| = %.2e\n", aewr_nom_diff))

bp_est <- bp_full %>% filter(year >= 2011L, year <= 2025L)

# Predetermined 2008-2010 CDL crop shares (long format: crop_type_label, acres)
cdl_0810 <- read_parquet(path_int("croplandcros_county_crop_type_acres.parquet")) %>%
  mutate(
    year = as.integer(year),
    county_fips = harmonize_county_fips_2010(county_fips),
    crop_grp = tolower(trimws(as.character(crop_type_label))),
    acres = as.numeric(acres)
  ) %>%
  filter(year %in% c(2008L, 2009L, 2010L), crop_grp != "non-crop") %>%
  group_by(county_fips) %>%
  summarise(
    tot = sum(acres, na.rm = TRUE),
    cdl_fruit = ifelse(tot > 0, sum(acres[crop_grp == "fruit & tree nuts"], na.rm = TRUE) / tot, 0),
    cdl_veg   = ifelse(tot > 0, sum(acres[crop_grp == "vegetables"], na.rm = TRUE) / tot, 0),
    cdl_field = ifelse(tot > 0, sum(acres[crop_grp == "field crops"], na.rm = TRUE) / tot, 0),
    cdl_hort  = ifelse(tot > 0, sum(acres[crop_grp == "horticulture"], na.rm = TRUE) / tot, 0),
    .groups = "drop"
  ) %>% select(-tot)

# Helper: within-region standardization (exact Phase 1 formula)
std_within <- function(x, grp) {
  out <- rep(NA_real_, length(x))
  for (g in unique(grp[!is.na(grp)])) {
    idx <- which(grp == g & is.finite(x))
    if (length(idx) >= 2L) {
      s <- stats::sd(x[idx])
      if (is.finite(s) && s > 0) out[idx] <- (x[idx] - mean(x[idx])) / s
    }
  }
  out
}

# Build 2008-2010 baseline H-2A contract duration (R4a)
ws_0810_dur <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date) %>%
  filter(fiscal_year %in% c(2008L, 2009L, 2010L)) %>% collect() %>%
  mutate(
    weeks = as.numeric(as.Date(period_end_date) - as.Date(period_begin_date)) / 7,
    workers = as.numeric(adjusted_nbr_workers_certified),
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    nc = str_count(county_fips_list, ",") + 1L
  ) %>%
  filter(is.finite(weeks), weeks > 0, weeks <= 60, is.finite(workers), workers > 0) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), w = workers / nc) %>%
  group_by(county_fips) %>%
  summarise(mean_weeks_0810 = sum(weeks * w) / sum(w), .groups = "drop")

cty_dur <- bp_full %>%
  filter(year %in% c(2008L, 2010L)) %>%
  group_by(county_fips, aewr_region_id) %>%
  summarise(aewr_0810 = mean(aewr, na.rm = TRUE), .groups = "drop") %>%
  left_join(ws_0810_dur, by = "county_fips") %>%
  mutate(
    w_tot = aewr_0810 * 40 * mean_weeks_0810,
    s_wage_2500 = ifelse(is.finite(w_tot), w_tot / (w_tot + 2500), NA_real_),
    s_wage_1500 = ifelse(is.finite(w_tot), w_tot / (w_tot + 1500), NA_real_),
    s_wage_4000 = ifelse(is.finite(w_tot), w_tot / (w_tot + 4000), NA_real_),
    E_dur      = std_within(s_wage_2500, aewr_region_id),
    E_dur_1500 = std_within(s_wage_1500, aewr_region_id),
    E_dur_4000 = std_within(s_wage_4000, aewr_region_id),
    E_weeks    = std_within(mean_weeks_0810, aewr_region_id)
  ) %>%
  filter(is.finite(E_dur))

# Build direct-grower vs H-2ALC county-year outcomes (FY2013-2025 + start-year 2013-2025)
disc_lc <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(fiscal_year, case_number, organization_flag, h2a_labor_contractor, type_of_employer_application) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2026L) %>% collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(
    org_raw = first(na.omit(as.character(organization_flag))),
    lc_raw  = first(na.omit(as.character(h2a_labor_contractor))),
    typ_raw = first(na.omit(as.character(type_of_employer_application))),
    .groups = "drop"
  ) %>%
  mutate(
    is_h2alc = case_when(
      fiscal_year >= 2020L ~ (toupper(trimws(coalesce(lc_raw, ""))) %in% c("Y", "YES", "TRUE", "1")) |
        grepl("LABOR CONTRACTOR|JOB CONTRACTOR", toupper(coalesce(typ_raw, ""))),
      TRUE ~ grepl("LABOR CONTRACTOR|H-2A LABOR CONTRACTOR|H2ALC|FARM LABOR CONTRACTOR", toupper(coalesce(org_raw, ""))) |
        grepl("LABOR CONTRACTOR|JOB CONTRACTOR", toupper(coalesce(typ_raw, "")))
    )
  ) %>% select(case_number, fiscal_year, is_h2alc)

nh_pre_mean <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, period_begin_date, number_of_hours) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2025L) %>% collect() %>%
  filter(!is.na(as.Date(period_begin_date))) %>%
  summarise(m = mean(suppressWarnings(as.numeric(number_of_hours)), na.rm = TRUE)) %>% pull(m)

ws_1326 <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified, period_begin_date, period_end_date, number_of_hours) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2026L) %>% collect() %>%
  mutate(
    fiscal_year = as.integer(fiscal_year),
    begin = as.Date(period_begin_date),
    end   = as.Date(period_end_date),
    start_year = as.integer(format(begin, "%Y"))
  ) %>%
  filter(!is.na(begin), !is.na(end)) %>%
  left_join(disc_lc, by = c("case_number", "fiscal_year")) %>%
  mutate(
    is_h2alc = coalesce(is_h2alc, FALSE),
    nh = coalesce(suppressWarnings(as.numeric(number_of_hours)), nh_pre_mean),
    workers = as.numeric(adjusted_nbr_workers_certified),
    hours = pmax(as.numeric(end - begin) / 7, 0) * nh * workers,
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    nc = str_count(county_fips_list, ",") + 1L
  ) %>%
  separate_rows(county_fips_list, sep = ",") %>%
  mutate(county_fips = harmonize_county_fips_2010(trimws(county_fips_list)), share = 1 / nc)

# Start-year 2013-2025 direct/all/h2alc totals (matches Phase 1 06_compile_phase1_triage.R)
ws_sy <- ws_1326 %>%
  filter(start_year >= 2013L, start_year <= 2025L) %>%
  group_by(county_fips, year = start_year) %>%
  summarise(
    direct_hours_sy   = sum(hours * share * (!is_h2alc), na.rm = TRUE),
    direct_workers_sy = sum(workers * share * (!is_h2alc), na.rm = TRUE),
    h2alc_hours_sy    = sum(hours * share * is_h2alc, na.rm = TRUE),
    all_hours_sy      = sum(hours * share, na.rm = TRUE),
    all_workers_sy    = sum(workers * share, na.rm = TRUE),
    .groups = "drop"
  )

# Strictly pre-FY2026 fiscal-year totals (FY2020-FY2025, ZERO FY2026 disclosures!)
ws_fy <- ws_1326 %>%
  filter(fiscal_year >= 2020L, fiscal_year <= 2025L) %>%
  group_by(county_fips, year = fiscal_year) %>%
  summarise(
    direct_hours_fy   = sum(hours * share * (!is_h2alc), na.rm = TRUE),
    direct_workers_fy = sum(workers * share * (!is_h2alc), na.rm = TRUE),
    .groups = "drop"
  )

# Assemble full enriched panels
p_r4_full <- bp_full %>%
  inner_join(cty_dur %>% select(county_fips, mean_weeks_0810, s_wage_2500, s_wage_1500, s_wage_4000, E_dur, E_dur_1500, E_dur_4000, E_weeks), by = "county_fips") %>%
  left_join(cdl_0810, by = "county_fips") %>%
  left_join(ws_sy, by = c("county_fips", "year")) %>%
  mutate(
    direct_hours_sy   = coalesce(direct_hours_sy, 0),
    direct_workers_sy = coalesce(direct_workers_sy, 0),
    E_dur_x_ln_aewr      = E_dur * ln_aewr,
    E_dur_x_ln_aewr_real = E_dur * ln_aewr_real,
    E_dur_x_ln_ppi_2012  = E_dur * ln_ppi_2012,
    s_wage_x_ln_aewr     = s_wage_2500 * ln_aewr,
    E_dur1500_x_ln_aewr  = E_dur_1500 * ln_aewr,
    E_dur4000_x_ln_aewr  = E_dur_4000 * ln_aewr,
    E_weeks_x_ln_aewr    = E_weeks * ln_aewr
  )
p_r4_est <- p_r4_full %>% filter(year >= 2011L, year <= 2025L)

p_r2_1325 <- bp_est %>%
  filter(year >= 2013L, year <= 2025L) %>%
  left_join(ws_sy, by = c("county_fips", "year")) %>%
  left_join(ws_fy, by = c("county_fips", "year")) %>%
  left_join(cdl_0810, by = "county_fips") %>%
  mutate(
    direct_hours_sy   = coalesce(direct_hours_sy, 0),
    direct_workers_sy = coalesce(direct_workers_sy, 0),
    h2alc_hours_sy    = coalesce(h2alc_hours_sy, 0),
    all_hours_sy      = coalesce(all_hours_sy, 0),
    all_workers_sy    = coalesce(all_workers_sy, 0),
    direct_hours_fy   = coalesce(direct_hours_fy, 0),
    direct_workers_fy = coalesce(direct_workers_fy, 0),
    E_p25_x_ln_aewr_real = E_p25 * ln_aewr_real
  )
p_r2_2025 <- p_r2_1325 %>% filter(year >= 2020L)
pos_dg20  <- p_r2_2025 %>% group_by(county_fips) %>% summarise(s = sum(direct_hours_sy), .groups = "drop") %>% filter(s > 0) %>% pull(county_fips)
p_r2_est  <- p_r2_2025 %>% filter(county_fips %in% pos_dg20)

cat(sprintf("Panel R4a (2011-2025): %d rows, %d baseline-H2A counties\n", nrow(p_r4_est), n_distinct(p_r4_est$county_fips)))
cat(sprintf("Panel R2a (2020-2025): %d rows, %d direct-grower counties\n", nrow(p_r2_est), n_distinct(p_r2_est$county_fips)))

# Region metadata for LOO table
region_meta <- bp_est %>%
  group_by(aewr_region_id) %>%
  summarise(
    states = paste(sort(unique(state_abbrev)), collapse = ","),
    n_states = n_distinct(state_fips),
    hours_1125 = sum(man_hours_certified_start_year, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(hours_share_1125 = hours_1125 / sum(hours_1125))

# ==============================================================================
# 2. Primary Tier 2 Hypotheses (P1-P4) + Holm Step-Down + G_eff + LOO
# ==============================================================================
cat("\n=== PRIMARY PHASE 2 HYPOTHESES (P1-P4, LOCKED REGISTRY) ===\n")

fit_P1 <- bite_fit_cv3j(
  data = p_r4_est,
  formula = man_hours_certified_start_year ~ E_dur_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_dur_x_ln_aewr", estimator = "fepois", cluster_col = "aewr_region_id",
  spec_id = "P1_R4a_h2a_duration_ry", spec_group = "primary_phase2",
  label = "P1: R4a Duration Bite (2011-2025, County + Reg x Yr)",
  outcome = "man_hours_certified_start_year", exposure_name = "E_dur",
  fe_label = "County + Region x Year", sample_label = "2011-2025 baseline-H2A counties (1,615 active)"
)

fit_P2 <- bite_fit_cv3j(
  data = p_r4_est,
  formula = man_hours_certified_start_year ~ E_dur_x_ln_aewr | county_fips + state_fips^year,
  terms = "E_dur_x_ln_aewr", estimator = "fepois", cluster_col = "aewr_region_id",
  spec_id = "P2_R4a_h2a_duration_sy", spec_group = "primary_phase2",
  label = "P2: R4a Duration Bite (2011-2025, County + State x Yr)",
  outcome = "man_hours_certified_start_year", exposure_name = "E_dur",
  fe_label = "County + State x Year", sample_label = "2011-2025 baseline-H2A counties (1,615 active)"
)

fit_P3 <- bite_fit_cv3j(
  data = p_r2_est,
  formula = direct_hours_sy ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr", estimator = "fepois", cluster_col = "aewr_region_id",
  spec_id = "P3_R2a_direct_2020_2025_1b", spec_group = "primary_phase2",
  label = "P3: R2a Direct-Grower Bite 1b (2020-2025, County + Reg x Yr)",
  outcome = "direct_hours_sy", exposure_name = "E_p25",
  fe_label = "County + Region x Year", sample_label = "2020-2025 direct-grower active counties (2,913)"
)

fit_P4 <- bite_fit_cv3j(
  data = p_r2_est,
  formula = direct_hours_sy ~ E_p25_x_ln_aewr + i(year, E_p25, ref = 2020) | county_fips + aewr_region_id^year,
  terms = "E_p25_x_ln_aewr", estimator = "fepois", cluster_col = "aewr_region_id",
  spec_id = "P4_R2a_direct_2020_2025_1a", spec_group = "primary_phase2",
  label = "P4: R2a Direct-Grower Bite 1a (2020-2025, + E_p25 x Yr)",
  outcome = "direct_hours_sy", exposure_name = "E_p25",
  fe_label = "County + Region x Year + E_p25 x Year", sample_label = "2020-2025 direct-grower active counties (2,913)"
)

# Exact Fisher-info G_eff on realized PPML fits
lev_P1 <- bite_cluster_leverage(fit_P1$fit, fit_P1$used, "E_dur_x_ln_aewr", "county_fips + aewr_region_id^year", "aewr_region_id", "man_hours_certified_start_year")
lev_P2 <- bite_cluster_leverage(fit_P2$fit, fit_P2$used, "E_dur_x_ln_aewr", "county_fips + state_fips^year", "aewr_region_id", "man_hours_certified_start_year")
lev_P3 <- bite_cluster_leverage(fit_P3$fit, fit_P3$used, "E_p25_x_ln_aewr", "county_fips + aewr_region_id^year", "aewr_region_id", "direct_hours_sy")
# For P4 (1a-PPML), compute Fisher info share residualizing E_p25_x_ln_aewr on C+RY + E_p25 x Year
u4 <- fit_P4$used
u4$.mu_hat <- as.numeric(stats::fitted(fit_P4$fit))
f4_x <- feols(E_p25_x_ln_aewr ~ i(year, E_p25, ref = 2020) | county_fips + aewr_region_id^year, data = u4, weights = ~.mu_hat, warn = FALSE, notes = FALSE)
u4$.x_til <- as.numeric(stats::residuals(f4_x))
inf4 <- u4 %>% group_by(cluster_id = as.character(aewr_region_id)) %>%
  summarise(info = sum(.mu_hat * (.x_til^2), na.rm = TRUE), .groups = "drop") %>%
  mutate(fisher_info_share = info / sum(info))
g_eff_P4 <- 1 / sum(inf4$fisher_info_share^2)
max_sh_P4 <- max(inf4$fisher_info_share)
max_cl_P4 <- inf4$cluster_id[which.max(inf4$fisher_info_share)]

MDE_MULT_T16 <- stats::qt(0.975, df = 16L) + stats::qt(0.80, df = 16L)

primary_tbl <- bind_rows(fit_P1$summary, fit_P2$summary, fit_P3$summary, fit_P4$summary) %>%
  mutate(
    mde_1sd_2p8 = 2.8 * se_cv3j,
    mde_2sd_2p8 = 2 * 2.8 * se_cv3j,
    mde_1sd_t16 = MDE_MULT_T16 * se_cv3j,
    mde_2sd_t16 = 2 * MDE_MULT_T16 * se_cv3j,
    p_holm_cv3j = stats::p.adjust(p_cv3j, method = "holm"),
    g_eff_realized = c(lev_P1$g_eff_fisher, lev_P2$g_eff_fisher, lev_P3$g_eff_fisher, g_eff_P4),
    max_cluster_info_share = c(
      max(lev_P1$by_cluster$fisher_info_share),
      max(lev_P2$by_cluster$fisher_info_share),
      max(lev_P3$by_cluster$fisher_info_share),
      max_sh_P4
    ),
    max_cluster_id = c(
      lev_P1$by_cluster$cluster_id[which.max(lev_P1$by_cluster$fisher_info_share)],
      lev_P2$by_cluster$cluster_id[which.max(lev_P2$by_cluster$fisher_info_share)],
      lev_P3$by_cluster$cluster_id[which.max(lev_P3$by_cluster$fisher_info_share)],
      max_cl_P4
    ),
    registry_sha256 = actual_sha
  )

for (i in seq_len(nrow(primary_tbl))) {
  r <- primary_tbl[i, ]
  cat(sprintf(
    "%-28s: est=%+.4f [CV3J %.4f] 95%%CI=[%+.4f, %+.4f] p=%.4f p_holm=%.4f | t16 MDE 1SD=%.3f 2SD=%.3f (2.8x: %.3f/%.3f) | G_eff=%.2f (max reg %s=%.1f%%) | LOO=[%+.4f, %+.4f]\n",
    r$spec_id, r$estimate, r$se_cv3j, r$ci95_low_cv3j, r$ci95_high_cv3j,
    r$p_cv3j, r$p_holm_cv3j, r$mde_1sd_t16, r$mde_2sd_t16, r$mde_1sd_2p8, r$mde_2sd_2p8,
    r$g_eff_realized, r$max_cluster_id, 100 * r$max_cluster_info_share, r$loo_min, r$loo_max
  ))
}
write_csv(primary_tbl, "explorations/bite_power_rescue/phase2_estimates.csv")

# Full 17-region LOO influence table
loo_tbl <- bind_rows(
  fit_P1$loo %>% rename(loo_estimate = estimate) %>% left_join(lev_P1$by_cluster %>% transmute(omitted_cluster = cluster_id, fisher_info_share), by = "omitted_cluster"),
  fit_P2$loo %>% rename(loo_estimate = estimate) %>% left_join(lev_P2$by_cluster %>% transmute(omitted_cluster = cluster_id, fisher_info_share), by = "omitted_cluster"),
  fit_P3$loo %>% rename(loo_estimate = estimate) %>% left_join(lev_P3$by_cluster %>% transmute(omitted_cluster = cluster_id, fisher_info_share), by = "omitted_cluster"),
  fit_P4$loo %>% rename(loo_estimate = estimate) %>% left_join(inf4 %>% transmute(omitted_cluster = cluster_id, fisher_info_share), by = "omitted_cluster")
) %>%
  left_join(region_meta, by = c("omitted_cluster" = "aewr_region_id")) %>%
  left_join(primary_tbl %>% select(spec_id, full_est = estimate), by = "spec_id") %>%
  mutate(delta_loo = loo_estimate - full_est) %>%
  arrange(spec_id, as.integer(omitted_cluster))
write_csv(loo_tbl, "explorations/bite_power_rescue/phase2_loo.csv")

# ==============================================================================
# 3. Exact Matched Pre-Policy Placebos
# ==============================================================================
cat("\n=== MATCHED PRE-POLICY PLACEBOS ===\n")
pre_terms_r4 <- paste0("year::", c(2008L, 2009L, 2010L), ":E_dur")
ev_terms_r4  <- paste0("year::", setdiff(2008L:2025L, 2011L), ":E_dur")

es_r4_ry_h <- bite_fit_cv3j(p_r4_full, man_hours_certified_start_year ~ i(year, E_dur, ref = 2011) | county_fips + aewr_region_id^year, ev_terms_r4, "fepois", "aewr_region_id", "es_r4_ry_h", "placebo", "R4a ES C+RY Hours", "hours", "E_dur", "C+RY", "2008-2025")
es_r4_ry_w <- bite_fit_cv3j(p_r4_full, nbr_workers_certified_start_year ~ i(year, E_dur, ref = 2011) | county_fips + aewr_region_id^year, ev_terms_r4, "fepois", "aewr_region_id", "es_r4_ry_w", "placebo", "R4a ES C+RY Workers", "workers", "E_dur", "C+RY", "2008-2025")
es_r4_sy_h <- bite_fit_cv3j(p_r4_full, man_hours_certified_start_year ~ i(year, E_dur, ref = 2011) | county_fips + state_fips^year, ev_terms_r4, "fepois", "aewr_region_id", "es_r4_sy_h", "placebo", "R4a ES C+SY Hours", "hours", "E_dur", "C+SY", "2008-2025")
es_r4_sy_w <- bite_fit_cv3j(p_r4_full, nbr_workers_certified_start_year ~ i(year, E_dur, ref = 2011) | county_fips + state_fips^year, ev_terms_r4, "fepois", "aewr_region_id", "es_r4_sy_w", "placebo", "R4a ES C+SY Workers", "workers", "E_dur", "C+SY", "2008-2025")

w_r4_ry_h <- bite_joint_wald(coef(es_r4_ry_h$fit)[pre_terms_r4], es_r4_ry_h$vcov_cv3j[pre_terms_r4, pre_terms_r4], 16L)
w_r4_ry_w <- bite_joint_wald(coef(es_r4_ry_w$fit)[pre_terms_r4], es_r4_ry_w$vcov_cv3j[pre_terms_r4, pre_terms_r4], 16L)
w_r4_sy_h <- bite_joint_wald(coef(es_r4_sy_h$fit)[pre_terms_r4], es_r4_sy_h$vcov_cv3j[pre_terms_r4, pre_terms_r4], 16L)
w_r4_sy_w <- bite_joint_wald(coef(es_r4_sy_w$fit)[pre_terms_r4], es_r4_sy_w$vcov_cv3j[pre_terms_r4, pre_terms_r4], 16L)

cat(sprintf("R4a 2008-2010 joint pre-trend C+RY: hours F(3,16)=%.3f (p=%.4f), workers F(3,16)=%.3f (p=%.4f)\n", w_r4_ry_h$f_stat, w_r4_ry_h$p_value, w_r4_ry_w$f_stat, w_r4_ry_w$p_value))
cat(sprintf("R4a 2008-2010 joint pre-trend C+SY: hours F(3,16)=%.3f (p=%.4f), workers F(3,16)=%.3f (p=%.4f)\n", w_r4_sy_h$f_stat, w_r4_sy_h$p_value, w_r4_sy_w$f_stat, w_r4_sy_w$p_value))

# Save R4a event-study path (2008-2025) so we can inspect temporal dynamics!
r4_es_path <- es_r4_ry_h$summary %>%
  transmute(year = as.integer(sub("^year::([0-9]{4}):E_dur$", "\\1", term)),
            est_ry_h = estimate, se_ry_h = se_cv3j, p_ry_h = p_cv3j) %>%
  left_join(es_r4_sy_h$summary %>%
    transmute(year = as.integer(sub("^year::([0-9]{4}):E_dur$", "\\1", term)),
              est_sy_h = estimate, se_sy_h = se_cv3j, p_sy_h = p_cv3j), by = "year") %>%
  left_join(es_r4_ry_w$summary %>%
    transmute(year = as.integer(sub("^year::([0-9]{4}):E_dur$", "\\1", term)),
              est_ry_w = estimate, se_ry_w = se_cv3j, p_ry_w = p_cv3j), by = "year") %>%
  arrange(year)
cat("\nR4a Event-Study Path (ref=2011):\n")
print(r4_es_path, n = 20, width = 160)
write_csv(r4_es_path, "explorations/bite_power_rescue/phase2_r4a_event_study.csv")

# R2a 2013-2019 pre-2020 event study + 2013-2019 1b and 1a placebos on direct growers
p_r2_pre <- p_r2_1325 %>% filter(year <= 2019L, county_fips %in% pos_dg20)
pre_terms_r2 <- paste0("year::", 2013L:2018L, ":E_p25")
es_r2_pre_h <- bite_fit_cv3j(p_r2_pre, direct_hours_sy ~ i(year, E_p25, ref = 2019) | county_fips + aewr_region_id^year, pre_terms_r2, "fepois", "aewr_region_id", "es_r2_pre_h", "placebo", "R2a Direct ES 2013-19 Hours", "direct_hours_sy", "E_p25", "C+RY", "2013-2019")
es_r2_pre_w <- bite_fit_cv3j(p_r2_pre, direct_workers_sy ~ i(year, E_p25, ref = 2019) | county_fips + aewr_region_id^year, pre_terms_r2, "fepois", "aewr_region_id", "es_r2_pre_w", "placebo", "R2a Direct ES 2013-19 Workers", "direct_workers_sy", "E_p25", "C+RY", "2013-2019")
w_r2_pre_h <- bite_joint_wald(coef(es_r2_pre_h$fit)[pre_terms_r2], es_r2_pre_h$vcov_cv3j[pre_terms_r2, pre_terms_r2], 16L)
w_r2_pre_w <- bite_joint_wald(coef(es_r2_pre_w$fit)[pre_terms_r2], es_r2_pre_w$vcov_cv3j[pre_terms_r2, pre_terms_r2], 16L)
cat(sprintf("\nR2a 2013-2019 Direct-Grower joint pre-trend: hours F(6,16)=%.3f (p=%.4f), workers F(6,16)=%.3f (p=%.4f)\n",
            w_r2_pre_h$f_stat, w_r2_pre_h$p_value, w_r2_pre_w$f_stat, w_r2_pre_w$p_value))

# ==============================================================================
# 4. Comprehensive Adversarial Referee Audit Battery
# ==============================================================================
cat("\n=== ADVERSARIAL REFEREE AUDIT BATTERY ===\n")

# Helper wrapper for compact referee rows
run_ref <- function(spec_id, grp, lbl, data, fml, term, outcome, exp_name, fe_lbl, samp_lbl, notes = "") {
  res <- bite_fit_cv3j(data, fml, term, "fepois", "aewr_region_id", spec_id, grp, lbl, outcome, exp_name, fe_lbl, samp_lbl, notes = notes)
  s <- res$summary %>% mutate(mde_2sd = 2 * mde_cv3j)
  cat(sprintf(
    "  %-38s: est=%+.4f [CV3J %.4f] 95%%CI=[%+.4f,%+.4f] p=%.4f MDE_1SD=%.3f MDE_2SD=%.3f LOO=[%+.3f,%+.3f]\n",
    spec_id, s$estimate, s$se_cv3j, s$ci95_low_cv3j, s$ci95_high_cv3j, s$p_cv3j, s$mde_cv3j, s$mde_2sd, s$loo_min, s$loo_max
  ))
  res
}

# --- 4A. R4a Referee Battery ---
cat("\n-- 4A. R4a Duration-Amortization Bite Referee Checks --\n")
# 1. Raw s_wage structural cost-share elasticity
r4_raw_ry <- run_ref("R4a_raw_swage_eps_cost_ry", "r4_structural", "R4a Raw s_wage x ln(AEWR) [eps_cost] C+RY",
                     p_r4_est, man_hours_certified_start_year ~ s_wage_x_ln_aewr | county_fips + aewr_region_id^year,
                     "s_wage_x_ln_aewr", "hours", "s_wage_2500", "C+RY", "2011-2025", "Direct structural cost-share elasticity")
r4_raw_sy <- run_ref("R4a_raw_swage_eps_cost_sy", "r4_structural", "R4a Raw s_wage x ln(AEWR) [eps_cost] C+SY",
                     p_r4_est, man_hours_certified_start_year ~ s_wage_x_ln_aewr | county_fips + state_fips^year,
                     "s_wage_x_ln_aewr", "hours", "s_wage_2500", "C+SY", "2011-2025", "Direct structural cost-share elasticity")

# 2. 1a-PPML (+ E_dur x Year FE) and variance retention
vd_r4 <- bind_rows(
  bite_variance_decomposition(p_r4_est, "E_dur_x_ln_aewr", "E_dur", weights_vec = NULL),
  bite_variance_decomposition(fit_P1$used, "E_dur_x_ln_aewr", "E_dur", weights_vec = as.numeric(stats::fitted(fit_P1$fit)))
)
cat(sprintf("  R4a Variance Decomposition: 1a retains %.2f%% (unwt) / %.2f%% (PPML-wt) of 1b variance; C+SY retains %.1f%% (PPML-wt)\n",
            100 * vd_r4$share_1a_of_1b[1], 100 * vd_r4$share_1a_of_1b[2], 100 * vd_r4$share_within_SY_of_1b[2]))
r4_1a_ry <- run_ref("R4a_1a_ppml_plus_Edur_x_Yr", "r4_1a_trend", "R4a 1a-PPML (+ E_dur x Year FE)",
                    p_r4_est, man_hours_certified_start_year ~ E_dur_x_ln_aewr + i(year, E_dur, ref = 2011) | county_fips + aewr_region_id^year,
                    "E_dur_x_ln_aewr", "hours", "E_dur", "C+RY+Edur*Y", "2011-2025",
                    sprintf("Retains %.2f%% PPML-wt var of 1b", 100 * vd_r4$share_1a_of_1b[2]))

# 3. Margin decomposition: Positions + Paired CV3J Implied Duration (MEAN REVERSION DETECTOR!)
r4_pos_ry <- run_ref("R4a_positions_ry", "r4_margin", "R4a Certified Positions C+RY",
                     p_r4_est, nbr_workers_certified_start_year ~ E_dur_x_ln_aewr | county_fips + aewr_region_id^year,
                     "E_dur_x_ln_aewr", "workers", "E_dur", "C+RY", "2011-2025")
r4_pos_sy <- run_ref("R4a_positions_sy", "r4_margin", "R4a Certified Positions C+SY",
                     p_r4_est, nbr_workers_certified_start_year ~ E_dur_x_ln_aewr | county_fips + state_fips^year,
                     "E_dur_x_ln_aewr", "workers", "E_dur", "C+SY", "2011-2025")
r4_dur_ry <- bite_implied_difference(fit_P1, r4_pos_ry, "E_dur_x_ln_aewr", "R4a_implied_duration_ry", "r4_margin",
                                     "R4a Implied Hours/Pos (theta_H - theta_N) C+RY", "implied_hours_per_pos", "E_dur", "C+RY", "2011-2025",
                                     "Tests mechanical mean reversion in baseline contract duration")
r4_dur_sy <- bite_implied_difference(fit_P2, r4_pos_sy, "E_dur_x_ln_aewr", "R4a_implied_duration_sy", "r4_margin",
                                     "R4a Implied Hours/Pos (theta_H - theta_N) C+SY", "implied_hours_per_pos", "E_dur", "C+SY", "2011-2025",
                                     "Tests mechanical mean reversion in baseline contract duration")
cat(sprintf("  %-38s: est=%+.4f [CV3J %.4f] 95%%CI=[%+.4f,%+.4f] p=%.4f\n",
            "R4a_implied_duration_ry (H - N)", r4_dur_ry$summary$estimate, r4_dur_ry$summary$se_cv3j, r4_dur_ry$summary$ci95_low_cv3j, r4_dur_ry$summary$ci95_high_cv3j, r4_dur_ry$summary$p_cv3j))
cat(sprintf("  %-38s: est=%+.4f [CV3J %.4f] 95%%CI=[%+.4f,%+.4f] p=%.4f\n",
            "R4a_implied_duration_sy (H - N)", r4_dur_sy$summary$estimate, r4_dur_sy$summary$se_cv3j, r4_dur_sy$summary$ci95_low_cv3j, r4_dur_sy$summary$ci95_high_cv3j, r4_dur_sy$summary$p_cv3j))

# 4. Fixed-cost F sensitivity & linear weeks
r4_f1500 <- run_ref("R4a_F1500_ry", "r4_param", "R4a Fixed Cost F=$1,500 (C+RY)", p_r4_est, man_hours_certified_start_year ~ E_dur1500_x_ln_aewr | county_fips + aewr_region_id^year, "E_dur1500_x_ln_aewr", "hours", "E_dur_1500", "C+RY", "2011-2025")
r4_f4000 <- run_ref("R4a_F4000_ry", "r4_param", "R4a Fixed Cost F=$4,000 (C+RY)", p_r4_est, man_hours_certified_start_year ~ E_dur4000_x_ln_aewr | county_fips + aewr_region_id^year, "E_dur4000_x_ln_aewr", "hours", "E_dur_4000", "C+RY", "2011-2025")
r4_wks   <- run_ref("R4a_linear_weeks_ry", "r4_param", "R4a Linear mean_weeks_0810 (C+RY)", p_r4_est, man_hours_certified_start_year ~ E_weeks_x_ln_aewr | county_fips + aewr_region_id^year, "E_weeks_x_ln_aewr", "hours", "E_weeks", "C+RY", "2011-2025")

# 5. Drop 2025 (2011-2024 strictly pre-Oct-2-2025), direct growers, and E_p25 on SAME 1,615 counties
r4_1124_ry <- run_ref("R4a_drop2025_2011_2024_ry", "r4_window", "R4a 2011-2024 (Drop Provisional 2025) C+RY",
                      p_r4_est %>% filter(year <= 2024L), man_hours_certified_start_year ~ E_dur_x_ln_aewr | county_fips + aewr_region_id^year,
                      "E_dur_x_ln_aewr", "hours", "E_dur", "C+RY", "2011-2024")
r4_1124_sy <- run_ref("R4a_drop2025_2011_2024_sy", "r4_window", "R4a 2011-2024 (Drop Provisional 2025) C+SY",
                      p_r4_est %>% filter(year <= 2024L), man_hours_certified_start_year ~ E_dur_x_ln_aewr | county_fips + state_fips^year,
                      "E_dur_x_ln_aewr", "hours", "E_dur", "C+SY", "2011-2024")
r4_dg1325  <- run_ref("R4a_direct_grower_2013_2025_ry", "r4_sample", "R4a Direct Growers Only (2013-2025) C+RY",
                      p_r4_est %>% filter(year >= 2013L), direct_hours_sy ~ E_dur_x_ln_aewr | county_fips + aewr_region_id^year,
                      "E_dur_x_ln_aewr", "direct_hours_sy", "E_dur", "C+RY", "2013-2025")
r4_dg2025  <- run_ref("R4a_direct_grower_2020_2025_ry", "r4_sample", "R4a Direct Growers Only (2020-2025) C+RY",
                      p_r4_est %>% filter(year >= 2020L), direct_hours_sy ~ E_dur_x_ln_aewr | county_fips + aewr_region_id^year,
                      "E_dur_x_ln_aewr", "direct_hours_sy", "E_dur", "C+RY", "2020-2025")
r4_ep25_same <- run_ref("R4a_Ep25_on_same_1615_ctys_ry", "r4_sample", "E_p25 on Identical 1,615 Baseline-H2A Counties C+RY",
                        p_r4_est, man_hours_certified_start_year ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
                        "E_p25_x_ln_aewr", "hours", "E_p25", "C+RY", "2011-2025 same 1,615 counties")

# 6. PPI decoupling, nominal AEWR, X_pre x Year, CDL x Year, drop top 1%
r4_ppi <- run_ref("R4a_ppi_decouple_ry", "r4_confound", "R4a + WPU01 PPI Decoupling C+RY",
                  p_r4_est, man_hours_certified_start_year ~ E_dur_x_ln_aewr + E_dur_x_ln_ppi_2012 | county_fips + aewr_region_id^year,
                  "E_dur_x_ln_aewr", "hours", "E_dur", "C+RY+PPI", "2011-2025")
r4_nom <- run_ref("R4a_real_aewr_ppi_ry", "r4_confound", "R4a Real ln(AEWR_ppi) C+RY",
                  p_r4_est, man_hours_certified_start_year ~ E_dur_x_ln_aewr_real | county_fips + aewr_region_id^year,
                  "E_dur_x_ln_aewr_real", "hours", "E_dur", "C+RY", "2011-2025 real PPI")
r4_xpre <- run_ref("R4a_xpre_x_year_ry", "r4_confound", "R4a + X_pre x Year FE C+RY",
                   p_r4_est, man_hours_certified_start_year ~ E_dur_x_ln_aewr + i(year, ln_pop_2010, ref = 2011) + i(year, farm_emp_share_0810, ref = 2011) + i(year, ln_emp_farm_0810, ref = 2011) | county_fips + aewr_region_id^year,
                   "E_dur_x_ln_aewr", "hours", "E_dur", "C+RY+Xpre*Y", "2011-2025")
r4_cdl <- run_ref("R4a_cdl0810_x_year_ry", "r4_confound", "R4a + CDL_0810 x Year FE C+RY",
                  p_r4_est, man_hours_certified_start_year ~ E_dur_x_ln_aewr + i(year, cdl_fruit, ref = 2011) + i(year, cdl_veg, ref = 2011) + i(year, cdl_field, ref = 2011) + i(year, cdl_hort, ref = 2011) | county_fips + aewr_region_id^year,
                  "E_dur_x_ln_aewr", "hours", "E_dur", "C+RY+CDL*Y", "2011-2025")

r4_cum <- p_r4_est %>% group_by(county_fips) %>% summarise(ch = sum(man_hours_certified_start_year), .groups = "drop")
r4_keep99 <- r4_cum$county_fips[r4_cum$ch <= quantile(r4_cum$ch, 0.99, names = FALSE)]
r4_top1 <- run_ref("R4a_drop_top1pct_ry", "r4_leverage", "R4a Drop Top 1% H-2A Counties C+RY",
                   p_r4_est %>% filter(county_fips %in% r4_keep99), man_hours_certified_start_year ~ E_dur_x_ln_aewr | county_fips + aewr_region_id^year,
                   "E_dur_x_ln_aewr", "hours", "E_dur", "C+RY", "2011-2025 drop top 1%")

# --- 4B. R2a Direct-Grower 2020-2025 Referee Battery ---
cat("\n-- 4B. R2a Direct-Grower Post-2020 Bite Referee Checks --\n")
# Inline variance decomposition for 2020-2025 (ref = 2020)
calc_vd_2020 <- function(df, w_vec = NULL) {
  d0 <- df
  if (is.null(w_vec)) {
    f1b <- feols(E_p25_x_ln_aewr ~ 1 | county_fips + aewr_region_id^year, data = d0, warn = FALSE, notes = FALSE)
    f1a <- feols(E_p25_x_ln_aewr ~ i(year, E_p25, ref = 2020) | county_fips + aewr_region_id^year, data = d0, warn = FALSE, notes = FALSE)
    fsy <- feols(E_p25_x_ln_aewr ~ 1 | county_fips + state_fips^year, data = d0, warn = FALSE, notes = FALSE)
    v1b <- sum(residuals(f1b)^2); v1a <- sum(residuals(f1a)^2); vsy <- sum(residuals(fsy)^2)
  } else {
    d0$.w <- w_vec
    f1b <- feols(E_p25_x_ln_aewr ~ 1 | county_fips + aewr_region_id^year, data = d0, weights = ~.w, warn = FALSE, notes = FALSE)
    f1a <- feols(E_p25_x_ln_aewr ~ i(year, E_p25, ref = 2020) | county_fips + aewr_region_id^year, data = d0, weights = ~.w, warn = FALSE, notes = FALSE)
    fsy <- feols(E_p25_x_ln_aewr ~ 1 | county_fips + state_fips^year, data = d0, weights = ~.w, warn = FALSE, notes = FALSE)
    v1b <- sum(d0$.w[obs(f1b)] * residuals(f1b)^2)
    v1a <- sum(d0$.w[obs(f1a)] * residuals(f1a)^2)
    vsy <- sum(d0$.w[obs(fsy)] * residuals(fsy)^2)
  }
  c(share_1a = v1a / v1b, share_sy = vsy / v1b)
}
vd_r2_unw <- calc_vd_2020(p_r2_est, NULL)
vd_r2_wt  <- calc_vd_2020(fit_P3$used, as.numeric(stats::fitted(fit_P3$fit)))
cat(sprintf("  R2a (2020-25 Direct) Variance Decomposition: 1a retains %.2f%% (unwt) / %.2f%% (PPML-wt) of 1b variance; C+SY retains %.1f%% (PPML-wt)\n",
            100 * vd_r2_unw[["share_1a"]], 100 * vd_r2_wt[["share_1a"]], 100 * vd_r2_wt[["share_sy"]]))

# 1. Pre-2020 placebo slopes (2013-2019 1b and 1a)
r2_pl_1b <- run_ref("R2a_placebo_2013_2019_1b", "r2_placebo", "R2a Direct Pre-2020 Placebo 1b (2013-2019)",
                    p_r2_pre, direct_hours_sy ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
                    "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+RY", "2013-2019")
r2_pl_1a <- run_ref("R2a_placebo_2013_2019_1a", "r2_placebo", "R2a Direct Pre-2020 Placebo 1a (2013-2019)",
                    p_r2_pre, direct_hours_sy ~ E_p25_x_ln_aewr + i(year, E_p25, ref = 2013) | county_fips + aewr_region_id^year,
                    "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+RY+E*Y", "2013-2019")

# 2. Strict pre-Oct-2-2025 checks: Drop 2025 (2020-2024) AND FY2020-FY2025 (zero FY2026 disclosures!)
r2_2024_1b <- run_ref("R2a_drop2025_2020_2024_1b", "r2_cutoff", "R2a Direct 2020-2024 (Strict Pre-Oct-2025) 1b",
                      p_r2_est %>% filter(year <= 2024L), direct_hours_sy ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
                      "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+RY", "2020-2024")
r2_2024_1a <- run_ref("R2a_drop2025_2020_2024_1a", "r2_cutoff", "R2a Direct 2020-2024 (Strict Pre-Oct-2025) 1a",
                      p_r2_est %>% filter(year <= 2024L), direct_hours_sy ~ E_p25_x_ln_aewr + i(year, E_p25, ref = 2020) | county_fips + aewr_region_id^year,
                      "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+RY+E*Y", "2020-2024")
r2_fy_1b   <- run_ref("R2a_fy2020_2025_noFY26_1b", "r2_cutoff", "R2a Direct FY2020-FY2025 (Zero FY2026 Filings) 1b",
                      p_r2_est, direct_hours_fy ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
                      "E_p25_x_ln_aewr", "direct_hours_fy", "E_p25", "C+RY", "FY2020-FY2025")
r2_fy_1a   <- run_ref("R2a_fy2020_2025_noFY26_1a", "r2_cutoff", "R2a Direct FY2020-FY2025 (Zero FY2026 Filings) 1a",
                      p_r2_est, direct_hours_fy ~ E_p25_x_ln_aewr + i(year, E_p25, ref = 2020) | county_fips + aewr_region_id^year,
                      "E_p25_x_ln_aewr", "direct_hours_fy", "E_p25", "C+RY+E*Y", "FY2020-FY2025")

# 3. Employer-type & window decomposition (All 2020-25, H-2ALC 2020-25, Direct 2013-25, C+SY)
r2_all_1b <- run_ref("R2a_all_employers_2020_2025_1b", "r2_decomp", "All Employers 2020-2025 1b (Window vs DG Check)",
                     p_r2_2025, all_hours_sy ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
                     "E_p25_x_ln_aewr", "all_hours_sy", "E_p25", "C+RY", "2020-2025 all employers")
r2_all_1a <- run_ref("R2a_all_employers_2020_2025_1a", "r2_decomp", "All Employers 2020-2025 1a (Window vs DG Check)",
                     p_r2_2025, all_hours_sy ~ E_p25_x_ln_aewr + i(year, E_p25, ref = 2020) | county_fips + aewr_region_id^year,
                     "E_p25_x_ln_aewr", "all_hours_sy", "E_p25", "C+RY+E*Y", "2020-2025 all employers")
r2_lc_1b  <- run_ref("R2a_h2alc_only_2020_2025_1b", "r2_decomp", "H-2ALC Only 2020-2025 1b",
                     p_r2_2025, h2alc_hours_sy ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
                     "E_p25_x_ln_aewr", "h2alc_hours_sy", "E_p25", "C+RY", "2020-2025 H-2ALC only")
r2_1325_1b <- run_ref("R2a_direct_2013_2025_1b", "r2_decomp", "Direct Growers 2013-2025 1b",
                      p_r2_1325, direct_hours_sy ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
                      "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+RY", "2013-2025 direct")
r2_sy_1b  <- run_ref("R2a_direct_2020_2025_1b_stxyr", "r2_fe", "R2a Direct 2020-2025 1b + State x Year FE",
                     p_r2_est, direct_hours_sy ~ E_p25_x_ln_aewr | county_fips + state_fips^year,
                     "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+SY", "2020-2025")
r2_sy_1a  <- run_ref("R2a_direct_2020_2025_1a_stxyr", "r2_fe", "R2a Direct 2020-2025 1a + State x Year FE",
                     p_r2_est, direct_hours_sy ~ E_p25_x_ln_aewr + i(year, E_p25, ref = 2020) | county_fips + state_fips^year,
                     "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+SY+E*Y", "2020-2025")

# 4. Margin decomposition: Positions + Paired CV3J Implied Duration
r2_pos_1b <- run_ref("R2a_direct_positions_1b", "r2_margin", "R2a Direct 2020-2025 Positions 1b",
                     p_r2_est, direct_workers_sy ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
                     "E_p25_x_ln_aewr", "direct_workers_sy", "E_p25", "C+RY", "2020-2025")
r2_pos_1a <- run_ref("R2a_direct_positions_1a", "r2_margin", "R2a Direct 2020-2025 Positions 1a",
                     p_r2_est, direct_workers_sy ~ E_p25_x_ln_aewr + i(year, E_p25, ref = 2020) | county_fips + aewr_region_id^year,
                     "E_p25_x_ln_aewr", "direct_workers_sy", "E_p25", "C+RY+E*Y", "2020-2025")
r2_dur_1b <- bite_implied_difference(fit_P3, r2_pos_1b, "E_p25_x_ln_aewr", "R2a_implied_duration_1b", "r2_margin",
                                     "R2a Direct Implied Hours/Pos (theta_H - theta_N) 1b", "implied_hours_per_pos", "E_p25", "C+RY", "2020-2025")
r2_dur_1a <- bite_implied_difference(fit_P4, r2_pos_1a, "E_p25_x_ln_aewr", "R2a_implied_duration_1a", "r2_margin",
                                     "R2a Direct Implied Hours/Pos (theta_H - theta_N) 1a", "implied_hours_per_pos", "E_p25", "C+RY+E*Y", "2020-2025")
cat(sprintf("  %-38s: est=%+.4f [CV3J %.4f] 95%%CI=[%+.4f,%+.4f] p=%.4f\n",
            "R2a_implied_duration_1b (H - N)", r2_dur_1b$summary$estimate, r2_dur_1b$summary$se_cv3j, r2_dur_1b$summary$ci95_low_cv3j, r2_dur_1b$summary$ci95_high_cv3j, r2_dur_1b$summary$p_cv3j))
cat(sprintf("  %-38s: est=%+.4f [CV3J %.4f] 95%%CI=[%+.4f,%+.4f] p=%.4f\n",
            "R2a_implied_duration_1a (H - N)", r2_dur_1a$summary$estimate, r2_dur_1a$summary$se_cv3j, r2_dur_1a$summary$ci95_low_cv3j, r2_dur_1a$summary$ci95_high_cv3j, r2_dur_1a$summary$p_cv3j))

# 5. Confounding & deflation checks (CRITICAL for 1b rurality balance!)
r2_ppi_1b <- run_ref("R2a_ppi_decouple_1b", "r2_confound", "R2a Direct 1b + WPU01 PPI Decoupling",
                     p_r2_est, direct_hours_sy ~ E_p25_x_ln_aewr + E_p25_x_ln_ppi_2012 | county_fips + aewr_region_id^year,
                     "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+RY+PPI", "2020-2025")
r2_nom_1b <- run_ref("R2a_real_aewr_ppi_1b", "r2_confound", "R2a Direct 1b Real ln(AEWR_ppi)",
                     p_r2_est, direct_hours_sy ~ E_p25_x_ln_aewr_real | county_fips + aewr_region_id^year,
                     "E_p25_x_ln_aewr_real", "direct_hours_sy", "E_p25", "C+RY", "2020-2025 real PPI")
r2_nom_1a <- run_ref("R2a_real_aewr_ppi_1a", "r2_confound", "R2a Direct 1a Real ln(AEWR_ppi)",
                     p_r2_est, direct_hours_sy ~ E_p25_x_ln_aewr_real + i(year, E_p25, ref = 2020) | county_fips + aewr_region_id^year,
                     "E_p25_x_ln_aewr_real", "direct_hours_sy", "E_p25", "C+RY+E*Y", "2020-2025 real PPI")
r2_xpre_1b <- run_ref("R2a_xpre_x_year_1b", "r2_confound", "R2a Direct 1b + X_pre x Year FE (Rurality Balance)",
                      p_r2_est, direct_hours_sy ~ E_p25_x_ln_aewr + i(year, ln_pop_2010, ref = 2020) + i(year, farm_emp_share_0810, ref = 2020) + i(year, ln_emp_farm_0810, ref = 2020) | county_fips + aewr_region_id^year,
                      "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+RY+Xpre*Y", "2020-2025")
r2_xpre_1a <- run_ref("R2a_xpre_x_year_1a", "r2_confound", "R2a Direct 1a + X_pre x Year FE",
                      p_r2_est, direct_hours_sy ~ E_p25_x_ln_aewr + i(year, E_p25, ref = 2020) + i(year, ln_pop_2010, ref = 2020) + i(year, farm_emp_share_0810, ref = 2020) + i(year, ln_emp_farm_0810, ref = 2020) | county_fips + aewr_region_id^year,
                      "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+RY+E*Y+Xpre*Y", "2020-2025")
r2_cdl_1b <- run_ref("R2a_cdl0810_x_year_1b", "r2_confound", "R2a Direct 1b + CDL_0810 x Year FE",
                     p_r2_est, direct_hours_sy ~ E_p25_x_ln_aewr + i(year, cdl_fruit, ref = 2020) + i(year, cdl_veg, ref = 2020) + i(year, cdl_field, ref = 2020) + i(year, cdl_hort, ref = 2020) | county_fips + aewr_region_id^year,
                     "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+RY+CDL*Y", "2020-2025")
r2_cdl_1a <- run_ref("R2a_cdl0810_x_year_1a", "r2_confound", "R2a Direct 1a + CDL_0810 x Year FE",
                     p_r2_est, direct_hours_sy ~ E_p25_x_ln_aewr + i(year, E_p25, ref = 2020) + i(year, cdl_fruit, ref = 2020) + i(year, cdl_veg, ref = 2020) + i(year, cdl_field, ref = 2020) + i(year, cdl_hort, ref = 2020) | county_fips + aewr_region_id^year,
                     "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+RY+E*Y+CDL*Y", "2020-2025")

# 6. Drop top 1% direct-grower counties
r2_cum <- p_r2_est %>% group_by(county_fips) %>% summarise(ch = sum(direct_hours_sy), .groups = "drop")
r2_keep99 <- r2_cum$county_fips[r2_cum$ch <= quantile(r2_cum$ch, 0.99, names = FALSE)]
r2_top1_1b <- run_ref("R2a_drop_top1pct_1b", "r2_leverage", "R2a Direct 1b Drop Top 1% Counties",
                      p_r2_est %>% filter(county_fips %in% r2_keep99), direct_hours_sy ~ E_p25_x_ln_aewr | county_fips + aewr_region_id^year,
                      "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+RY", "2020-2025 drop top 1%")
r2_top1_1a <- run_ref("R2a_drop_top1pct_1a", "r2_leverage", "R2a Direct 1a Drop Top 1% Counties",
                      p_r2_est %>% filter(county_fips %in% r2_keep99), direct_hours_sy ~ E_p25_x_ln_aewr + i(year, E_p25, ref = 2020) | county_fips + aewr_region_id^year,
                      "E_p25_x_ln_aewr", "direct_hours_sy", "E_p25", "C+RY+E*Y", "2020-2025 drop top 1%")

# Combine and save complete referee audit table
ref_tbl <- bind_rows(
  r4_raw_ry$summary, r4_raw_sy$summary, r4_1a_ry$summary,
  r4_pos_ry$summary, r4_pos_sy$summary, r4_dur_ry$summary, r4_dur_sy$summary,
  r4_f1500$summary, r4_f4000$summary, r4_wks$summary,
  r4_1124_ry$summary, r4_1124_sy$summary, r4_dg1325$summary, r4_dg2025$summary, r4_ep25_same$summary,
  r4_ppi$summary, r4_nom$summary, r4_xpre$summary, r4_cdl$summary, r4_top1$summary,
  r2_pl_1b$summary, r2_pl_1a$summary,
  r2_2024_1b$summary, r2_2024_1a$summary, r2_fy_1b$summary, r2_fy_1a$summary,
  r2_all_1b$summary, r2_all_1a$summary, r2_lc_1b$summary, r2_1325_1b$summary, r2_sy_1b$summary, r2_sy_1a$summary,
  r2_pos_1b$summary, r2_pos_1a$summary, r2_dur_1b$summary, r2_dur_1a$summary,
  r2_ppi_1b$summary, r2_nom_1b$summary, r2_nom_1a$summary, r2_xpre_1b$summary, r2_xpre_1a$summary,
  r2_cdl_1b$summary, r2_cdl_1a$summary, r2_top1_1b$summary, r2_top1_1a$summary
) %>% mutate(mde_1sd_t16 = MDE_MULT_T16 * se_cv3j, mde_2sd_t16 = 2 * mde_1sd_t16, mde_2sd_2p8 = 2 * mde_cv3j)

write_csv(ref_tbl, "explorations/bite_power_rescue/phase2_referee_audit.csv")
cat(sprintf("\nWrote phase2_estimates.csv (%d rows), phase2_loo.csv (%d rows), phase2_referee_audit.csv (%d rows)\n",
            nrow(primary_tbl), nrow(loo_tbl), nrow(ref_tbl)))
