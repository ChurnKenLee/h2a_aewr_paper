# Purpose: Estimate the FY2026 Q4 mechanism heterogeneity module (Panels 1-5)
# explaining the near-zero one-season H-2A quantity response under the October 2,
# 2025 statewide AEWR rule using cached local disclosure, worksite, filer, and
# CDL crop-acreage inputs.
# Inputs:  $POLICY2025_WORK/quantity_county_fy.parquet,
#          $POLICY2025_WORK/quantity_estimates.csv,
#          $POLICY2025_WORK/quantity_state_deletions.csv,
#          $POLICY2025_WORK/filer/filer_ids.parquet,
#          data/intermediate/h2a_disclosure_records.parquet,
#          data/intermediate/h2a_allocated_worksites.parquet,
#          data/intermediate/h2a_with_fips.parquet,
#          data/intermediate/croplandcros_county_crop_type_acres.parquet
# Outputs: written to $POLICY2025_WORK (published by 12_publish.R after validation)

source("code/designs/policy_2025/helpers.R")
out <- id_output_dir()
cd_verify()

suppressPackageStartupMessages({
  library(ggplot2)
  library(tibble)
})

LAST_FY <- 2026L
FE_BASE <- "county_fips + aewr_region_id^year"

# ------------------------------------------------------------------------------
# 1. Exact Track 1 Multi-State Frame & Anchor Series (2,921 counties, 46 states)
# ------------------------------------------------------------------------------
qp_path <- file.path(out, "quantity_county_fy.parquet")
q_est_path <- file.path(out, "quantity_estimates.csv")
q_loo_path <- file.path(out, "quantity_state_deletions.csv")
filer_ids_path <- file.path(out, "filer", "filer_ids.parquet")

stopifnot(
  file.exists(qp_path),
  file.exists(q_est_path),
  file.exists(q_loo_path),
  file.exists(filer_ids_path)
)

qp <- read_parquet(
  qp_path,
  col_select = c(
    "county_fips", "state_fips", "aewr_region_id", "emp_farm_2011",
    "year", "x_us_skill_1", "x_placebo", "dose", "cert_full_fy", "hours_full_fy"
  )
) %>%
  mutate(
    county_fips = harmonize_county_fips_2010(county_fips),
    state_fips = state_fips(state_fips),
    aewr_region_id = aewr_region_id(aewr_region_id),
    year = as.integer(year)
  )

assert_geo_columns(qp, c("county_fips", "state_fips", "aewr_region_id"))

counties <- qp %>%
  filter(year == 2025L) %>%
  select(county_fips, state_fips, aewr_region_id, emp_farm_2011) %>%
  arrange(aewr_region_id, state_fips, county_fips)

cd_keys(counties, "county_fips")
all_46_states <- sort(unique(counties$state_fips))
stopifnot(
  nrow(counties) == 2921L,
  length(all_46_states) == 46L,
  n_distinct(counties$aewr_region_id) == 15L
)

qp_24_26 <- qp %>%
  filter(year %in% 2024L:2026L) %>%
  arrange(county_fips, year)

cd_keys(qp_24_26, c("county_fips", "year"))
stopifnot(nrow(qp_24_26) == 2921L * 3L)

# Baseline (predetermined FY2024-FY2025) positive-H2A county set for CDL splits
base_pos_h2a_counties <- qp_24_26 %>%
  filter(year %in% 2024L:2025L) %>%
  group_by(county_fips) %>%
  summarise(base_h2a_hours = sum(hours_full_fy, na.rm = TRUE), .groups = "drop") %>%
  filter(base_h2a_hours > 0)

# ------------------------------------------------------------------------------
# 2. Case Disclosure Pipeline (FY2021-FY2026) & Worksite Allocations
# ------------------------------------------------------------------------------
plausible_date <- function(value, fy) {
  y <- as.integer(format(value, "%Y"))
  value[!is.na(y) & abs(y - fy) > 1L] <- as.Date(NA)
  value
}

first_char <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x) & nzchar(trimws(x))]
  if (length(x) > 0L) x[[1L]] else NA_character_
}

first_num <- function(x) {
  v <- suppressWarnings(as.numeric(x))
  v <- v[is.finite(v)]
  if (length(v) > 0L) v[[1L]] else NA_real_
}

# Fixed FY2012-FY2025 pooled weekly-hours imputation mean (matches 11_estimate_quantity.R)
nh_pre <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(fiscal_year, period_begin_date, number_of_hours) %>%
  filter(fiscal_year >= 2012L, fiscal_year <= 2025L) %>%
  collect() %>%
  filter(!is.na(as.Date(period_begin_date)))

nh_pre_mean <- mean(suppressWarnings(as.numeric(nh_pre$number_of_hours)), na.rm = TRUE)
rm(nh_pre)
stopifnot(is.finite(nh_pre_mean), nh_pre_mean > 0)

disc_all <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
  select(
    fiscal_year, case_number, decision_date, case_received_date, case_status,
    nbr_workers_requested, nbr_workers_certified, h2a_labor_contractor,
    number_of_hours, period_begin_date, period_end_date,
    requested_begin_date, requested_end_date
  ) %>%
  filter(fiscal_year >= 2021L, fiscal_year <= LAST_FY) %>%
  collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(
    decision_raw = first_char(decision_date),
    received_raw = first_char(case_received_date),
    status_raw = first_char(case_status),
    req_workers = first_num(nbr_workers_requested),
    cert_workers = first_num(nbr_workers_certified),
    h2alc_raw = first_char(h2a_labor_contractor),
    nh_case_raw = first_num(number_of_hours),
    per_begin_raw = first_char(period_begin_date),
    per_end_raw = first_char(period_end_date),
    req_begin_raw = first_char(requested_begin_date),
    req_end_raw = first_char(requested_end_date),
    .groups = "drop"
  ) %>%
  mutate(
    decision_date = plausible_date(as.Date(substr(as.character(decision_raw), 1L, 10L)), fiscal_year),
    received = as.Date(substr(as.character(received_raw), 1L, 10L)),
    fy_start = as.Date(sprintf("%d-10-01", fiscal_year - 1L)),
    fy_end = as.Date(sprintf("%d-09-30", fiscal_year)),
    w_full_fy = !is.na(decision_date) & decision_date >= fy_start & decision_date <= fy_end,
    status_up = toupper(trimws(coalesce(status_raw, ""))),
    status_grp = case_when(
      grepl("PARTIAL", status_up) ~ "partial",
      grepl("DENIED", status_up) ~ "denied",
      grepl("WITHDRAWN", status_up) ~ "withdrawn",
      grepl("CERTIF", status_up) ~ "certified",
      TRUE ~ "other"
    ),
    h2alc_up = toupper(trimws(coalesce(h2alc_raw, ""))),
    is_h2alc = h2alc_up %in% c("Y", "YES", "TRUE", "1"),
    req_begin = as.Date(substr(as.character(req_begin_raw), 1L, 10L)),
    req_end = as.Date(substr(as.character(req_end_raw), 1L, 10L)),
    per_begin = as.Date(substr(as.character(per_begin_raw), 1L, 10L)),
    per_end = as.Date(substr(as.character(per_end_raw), 1L, 10L)),
    req_days_raw = as.numeric(req_end - req_begin),
    per_days_raw = as.numeric(per_end - per_begin),
    dur_days_case = ifelse(
      is.finite(req_days_raw) & req_days_raw > 0 & req_days_raw <= 400,
      req_days_raw,
      ifelse(is.finite(per_days_raw) & per_days_raw > 0 & per_days_raw <= 400, per_days_raw, NA_real_)
    ),
    nh_case = ifelse(is.finite(nh_case_raw) & nh_case_raw > 0 & nh_case_raw <= 100, nh_case_raw, nh_pre_mean),
    req_workers = coalesce(req_workers, 0),
    cert_workers = coalesce(cert_workers, 0),
    req_hours_case = ifelse(is.finite(dur_days_case) & dur_days_case > 0, dur_days_case / 7 * nh_case * req_workers, 0)
  ) %>%
  select(
    case_number, fiscal_year, decision_date, received, w_full_fy,
    status_grp, is_h2alc, req_workers, cert_workers, req_hours_case
  )

cd_keys(disc_all, c("case_number", "fiscal_year"))

filer_ids <- read_parquet(filer_ids_path) %>%
  transmute(
    fiscal_year = as.integer(fiscal_year),
    case_number = as.character(case_number),
    filer_R7 = as.character(filer_R7)
  ) %>%
  filter(fiscal_year >= 2023L, fiscal_year <= LAST_FY)

cd_keys(filer_ids, c("case_number", "fiscal_year"))

# Worksite allocations across FY2021-FY2026 (all post-FY2020 Addendum B)
ws_all <- open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
  select(
    fiscal_year, case_number, county_fips_list, adjusted_nbr_workers_certified,
    period_begin_date, period_end_date, number_of_hours,
    employer_is_worksite_business, employer_id_balanced
  ) %>%
  filter(fiscal_year >= 2021L, fiscal_year <= LAST_FY) %>%
  collect() %>%
  mutate(
    fiscal_year = as.integer(fiscal_year),
    begin = as.Date(period_begin_date),
    end = as.Date(period_end_date)
  ) %>%
  filter(!is.na(begin)) %>%
  left_join(
    disc_all %>% select(case_number, fiscal_year, decision_date, received, w_full_fy, is_h2alc),
    by = c("case_number", "fiscal_year"),
    relationship = "many-to-one"
  ) %>%
  filter(w_full_fy) %>%
  left_join(filer_ids, by = c("case_number", "fiscal_year"), relationship = "many-to-one") %>%
  mutate(
    workers = as.numeric(adjusted_nbr_workers_certified),
    nh_raw = suppressWarnings(as.numeric(number_of_hours)),
    nh = ifelse(is.na(nh_raw), nh_pre_mean, nh_raw),
    dur_weeks = as.numeric(end - begin) / 7,
    hours = dur_weeks * nh * workers,
    bus_id = ifelse(
      !is.na(employer_is_worksite_business) & as.logical(employer_is_worksite_business) &
        !is.na(employer_id_balanced) & nzchar(trimws(as.character(employer_id_balanced))),
      trimws(as.character(employer_id_balanced)),
      NA_character_
    ),
    is_wb = !is.na(bus_id),
    r7_id = ifelse(!is.na(filer_R7) & nzchar(trimws(filer_R7)), trimws(filer_R7), NA_character_),
    is_r7 = !is.na(r7_id),
    begin_month = as.integer(format(begin, "%m")),
    is_late_start = begin_month %in% 4L:9L,
    is_late_receipt = !is.na(received) & received >= as.Date(sprintf("%d-01-01", fiscal_year)),
    is_early_receipt = !is.na(received) & received < as.Date(sprintf("%d-01-01", fiscal_year)),
    county_fips_list = ifelse(is.na(county_fips_list) | county_fips_list == "", "00000", county_fips_list),
    n_counties = stringr::str_count(county_fips_list, ",") + 1L
  )

# Symmetric national incumbent lookbacks (EMPLOYER-001 worksite businesses + R7 filers)
wb_active <- ws_all %>%
  filter(is_wb, is.finite(workers), workers > 0) %>%
  distinct(bus_id, fiscal_year)

wb_status <- bind_rows(lapply(2024L:2026L, function(fy) {
  cur_ids <- unique(wb_active$bus_id[wb_active$fiscal_year == fy])
  prev_1 <- unique(wb_active$bus_id[wb_active$fiscal_year == fy - 1L])
  prev_3 <- unique(wb_active$bus_id[wb_active$fiscal_year %in% (fy - 3L):(fy - 1L)])
  tibble(
    bus_id = cur_ids,
    fiscal_year = fy,
    is_inc_1yr = cur_ids %in% prev_1,
    is_inc_3yr = cur_ids %in% prev_3
  )
}))
cd_keys(wb_status, c("bus_id", "fiscal_year"))

r7_active <- ws_all %>%
  filter(fiscal_year >= 2023L, is_r7, is.finite(workers), workers > 0) %>%
  distinct(r7_id, fiscal_year)

r7_status <- bind_rows(lapply(2024L:2026L, function(fy) {
  cur_ids <- unique(r7_active$r7_id[r7_active$fiscal_year == fy])
  prev_1 <- unique(r7_active$r7_id[r7_active$fiscal_year == fy - 1L])
  tibble(
    r7_id = cur_ids,
    fiscal_year = fy,
    is_r7_inc_1yr = cur_ids %in% prev_1
  )
}))
cd_keys(r7_status, c("r7_id", "fiscal_year"))

ws_24_26 <- ws_all %>%
  filter(fiscal_year %in% 2024L:2026L) %>%
  left_join(wb_status, by = c("bus_id", "fiscal_year"), relationship = "many-to-one") %>%
  left_join(r7_status, by = c("r7_id", "fiscal_year"), relationship = "many-to-one") %>%
  mutate(
    is_inc_1yr = coalesce(is_inc_1yr, FALSE),
    is_inc_3yr = coalesce(is_inc_3yr, FALSE),
    is_r7_inc_1yr = coalesce(is_r7_inc_1yr, FALSE)
  ) %>%
  tidyr::separate_rows(county_fips_list, sep = ",") %>%
  mutate(
    county_fips = harmonize_county_fips_2010(trimws(county_fips_list)),
    share = 1 / n_counties
  )

rm(ws_all)

# County x FY worksite mechanism aggregates (Panels 2, 3, 4A, 5)
ws_county <- ws_24_26 %>%
  semi_join(counties, by = "county_fips") %>%
  group_by(county_fips, year = fiscal_year) %>%
  summarise(
    cert_check = sum(workers * share, na.rm = TRUE),
    hours_check = sum(hours * share, na.rm = TRUE),
    # Panel 2: Incumbent vs Entrant (worksite business 1yr, 3yr, plus filer_R7)
    worksite_bus_hours = sum(hours * share * is_wb, na.rm = TRUE),
    non_wb_hours = sum(hours * share * (!is_wb), na.rm = TRUE),
    incumbent_1yr_hours = sum(hours * share * (is_wb & is_inc_1yr), na.rm = TRUE),
    entrant_1yr_hours = sum(hours * share * (is_wb & !is_inc_1yr), na.rm = TRUE),
    incumbent_1yr_positions = sum(workers * share * (is_wb & is_inc_1yr), na.rm = TRUE),
    entrant_1yr_positions = sum(workers * share * (is_wb & !is_inc_1yr), na.rm = TRUE),
    incumbent_3yr_hours = sum(hours * share * (is_wb & is_inc_3yr), na.rm = TRUE),
    entrant_3yr_hours = sum(hours * share * (is_wb & !is_inc_3yr), na.rm = TRUE),
    filer_r7_identified_hours = sum(hours * share * is_r7, na.rm = TRUE),
    filer_r7_incumbent_1yr_hours = sum(hours * share * (is_r7 & is_r7_inc_1yr), na.rm = TRUE),
    filer_r7_entrant_1yr_hours = sum(hours * share * (is_r7 & !is_r7_inc_1yr), na.rm = TRUE),
    n_businesses = as.numeric(n_distinct(bus_id[is_wb & is.finite(workers) & workers > 0])),
    n_incumbent_1yr_bus = as.numeric(n_distinct(bus_id[is_wb & is_inc_1yr & is.finite(workers) & workers > 0])),
    n_entrant_1yr_bus = as.numeric(n_distinct(bus_id[is_wb & !is_inc_1yr & is.finite(workers) & workers > 0])),
    # Panel 3: H-2ALC Contractors vs Direct Growers
    h2alc_hours = sum(hours * share * is_h2alc, na.rm = TRUE),
    direct_hours = sum(hours * share * (!is_h2alc), na.rm = TRUE),
    h2alc_positions = sum(workers * share * is_h2alc, na.rm = TRUE),
    direct_positions = sum(workers * share * (!is_h2alc), na.rm = TRUE),
    # Panel 4A: Planning Lead-Time (start month & receipt date)
    early_start_hours = sum(hours * share * (!is_late_start), na.rm = TRUE),
    late_start_hours = sum(hours * share * is_late_start, na.rm = TRUE),
    early_receipt_hours = sum(hours * share * is_early_receipt, na.rm = TRUE),
    late_receipt_hours = sum(hours * share * is_late_receipt, na.rm = TRUE),
    missing_receipt_hours = sum(hours * share * is.na(received), na.rm = TRUE),
    # Panel 5: Intensive Contract Duration & Observed Weekly Hours
    dur_weeks_num = sum(dur_weeks * workers * share * (is.finite(dur_weeks) & dur_weeks > 0), na.rm = TRUE),
    dur_weeks_den = sum(workers * share * (is.finite(dur_weeks) & dur_weeks > 0), na.rm = TRUE),
    obs_nh_num = sum(nh_raw * workers * share * (is.finite(nh_raw) & nh_raw > 0 & nh_raw <= 100), na.rm = TRUE),
    obs_nh_den = sum(workers * share * (is.finite(nh_raw) & nh_raw > 0 & nh_raw <= 100), na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------------------------
# 3. Full Application Pipeline County Weights (incl. Denied & Withdrawn Cases)
# ------------------------------------------------------------------------------
disc_24_26 <- disc_all %>% filter(fiscal_year %in% 2024L:2026L, w_full_fy)
rm(disc_all)

alloc_case_weights <- ws_24_26 %>%
  group_by(case_number, fiscal_year, county_fips) %>%
  summarise(
    w_cert = sum(workers * share, na.rm = TRUE),
    w_row = sum(share, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(case_number, fiscal_year) %>%
  mutate(
    tot_cert = sum(w_cert),
    tot_row = sum(w_row),
    w_c = ifelse(tot_cert > 0, w_cert / tot_cert, w_row / tot_row)
  ) %>%
  ungroup() %>%
  select(case_number, fiscal_year, county_fips, w_c)

rm(ws_24_26)

alloc_keys <- alloc_case_weights %>% distinct(case_number, fiscal_year)
unalloc_cases <- disc_24_26 %>%
  select(case_number, fiscal_year) %>%
  anti_join(alloc_keys, by = c("case_number", "fiscal_year"))

unalloc_raw <- open_dataset(path_int("h2a_with_fips.parquet")) %>%
  select(case_number, fiscal_year, county_fips_list) %>%
  filter(fiscal_year >= 2024L, fiscal_year <= LAST_FY) %>%
  collect() %>%
  mutate(fiscal_year = as.integer(fiscal_year)) %>%
  semi_join(unalloc_cases, by = c("case_number", "fiscal_year")) %>%
  mutate(
    county_fips_list = ifelse(is.na(county_fips_list) | trimws(county_fips_list) == "", "00000", trimws(county_fips_list)),
    n_c = stringr::str_count(county_fips_list, ",") + 1L
  ) %>%
  tidyr::separate_rows(county_fips_list, sep = ",") %>%
  mutate(
    tok = trimws(county_fips_list),
    tok = ifelse(grepl("^[0-9]{1,5}$", tok), tok, "00000"),
    county_fips = harmonize_county_fips_2010(tok),
    w_row = 1 / n_c
  ) %>%
  select(-tok)

unalloc_case_weights <- if (nrow(unalloc_raw) > 0L) {
  unalloc_raw %>%
    group_by(case_number, fiscal_year, county_fips) %>%
    summarise(w_row = sum(w_row, na.rm = TRUE), .groups = "drop") %>%
    group_by(case_number, fiscal_year) %>%
    mutate(w_c = w_row / sum(w_row)) %>%
    ungroup() %>%
    select(case_number, fiscal_year, county_fips, w_c)
} else {
  tibble(case_number = character(), fiscal_year = integer(), county_fips = character(), w_c = numeric())
}
rm(unalloc_raw)

case_geo_weights <- bind_rows(alloc_case_weights, unalloc_case_weights)
wt_check <- case_geo_weights %>%
  group_by(case_number, fiscal_year) %>%
  summarise(sum_w = sum(w_c), .groups = "drop")
stopifnot(max(abs(wt_check$sum_w - 1)) < 1e-10)

pipe_county <- disc_24_26 %>%
  inner_join(case_geo_weights, by = c("case_number", "fiscal_year"), relationship = "one-to-many") %>%
  semi_join(counties, by = "county_fips") %>%
  group_by(county_fips, year = fiscal_year) %>%
  summarise(
    requested_positions = sum(req_workers * w_c, na.rm = TRUE),
    requested_hours = sum(req_hours_case * w_c, na.rm = TRUE),
    cert_pipeline_positions = sum(cert_workers * w_c, na.rm = TRUE),
    filed_apps = sum(w_c, na.rm = TRUE),
    certified_apps = sum(w_c * (status_grp %in% c("certified", "partial")), na.rm = TRUE),
    denied_withdrawn_apps = sum(w_c * (status_grp %in% c("denied", "withdrawn")), na.rm = TRUE),
    other_apps = sum(w_c * (status_grp == "other"), na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------------------------
# 4. Predetermined 2022-2024 CDL Crop Capital Shares (2,921/2,921 coverage)
# ------------------------------------------------------------------------------
cdl_base <- read_parquet(path_int("croplandcros_county_crop_type_acres.parquet")) %>%
  mutate(
    county_fips = harmonize_county_fips_2010(county_fips),
    year = as.integer(year),
    crop_grp = tolower(trimws(as.character(crop_type_label))),
    acres = as.numeric(acres)
  ) %>%
  filter(year %in% 2022L:2024L, crop_grp != "non-crop") %>%
  group_by(county_fips) %>%
  summarise(
    crop_acres_22_24 = sum(acres, na.rm = TRUE),
    perennial_acres_22_24 = sum(acres[crop_grp == "fruit & tree nuts"], na.rm = TRUE),
    vegetable_acres_22_24 = sum(acres[crop_grp == "vegetables"], na.rm = TRUE),
    field_acres_22_24 = sum(acres[crop_grp == "field crops"], na.rm = TRUE),
    .groups = "drop"
  ) %>%
  right_join(counties %>% select(county_fips), by = "county_fips") %>%
  mutate(
    crop_acres_22_24 = coalesce(crop_acres_22_24, 0),
    perennial_acres_22_24 = coalesce(perennial_acres_22_24, 0),
    vegetable_acres_22_24 = coalesce(vegetable_acres_22_24, 0),
    field_acres_22_24 = coalesce(field_acres_22_24, 0),
    perennial_share = ifelse(crop_acres_22_24 > 0, perennial_acres_22_24 / crop_acres_22_24, 0),
    vegetable_share = ifelse(crop_acres_22_24 > 0, vegetable_acres_22_24 / crop_acres_22_24, 0)
  )

cd_keys(cdl_base, "county_fips")
stopifnot(nrow(cdl_base) == 2921L, all(is.finite(cdl_base$perennial_share)), all(is.finite(cdl_base$vegetable_share)))

cdl_pos <- cdl_base %>% semi_join(base_pos_h2a_counties, by = "county_fips")
per_p75 <- unname(stats::quantile(cdl_pos$perennial_share, probs = 0.75, na.rm = TRUE))
veg_p75 <- unname(stats::quantile(cdl_pos$vegetable_share, probs = 0.75, na.rm = TRUE))
per_mean <- mean(cdl_pos$perennial_share, na.rm = TRUE)
per_sd <- stats::sd(cdl_pos$perennial_share, na.rm = TRUE)
veg_mean <- mean(cdl_pos$vegetable_share, na.rm = TRUE)
veg_sd <- stats::sd(cdl_pos$vegetable_share, na.rm = TRUE)

stopifnot(is.finite(per_sd), per_sd > 0, is.finite(veg_sd), veg_sd > 0)

cdl_base <- cdl_base %>%
  mutate(
    perennial_share_std = (perennial_share - per_mean) / per_sd,
    vegetable_share_std = (vegetable_share - veg_mean) / veg_sd,
    perennial_high = perennial_share >= per_p75 & perennial_share > 0,
    vegetable_high = vegetable_share >= veg_p75 & vegetable_share > 0
  )

# ------------------------------------------------------------------------------
# 5. Assemble County x FY Mechanism Panel & Verify Strict Conservation
# ------------------------------------------------------------------------------
mech_panel <- qp_24_26 %>%
  left_join(ws_county, by = c("county_fips", "year"), relationship = "one-to-one") %>%
  left_join(pipe_county, by = c("county_fips", "year"), relationship = "one-to-one") %>%
  left_join(
    cdl_base %>% select(
      county_fips, crop_acres_22_24, perennial_share, vegetable_share,
      perennial_share_std, vegetable_share_std, perennial_high, vegetable_high
    ),
    by = "county_fips",
    relationship = "many-to-one"
  ) %>%
  mutate(
    across(
      c(
        cert_check, hours_check, worksite_bus_hours, non_wb_hours,
        incumbent_1yr_hours, entrant_1yr_hours, incumbent_1yr_positions, entrant_1yr_positions,
        incumbent_3yr_hours, entrant_3yr_hours,
        filer_r7_identified_hours, filer_r7_incumbent_1yr_hours, filer_r7_entrant_1yr_hours,
        n_businesses, n_incumbent_1yr_bus, n_entrant_1yr_bus,
        h2alc_hours, direct_hours, h2alc_positions, direct_positions,
        early_start_hours, late_start_hours, early_receipt_hours, late_receipt_hours, missing_receipt_hours,
        dur_weeks_num, dur_weeks_den, obs_nh_num, obs_nh_den,
        requested_positions, requested_hours, cert_pipeline_positions,
        filed_apps, certified_apps, denied_withdrawn_apps, other_apps
      ),
      ~ coalesce(.x, 0)
    ),
    approval_rate = ifelse(requested_positions > 0, cert_pipeline_positions / requested_positions, NA_real_),
    denial_withdrawal_share = ifelse(filed_apps > 0, denied_withdrawn_apps / filed_apps, NA_real_),
    h2alc_hours_share = ifelse(hours_full_fy > 0, h2alc_hours / hours_full_fy, NA_real_),
    mean_contract_weeks = ifelse(dur_weeks_den > 0, dur_weeks_num / dur_weeks_den, NA_real_),
    mean_weekly_hours = ifelse(obs_nh_den > 0, obs_nh_num / obs_nh_den, NA_real_)
  )

assert_geo_columns(mech_panel, c("county_fips", "state_fips", "aewr_region_id"))
cd_keys(mech_panel, c("county_fips", "year"))

# Strict conservation assertions (< 1e-8)
stopifnot(
  nrow(mech_panel) == 2921L * 3L,
  max(abs(mech_panel$hours_check - mech_panel$hours_full_fy)) < 1e-8,
  max(abs(mech_panel$cert_check - mech_panel$cert_full_fy)) < 1e-8,
  max(abs((mech_panel$incumbent_1yr_hours + mech_panel$entrant_1yr_hours) - mech_panel$worksite_bus_hours)) < 1e-8,
  max(abs((mech_panel$worksite_bus_hours + mech_panel$non_wb_hours) - mech_panel$hours_full_fy)) < 1e-8,
  max(abs((mech_panel$incumbent_3yr_hours + mech_panel$entrant_3yr_hours) - mech_panel$worksite_bus_hours)) < 1e-8,
  max(abs((mech_panel$filer_r7_incumbent_1yr_hours + mech_panel$filer_r7_entrant_1yr_hours) - mech_panel$filer_r7_identified_hours)) < 1e-8,
  max(abs((mech_panel$n_incumbent_1yr_bus + mech_panel$n_entrant_1yr_bus) - mech_panel$n_businesses)) < 1e-10,
  max(abs((mech_panel$h2alc_hours + mech_panel$direct_hours) - mech_panel$hours_full_fy)) < 1e-8,
  max(abs((mech_panel$h2alc_positions + mech_panel$direct_positions) - mech_panel$cert_full_fy)) < 1e-8,
  max(abs((mech_panel$early_start_hours + mech_panel$late_start_hours) - mech_panel$hours_full_fy)) < 1e-8,
  max(abs((mech_panel$early_receipt_hours + mech_panel$late_receipt_hours + mech_panel$missing_receipt_hours) - mech_panel$hours_full_fy)) < 1e-8,
  max(abs((mech_panel$certified_apps + mech_panel$denied_withdrawn_apps + mech_panel$other_apps) - mech_panel$filed_apps)) < 1e-8
)

mech_panel <- mech_panel %>% select(-cert_check, -hours_check, -dur_weeks_num, -dur_weeks_den, -obs_nh_num, -obs_nh_den)
write_parquet(mech_panel, file.path(out, "mechanisms_county_fy.parquet"))

# Descriptive subgroup totals & shares (FY2024-FY2026)
desc_tbl <- mech_panel %>%
  group_by(year) %>%
  summarise(
    counties_frame = n(),
    hours_full_fy = sum(hours_full_fy),
    cert_full_fy = sum(cert_full_fy),
    requested_hours = sum(requested_hours),
    requested_positions = sum(requested_positions),
    cert_pipeline_positions = sum(cert_pipeline_positions),
    filed_apps = sum(filed_apps),
    certified_apps = sum(certified_apps),
    denied_withdrawn_apps = sum(denied_withdrawn_apps),
    pipeline_approval_ratio = sum(cert_pipeline_positions) / sum(requested_positions),
    worksite_bus_hours_share = sum(worksite_bus_hours) / sum(hours_full_fy),
    incumbent_1yr_hours_share = sum(incumbent_1yr_hours) / sum(worksite_bus_hours),
    entrant_1yr_hours_share = sum(entrant_1yr_hours) / sum(worksite_bus_hours),
    incumbent_3yr_hours_share = sum(incumbent_3yr_hours) / sum(worksite_bus_hours),
    filer_r7_incumbent_hours_share = sum(filer_r7_incumbent_1yr_hours) / sum(filer_r7_identified_hours),
    h2alc_hours_share = sum(h2alc_hours) / sum(hours_full_fy),
    direct_hours_share = sum(direct_hours) / sum(hours_full_fy),
    early_start_hours_share = sum(early_start_hours) / sum(hours_full_fy),
    late_start_hours_share = sum(late_start_hours) / sum(hours_full_fy),
    early_receipt_hours_share = sum(early_receipt_hours) / sum(hours_full_fy),
    late_receipt_hours_share = sum(late_receipt_hours) / sum(hours_full_fy),
    perennial_high_hours_share = sum(hours_full_fy[perennial_high]) / sum(hours_full_fy),
    vegetable_high_hours_share = sum(hours_full_fy[vegetable_high]) / sum(hours_full_fy),
    perennial_p75_cutoff = per_p75,
    perennial_mean_pos_h2a = per_mean,
    perennial_sd_pos_h2a = per_sd,
    vegetable_p75_cutoff = veg_p75,
    vegetable_mean_pos_h2a = veg_mean,
    vegetable_sd_pos_h2a = veg_sd,
    total_businesses_county_sum = sum(n_businesses),
    mean_contract_weeks_unweighted = mean(mean_contract_weeks, na.rm = TRUE),
    mean_weekly_hours_unweighted = mean(mean_weekly_hours, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(year)

cd_write(desc_tbl, out, "mechanism_descriptives.csv")

# ------------------------------------------------------------------------------
# 6. Estimation Machinery (Exact CV3J Jackknife + Paired 46-State Differences)
# ------------------------------------------------------------------------------
complete_df <- function(d, cols) {
  d <- as.data.frame(d)
  for (v in cols) {
    keep <- if (is.numeric(d[[v]])) is.finite(d[[v]]) else (!is.na(d[[v]]) & nzchar(as.character(d[[v]])))
    d <- d[keep, , drop = FALSE]
  }
  d
}

balanced_two_yr <- function(d, y_col) {
  d %>%
    group_by(county_fips) %>%
    filter(sum(is.finite(.data[[y_col]])) == 2L) %>%
    ungroup()
}

fit_mech_model <- function(d, fml, ppml = TRUE) {
  tryCatch(
    if (ppml) {
      fixest::fepois(fml, data = d, vcov = ~state_fips, notes = FALSE)
    } else {
      fixest::feols(fml, data = d, vcov = ~state_fips, notes = FALSE)
    },
    error = function(e) NULL
  )
}

mech_results <- list()
mech_loos <- list()

run_mech <- function(
  panel_id,
  contrast,
  specification,
  outcome_label,
  d,
  y_col,
  x_col,
  rhs_extra = NULL,
  term_col = x_col,
  ppml = TRUE,
  scale_mode = c("ppml_price", "ols_elasticity", "level_rate"),
  in_primary_family = FALSE
) {
  scale_mode <- match.arg(scale_mode)
  needed <- unique(c(y_col, x_col, "county_fips", "state_fips", "aewr_region_id", "year"))
  if (!is.null(rhs_extra)) {
    extra_vars <- intersect(all.vars(stats::as.formula(paste("~", paste(rhs_extra, collapse = " + ")))), names(d))
    needed <- unique(c(needed, extra_vars))
  }
  cc <- complete_df(d, needed)
  rhs_str <- paste(c(x_col, rhs_extra), collapse = " + ")
  fml <- stats::as.formula(paste0(y_col, " ~ ", rhs_str, " | ", FE_BASE))
  ctrl_lbl <- if (is.null(rhs_extra)) "none" else paste(rhs_extra, collapse = "+")

  m <- fit_mech_model(cc, fml, ppml = ppml)
  if (is.null(m) || !term_col %in% names(stats::coef(m))) {
    row <- tibble(
      track = "mechanisms", panel = panel_id, contrast = contrast,
      specification = specification, outcome = outcome_label,
      estimator = if (ppml) "PPML" else "OLS", treatment = x_col, term = term_col,
      controls = ctrl_lbl, fe = FE_BASE, status = "fit_failed",
      n = as.integer(nrow(cc)), counties = as.integer(n_distinct(cc$county_fips)),
      states = as.integer(n_distinct(cc$state_fips)), regions = as.integer(n_distinct(cc$aewr_region_id)),
      first_year = if (nrow(cc) > 0L) as.integer(min(cc$year)) else NA_integer_,
      last_year = if (nrow(cc) > 0L) as.integer(max(cc$year)) else NA_integer_,
      mean_y = NA_real_, mean_price = NA_real_,
      estimate = NA_real_, se_jackknife = NA_real_, ci95_low_jk = NA_real_, ci95_high_jk = NA_real_,
      p_jackknife = NA_real_, se_state_crv1 = NA_real_, p_state_crv1 = NA_real_,
      se_region_crv1 = NA_real_, p_region_crv1 = NA_real_,
      elasticity = NA_real_, elasticity_se_jk = NA_real_,
      elasticity_low_jk = NA_real_, elasticity_high_jk = NA_real_, mde_elasticity_jk = NA_real_,
      in_primary_family = isTRUE(in_primary_family)
    )
    mech_results[[length(mech_results) + 1L]] <<- row
    return(invisible(list(row = row, loo = tibble(), ok = FALSE, mean_p = NA_real_, scale_el = NA_real_)))
  }

  dd <- cc[fixest::obs(m), , drop = FALSE]
  states <- sort(unique(dd$state_fips))
  G <- length(states)
  R_n <- n_distinct(dd$aewr_region_id)
  b <- unname(stats::coef(m)[[term_col]])

  loo_vals <- vapply(states, function(s) {
    mm <- fit_mech_model(cc[cc$state_fips != s, , drop = FALSE], fml, ppml = ppml)
    if (is.null(mm) || !term_col %in% names(stats::coef(mm))) NA_real_ else unname(stats::coef(mm)[[term_col]])
  }, numeric(1))

  ok_jk <- is.finite(b) && G >= 2L && length(loo_vals) == G && all(is.finite(loo_vals))
  se_jk <- if (ok_jk) sqrt((G - 1) / G * sum((loo_vals - mean(loo_vals))^2)) else NA_real_
  tq <- if (G >= 2L) stats::qt(0.975, df = G - 1L) else NA_real_
  p_jk <- if (ok_jk && se_jk > 0) 2 * stats::pt(-abs(b / se_jk), df = G - 1L) else NA_real_

  s_st <- summary(m, vcov = ~state_fips)
  se_st <- unname(fixest::se(s_st)[[term_col]])
  p_st <- if (is.finite(se_st) && se_st > 0 && G >= 2L) 2 * stats::pt(-abs(b / se_st), df = G - 1L) else NA_real_

  s_reg <- summary(m, vcov = ~aewr_region_id)
  se_reg <- unname(fixest::se(s_reg)[[term_col]])
  p_reg <- if (is.finite(se_reg) && se_reg > 0 && R_n >= 2L) 2 * stats::pt(-abs(b / se_reg), df = R_n - 1L) else NA_real_

  mean_y <- mean(dd[[y_col]])
  mean_p <- mean(dd[[x_col]])

  if (scale_mode == "ppml_price") {
    scale_el <- mean_p
  } else if (scale_mode == "ols_elasticity") {
    scale_el <- mean_p / mean_y
  } else {
    scale_el <- NA_real_
  }

  el_b <- if (is.finite(scale_el)) b * scale_el else NA_real_
  el_se <- if (ok_jk && is.finite(scale_el)) se_jk * scale_el else NA_real_
  el_lo <- if (ok_jk && is.finite(scale_el)) (b - tq * se_jk) * scale_el else NA_real_
  el_hi <- if (ok_jk && is.finite(scale_el)) (b + tq * se_jk) * scale_el else NA_real_
  el_mde <- if (ok_jk && is.finite(scale_el)) 2.8 * se_jk * scale_el else NA_real_

  row <- tibble(
    track = "mechanisms",
    panel = panel_id,
    contrast = contrast,
    specification = specification,
    outcome = outcome_label,
    estimator = if (ppml) "PPML" else "OLS",
    treatment = x_col,
    term = term_col,
    controls = ctrl_lbl,
    fe = FE_BASE,
    status = if (ok_jk) "estimated" else "jackknife_not_identified",
    n = as.integer(stats::nobs(m)),
    counties = as.integer(n_distinct(dd$county_fips)),
    states = as.integer(G),
    regions = as.integer(R_n),
    first_year = as.integer(min(dd$year)),
    last_year = as.integer(max(dd$year)),
    mean_y = mean_y,
    mean_price = mean_p,
    estimate = b,
    se_jackknife = se_jk,
    ci95_low_jk = if (ok_jk) b - tq * se_jk else NA_real_,
    ci95_high_jk = if (ok_jk) b + tq * se_jk else NA_real_,
    p_jackknife = p_jk,
    se_state_crv1 = se_st,
    p_state_crv1 = p_st,
    se_region_crv1 = se_reg,
    p_region_crv1 = p_reg,
    elasticity = el_b,
    elasticity_se_jk = el_se,
    elasticity_low_jk = el_lo,
    elasticity_high_jk = el_hi,
    mde_elasticity_jk = el_mde,
    in_primary_family = isTRUE(in_primary_family)
  )

  loo_tbl <- tibble(
    panel = panel_id,
    contrast = contrast,
    specification = specification,
    outcome = outcome_label,
    estimator = if (ppml) "PPML" else "OLS",
    term = term_col,
    omitted_state = states,
    estimate = loo_vals,
    elasticity = if (is.finite(scale_el)) loo_vals * scale_el else NA_real_
  )

  mech_results[[length(mech_results) + 1L]] <<- row
  mech_loos[[length(mech_loos) + 1L]] <<- loo_tbl
  invisible(list(row = row, loo = loo_tbl, ok = ok_jk, mean_p = mean_p, scale_el = scale_el))
}

# Exact delete-one-state paired difference over the 46-state universe
loo_universe_vec <- function(res_obj, all_states, use_el = TRUE) {
  if (!isTRUE(res_obj$ok)) return(NULL)
  base_val <- if (use_el) res_obj$row$elasticity[[1]] else res_obj$row$estimate[[1]]
  v <- stats::setNames(rep(base_val, length(all_states)), all_states)
  loo_d <- res_obj$loo
  col_nm <- if (use_el) "elasticity" else "estimate"
  idx <- match(loo_d$omitted_state, all_states)
  v[idx] <- loo_d[[col_nm]]
  v
}

run_paired_diff <- function(
  panel_id,
  contrast,
  specification,
  outcome_label,
  res_a,
  res_b,
  estimator_label = "PPML_DIFF_JK",
  in_primary_family = FALSE
) {
  ok_both <- isTRUE(res_a$ok) && isTRUE(res_b$ok)
  G_u <- length(all_46_states)
  tq <- stats::qt(0.975, df = G_u - 1L)

  if (!ok_both) {
    row <- tibble(
      track = "mechanisms", panel = panel_id, contrast = contrast,
      specification = specification, outcome = outcome_label,
      estimator = estimator_label, treatment = res_a$row$treatment[[1]], term = "paired_diff",
      controls = "none", fe = FE_BASE, status = "jackknife_not_identified",
      n = as.integer(max(res_a$row$n[[1]], res_b$row$n[[1]], na.rm = TRUE)),
      counties = as.integer(max(res_a$row$counties[[1]], res_b$row$counties[[1]], na.rm = TRUE)),
      states = as.integer(G_u), regions = 15L,
      first_year = res_a$row$first_year[[1]], last_year = res_a$row$last_year[[1]],
      mean_y = NA_real_, mean_price = res_a$row$mean_price[[1]],
      estimate = NA_real_, se_jackknife = NA_real_, ci95_low_jk = NA_real_, ci95_high_jk = NA_real_,
      p_jackknife = NA_real_, se_state_crv1 = NA_real_, p_state_crv1 = NA_real_,
      se_region_crv1 = NA_real_, p_region_crv1 = NA_real_,
      elasticity = NA_real_, elasticity_se_jk = NA_real_,
      elasticity_low_jk = NA_real_, elasticity_high_jk = NA_real_, mde_elasticity_jk = NA_real_,
      in_primary_family = isTRUE(in_primary_family)
    )
    mech_results[[length(mech_results) + 1L]] <<- row
    return(invisible(list(row = row, ok = FALSE)))
  }

  v_b_a <- loo_universe_vec(res_a, all_46_states, use_el = FALSE)
  v_b_b <- loo_universe_vec(res_b, all_46_states, use_el = FALSE)
  diff_b_loo <- v_b_a - v_b_b
  b_diff <- res_a$row$estimate[[1]] - res_b$row$estimate[[1]]
  se_b_jk <- sqrt((G_u - 1) / G_u * sum((diff_b_loo - mean(diff_b_loo))^2))

  v_el_a <- loo_universe_vec(res_a, all_46_states, use_el = TRUE)
  v_el_b <- loo_universe_vec(res_b, all_46_states, use_el = TRUE)
  diff_el_loo <- v_el_a - v_el_b
  el_diff <- res_a$row$elasticity[[1]] - res_b$row$elasticity[[1]]
  se_el_jk <- sqrt((G_u - 1) / G_u * sum((diff_el_loo - mean(diff_el_loo))^2))
  p_jk <- if (is.finite(se_el_jk) && se_el_jk > 0) 2 * stats::pt(-abs(el_diff / se_el_jk), df = G_u - 1L) else NA_real_

  row <- tibble(
    track = "mechanisms",
    panel = panel_id,
    contrast = contrast,
    specification = specification,
    outcome = outcome_label,
    estimator = estimator_label,
    treatment = res_a$row$treatment[[1]],
    term = "paired_diff",
    controls = "none",
    fe = FE_BASE,
    status = "estimated",
    n = as.integer(max(res_a$row$n[[1]], res_b$row$n[[1]])),
    counties = as.integer(max(res_a$row$counties[[1]], res_b$row$counties[[1]])),
    states = as.integer(G_u),
    regions = 15L,
    first_year = res_a$row$first_year[[1]],
    last_year = res_a$row$last_year[[1]],
    mean_y = NA_real_,
    mean_price = mean(c(res_a$row$mean_price[[1]], res_b$row$mean_price[[1]]), na.rm = TRUE),
    estimate = b_diff,
    se_jackknife = se_b_jk,
    ci95_low_jk = b_diff - tq * se_b_jk,
    ci95_high_jk = b_diff + tq * se_b_jk,
    p_jackknife = p_jk,
    se_state_crv1 = NA_real_,
    p_state_crv1 = NA_real_,
    se_region_crv1 = NA_real_,
    p_region_crv1 = NA_real_,
    elasticity = el_diff,
    elasticity_se_jk = se_el_jk,
    elasticity_low_jk = el_diff - tq * se_el_jk,
    elasticity_high_jk = el_diff + tq * se_el_jk,
    mde_elasticity_jk = 2.8 * se_el_jk,
    in_primary_family = isTRUE(in_primary_family)
  )

  loo_tbl <- tibble(
    panel = panel_id,
    contrast = contrast,
    specification = specification,
    outcome = outcome_label,
    estimator = estimator_label,
    term = "paired_diff",
    omitted_state = all_46_states,
    estimate = unname(diff_b_loo),
    elasticity = unname(diff_el_loo)
  )

  mech_results[[length(mech_results) + 1L]] <<- row
  mech_loos[[length(mech_loos) + 1L]] <<- loo_tbl
  invisible(list(row = row, loo = loo_tbl, ok = TRUE))
}

# Load Track 1 headline anchors (hours & cert PPML) for paired Panel 1 & Panel 5 diffs
q_est <- read_csv(q_est_path, show_col_types = FALSE)
q_loo <- read_csv(q_loo_path, col_types = cols(omitted_state = "c", .default = col_guess()))

load_q_anchor <- function(contrast_nm, out_nm) {
  spec_nm <- if (contrast_nm == "fy2025_2026") "B_primary" else "placebo_fy2024_fy2025"
  r <- q_est %>% filter(window == "full_fy", contrast == contrast_nm, specification == spec_nm, outcome == out_nm, estimator == "PPML")
  stopifnot(nrow(r) == 1L)
  l <- q_loo %>%
    filter(window == "full_fy", contrast == contrast_nm, specification == spec_nm, outcome == out_nm, estimator == "PPML") %>%
    transmute(omitted_state = state_fips(omitted_state), estimate = estimate, elasticity = estimate * r$mean_price[[1]])
  stopifnot(nrow(l) == r$states[[1]])
  list(row = r, loo = l, ok = (r$status[[1]] == "estimated"), mean_p = r$mean_price[[1]], scale_el = r$mean_price[[1]])
}

# ------------------------------------------------------------------------------
# 7. Estimate All 5 Panels for FY2025 vs FY2026 AND FY2024 vs FY2025 Placebo
# ------------------------------------------------------------------------------
run_contrast_suite <- function(contrast_nm, years_pair, x_col, post_yr, is_main) {
  d <- mech_panel %>%
    filter(year %in% years_pair) %>%
    mutate(
      z_post_per = perennial_share_std * as.numeric(year == post_yr),
      x_per_int = .data[[x_col]] * perennial_share_std,
      z_post_veg = vegetable_share_std * as.numeric(year == post_yr),
      x_veg_int = .data[[x_col]] * vegetable_share_std
    )

  anc_hours <- load_q_anchor(contrast_nm, "hours")
  anc_cert <- load_q_anchor(contrast_nm, "cert")

  # --- Panel 1: Application Pipeline & Bottlenecks (8 rows) ---
  r_req_h <- run_mech("P1_pipeline", contrast_nm, "pipeline_requested_hours", "requested_hours", d, "requested_hours", x_col, ppml = TRUE, scale_mode = "ppml_price", in_primary_family = is_main)
  run_mech("P1_pipeline", contrast_nm, "pipeline_requested_positions", "requested_positions", d, "requested_positions", x_col, ppml = TRUE, scale_mode = "ppml_price", in_primary_family = is_main)
  run_mech("P1_pipeline", contrast_nm, "pipeline_filed_apps", "filed_apps", d, "filed_apps", x_col, ppml = TRUE, scale_mode = "ppml_price")
  r_cert_apps <- run_mech("P1_pipeline", contrast_nm, "pipeline_certified_apps", "certified_apps", d, "certified_apps", x_col, ppml = TRUE, scale_mode = "ppml_price")
  run_mech("P1_pipeline", contrast_nm, "pipeline_denied_withdrawn_apps", "denied_withdrawn_apps", d, "denied_withdrawn_apps", x_col, ppml = TRUE, scale_mode = "ppml_price")
  run_mech("P1_pipeline", contrast_nm, "pipeline_approval_rate", "approval_rate", balanced_two_yr(d, "approval_rate"), "approval_rate", x_col, ppml = FALSE, scale_mode = "level_rate", in_primary_family = is_main)
  run_mech("P1_pipeline", contrast_nm, "pipeline_denial_withdrawal_share", "denial_withdrawal_share", balanced_two_yr(d, "denial_withdrawal_share"), "denial_withdrawal_share", x_col, ppml = FALSE, scale_mode = "level_rate")
  run_paired_diff("P1_pipeline", contrast_nm, "pipeline_diff_req_minus_cert_hours", "diff_requested_minus_certified_hours", r_req_h, anc_hours, estimator_label = "PPML_DIFF_JK")

  # --- Panel 2: Incumbent Capacity vs Employer Entry (14 rows) ---
  r_inc1_h <- run_mech("P2_incumbent_entry", contrast_nm, "wb_1yr_incumbent_hours", "incumbent_1yr_hours", d, "incumbent_1yr_hours", x_col, ppml = TRUE, scale_mode = "ppml_price", in_primary_family = is_main)
  r_ent1_h <- run_mech("P2_incumbent_entry", contrast_nm, "wb_1yr_entrant_hours", "entrant_1yr_hours", d, "entrant_1yr_hours", x_col, ppml = TRUE, scale_mode = "ppml_price", in_primary_family = is_main)
  run_paired_diff("P2_incumbent_entry", contrast_nm, "wb_1yr_diff_entrant_minus_incumbent_hours", "diff_entrant_minus_incumbent_1yr_hours", r_ent1_h, r_inc1_h, estimator_label = "PPML_DIFF_JK", in_primary_family = is_main)
  run_mech("P2_incumbent_entry", contrast_nm, "wb_1yr_incumbent_positions", "incumbent_1yr_positions", d, "incumbent_1yr_positions", x_col, ppml = TRUE, scale_mode = "ppml_price")
  run_mech("P2_incumbent_entry", contrast_nm, "wb_1yr_entrant_positions", "entrant_1yr_positions", d, "entrant_1yr_positions", x_col, ppml = TRUE, scale_mode = "ppml_price")
  r_nbus <- run_mech("P2_incumbent_entry", contrast_nm, "wb_total_businesses", "n_businesses", d, "n_businesses", x_col, ppml = TRUE, scale_mode = "ppml_price")
  run_mech("P2_incumbent_entry", contrast_nm, "wb_1yr_incumbent_businesses", "n_incumbent_1yr_bus", d, "n_incumbent_1yr_bus", x_col, ppml = TRUE, scale_mode = "ppml_price")
  run_mech("P2_incumbent_entry", contrast_nm, "wb_1yr_entrant_businesses", "n_entrant_1yr_bus", d, "n_entrant_1yr_bus", x_col, ppml = TRUE, scale_mode = "ppml_price")
  r_inc3_h <- run_mech("P2_incumbent_entry", contrast_nm, "wb_3yr_incumbent_hours", "incumbent_3yr_hours", d, "incumbent_3yr_hours", x_col, ppml = TRUE, scale_mode = "ppml_price")
  r_ent3_h <- run_mech("P2_incumbent_entry", contrast_nm, "wb_3yr_entrant_hours", "entrant_3yr_hours", d, "entrant_3yr_hours", x_col, ppml = TRUE, scale_mode = "ppml_price")
  run_paired_diff("P2_incumbent_entry", contrast_nm, "wb_3yr_diff_entrant_minus_incumbent_hours", "diff_entrant_minus_incumbent_3yr_hours", r_ent3_h, r_inc3_h, estimator_label = "PPML_DIFF_JK")
  r_r7_inc <- run_mech("P2_incumbent_entry", contrast_nm, "filer_r7_1yr_incumbent_hours", "filer_r7_incumbent_1yr_hours", d, "filer_r7_incumbent_1yr_hours", x_col, ppml = TRUE, scale_mode = "ppml_price")
  r_r7_ent <- run_mech("P2_incumbent_entry", contrast_nm, "filer_r7_1yr_entrant_hours", "filer_r7_entrant_1yr_hours", d, "filer_r7_entrant_1yr_hours", x_col, ppml = TRUE, scale_mode = "ppml_price")
  run_paired_diff("P2_incumbent_entry", contrast_nm, "filer_r7_1yr_diff_entrant_minus_incumbent_hours", "diff_filer_r7_entrant_minus_incumbent_hours", r_r7_ent, r_r7_inc, estimator_label = "PPML_DIFF_JK")

  # --- Panel 3: H-2ALC Contractors vs Direct Growers (7 rows) ---
  r_alc_h <- run_mech("P3_h2alc_direct", contrast_nm, "contractor_h2alc_hours", "h2alc_hours", d, "h2alc_hours", x_col, ppml = TRUE, scale_mode = "ppml_price", in_primary_family = is_main)
  r_dir_h <- run_mech("P3_h2alc_direct", contrast_nm, "contractor_direct_hours", "direct_hours", d, "direct_hours", x_col, ppml = TRUE, scale_mode = "ppml_price", in_primary_family = is_main)
  run_paired_diff("P3_h2alc_direct", contrast_nm, "contractor_diff_h2alc_minus_direct_hours", "diff_h2alc_minus_direct_hours", r_alc_h, r_dir_h, estimator_label = "PPML_DIFF_JK", in_primary_family = is_main)
  r_alc_p <- run_mech("P3_h2alc_direct", contrast_nm, "contractor_h2alc_positions", "h2alc_positions", d, "h2alc_positions", x_col, ppml = TRUE, scale_mode = "ppml_price")
  r_dir_p <- run_mech("P3_h2alc_direct", contrast_nm, "contractor_direct_positions", "direct_positions", d, "direct_positions", x_col, ppml = TRUE, scale_mode = "ppml_price")
  run_paired_diff("P3_h2alc_direct", contrast_nm, "contractor_diff_h2alc_minus_direct_positions", "diff_h2alc_minus_direct_positions", r_alc_p, r_dir_p, estimator_label = "PPML_DIFF_JK")
  run_mech("P3_h2alc_direct", contrast_nm, "contractor_h2alc_hours_share", "h2alc_hours_share", balanced_two_yr(d, "h2alc_hours_share"), "h2alc_hours_share", x_col, ppml = FALSE, scale_mode = "level_rate")

  # --- Panel 4: Planning Lead-Time & Predetermined CDL Crop Capital (14 rows) ---
  r_early_s <- run_mech("P4_leadtime_crops", contrast_nm, "leadtime_early_start_hours", "early_start_hours", d, "early_start_hours", x_col, ppml = TRUE, scale_mode = "ppml_price")
  r_late_s <- run_mech("P4_leadtime_crops", contrast_nm, "leadtime_late_start_hours", "late_start_hours", d, "late_start_hours", x_col, ppml = TRUE, scale_mode = "ppml_price", in_primary_family = is_main)
  run_paired_diff("P4_leadtime_crops", contrast_nm, "leadtime_diff_late_minus_early_start_hours", "diff_late_minus_early_start_hours", r_late_s, r_early_s, estimator_label = "PPML_DIFF_JK", in_primary_family = is_main)
  r_early_r <- run_mech("P4_leadtime_crops", contrast_nm, "leadtime_early_receipt_hours", "early_receipt_hours", d, "early_receipt_hours", x_col, ppml = TRUE, scale_mode = "ppml_price")
  r_late_r <- run_mech("P4_leadtime_crops", contrast_nm, "leadtime_late_receipt_hours", "late_receipt_hours", d, "late_receipt_hours", x_col, ppml = TRUE, scale_mode = "ppml_price")
  run_paired_diff("P4_leadtime_crops", contrast_nm, "leadtime_diff_late_minus_early_receipt_hours", "diff_late_minus_early_receipt_hours", r_late_r, r_early_r, estimator_label = "PPML_DIFF_JK")

  r_per_hi <- run_mech("P4_leadtime_crops", contrast_nm, "cdl_perennial_high_hours", "hours_perennial_high", d %>% filter(perennial_high), "hours_full_fy", x_col, ppml = TRUE, scale_mode = "ppml_price")
  r_per_lo <- run_mech("P4_leadtime_crops", contrast_nm, "cdl_annual_low_perennial_hours", "hours_annual_low_perennial", d %>% filter(!perennial_high), "hours_full_fy", x_col, ppml = TRUE, scale_mode = "ppml_price")
  run_paired_diff("P4_leadtime_crops", contrast_nm, "cdl_diff_annual_minus_perennial_hours", "diff_annual_minus_perennial_hours", r_per_lo, r_per_hi, estimator_label = "PPML_DIFF_JK")
  run_mech("P4_leadtime_crops", contrast_nm, "cdl_perennial_continuous_interaction", "hours_interaction_perennial_std", d, "hours_full_fy", x_col, rhs_extra = c("x_per_int", "i(aewr_region_id, z_post_per)"), term_col = "x_per_int", ppml = TRUE, scale_mode = "ppml_price", in_primary_family = is_main)

  r_veg_hi <- run_mech("P4_leadtime_crops", contrast_nm, "cdl_vegetable_high_hours", "hours_veg_high", d %>% filter(vegetable_high), "hours_full_fy", x_col, ppml = TRUE, scale_mode = "ppml_price")
  r_veg_lo <- run_mech("P4_leadtime_crops", contrast_nm, "cdl_nonveg_low_hours", "hours_nonveg_low", d %>% filter(!vegetable_high), "hours_full_fy", x_col, ppml = TRUE, scale_mode = "ppml_price")
  run_paired_diff("P4_leadtime_crops", contrast_nm, "cdl_diff_veg_minus_nonveg_hours", "diff_veg_minus_nonveg_hours", r_veg_hi, r_veg_lo, estimator_label = "PPML_DIFF_JK")
  run_mech("P4_leadtime_crops", contrast_nm, "cdl_vegetable_continuous_interaction", "hours_interaction_veg_std", d, "hours_full_fy", x_col, rhs_extra = c("x_veg_int", "i(aewr_region_id, z_post_veg)"), term_col = "x_veg_int", ppml = TRUE, scale_mode = "ppml_price")

  # --- Panel 5: Intensive Contract-Length & Scale Margins (5 rows) ---
  run_paired_diff("P5_intensive_margins", contrast_nm, "intensive_implied_hours_per_position", "implied_hours_per_position", anc_hours, anc_cert, estimator_label = "PPML_IMPLIED_JK")
  run_paired_diff("P5_intensive_margins", contrast_nm, "intensive_implied_positions_per_cert_app", "implied_positions_per_cert_app", anc_cert, r_cert_apps, estimator_label = "PPML_IMPLIED_JK")
  run_paired_diff("P5_intensive_margins", contrast_nm, "intensive_implied_positions_per_business", "implied_positions_per_business", anc_cert, r_nbus, estimator_label = "PPML_IMPLIED_JK")
  run_mech("P5_intensive_margins", contrast_nm, "intensive_mean_contract_weeks", "mean_contract_weeks", balanced_two_yr(d, "mean_contract_weeks"), "mean_contract_weeks", x_col, ppml = FALSE, scale_mode = "ols_elasticity")
  run_mech("P5_intensive_margins", contrast_nm, "intensive_mean_weekly_hours", "mean_weekly_hours", balanced_two_yr(d, "mean_weekly_hours"), "mean_weekly_hours", x_col, ppml = FALSE, scale_mode = "ols_elasticity")
}

run_contrast_suite("fy2025_2026", 2025L:2026L, "x_us_skill_1", 2026L, is_main = TRUE)
run_contrast_suite("placebo_fy2024_2025", 2024L:2025L, "x_placebo", 2025L, is_main = FALSE)

mech_estimates <- bind_rows(mech_results)
mech_deletions <- bind_rows(mech_loos)

# Holm step-down correction across the 12 declared primary FY2025-FY2026 hypotheses
stopifnot(nrow(mech_estimates) == 96L, sum(mech_estimates$in_primary_family) == 12L)
mech_estimates$p_holm_family <- NA_real_
fam_idx <- which(mech_estimates$in_primary_family)
mech_estimates$p_holm_family[fam_idx] <- stats::p.adjust(
  mech_estimates$p_jackknife[fam_idx],
  method = "holm",
  n = 12L
)

cd_keys(mech_estimates, c("track", "panel", "contrast", "specification", "outcome", "estimator", "term"))
cd_write(mech_estimates, out, "mechanism_estimates.csv")
cd_write(mech_deletions, out, "mechanism_state_deletions.csv")

# ------------------------------------------------------------------------------
# 8. Booktabs LaTeX Table: table_policy_2025_mechanisms.tex
# ------------------------------------------------------------------------------
fmt_n <- function(x, d = 3L) if (is.finite(x)) sprintf(paste0("%.", d, "f"), x) else "---"
stars <- function(p) if (!is.finite(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""

pick_m <- function(contrast_nm, spec_nm) {
  r <- mech_estimates %>% filter(contrast == !!contrast_nm, specification == !!spec_nm)
  stopifnot(nrow(r) == 1L)
  r[1L, ]
}

fmt_mrow <- function(spec_nm, lbl, is_rate = FALSE) {
  r <- pick_m("fy2025_2026", spec_nm)
  rp <- pick_m("placebo_fy2024_2025", spec_nm)
  if (is_rate) {
    val <- r$estimate
    se_v <- r$se_jackknife
    lo_v <- r$ci95_low_jk
    hi_v <- r$ci95_high_jk
    mde_txt <- "---"
    p_val <- rp$estimate
    p_se <- rp$se_jackknife
    d_dig <- 4L
  } else {
    val <- r$elasticity
    se_v <- r$elasticity_se_jk
    lo_v <- r$elasticity_low_jk
    hi_v <- r$elasticity_high_jk
    mde_txt <- fmt_n(r$mde_elasticity_jk, 2L)
    p_val <- rp$elasticity
    p_se <- rp$elasticity_se_jk
    d_dig <- 3L
  }
  ci_txt <- if (is.finite(lo_v) && is.finite(hi_v)) paste0("[", fmt_n(lo_v, d_dig), ", ", fmt_n(hi_v, d_dig), "]") else "---"
  holm_txt <- if (is.finite(r$p_holm_family)) fmt_n(r$p_holm_family, 3L) else "---"
  plac_txt <- if (is.finite(p_val) && is.finite(p_se)) paste0(fmt_n(p_val, d_dig), " (", fmt_n(p_se, d_dig), ")") else "---"
  paste0(
    lbl, " & ",
    fmt_n(val, d_dig), stars(r$p_jackknife), " & ",
    "[", fmt_n(se_v, d_dig), "] & ",
    ci_txt, " & ",
    mde_txt, " & ",
    holm_txt, " & ",
    plac_txt, " \\\\"
  )
}

tex_mech <- c(
  "\\begin{table}[!htbp]",
  "\\centering",
  "\\caption{2025 Statewide AEWR Rule: Mechanism Heterogeneity Across Pipeline, Employer Entry, Contractors, Lead-Time, and Crop Capital (FY2025--FY2026)}",
  "\\label{tab:policy_2025_mechanisms}",
  "\\small",
  "\\begin{tabular}{lcccccc}",
  "\\toprule",
  "Mechanism / Subgroup & Estimate & JK SE & 95\\% JK CI & MDE & Holm $p$ & Placebo (FY24--25) \\\\",
  "\\midrule",
  "\\multicolumn{7}{l}{\\textit{Panel A: Application Pipeline \\& Certification Bottlenecks}} \\\\",
  fmt_mrow("pipeline_requested_hours", "Requested hours (all filed, PPML)$^{\\dagger}$"),
  fmt_mrow("pipeline_requested_positions", "Requested positions (all filed, PPML)$^{\\dagger}$"),
  fmt_mrow("pipeline_approval_rate", "Approval rate (cert./req., OLS \\$/hr)$^{\\dagger}$", is_rate = TRUE),
  fmt_mrow("pipeline_diff_req_minus_cert_hours", "Paired diff.: requested $-$ certified hours"),
  "\\midrule",
  "\\multicolumn{7}{l}{\\textit{Panel B: Incumbent Capacity vs.\\ Employer Entry (Worksite Business, 1-Yr Lookback)}} \\\\",
  fmt_mrow("wb_1yr_incumbent_hours", "Incumbent business hours (PPML)$^{\\dagger}$"),
  fmt_mrow("wb_1yr_entrant_hours", "Entrant business hours (PPML)$^{\\dagger}$"),
  fmt_mrow("wb_1yr_diff_entrant_minus_incumbent_hours", "Paired diff.: entrant $-$ incumbent hours$^{\\dagger}$"),
  fmt_mrow("wb_total_businesses", "Distinct worksite businesses (PPML)"),
  fmt_mrow("filer_r7_1yr_diff_entrant_minus_incumbent_hours", "Sensitivity: R7 filer entrant $-$ incumbent"),
  "\\midrule",
  "\\multicolumn{7}{l}{\\textit{Panel C: H-2ALC Labor Contractors vs.\\ Direct Growers}} \\\\",
  fmt_mrow("contractor_h2alc_hours", "H-2ALC contractor hours (PPML)$^{\\dagger}$"),
  fmt_mrow("contractor_direct_hours", "Direct-grower hours (PPML)$^{\\dagger}$"),
  fmt_mrow("contractor_diff_h2alc_minus_direct_hours", "Paired diff.: H-2ALC $-$ direct hours$^{\\dagger}$"),
  fmt_mrow("contractor_h2alc_hours_share", "H-2ALC share of hours (OLS \\$/hr)", is_rate = TRUE),
  "\\midrule",
  "\\multicolumn{7}{l}{\\textit{Panel D: Planning Lead-Time \\& Predetermined 2022--2024 CDL Crop Capital}} \\\\",
  fmt_mrow("leadtime_early_start_hours", "Early starts (Oct--Mar, PPML)"),
  fmt_mrow("leadtime_late_start_hours", "Late starts (Apr--Sep, PPML)$^{\\dagger}$"),
  fmt_mrow("leadtime_diff_late_minus_early_start_hours", "Paired diff.: late $-$ early starts$^{\\dagger}$"),
  fmt_mrow("cdl_perennial_high_hours", "Perennial-intensive counties ($\\ge p_{75}$, PPML)"),
  fmt_mrow("cdl_annual_low_perennial_hours", "Annual-dominated counties ($< p_{75}$, PPML)"),
  fmt_mrow("cdl_perennial_continuous_interaction", "Continuous interaction: dose $\\times$ perennial SD$^{\\dagger}$"),
  "\\midrule",
  "\\multicolumn{7}{l}{\\textit{Panel E: Intensive Contract-Length \\& Scale Margins}} \\\\",
  fmt_mrow("intensive_implied_hours_per_position", "Implied $\\ln(\\text{hours} / \\text{position})$ (paired JK)"),
  fmt_mrow("intensive_implied_positions_per_business", "Implied $\\ln(\\text{positions} / \\text{business})$ (paired JK)"),
  fmt_mrow("intensive_mean_contract_weeks", "Mean contract weeks (continuing OLS)"),
  fmt_mrow("intensive_mean_weekly_hours", "Mean observed weekly hours (continuing OLS)"),
  "\\bottomrule",
  "\\end{tabular}",
  "\\begin{minipage}{0.96\\linewidth}",
  "\\vspace{0.4em}\\footnotesize",
  paste0(
    "\\textit{Notes:} All models use the Track 1 multi-state frame (2,921 counties, 46 states, 15 former AEWR regions) with County and ",
    "Former-Region $\\times$ FY fixed effects and exact delete-one-state jackknife (CV3J, $t(G-1)$) inference. ",
    "PPML rows report statutory elasticities ($\\hat{\\beta}\\bar{p}$); paired differences and implied intensive margins use exact paired ",
    "46-state jackknife draws; OLS rate rows report slopes per statutory dollar. ",
    "The continuous CDL interaction absorbs region-specific post-period perennial slopes ($\\text{Region} \\times \\text{Post} \\times Z_c$) so ",
    "identification uses within-former-region dose dispersion only. ",
    "$^{\\dagger}$Marks the 12 declared primary FY2025--FY2026 mechanism hypotheses jointly adjusted via Holm step-down $p$-values (column 6). ",
    "Matched FY2024--FY2025 placebo estimates (JK SE in parentheses) appear in column 7. ",
    "Post-inspection mechanism audit (not a preregistration); $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$ (unadjusted JK $p$)."
  ),
  "\\end{minipage}",
  "\\end{table}"
)

writeLines(tex_mech, file.path(out, "table_policy_2025_mechanisms.tex"))

# ------------------------------------------------------------------------------
# 9. Two-Panel Mechanism Forest Plot: fig_policy_2025_mechanisms.{pdf,png}
# ------------------------------------------------------------------------------
q_main_h <- load_q_anchor("fy2025_2026", "hours")$row
q_plac_h <- load_q_anchor("placebo_fy2024_2025", "hours")$row

plot_specs <- tibble(
  specification = c(
    "pipeline_requested_hours",
    "wb_1yr_incumbent_hours",
    "wb_1yr_entrant_hours",
    "contractor_h2alc_hours",
    "contractor_direct_hours",
    "leadtime_early_start_hours",
    "leadtime_late_start_hours",
    "cdl_perennial_high_hours",
    "cdl_annual_low_perennial_hours",
    "intensive_implied_hours_per_position",
    "intensive_implied_positions_per_business"
  ),
  label = c(
    "P1: Requested hours (all filed)",
    "P2: Incumbent businesses (1-yr)",
    "P2: Entrant businesses (1-yr)",
    "P3: H-2ALC labor contractors",
    "P3: Direct growers",
    "P4: Early starts (Oct-Mar)",
    "P4: Late starts (Apr-Sep)",
    "P4: Perennial-intensive (>=p75)",
    "P4: Annual-dominated (<p75)",
    "P5: Implied hours / position",
    "P5: Implied positions / business"
  )
)

anchor_df <- bind_rows(
  tibble(
    contrast = "fy2025_2026",
    label = "Track 1 Benchmark: Certified hours",
    elasticity = q_main_h$elasticity[[1]],
    low = q_main_h$elasticity_low_jk[[1]],
    high = q_main_h$elasticity_high_jk[[1]]
  ),
  tibble(
    contrast = "placebo_fy2024_2025",
    label = "Track 1 Benchmark: Certified hours",
    elasticity = q_plac_h$elasticity[[1]],
    low = q_plac_h$elasticity_low_jk[[1]],
    high = q_plac_h$elasticity_high_jk[[1]]
  )
)

mech_plot_df <- mech_estimates %>%
  inner_join(plot_specs, by = "specification") %>%
  transmute(
    contrast = contrast,
    label = label,
    elasticity = elasticity,
    low = elasticity_low_jk,
    high = elasticity_high_jk
  ) %>%
  bind_rows(anchor_df) %>%
  filter(is.finite(elasticity), is.finite(low), is.finite(high)) %>%
  mutate(
    label = factor(label, levels = rev(c("Track 1 Benchmark: Certified hours", plot_specs$label))),
    panel_label = factor(
      ifelse(contrast == "fy2025_2026", "A. Policy Contrast (FY2025 vs. FY2026)", "B. Pre-Policy Placebo (FY2024 vs. FY2025)"),
      levels = c("A. Policy Contrast (FY2025 vs. FY2026)", "B. Pre-Policy Placebo (FY2024 vs. FY2025)")
    ),
    is_anchor = label == "Track 1 Benchmark: Certified hours"
  )

p_mech <- ggplot(mech_plot_df, aes(x = elasticity, y = label, color = contrast, shape = is_anchor)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "#4A5568", linewidth = 0.45) +
  geom_vline(xintercept = -1, linetype = "dotted", color = "#A0AEC0", linewidth = 0.45) +
  geom_errorbarh(aes(xmin = low, xmax = high), height = 0.22, linewidth = 0.65) +
  geom_point(size = 2.4) +
  facet_wrap(~panel_label, ncol = 2) +
  scale_color_manual(
    values = c("fy2025_2026" = "#1B4F72", "placebo_fy2024_2025" = "#6C7A89"),
    guide = "none"
  ) +
  scale_shape_manual(values = c("FALSE" = 16, "TRUE" = 18), guide = "none") +
  labs(
    title = "2025 Statewide AEWR Rule: Statutory Quantity Elasticities Across Mechanism Margins",
    subtitle = "County + Former-Region x FY FE PPML elasticities with 95% delete-one-state jackknife (CV3J) intervals",
    x = "Statutory AEWR elasticity (dashed = 0; dotted = -1.0 unit elastic)",
    y = NULL,
    caption = paste0(
      "Notes: Estimates cover 2,921 positive-2011-farm-employment counties in 15 multi-state former AEWR regions (46 states).\n",
      "Panel A shows FY2025-FY2026 policy elasticities; Panel B shows matched FY2024-FY2025 placebos. Post-inspection mechanism audit."
    )
  ) +
  theme_minimal(base_size = 10.5) +
  theme(
    plot.title = element_text(face = "bold", size = 11.5, hjust = 0),
    plot.subtitle = element_text(size = 9.5, color = "#4A5568", margin = margin(b = 8)),
    plot.caption = element_text(size = 8.2, color = "#4A5568", hjust = 0, margin = margin(t = 8)),
    strip.text = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "#EDF2F7", linewidth = 0.35),
    axis.text.y = element_text(size = 9.0, color = "#1A202C"),
    plot.margin = margin(10, 12, 10, 10)
  )

ggsave(file.path(out, "fig_policy_2025_mechanisms.pdf"), p_mech, width = 10.5, height = 6.2, device = "pdf")
ggsave(file.path(out, "fig_policy_2025_mechanisms.png"), p_mech, width = 10.5, height = 6.2, dpi = 300)

cat(
  sprintf(
    "Estimated policy_2025 mechanism heterogeneity module (%d rows, %d primary-family Holm tests; min p_holm=%.3f)\n",
    nrow(mech_estimates),
    sum(mech_estimates$in_primary_family),
    min(mech_estimates$p_holm_family, na.rm = TRUE)
  )
)
