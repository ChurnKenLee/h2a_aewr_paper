# Identification search, Phase 1 triage (credibility-first): treatment-side
# support statistics. No outcome regressions. Reads raw FY2018-FY2025
# disclosures (job-requirement and housing fields that the retained
# disclosure parquet does not carry), the AEWR schedules, and Phase 0 shared
# files. FY2026 rows are NOT read except the published FY2026 statutory rates.
#
#   Rscript --vanilla explorations/identification_search/triage/credibility/01_support.R "$S/triage"
suppressPackageStartupMessages({library(dplyr); library(readxl); library(arrow); library(data.table)})
args <- commandArgs(trailingOnly = TRUE); out <- args[1]; dir.create(out, showWarnings = FALSE, recursive = TRUE)
S <- dirname(out)

files <- c(`2018` = "data/raw/h2a/H-2A_Disclosure_Data_FY2018_EOY.xlsx", `2019` = "data/raw/h2a/H-2A_Disclosure_Data_FY2019.xlsx",
  `2020` = "data/raw/h2a/H-2A_Disclosure_Data_FY2020.xlsx", `2021` = "data/raw/h2a/H-2A_Disclosure_Data_FY2021.xlsx",
  `2022` = "data/raw/h2a/H-2A_Disclosure_Data_FY2022_Q4.xlsx", `2023` = "data/raw/h2a/H-2A_Disclosure_Data_FY2023_Q4.xlsx",
  `2024` = "data/raw/h2a/H-2A_Disclosure_Data_FY2024_Q4.xlsx", `2025` = "data/raw/h2a/H-2A_Disclosure_Data_FY2025_Q4.xlsx")
sdate <- function(x) { if (inherits(x, c("Date", "POSIXct"))) return(as.Date(x)); x <- as.character(x); out <- suppressWarnings(as.Date(x, "%Y-%m-%d")); i <- is.na(out); out[i] <- suppressWarnings(as.Date(x[i], "%m/%d/%Y")); i <- is.na(out) & grepl("^[0-9]+(\\.[0-9]+)?$", x); out[i] <- as.Date(as.numeric(x[i]), origin = "1899-12-30"); out }
pick <- function(d, opts) { for (o in opts) if (o %in% names(d)) return(d[[o]]); rep(NA, nrow(d)) }
rows <- list()
for (fy in names(files)) {
  d <- read_excel(files[[fy]], guess_max = 50000)
  rows[[fy]] <- data.frame(fy = as.integer(fy), case_number = pick(d, "CASE_NUMBER"), status = pick(d, "CASE_STATUS"),
    state = pick(d, c("WORKSITE_STATE")), soc = substr(gsub("[^0-9-]", "", as.character(pick(d, "SOC_CODE"))), 1, 7),
    exp_months = suppressWarnings(as.numeric(pick(d, c("WORK_EXPERIENCE_MONTHS", "EMP_EXP_NUM_MONTHS")))),
    exp_reqd = as.character(pick(d, "EMP_EXPERIENCE_REQD")),
    supervise = as.character(pick(d, "SUPERVISE_OTHER_EMP")),
    cert = suppressWarnings(as.numeric(pick(d, c("TOTAL_WORKERS_H2A_CERTIFIED", "NBR_WORKERS_CERTIFIED")))),
    needed = suppressWarnings(as.numeric(pick(d, "TOTAL_WORKERS_NEEDED"))),
    req = suppressWarnings(as.numeric(pick(d, c("TOTAL_WORKERS_H2A_REQUESTED", "NBR_WORKERS_REQUESTED")))),
    begin = sdate(pick(d, c("EMPLOYMENT_BEGIN_DATE", "JOB_START_DATE"))), end = sdate(pick(d, c("EMPLOYMENT_END_DATE", "JOB_END_DATE"))),
    hours_wk = suppressWarnings(as.numeric(pick(d, c("ANTICIPATED_NUMBER_OF_HOURS", "BASIC_NUMBER_OF_HOURS")))),
    wage = suppressWarnings(as.numeric(pick(d, c("WAGE_OFFER", "BASIC_RATE_OF_PAY")))), per = as.character(pick(d, c("PER", "BASIC_UNIT_OF_PAY"))),
    housing_type = as.character(pick(d, "TYPE_OF_HOUSING")), housing_transport = as.character(pick(d, "HOUSING_TRANSPORTATION")),
    flc = as.character(pick(d, "H2A_LABOR_CONTRACTOR")), fein = as.character(pick(d, "EMPLOYER_FEIN")),
    employer = toupper(trimws(as.character(pick(d, "EMPLOYER_NAME")))), stringsAsFactors = FALSE)
  cat(fy, nrow(d), "\n")
}
jo <- bind_rows(rows) |>
  mutate(certified = grepl("^Determination Issued - Certification|^CERTIFIED|^Certified", status, ignore.case = TRUE) |
           grepl("Partial", status, ignore.case = TRUE),
    weeks = pmax(as.numeric(end - begin), 0) / 7, hours = cert * weeks * hours_wk,
    exp_class = case_when(is.na(exp_months) ~ "NA", exp_months == 0 ~ "0", exp_months < 3 ~ "1-2", exp_months == 3 ~ "3",
      exp_months <= 6 ~ "4-6", TRUE ~ "7+"),
    skill2_pred = !is.na(exp_months) & exp_months >= 3)
write_parquet(jo, file.path(out, "cred_job_orders_fy2018_2025.parquet"))

# 1. Experience-requirement distribution by FY (certified rows with positive certified workers)
jc <- jo |> filter(!is.na(cert), cert > 0)
dist <- jc |> group_by(fy, exp_class) |> summarise(cases = n(), cert = sum(cert), hours = sum(hours, na.rm = TRUE), .groups = "drop") |>
  group_by(fy) |> mutate(cert_share = cert / sum(cert), hours_share = hours / sum(hours), case_share = cases / sum(cases))
write.csv(dist, file.path(out, "cred_exp_distribution.csv"), row.names = FALSE)
sup <- jc |> group_by(fy, supervise = toupper(substr(supervise, 1, 1))) |> summarise(cert = sum(cert), .groups = "drop") |> group_by(fy) |> mutate(share = cert / sum(cert))
write.csv(sup, file.path(out, "cred_supervise_distribution.csv"), row.names = FALSE)

# 2. 2025 rule skill gap by state (FIELD_LIVESTOCK_COMBINED, 2025-10-02 schedule) and FY2025 reference AEWR
occ <- read_parquet("data/intermediate/aewr_occupation_skill_rates.parquet") |>
  filter(effective_start == as.Date("2025-10-02"), soc_code == "FIELD_LIVESTOCK_COMBINED")
rates <- occ |> select(state_fips, state_name, skill_level, worker_category, hourly_aewr) |>
  mutate(k = paste0(worker_category, "_s", skill_level)) |> select(-skill_level, -worker_category) |>
  tidyr::pivot_wider(names_from = k, values_from = hourly_aewr)
sched <- read_parquet("data/intermediate/aewr_rate_schedule.parquet")
old <- sched |> mutate(es = as.Date(as.character(effective_start)), ee = as.Date(as.character(effective_end_exclusive))) |>
  filter(es <= as.Date("2025-04-01"), ee > as.Date("2025-04-01"), skill_level == "not_differentiated") |> group_by(state_fips) |> summarise(aewr_fy2025 = max(aewr), .groups = "drop")
rates <- rates |> left_join(old, by = "state_fips") |>
  mutate(gap_us = US_s2 - US_s1, gap_h2a = H2A_s2 - H2A_s1, housing_adj = US_s1 - H2A_s1,
    dose_s1_h2a = H2A_s1 - aewr_fy2025, dose_s2_h2a = H2A_s2 - aewr_fy2025)
st <- read_parquet("data/processed/county_year_panel.parquet", col_select = c("state_fips", "aewr_region_id")) |> distinct()
if (!"aewr_region_id" %in% names(st)) st <- NULL
write.csv(rates, file.path(out, "cred_skill_gap_by_state.csv"), row.names = FALSE)

# 3. Within-state skill contrast support: FY2025 certified hours by state x predetermined class
fips <- read_parquet("data/intermediate/aewr_occupation_skill_rates.parquet") |> distinct(state_name, state_fips)
ab <- data.frame(state = c(state.abb, "DC", "PR"), state_name = c(state.name, "District of Columbia", "Puerto Rico"))
j25 <- jc |> filter(fy == 2025) |> left_join(ab, by = "state") |> left_join(rates, by = "state_name") |>
  mutate(dose = ifelse(skill2_pred, dose_s2_h2a, dose_s1_h2a))
sc <- j25 |> group_by(state, skill2_pred) |> summarise(cases = n(), cert = sum(cert), hours = sum(hours, na.rm = TRUE),
  .groups = "drop")
write.csv(sc, file.path(out, "cred_skill_contrast_state_fy2025.csv"), row.names = FALSE)
# dose SD after state FE, job-order level, hours-weighted
v <- j25 |> filter(!is.na(dose), !is.na(hours), hours > 0)
v <- v |> group_by(state) |> mutate(dm = dose - weighted.mean(dose, hours)) |> ungroup()
both <- v |> group_by(state) |> summarise(n2 = sum(skill2_pred), n1 = sum(!skill2_pred), h2 = sum(hours[skill2_pred]), h = sum(hours), .groups = "drop")
s3 <- data.frame(stat = c("job_orders", "dose_sd_after_state_fe_hours_weighted", "dose_sd_raw_hours_weighted", "states_with_both_classes",
  "states_with_5plus_skill2_orders", "skill2_hours_share", "mean_gap_h2a", "sd_gap_h2a", "min_gap_h2a", "max_gap_h2a",
  "emp_with_any_skill2_order", "employers"),
  value = c(nrow(v), sqrt(weighted.mean(v$dm^2, v$hours)), sqrt(weighted.mean((v$dose - weighted.mean(v$dose, v$hours))^2, v$hours)),
    sum(both$n1 > 0 & both$n2 > 0), sum(both$n2 >= 5), sum(both$h2) / sum(both$h),
    mean(rates$gap_h2a, na.rm = TRUE), sd(rates$gap_h2a, na.rm = TRUE), min(rates$gap_h2a, na.rm = TRUE), max(rates$gap_h2a, na.rm = TRUE),
    n_distinct(v$employer[v$skill2_pred]), n_distinct(v$employer)))
write.csv(s3, file.path(out, "cred_skill_contrast_support.csv"), row.names = FALSE)

# 4. Wage offers relative to the new skill rates for FY2025 hourly job orders (is the predetermined skill-II floor binding?)
w <- j25 |> filter(per %in% c("Hour", "HOUR", "hour"), !is.na(wage), wage > 0, wage < 100) |>
  mutate(at_old = abs(wage - aewr_fy2025) <= 0.05, above_s2 = wage >= H2A_s2 - 0.005, above_us_s2 = wage >= US_s2 - 0.005)
ws <- w |> group_by(skill2_pred) |> summarise(cert_total = sum(cert), share_at_old_aewr = weighted.mean(at_old, cert, na.rm = TRUE),
  share_ge_h2a_s2 = weighted.mean(above_s2, cert, na.rm = TRUE), share_ge_us_s2 = weighted.mean(above_us_s2, cert, na.rm = TRUE),
  mean_wage_minus_old = weighted.mean(wage - aewr_fy2025, cert, na.rm = TRUE), .groups = "drop")
write.csv(ws, file.path(out, "cred_skill_contrast_wages.csv"), row.names = FALSE)

# 5. Housing and transport fields FY2020-2025
hz <- jc |> filter(fy >= 2020) |> group_by(fy, housing_type) |> summarise(cert = sum(cert), .groups = "drop") |> group_by(fy) |> mutate(share = cert / sum(cert))
write.csv(hz, file.path(out, "cred_housing_type.csv"), row.names = FALSE)
# needed vs requested (corresponding domestic share)
nd <- jc |> filter(fy >= 2020, !is.na(needed), !is.na(req), req > 0) |> group_by(fy) |>
  summarise(needed = sum(needed), req = sum(req), share_needed_gt_req = mean(needed > req), .groups = "drop")
write.csv(nd, file.path(out, "cred_needed_vs_requested.csv"), row.names = FALSE)
# FLC share
fl <- jc |> group_by(fy, flc = toupper(substr(flc, 1, 1))) |> summarise(cert = sum(cert), .groups = "drop") |> group_by(fy) |> mutate(share = cert / sum(cert))
write.csv(fl, file.path(out, "cred_flc_share.csv"), row.names = FALSE)
# contract length distribution (weeks) by FY: fixed per-worker cost exposure
cl <- jc |> filter(weeks > 0, weeks < 60) |> group_by(fy) |> summarise(p10 = quantile(weeks, .1), p25 = quantile(weeks, .25), p50 = median(weeks),
  p75 = quantile(weeks, .75), p90 = quantile(weeks, .9), hours_per_pos_p10 = quantile(weeks * hours_wk, .1, na.rm = TRUE),
  hours_per_pos_p90 = quantile(weeks * hours_wk, .9, na.rm = TRUE), .groups = "drop")
write.csv(cl, file.path(out, "cred_contract_length.csv"), row.names = FALSE)
cat("done\n")
