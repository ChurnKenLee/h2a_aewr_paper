# Triage (credibility-first): further treatment-side support statistics.
# No outcome regressions. Writes to "$S/triage".
#   Rscript --vanilla explorations/identification_search/triage/credibility/04_more_support.R "$S/triage"
suppressPackageStartupMessages({library(dplyr); library(arrow)})
out <- commandArgs(trailingOnly = TRUE)[1]; S <- dirname(out)
rates <- read.csv(file.path(out, "cred_skill_gap_by_state.csv"), colClasses = c(state_fips = "character"))
reg <- read.csv("data/raw/aewr/aewr_regions.csv") |>
  left_join(data.frame(state_abbrev = c(state.abb, "DC"), state_name = c(state.name, "District of Columbia")), by = "state_abbrev")
mw <- read_parquet("data/intermediate/state_real_minwages.parquet") |> filter(year %in% c(2025, 2026)) |>
  select(state_fips, year, prevailing_ag_min_wage) |> tidyr::pivot_wider(names_from = year, values_from = prevailing_ag_min_wage, names_prefix = "mw")
x <- rates |> left_join(reg, by = "state_name") |> left_join(mw, by = "state_fips") |>
  filter(!is.na(aewr_fy2025), !is.na(aewr_region_num)) |>
  mutate(mw2025 = coalesce(mw2025, 7.25), mw2026 = coalesce(mw2026, mw2025),
    stat_cut_us = US_s1 - aewr_fy2025, stat_cut_h2a = H2A_s1 - aewr_fy2025,
    eff_floor_2025q4 = pmax(H2A_s1, mw2025), eff_floor_2026 = pmax(H2A_s1, mw2026),
    eff_cut_h2a = eff_floor_2025q4 - aewr_fy2025, trunc = pmax(0, mw2025 - H2A_s1), jan_step = eff_floor_2026 - eff_floor_2025q4) |>
  group_by(aewr_region_num) |> mutate(n_reg = n()) |> ungroup()
w <- x |> filter(n_reg > 1) |> group_by(aewr_region_num) |>
  mutate(across(c(stat_cut_us, stat_cut_h2a, eff_cut_h2a, trunc, housing_adj, gap_h2a), ~ .x - mean(.x), .names = "dm_{.col}")) |> ungroup()
sumtab <- data.frame(stat = c("states_multi_state_regions", "sd_within_region_stat_cut_us", "sd_within_region_stat_cut_h2a",
  "sd_within_region_eff_cut_h2a_mw", "sd_within_region_truncation", "states_truncated_multi_region", "sd_within_region_housing_adj",
  "sd_within_region_skill_gap", "cor_trunc_statcut_within", "states_with_jan2026_floor_step", "mean_jan_step_if_pos"),
  value = c(nrow(w), sd(w$dm_stat_cut_us), sd(w$dm_stat_cut_h2a), sd(w$dm_eff_cut_h2a), sd(w$dm_trunc), sum(w$trunc > 0),
    sd(w$dm_housing_adj), sd(w$dm_gap_h2a), cor(w$dm_trunc, w$dm_stat_cut_h2a), sum(x$jan_step > 0.001), mean(x$jan_step[x$jan_step > 0.001])))
write.csv(sumtab, file.path(out, "cred_state_dose_support.csv"), row.names = FALSE)
write.csv(x, file.path(out, "cred_state_dose_table.csv"), row.names = FALSE)
print(sumtab)

# Multi-state employers in FY2025 (within-employer cross-state design N3)
cr <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("case_number", "fiscal_year", "employer_id_balanced", "state_fips", "cert_hours", "cert_pos")) |> filter(fiscal_year %in% 2023:2025)
ms <- cr |> filter(!is.na(state_fips), state_fips != "") |> group_by(fiscal_year, employer_id_balanced) |>
  summarise(ns = n_distinct(state_fips), h = sum(cert_hours, na.rm = TRUE), .groups = "drop") |>
  group_by(fiscal_year) |> summarise(multi_state_employers = sum(ns >= 2), multi_state_hours_share = sum(h[ns >= 2]) / sum(h), .groups = "drop")
# within former region multi-state
cr2 <- cr |> left_join(x |> select(state_fips, aewr_region_num), by = "state_fips")
msr <- cr2 |> filter(fiscal_year == 2025, !is.na(aewr_region_num)) |> group_by(employer_id_balanced, aewr_region_num) |>
  summarise(ns = n_distinct(state_fips), h = sum(cert_hours, na.rm = TRUE), .groups = "drop")
ms <- ms |> mutate(fy2025_emp_multi_state_within_region = sum(msr$ns >= 2), fy2025_hours_share_multi_within_region = sum(msr$h[msr$ns >= 2]) / sum(msr$h))
write.csv(ms, file.path(out, "cred_multistate_support.csv"), row.names = FALSE); print(ms)

# Case size distribution (per-petition fixed-fee exposure)
cs <- read_parquet(file.path(out, "cred_job_orders_fy2018_2025.parquet")) |> filter(fy %in% 2023:2025, cert > 0, !is.na(hours), hours > 0) |>
  mutate(fee_dose_per_hour = 670 / hours) # USCIS 2024 fee change for an unnamed-beneficiary petition, from memory (see report)
cst <- cs |> summarise(cases = n(), p10_hours = quantile(hours, .1), p50_hours = median(hours), p90_hours = quantile(hours, .9),
  hours_wtd_mean_fee_dose = weighted.mean(fee_dose_per_hour, hours), share_hours_dose_gt_0.25 = sum(hours[fee_dose_per_hour > 0.25]) / sum(hours),
  share_cases_dose_gt_0.25 = mean(fee_dose_per_hour > 0.25))
write.csv(cst, file.path(out, "cred_fee_exposure.csv"), row.names = FALSE); print(cst)

# 2009 nonstatutory application wages vs FY2008 AEWR (state-level A2 dose proxy)
a9 <- read_parquet("data/intermediate/aewr_2009_nonstatutory_wages.parquet")
sch <- read_parquet("data/intermediate/aewr_rate_schedule.parquet") |> mutate(es = as.Date(as.character(effective_start)), ee = as.Date(as.character(effective_end_exclusive))) |>
  filter(es <= as.Date("2008-07-01"), ee > as.Date("2008-07-01")) |> group_by(state_fips) |> summarise(aewr2008 = max(aewr), .groups = "drop")
a2 <- a9 |> left_join(sch, by = "state_fips") |> left_join(reg, by = "state_name") |>
  mutate(d = aewr_nonstatutory_application_wage_2009 - aewr2008) |> group_by(aewr_region_num) |> mutate(dm = d - mean(d), n = n()) |> ungroup()
a2s <- data.frame(stat = c("states", "mean_dose", "sd_dose", "sd_within_region_state_level", "mean_log_dose"),
  value = c(sum(!is.na(a2$d)), mean(a2$d, na.rm = TRUE), sd(a2$d, na.rm = TRUE), sd(a2$dm[a2$n > 1], na.rm = TRUE),
    mean(log(a2$aewr_nonstatutory_application_wage_2009 / a2$aewr2008), na.rm = TRUE)))
write.csv(a2s, file.path(out, "cred_a2_state_dose.csv"), row.names = FALSE); print(a2s)
print(unique(a9$measure)); print(unique(a9$source_document))
