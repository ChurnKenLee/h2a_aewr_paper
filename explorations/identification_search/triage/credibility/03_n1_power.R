# Triage (credibility-first): rough design-stage power for new strategy N1
# (predetermined skill class under the 2025 rule, within state x FY) and, on
# the same employer frame, an A1-style statewide dose with former-region x FY
# FE for comparison. Uses only FY2024-FY2025 pre-period outcome residual
# variance (calibration) through the shared harness; no dose regression on
# any outcome. Statutory ITT units (beta_e1 = 1 / mean statutory price).
#   Rscript --vanilla explorations/identification_search/triage/credibility/03_n1_power.R "$S/triage"
suppressPackageStartupMessages({library(dplyr); library(arrow)})
source("explorations/identification_search/shared/power_harness.R")
out <- commandArgs(trailingOnly = TRUE)[1]; S <- dirname(out)

jo <- read_parquet(file.path(out, "cred_job_orders_fy2018_2025.parquet")) |> filter(fy == 2025, !is.na(cert), cert > 0)
cr <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("case_number", "fiscal_year", "employer_id_balanced", "state_fips", "cert_hours")) |> filter(fiscal_year == 2025)
emp_state <- cr |> group_by(employer_id_balanced, state_fips) |> summarise(h = sum(cert_hours, na.rm = TRUE), .groups = "drop") |>
  group_by(employer_id_balanced) |> slice_max(h, n = 1, with_ties = FALSE) |> ungroup() |> select(employer_id_balanced, state_fips)
case_emp <- cr |> distinct(case_number, employer_id_balanced)
cls <- jo |> inner_join(case_emp, by = "case_number", relationship = "many-to-many") |>
  group_by(employer_id_balanced) |> summarise(s2 = weighted.mean(exp_months >= 3, pmax(hours, 1), na.rm = TRUE), .groups = "drop")
rates <- read.csv(file.path(out, "cred_skill_gap_by_state.csv"), colClasses = c(state_fips = "character"))
reg <- read.csv("data/raw/aewr/aewr_regions.csv") |>
  left_join(data.frame(state_abbrev = c(state.abb, "DC"), state_name = c(state.name, "District of Columbia")), by = "state_abbrev")
rates <- rates |> left_join(reg, by = "state_name") |> filter(!is.na(aewr_fy2025), !is.na(aewr_region_num))

ep <- read_parquet(file.path(S, "shared", "outcomes_employer_fy.parquet"), col_select = c("employer_id_balanced", "year", "cert_hours_dq13")) |>
  filter(year %in% c(2024, 2025)) |> mutate(cert_hours_dq13 = coalesce(cert_hours_dq13, 0))
units <- emp_state |> inner_join(cls, by = "employer_id_balanced") |> inner_join(rates, by = "state_fips") |>
  filter(!is.na(s2))
pre <- ep |> inner_join(units, by = "employer_id_balanced") |>
  mutate(state_year = paste(state_fips, year), region_year = paste(aewr_region_num, year), region = aewr_region_num)
lin <- ish_ppml_linearize(pre, pre, "cert_hours_dq13", "employer_id_balanced")
pre_l <- lin$pre
pol <- bind_rows(pre_l |> filter(year == 2025) |> mutate(year = 2025L),
                 pre_l |> filter(year == 2025) |> mutate(year = 2026L)) |>
  mutate(state_year = paste(state_fips, year), region_year = paste(aewr_region_num, year),
    dose_n1 = ifelse(year == 2026, H2A_s1 + s2 * gap_h2a - aewr_fy2025, 0),
    dose_a1 = ifelse(year == 2026, H2A_s1 - aewr_fy2025, 0))
pre_l <- pre_l |> mutate(year = as.integer(year)); pol <- pol |> mutate(year = as.integer(year))
wbar <- weighted.mean(pre_l$aewr_fy2025, pre_l$ppml_weight)
res <- list()
specs <- list(
  N1 = list(x = "dose_n1", fe = c("employer_id_balanced", "state_year")),
  A1_employer = list(x = "dose_a1", fe = c("employer_id_balanced", "region_year")))
for (nm in names(specs)) {
  sp <- specs[[nm]]
  r <- ish_power(pol, pre_l, y = lin$y, x = sp$x, unit = "employer_id_balanced", period = "year", fe = sp$fe,
    clusters = c("state_fips", "region"), jk_cluster = "state_fips", node = "state_fips", group = "region",
    weights = lin$weights, beta_e1 = 1 / wbar, dgps = c("county_serial", "shared_region_serial"), label = list(spec = nm))
  res[[nm]] <- r
  write.csv(r$support, file.path(out, paste0("n1_power_support_", nm, ".csv")), row.names = FALSE)
  write.csv(r$summary, file.path(out, paste0("n1_power_summary_", nm, ".csv")), row.names = FALSE)
  write.csv(r$mde, file.path(out, paste0("n1_power_mde_", nm, ".csv")), row.names = FALSE)
  print(r$support); print(r$summary)
}
share <- ish_node_period_share(pre_l, lin$y, "employer_id_balanced", "year", c("employer_id_balanced", "state_year"), "state_fips", lin$weights)
print(share)
cat("wbar", wbar, "employers", n_distinct(pre_l$employer_id_balanced), "\n")
