# Triage (credibility-first): post-rule TAKE-UP of the predetermined skill
# class only. Reads FY2026 Q3 raw disclosure columns needed for the
# treatment side (experience requirement, hourly wage offer, employer, state).
# It reads no certified/requested counts as outcomes: certified workers are
# used only as weights for shares within the FY2026 set. No quantity outcome
# is computed or compared across years.
#   Rscript --vanilla explorations/identification_search/triage/credibility/02_fy2026_takeup.R "$S/triage"
suppressPackageStartupMessages({library(dplyr); library(readxl); library(arrow)})
out <- commandArgs(trailingOnly = TRUE)[1]
d <- read_excel("data/raw/h2a/H-2A_Disclosure_Data_FY2026_Q3.xlsx", guess_max = 50000)
j26 <- data.frame(employer = toupper(trimws(d$EMPLOYER_NAME)), state = d$WORKSITE_STATE,
  exp_months = suppressWarnings(as.numeric(d$WORK_EXPERIENCE_MONTHS)), per = d$PER,
  wage = suppressWarnings(as.numeric(d$WAGE_OFFER)), w = suppressWarnings(as.numeric(d$TOTAL_WORKERS_H2A_CERTIFIED)),
  received = as.Date(d$RECEIVED_DATE)) |> filter(!is.na(w), w > 0, received >= as.Date("2025-10-02"))
j25 <- read_parquet(file.path(out, "cred_job_orders_fy2018_2025.parquet")) |> filter(fy == 2025, !is.na(cert), cert > 0)
cls25 <- j25 |> group_by(employer, state) |> summarise(share_s2_25 = weighted.mean(exp_months >= 3, cert, na.rm = TRUE), .groups = "drop") |>
  mutate(class25 = case_when(share_s2_25 >= 0.999 ~ "all>=3", share_s2_25 <= 0.001 ~ "all<3", TRUE ~ "mixed"))
rates <- read.csv(file.path(out, "cred_skill_gap_by_state.csv"), colClasses = c(state_fips = "character"))
ab <- data.frame(state = c(state.abb, "DC", "PR"), state_name = c(state.name, "District of Columbia", "Puerto Rico"))
m <- j26 |> left_join(cls25, by = c("employer", "state")) |> left_join(ab, by = "state") |> left_join(rates, by = "state_name") |>
  mutate(class25 = ifelse(is.na(class25), "new_employer", class25), hourly = per == "Hour" & !is.na(wage) & wage > 0 & wage < 100)
t1 <- m |> group_by(class25) |> summarise(orders = n(), w_total = sum(w), share_exp_ge3_2026 = weighted.mean(exp_months >= 3, w, na.rm = TRUE),
  share_exp_eq0_2026 = weighted.mean(exp_months == 0, w, na.rm = TRUE),
  hourly_wage_minus_h2a_s1 = weighted.mean((wage - H2A_s1)[hourly], w[hourly], na.rm = TRUE),
  share_at_h2a_s1 = weighted.mean((abs(wage - H2A_s1) <= 0.05)[hourly], w[hourly], na.rm = TRUE),
  share_at_h2a_s2 = weighted.mean((abs(wage - H2A_s2) <= 0.05)[hourly], w[hourly], na.rm = TRUE),
  share_ge_h2a_s2 = weighted.mean((wage >= H2A_s2 - 0.05)[hourly], w[hourly], na.rm = TRUE),
  share_at_us_s1 = weighted.mean((abs(wage - US_s1) <= 0.05)[hourly], w[hourly], na.rm = TRUE),
  share_at_us_s2 = weighted.mean((abs(wage - US_s2) <= 0.05)[hourly], w[hourly], na.rm = TRUE),
  share_at_old_aewr = weighted.mean((abs(wage - aewr_fy2025) <= 0.05)[hourly], w[hourly], na.rm = TRUE),
  wage_minus_old_aewr = weighted.mean((wage - aewr_fy2025)[hourly], w[hourly], na.rm = TRUE), .groups = "drop")
write.csv(t1, file.path(out, "cred_fy2026_skill_takeup.csv"), row.names = FALSE)
print(t1, width = 200)
