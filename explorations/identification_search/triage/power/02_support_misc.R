# Identification search, Phase 1 triage (power-first): treatment-side support
# for A4 (range herders), D2 (other floors), the FY2026 minimum-wage overlap
# (NP2), and pre-period national forecast noise (NP5).
#
# Usage: Rscript --vanilla explorations/identification_search/triage/power/02_support_misc.R "$S"
# Outcome use: herder positions FY2008-FY2015 only (before the 2015-11-16 rule)
# and national totals FY2012-FY2025 (before the FY2026 statewide rule), for
# residual variance only. No dose regression.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr)})
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "triage")
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
wr <- function(d, k) { write_csv(d, file.path(out, paste0("power_note_", k, ".csv"))); print(d, width = 200, n = 60) }

d <- read_parquet(path_int("h2a_disclosure_records.parquet"),
  col_select = c("case_number", "fiscal_year", "case_status", "job_title", "soc_code", "wage_rate", "wage_unit", "nbr_workers_certified", "worksite_state", "employer_state", "piece_rate_offer")) %>%
  mutate(pos = suppressWarnings(as.numeric(nbr_workers_certified)), wage = suppressWarnings(as.numeric(wage_rate)),
    monthly = grepl("^(month|mth)", tolower(trimws(wage_unit))), st = toupper(trimws(coalesce(worksite_state, employer_state))))

# ---- A4: herders before the 2015-11-16 rule (FY2008-FY2015) -------------------
herd_rx <- "(?i)herd|range.*(?:livestock|cattle|calv)|(?:livestock|cattle).*range|sheep"
h <- d %>% filter(fiscal_year <= 2015, monthly, grepl(herd_rx, job_title, perl = TRUE), is.finite(pos), pos > 0)
hs <- h %>% group_by(st, fiscal_year) %>% summarise(pos = sum(pos), .groups = "drop")
nat <- hs %>% group_by(fiscal_year) %>% summarise(pos = sum(pos), states = n_distinct(st), .groups = "drop")
wr(nat, "A4_herder_positions_pre")
wage_st <- h %>% filter(fiscal_year %in% 2014:2015, is.finite(wage), wage > 300) %>% group_by(st) %>%
  summarise(pos = sum(pos), med_monthly = median(wage), .groups = "drop") %>% arrange(desc(pos)) %>%
  mutate(dose_log_2016 = log(pmax(1206.31, med_monthly) / med_monthly), dose_log_2018 = log(pmax(1584.22, med_monthly) / med_monthly))
wr(wage_st, "A4_prerule_wage_by_state")
ww <- wage_st$pos / sum(wage_st$pos)
top <- hs %>% filter(fiscal_year %in% 2012:2015) %>% group_by(st) %>% summarise(p = sum(pos)) %>% mutate(sh = p / sum(p)) %>% arrange(desc(sh))
# Residual noise of state log herder positions after state + FY FE, FY2008-15 (states present every year)
bal <- hs %>% group_by(st) %>% filter(n() == length(unique(hs$fiscal_year))) %>% ungroup() %>% mutate(ly = log(pos))
fit <- lm(ly ~ factor(st) + factor(fiscal_year), data = bal)
nl <- log(nat$pos); tr <- residuals(lm(nl ~ nat$fiscal_year))
wr(tibble(states_with_prerule_wage = nrow(wage_st), weighted_mean_dose_log_2018 = sum(ww * wage_st$dose_log_2018),
  weighted_sd_dose_log_2018 = sqrt(sum(ww * (wage_st$dose_log_2018 - sum(ww * wage_st$dose_log_2018))^2)),
  top_state = top$st[1], top_state_share = top$sh[1], top3_share = sum(head(top$sh, 3)),
  balanced_states = n_distinct(bal$st), state_resid_sd_log = sd(residuals(fit)), national_detrended_sd_log = sd(tr),
  rough_mde_national_its = 2.8 * sd(tr) * sqrt(1 + 1 / length(tr)) / sum(ww * wage_st$dose_log_2018)), "A4_summary")

# ---- D2: other floors -----------------------------------------------------------
pr <- d %>% filter(fiscal_year %in% 2024:2025, is.finite(pos), pos > 0) %>%
  mutate(pr_num = suppressWarnings(as.numeric(piece_rate_offer)), piece = is.finite(pr_num) & pr_num > 0)
wr(pr %>% group_by(fiscal_year) %>% summarise(share_pos_with_piece_rate = sum(pos[piece]) / sum(pos), cases = n(), .groups = "drop"), "D2_piece_rate_share")
oc <- read_parquet(file.path(S, "shared", "outcomes_county_fy.parquet"), col_select = c("county_fips", "state_fips", "year", "at_aewr_hours_dq13", "hourly_aewr_known_hours_dq13"))
ab <- oc %>% filter(year == 2025, hourly_aewr_known_hours_dq13 > 0) %>% mutate(above = 1 - at_aewr_hours_dq13 / hourly_aewr_known_hours_dq13)
wr(tibble(fy = 2025, hours_share_above_aewr = 1 - sum(ab$at_aewr_hours_dq13) / sum(ab$hourly_aewr_known_hours_dq13),
  counties = nrow(ab), counties_with_majority_above = sum(ab$above > 0.5)), "D2_above_aewr_fy2025")

# ---- NP2: FY2026 floors where the state minimum wage exceeds the H-2A cash rate --
sw <- read_parquet(path_int("aewr_occupation_skill_rates.parquet")) %>%
  filter(trimws(source_sheet) == "Statewide 655.120(b)(1)(i)", as.Date(effective_start) == as.Date("2025-10-02"), skill_level == 1) %>%
  select(state_fips, worker_category, r = hourly_aewr) %>% mutate(r = as.numeric(r)) %>% pivot_wider(names_from = worker_category, values_from = r)
cov <- read_csv("code/a01_sources/state_ag_minwage_coverage.csv", show_col_types = FALSE, col_types = cols(.default = "c"))
mw <- read_parquet(path_int("fred_state_minwages.parquet")) %>% filter(year %in% 2025:2026) %>%
  select(state_fips, year, smw = state_min_wage) %>% pivot_wider(names_from = year, values_from = smw, names_prefix = "smw_")
np2 <- sw %>% left_join(mw, by = "state_fips") %>% mutate(mw26 = pmax(coalesce(smw_2026, smw_2025, 7.25), 7.25),
  mw_binds_h2a = mw26 > H2A, gap = mw26 - H2A)
wr(np2 %>% arrange(desc(gap)) %>% head(20), "NP2_mw_vs_h2a_rate")
wr(tibble(states = nrow(np2), states_mw_above_h2a_skill1 = sum(np2$mw_binds_h2a, na.rm = TRUE),
  states_mw_above_us_skill1 = sum(np2$mw26 > np2$US, na.rm = TRUE), coverage_table_rows = nrow(cov),
  note = "state_min_wage from FRED (general minimum; agricultural coverage not applied: see code/a01_sources/state_ag_minwage_coverage.csv); 2026 missing -> 2025 -> federal"), "NP2_summary")

# ---- NP5: national forecast noise, FY2012-FY2025 (pre-rule) ---------------------
nw <- read_csv(file.path(S, "shared", "outcome_national_by_window.csv"), show_col_types = FALSE)
f1 <- function(win, var, yrs) {
  z <- nw %>% filter(window == win, year %in% yrs, is.finite(.data[[var]])) %>% arrange(year)
  ly <- log(z[[var]]); yr <- z$year
  # one-step-ahead forecast errors from a linear trend in logs fit on all earlier years (>= 4 points)
  fe <- sapply(seq_along(ly), function(i) if (i <= 4) NA_real_ else { m <- lm(ly[1:(i - 1)] ~ yr[1:(i - 1)]); ly[i] - sum(coef(m) * c(1, yr[i])) })
  g <- diff(ly)
  tibble(window = win, var = var, years = paste(range(yr), collapse = "-"), n = length(ly), sd_growth = sd(g),
    sd_growth_change = sd(diff(g)), rmse_trend_forecast = sqrt(mean(fe^2, na.rm = TRUE)))
}
its <- bind_rows(f1("dq13", "cert_hours", 2012:2025), f1("dq13", "cert_pos", 2012:2025), f1("rq13d", "req_pos", 2015:2025), f1("rq13d", "cert_hours", 2012:2025))
dlnC <- -2.18 / (17.7 * 1.23)  # national offered-wage change FY2025->FY2026 (report section 3) over C = w x 1.23
wr(its %>% mutate(dlnC = dlnC, rough_mde_growth_its = 2.8 * sd_growth_change / abs(dlnC), rough_mde_trend_its = 2.8 * rmse_trend_forecast / abs(dlnC)), "NP5_national_its")
