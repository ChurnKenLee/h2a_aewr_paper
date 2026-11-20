# Scratch diagnostic: design 1(a)/1(b) with the AEWR-relative-to-local-wage bite.
# Exposure frozen over 2008 and 2010 (2009 AEWR is missing):
#   gap  = real AEWR - max(ACS CZ p25 real hourly wage, real applicable ag minimum wage)
#   frac = share of the five ACS quantile point masses (p10..p90) below the AEWR
#          after the ag-minimum-wage floor (MC-style fraction affected)
# Both are centered within AEWR region and scaled to unit SD. Writes to argument 1.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
suppressMessages({library(arrow); library(dplyr); library(tidyr); library(fixest); library(readr)})
out_dir <- commandArgs(trailingOnly = TRUE)[1]

panel <- read_parquet(path_processed("county_year_panel.parquet")) %>%
  filter(year >= 2008, year <= 2024) %>%
  mutate(aewr_region_id = as.character(aewr_region_id),
    businesses = if_else(emp_farm_2011 > 0, nbr_employers_balanced_start_year / emp_farm_2011, NA_real_))
q <- c("wage_p10", "wage_p25", "wage_p50", "wage_p75", "wage_p90")
base <- panel %>% filter(year %in% c(2008, 2010), is.finite(aewr_ppi)) %>%
  mutate(floor = pmax(wage_p25, prevailing_ag_min_wage_ppi),
    gap = aewr_ppi - floor,
    frac = (as.numeric(pmax(wage_p10, prevailing_ag_min_wage_ppi) < aewr_ppi) +
      as.numeric(pmax(wage_p25, prevailing_ag_min_wage_ppi) < aewr_ppi) +
      as.numeric(pmax(wage_p50, prevailing_ag_min_wage_ppi) < aewr_ppi) +
      as.numeric(pmax(wage_p75, prevailing_ag_min_wage_ppi) < aewr_ppi) +
      as.numeric(pmax(wage_p90, prevailing_ag_min_wage_ppi) < aewr_ppi)) / 5) %>%
  group_by(county_fips, aewr_region_id, cz_id) %>%
  summarise(gap = mean(gap), frac = mean(frac), .groups = "drop") %>%
  filter(is.finite(gap), is.finite(frac)) %>%
  group_by(aewr_region_id) %>%
  mutate(gap_c = gap - mean(gap), frac_c = frac - mean(frac)) %>% ungroup() %>%
  mutate(E_gap = gap_c / sd(gap_c), E_frac = frac_c / sd(frac_c))
cat("Counties:", nrow(base), " CZs:", n_distinct(base$cz_id),
  "\nBaseline gap ($, real): mean", round(mean(base$gap), 2), " SD", round(sd(base$gap), 2),
  " within-region SD", round(sd(base$gap_c), 2),
  "\nFraction affected: mean", round(mean(base$frac), 2), " within-region SD", round(sd(base$frac_c), 2),
  "\ncor(E_gap, E_frac):", round(cor(base$E_gap, base$E_frac), 2), "\n")
cat("Share of gap variance within region:", round(var(base$gap_c) / var(base$gap), 2), "\n")

d <- panel %>% inner_join(base %>% select(county_fips, E_gap, E_frac), by = "county_fips")
# Relevance in levels: does a bigger baseline gap go with less H-2A use in 2011, within region?
lv <- d %>% filter(year == 2011)
for (ex in c("E_gap", "E_frac")) {
  m1 <- feols(as.formula(paste("I(nbr_workers_certified_start_year > 0) ~", ex, "| aewr_region_id")), lv, vcov = ~state_fips)
  m2 <- feols(as.formula(paste("h2a_cert_share_farm_workers_2011_start_year ~", ex, "| aewr_region_id")), lv, vcov = ~state_fips)
  cat(sprintf("2011 within region, %s: any use %.3f (%.3f); cert share %.4f (%.4f)\n", ex,
    coef(m1)[1], se(m1)[1], coef(m2)[1], se(m2)[1]))
}

outcomes <- c(cert = "h2a_cert_share_farm_workers_2011_start_year",
  hours = "h2a_cert_hours_per_farm_worker_2011_start_year",
  applications = "h2a_applications_per_farm_worker_2011_start_year",
  businesses = "businesses")
est <- d %>% filter(year >= 2011, is.finite(aewr_ppi))
rows <- list()
for (nm in names(outcomes)) {
  y <- outcomes[[nm]]
  for (ex in c("E_gap", "E_frac")) {
    dd <- est %>% filter(is.finite(.data[[y]])) %>% mutate(X = .data[[ex]] * aewr_ppi, Ex = .data[[ex]])
    b <- feols(as.formula(paste(y, "~ X | county_fips + aewr_region_id^year")), dd, vcov = ~state_fips)
    a <- feols(as.formula(paste(y, "~ X | county_fips + aewr_region_id^year + year[Ex]")), dd, vcov = ~state_fips)
    for (m in list(list("1b", b), list("1a", a))) {
      rows[[length(rows) + 1]] <- tibble::tibble(outcome = nm, exposure = ex, design = m[[1]],
        coef = coef(m[[2]])[["X"]], se_state = se(m[[2]])[["X"]],
        se_region = se(summary(m[[2]], vcov = ~aewr_region_id))[["X"]],
        nobs = nobs(m[[2]]), mean_y = mean(dd[[y]]))
    }
  }
}
res <- bind_rows(rows) %>% mutate(t = coef / se_state, mde = 2.8 * se_state)
write_csv(res, file.path(out_dir, "county_bite_acs_estimates.csv"))
print(as.data.frame(res), digits = 3)

es_rows <- list()
for (nm in names(outcomes)) {
  y <- outcomes[[nm]]
  es <- feols(as.formula(paste(y, "~ i(year, E_gap, ref = 2011) | county_fips + aewr_region_id^year")),
    d %>% filter(is.finite(.data[[y]])), vcov = ~state_fips)
  ct <- coeftable(es)
  es_rows[[nm]] <- tibble::tibble(outcome = nm, year = as.integer(sub(".*::(\\d+):E_gap", "\\1", rownames(ct))),
    coef = ct[, 1], se = ct[, 2])
}
es <- bind_rows(es_rows)
write_csv(es, file.path(out_dir, "county_bite_acs_event_study.csv"))
print(as.data.frame(es %>% filter(outcome %in% c("cert", "applications"))), digits = 3)
