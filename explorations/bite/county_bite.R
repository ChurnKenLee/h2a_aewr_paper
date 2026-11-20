# Scratch diagnostic: county AEWR-bite designs 1(a) and 1(b).
# Exposure E: minus log 2007 Census of Agriculture hired-labor expense per hired
# worker, centered within AEWR region, scaled to unit SD (higher = lower local
# farm pay = AEWR more likely to bind). Writes only to argument 1.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressMessages({library(arrow); library(dplyr); library(tidyr); library(fixest); library(readr)})
out_dir <- commandArgs(trailingOnly = TRUE)[1]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
MIN_WORKERS <- 20

census <- read_parquet(path_int("census_ag_hired_worker_duration_county.parquet")) %>%
  filter(year == 2007) %>%
  transmute(county_fips, workers = census_hired_workers_total,
    expense = census_hired_labor_expense, short_share = census_hired_worker_less_than_150_share)
panel <- read_parquet(path_processed("county_year_panel.parquet")) %>%
  filter(year >= 2008, year <= 2024) %>%
  mutate(aewr_region_id = as.character(aewr_region_id),
    businesses = if_else(emp_farm_2011 > 0, nbr_employers_balanced_start_year / emp_farm_2011, NA_real_),
    aewr_real = aewr / ppi_2012)
exposure <- panel %>% distinct(county_fips, aewr_region_id, state_fips) %>%
  inner_join(census, by = "county_fips") %>%
  filter(is.finite(workers), workers >= MIN_WORKERS, is.finite(expense), expense > 0) %>%
  mutate(lpay = log(expense / workers)) %>%
  group_by(aewr_region_id) %>%
  mutate(e_raw = -(lpay - mean(lpay))) %>% ungroup()
# Duration-adjusted variant: residual of log pay on the short-duration share within region.
adj <- feols(lpay ~ short_share | aewr_region_id, exposure %>% filter(is.finite(short_share)))
exposure$e_adj_raw <- NA_real_
exposure$e_adj_raw[obs(adj)] <- -resid(adj)
exposure <- exposure %>% mutate(E = e_raw / sd(e_raw), E_adj = e_adj_raw / sd(e_adj_raw, na.rm = TRUE))
cat("Exposure counties:", nrow(exposure), "; SD of log pay within region:", round(sd(exposure$e_raw), 3),
  "; cor(E, E_adj):", round(cor(exposure$E, exposure$E_adj, use = "complete.obs"), 3), "\n")

d <- panel %>% inner_join(exposure %>% select(county_fips, E, E_adj), by = "county_fips")

# Validation: does exposure predict later AEWR bindingness among users? -----
bind <- d %>% filter(year %in% 2011:2013) %>% group_by(county_fips, aewr_region_id, E) %>%
  summarise(at = sum(cert_hours_at_aewr_start_year, na.rm = TRUE),
    base = sum(cert_hours_with_hourly_wage_and_aewr_start_year, na.rm = TRUE), .groups = "drop") %>%
  filter(base > 0) %>% mutate(share_at_aewr = at / base)
vb <- feols(share_at_aewr ~ E | aewr_region_id, bind, vcov = ~aewr_region_id)
cat(sprintf("Validation (2011-13 users, n=%d): share of hourly H-2A hours at AEWR on E within region: %.3f (SE %.3f); mean share %.2f\n",
  nrow(bind), coef(vb)[1], se(vb)[1], mean(bind$share_at_aewr)))
vb_any <- feols(I(nbr_workers_certified_start_year > 0) ~ E | aewr_region_id, d %>% filter(year == 2011))
cat(sprintf("2011 any use on E within region: %.3f (SE %.3f)\n", coef(vb_any)[1], se(vb_any)[1]))

# Estimates -------------------------------------------------------------------
outcomes <- c(cert = "h2a_cert_share_farm_workers_2011_start_year",
  hours = "h2a_cert_hours_per_farm_worker_2011_start_year",
  applications = "h2a_applications_per_farm_worker_2011_start_year",
  businesses = "businesses")
est <- d %>% filter(year >= 2011, is.finite(aewr_real))
rows <- list()
for (nm in names(outcomes)) {
  y <- outcomes[[nm]]
  for (ex in c("E", "E_adj")) {
    dd <- est %>% filter(is.finite(.data[[y]]), is.finite(.data[[ex]])) %>% mutate(X = .data[[ex]] * aewr_real, Ex = .data[[ex]])
    b <- feols(as.formula(paste(y, "~ X | county_fips + aewr_region_id^year")), dd, vcov = ~state_fips)
    a <- feols(as.formula(paste(y, "~ X | county_fips + aewr_region_id^year + year[Ex]")), dd, vcov = ~state_fips)
    for (m in list(list("1b", b), list("1a", a))) {
      mod <- m[[2]]
      rows[[length(rows) + 1]] <- tibble::tibble(outcome = nm, exposure = ex, design = m[[1]],
        coef = coef(mod)[["X"]], se_state = se(mod)[["X"]],
        se_county = se(summary(mod, vcov = ~county_fips))[["X"]],
        se_region = se(summary(mod, vcov = ~aewr_region_id))[["X"]],
        nobs = nobs(mod), mean_y = mean(dd[[y]]))
    }
  }
}
res <- bind_rows(rows) %>% mutate(t_state = coef / se_state, mde_state = 2.8 * se_state)
write_csv(res, file.path(out_dir, "county_bite_estimates.csv"))
print(as.data.frame(res), digits = 3)

# Event study: E x year, reference 2011, years 2008-2024 ---------------------
es_rows <- list()
for (nm in names(outcomes)) {
  y <- outcomes[[nm]]
  dd <- d %>% filter(is.finite(.data[[y]]))
  es <- feols(as.formula(paste(y, "~ i(year, E, ref = 2011) | county_fips + aewr_region_id^year")), dd,
    vcov = ~state_fips)
  ct <- coeftable(es)
  es_rows[[nm]] <- tibble::tibble(outcome = nm, term = rownames(ct),
    year = as.integer(sub(".*::(\\d+):E", "\\1", rownames(ct))),
    coef = ct[, 1], se = ct[, 2])
}
es <- bind_rows(es_rows)
write_csv(es, file.path(out_dir, "county_bite_event_study.csv"))
print(as.data.frame(es %>% select(-term) %>% filter(outcome %in% c("cert", "applications"))), digits = 3)
# Real AEWR path (national mean) for comparison with event-study shape
path <- est %>% group_by(year) %>% summarise(aewr_real = mean(aewr_real))
write_csv(path, file.path(out_dir, "aewr_real_path.csv"))
