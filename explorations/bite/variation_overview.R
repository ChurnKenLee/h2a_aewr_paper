# Scratch diagnostic: where the panel-IV variation lives. Writes only to the
# output directory (argument 1).
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "specification.R"))
suppressMessages({
  library(arrow)
  library(dplyr)
  library(fixest)
  library(jsonlite)
})
out_dir <- commandArgs(trailingOnly = TRUE)[1]
spec <- panel_iv_read_specification(path_code("designs", "panel_iv", "estimation_spec.json"))
controls <- paste(spec$controls, collapse = " + ")
panel <- panel_iv_prepare_outcomes(
  read_parquet(path_processed("panel_iv_county_year.parquet")), spec
)
names_tbl <- read_parquet(path_int("fls_region.parquet")) %>%
  distinct(aewr_region_id = as.character(aewr_region_id), region_name)
panel <- panel %>% mutate(aewr_region_id = as.character(aewr_region_id)) %>%
  left_join(names_tbl, by = "aewr_region_id")
res <- list()
share <- function(part, total) round(var(part) / var(total), 3)

# 1. Treatment: region-year AEWR, 2011-2024 ----------------------------------
ry <- panel %>% filter(year >= 2011, year <= 2024) %>%
  group_by(aewr_region_id, region_name, year) %>%
  summarise(aewr_ppi = mean(aewr_ppi), aewr = mean(aewr), .groups = "drop")
m_r <- feols(aewr_ppi ~ 1 | aewr_region_id, ry)
m_y <- feols(aewr_ppi ~ 1 | year, ry)
m_ry <- feols(aewr_ppi ~ 1 | aewr_region_id + year, ry)
ry$resid <- resid(m_ry)
res$treatment <- list(
  cells = nrow(ry), sd_total = sd(ry$aewr_ppi), mean = mean(ry$aewr_ppi),
  r2_region = r2(m_r, "r2"), r2_year = r2(m_y, "r2"), r2_twoway = r2(m_ry, "r2"),
  sd_resid = sd(ry$resid), resid_share = 1 - r2(m_ry, "r2"),
  range_resid = range(ry$resid)
)
ry <- ry %>% arrange(aewr_region_id, year) %>% group_by(aewr_region_id) %>%
  mutate(resid_l1 = lag(resid), d = resid - lag(resid), d_l1 = lag(d)) %>% ungroup()
res$treatment$ar1_resid <- cor(ry$resid, ry$resid_l1, use = "complete.obs")
res$treatment$autocorr_diff <- cor(ry$d, ry$d_l1, use = "complete.obs")
res$treatment$sd_annual_change_real <- sd(ry$d, na.rm = TRUE)
res$region <- ry %>% group_by(aewr_region_id, region_name) %>%
  summarise(mean_aewr_ppi = mean(aewr_ppi), resid_sd = sd(resid),
    growth_real = last(aewr_ppi) / first(aewr_ppi) - 1, .groups = "drop") %>%
  arrange(desc(resid_sd))

# 2. Instrument vs a naive regional OEWS mean --------------------------------
naive <- panel %>% filter(oews_wage_observed, is.finite(oews_big_six_mean_hourly_wage)) %>%
  group_by(aewr_region_id, source = year) %>%
  summarise(oews_naive = mean(oews_big_six_mean_hourly_wage / ppi_2012), .groups = "drop") %>%
  transmute(aewr_region_id, year = source + 1L, oews_naive)
zry <- panel %>% filter(year >= 2011, year <= 2024, is.finite(z_wage_seasonal_composition_real)) %>%
  group_by(aewr_region_id, year) %>%
  summarise(z = mean(z_wage_seasonal_composition_real),
    z_wage = mean(z_wage_only_real, na.rm = TRUE), .groups = "drop")
ry2 <- ry %>% left_join(naive, by = c("aewr_region_id", "year")) %>%
  left_join(zry, by = c("aewr_region_id", "year")) %>%
  filter(is.finite(oews_naive), is.finite(z))
ry2$z_res <- resid(feols(z ~ 1 | aewr_region_id + year, ry2))
ry2$naive_res <- resid(feols(oews_naive ~ 1 | aewr_region_id + year, ry2))
ry2$a_res <- resid(feols(aewr_ppi ~ 1 | aewr_region_id + year, ry2))
fs_z <- feols(a_res ~ z_res, ry2)
fs_n <- feols(a_res ~ naive_res, ry2)
res$instrument <- list(
  cells = nrow(ry2), sd_z_resid = sd(ry2$z_res), sd_naive_resid = sd(ry2$naive_res),
  cor_z_aewr = cor(ry2$z_res, ry2$a_res), cor_naive_aewr = cor(ry2$naive_res, ry2$a_res),
  cor_z_naive = cor(ry2$z_res, ry2$naive_res),
  slope_z = coef(fs_z)[["z_res"]], slope_naive = coef(fs_n)[["naive_res"]],
  r2_z = r2(fs_z, "r2"), r2_naive = r2(fs_n, "r2")
)
# z inside the county-level estimation sample: region-year vs within region-year
est <- panel %>% filter(year >= 2011, year <= DISSIMILARITY_IV_BEA_CONTROL_END_YEAR,
  if_all(all_of(c("aewr_ppi", DISSIMILARITY_IV_CONTROL_COLUMNS,
    "z_wage_seasonal_composition_real")), ~ is.finite(.x)), !is.na(aewr_iv_cluster_id))
est$z_tw <- resid(feols(z_wage_seasonal_composition_real ~ 1 | county_fips + year, est))
est$a_tw <- resid(feols(aewr_ppi ~ 1 | county_fips + year, est))
est$a_full <- resid(feols(as.formula(paste("aewr_ppi ~", controls, "| county_fips + year")), est))
est <- est %>% group_by(aewr_region_id, year) %>%
  mutate(z_ry = mean(z_tw), a_ry = mean(a_tw)) %>% ungroup()
fs_county <- feols(as.formula(paste(
  "aewr_ppi ~ z_wage_seasonal_composition_real +", controls, "| county_fips + year")),
  est, vcov = ~aewr_iv_cluster_id)
res$instrument$county_obs <- nrow(est)
res$instrument$z_share_region_year <- share(est$z_ry, est$z_tw)
res$instrument$aewr_share_region_year <- share(est$a_ry, est$a_tw)
res$instrument$aewr_resid_sd_county <- sd(est$a_tw)
res$instrument$aewr_resid_sd_controls <- sd(est$a_full)
res$instrument$first_stage_coef <- coef(fs_county)[["z_wage_seasonal_composition_real"]]
res$instrument$first_stage_f <- coeftable(fs_county)[1, "t value"]^2
res$instrument$partial_r2 <- r2(fs_county, "wr2")

# 3. Outcomes ------------------------------------------------------------------
outs <- spec$outcomes$outcome[1:6]
res$outcomes <- lapply(outs, function(y) {
  d <- est %>% filter(is.finite(.data[[y]]))
  m <- feols(as.formula(paste(y, "~ 1 | county_fips + year")), d, notes = FALSE)
  if (!is.null(m$obs_selection$obsRemoved)) d <- d[m$obs_selection$obsRemoved, ]
  d$y_tw <- resid(m)
  d <- d %>% group_by(aewr_region_id, year) %>% mutate(y_ry = mean(y_tw)) %>% ungroup()
  ryo <- d %>% group_by(aewr_region_id, year) %>%
    summarise(y = mean(y_tw), a = mean(a_tw), .groups = "drop")
  list(outcome = y, obs = nrow(d), mean = mean(d[[y]]), share_zero = mean(d[[y]] == 0),
    sd = sd(d[[y]]), sd_resid = sd(d$y_tw),
    r2_county_year = 1 - var(d$y_tw) / var(d[[y]]),
    resid_share_region_year = share(d$y_ry, d$y_tw),
    cor_ry_with_aewr = cor(ryo$y, ryo$a))
})
# Concentration and take-up
cw <- panel %>% filter(year >= 2011, year <= 2023) %>% group_by(county_fips) %>%
  summarise(w = sum(nbr_workers_certified_start_year, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(w)) %>% mutate(cum = cumsum(w) / sum(w), rank = row_number())
n <- nrow(cw)
res$concentration <- list(counties = n, ever_users = sum(cw$w > 0),
  top1 = cw$cum[ceiling(.01 * n)], top5 = cw$cum[ceiling(.05 * n)],
  top10 = cw$cum[ceiling(.10 * n)], counties_for_half = which(cw$cum >= .5)[1])
res$takeup <- panel %>% filter(year >= 2011, year <= 2024) %>% group_by(year) %>%
  summarise(counties_using = sum(nbr_workers_certified_start_year > 0, na.rm = TRUE),
    certified = sum(nbr_workers_certified_start_year, na.rm = TRUE), .groups = "drop")
res$region_h2a <- panel %>% filter(year >= 2011, year <= 2023) %>%
  group_by(aewr_region_id, region_name) %>%
  summarise(counties = n_distinct(county_fips),
    certified = sum(nbr_workers_certified_start_year, na.rm = TRUE), .groups = "drop") %>%
  mutate(share = certified / sum(certified))
res$region_year <- ry2 %>% select(aewr_region_id, region_name, year, aewr_ppi,
  a_res, z_res, naive_res)
write_json(res, file.path(out_dir, "variation_overview.json"), auto_unbox = TRUE,
  digits = 6, pretty = TRUE)
str(res[c("treatment", "instrument", "concentration")])
print(res$region, n = 20)
for (o in res$outcomes) print(unlist(o))
