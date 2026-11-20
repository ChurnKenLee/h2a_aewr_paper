# Scratch diagnostic: do the dissimilar donor subregions share shocks with
# their target subregion? Compares target-donor co-movement with target vs
# non-donor same-region subregions and with subregions in other regions.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "specification.R"))
suppressMessages({library(arrow); library(dplyr); library(tidyr); library(fixest)})
spec <- panel_iv_read_specification(path_code("designs", "panel_iv", "estimation_spec.json"))
controls <- paste(spec$controls, collapse = " + ")
panel <- panel_iv_prepare_outcomes(read_parquet(path_processed("panel_iv_county_year.parquet")), spec) %>%
  filter(!is.na(target_cluster)) %>%
  mutate(aewr_region_id = as.character(aewr_region_id), target_cluster = as.integer(target_cluster))

donors <- read_parquet(path_int("panel_iv_county_donor_frame.parquet")) %>%
  filter(is_primary) %>%
  distinct(aewr_region_id = as.character(aewr_region_id), target = as.integer(target_cluster),
    donor = as.integer(donor_cluster))
subs <- panel %>% distinct(aewr_region_id, sub = target_cluster)
pairs <- subs %>% rename(target = sub) %>%
  inner_join(subs %>% rename(other = sub), by = "aewr_region_id", relationship = "many-to-many") %>%
  filter(target != other) %>%
  left_join(donors %>% mutate(is_donor = TRUE), by = c("aewr_region_id", "target", "other" = "donor")) %>%
  mutate(is_donor = coalesce(is_donor, FALSE))
cat("pairs:", nrow(pairs), " donor pairs:", sum(pairs$is_donor), "\n")

# Subregion-year means of two-way residuals ----------------------------------
sub_means <- function(data, formula, value) {
  m <- feols(formula, data, notes = FALSE, warn = FALSE)
  d <- data[obs(m), ]
  d$r <- resid(m)
  d %>% group_by(aewr_region_id, sub = target_cluster, year) %>%
    summarise(!!value := mean(r), n = n(), .groups = "drop") %>% select(-n)
}
yrs <- panel %>% filter(year >= 2010, year <= 2023)
wage <- yrs %>% filter(oews_wage_observed, is.finite(oews_big_six_mean_hourly_wage), ppi_2012 > 0) %>%
  mutate(lw = log(oews_big_six_mean_hourly_wage / ppi_2012))
series <- list(
  oews_log_wage = sub_means(wage, lw ~ 1 | county_fips + year, "v"),
  cert_share = sub_means(yrs %>% filter(year >= 2011, is.finite(h2a_cert_share_farm_workers_2011_start_year)),
    h2a_cert_share_farm_workers_2011_start_year ~ 1 | county_fips + year, "v"),
  cert_share_net_aewr = sub_means(yrs %>% filter(year >= 2011, is.finite(h2a_cert_share_farm_workers_2011_start_year)),
    h2a_cert_share_farm_workers_2011_start_year ~ aewr_ppi | county_fips + year, "v"),
  applications_net_aewr = sub_means(yrs %>% filter(year >= 2011, is.finite(h2a_applications_per_farm_worker_2011_start_year)),
    h2a_applications_per_farm_worker_2011_start_year ~ aewr_ppi | county_fips + year, "v")
)

pair_cor <- function(s, p) {
  x <- p %>% inner_join(s %>% rename(target = sub, vt = v), by = c("aewr_region_id", "target"),
      relationship = "many-to-many") %>%
    inner_join(s %>% rename(other = sub, vo = v), by = c("aewr_region_id", "other", "year"))
  x %>% group_by(is_donor) %>% summarise(cor = cor(vt, vo), pairs_years = n(), .groups = "drop")
}
cross_region <- function(s) {
  # Each subregion vs every subregion in other regions, same year.
  x <- s %>% inner_join(s, by = "year", suffix = c("_t", "_o"), relationship = "many-to-many") %>%
    filter(aewr_region_id_t != aewr_region_id_o)
  cor(x$v_t, x$v_o)
}
for (nm in names(series)) {
  s <- series[[nm]]
  pc <- pair_cor(s, pairs)
  cat(sprintf("%-24s donor corr %.3f | non-donor same-region corr %.3f | other-region corr %.3f\n", nm,
    pc$cor[pc$is_donor], pc$cor[!pc$is_donor], cross_region(s)))
}

# Share of a subregion's year-to-year wage variation that is region-wide ----
s <- series$oews_log_wage
ry <- s %>% group_by(aewr_region_id, year) %>% mutate(v_ry = mean(v)) %>% ungroup()
cat("OEWS subregion residual: share of variance common to region-year:",
  round(var(ry$v_ry) / var(ry$v), 3), "\n")

# OEWS reporting areas shared between target subregion and its donors -------
areas <- panel %>% filter(year == 2015, !is.na(oews_area_code)) %>%
  distinct(aewr_region_id, sub = target_cluster, oews_area_code)
shared <- donors %>%
  inner_join(areas %>% rename(target = sub), by = c("aewr_region_id", "target"), relationship = "many-to-many") %>%
  rename(t_area = oews_area_code) %>%
  inner_join(areas %>% rename(donor = sub, d_area = oews_area_code), by = c("aewr_region_id", "donor"),
    relationship = "many-to-many") %>%
  group_by(aewr_region_id, target, donor) %>%
  summarise(shares_area = any(t_area == d_area), .groups = "drop")
cat("target-donor subregion pairs sharing at least one primary OEWS area:",
  sum(shared$shares_area), "of", nrow(shared), "\n")

# Region-cluster bootstrap of the donor and non-donor correlations ---------
set.seed(20261102)
regions <- unique(pairs$aewr_region_id)
for (nm in names(series)) {
  s <- series[[nm]]
  x <- pairs %>% inner_join(s %>% rename(target = sub, vt = v), by = c("aewr_region_id", "target"),
      relationship = "many-to-many") %>%
    inner_join(s %>% rename(other = sub, vo = v), by = c("aewr_region_id", "other", "year"))
  split_x <- split(x, x$aewr_region_id)
  draws <- replicate(999, {
    b <- bind_rows(split_x[sample(regions, replace = TRUE)])
    c(d = cor(b$vt[b$is_donor], b$vo[b$is_donor]), n = cor(b$vt[!b$is_donor], b$vo[!b$is_donor]))
  })
  q <- function(v) sprintf("[%.2f, %.2f]", quantile(v, .025), quantile(v, .975))
  cat(sprintf("%-24s donor 95%% %s | non-donor %s | difference %s\n", nm, q(draws["d", ]),
    q(draws["n", ]), q(draws["d", ] - draws["n", ])))
}
