here::i_am("code/paths.R"); source(here::here("code", "paths.R"))
suppressMessages({library(arrow); library(dplyr); library(fixest)})
p <- read_parquet(path_processed("county_year_panel.parquet")) %>% filter(year %in% 2011:2013) %>%
  mutate(aewr_region_id = as.character(aewr_region_id))
b <- p %>% group_by(county_fips, aewr_region_id, state_fips) %>%
  summarise(at = sum(cert_hours_at_aewr_start_year, na.rm = TRUE),
    base = sum(cert_hours_with_hourly_wage_and_aewr_start_year, na.rm = TRUE),
    hours = sum(man_hours_certified_start_year, na.rm = TRUE),
    oews = mean(log(oews_big_six_mean_hourly_wage / aewr), na.rm = TRUE),
    p25 = mean(log(aewr_cz_p25), na.rm = TRUE), p10 = mean(log(wage_p10 / aewr), na.rm = TRUE),
    minw = mean(log(prevailing_min_wage / aewr), na.rm = TRUE), .groups = "drop") %>%
  filter(base > 0) %>% mutate(s = at / base)
cat("counties:", nrow(b), " share at AEWR: mean", round(mean(b$s), 3), " hours-weighted",
  round(sum(b$at) / sum(b$base), 3), " quantiles", round(quantile(b$s, c(.1, .25, .5, .75, .9)), 2), "\n")
cat("share of share-variance within region:", round(1 - r2(feols(s ~ 1 | aewr_region_id, b)), 3), "\n")
for (v in c("oews", "p25", "p10", "minw")) {
  m <- feols(as.formula(paste("s ~", v, "| aewr_region_id")), b %>% filter(is.finite(.data[[v]])), vcov = ~state_fips)
  cat(sprintf("%-5s coef %.3f (SE %.3f) within R2 %.4f n=%d\n", v, coef(m)[1], se(m)[1], r2(m, "wr2"), nobs(m)))
}
