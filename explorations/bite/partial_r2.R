here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "specification.R"))
suppressMessages({library(arrow); library(dplyr); library(fixest)})
spec <- panel_iv_read_specification(path_code("designs", "panel_iv", "estimation_spec.json"))
controls <- paste(spec$controls, collapse = " + ")
panel <- panel_iv_prepare_outcomes(read_parquet(path_processed("panel_iv_county_year.parquet")), spec)
est <- panel %>% filter(year >= 2011, year <= 2023,
  if_all(all_of(c("aewr_ppi", DISSIMILARITY_IV_CONTROL_COLUMNS, "z_wage_seasonal_composition_real")), ~ is.finite(.x)),
  !is.na(aewr_iv_cluster_id))
f <- function(v) resid(feols(as.formula(paste(v, "~", controls, "| county_fips + year")), est, notes = FALSE))
a <- f("aewr_ppi"); z <- f("z_wage_seasonal_composition_real")
cat("partial R2 of z:", cor(a, z)^2, " sd a:", sd(a), " sd z:", sd(z), "\n")
# OEWS naive regional mean as instrument, same sample
naive <- panel %>% filter(oews_wage_observed, is.finite(oews_big_six_mean_hourly_wage)) %>%
  group_by(aewr_region_id, year) %>% summarise(n = mean(oews_big_six_mean_hourly_wage / ppi_2012), .groups = "drop") %>%
  transmute(aewr_region_id, year = year + 1L, oews_naive = n)
est <- left_join(est, naive, by = c("aewr_region_id", "year")) %>% filter(is.finite(oews_naive)); cat("rows with naive:", nrow(est), "\n"); a <- f("aewr_ppi")
m <- feols(as.formula(paste("aewr_ppi ~ oews_naive +", controls, "| county_fips + year")), est, vcov = ~aewr_iv_cluster_id)
m2 <- feols(as.formula(paste("aewr_ppi ~ oews_naive +", controls, "| county_fips + year")), est, vcov = ~aewr_region_id)
n <- f("oews_naive")
cat("naive: coef", coef(m)[["oews_naive"]], " F(subregion)", coeftable(m)[1, 3]^2,
  " F(region)", coeftable(m2)[1, 3]^2, " partial R2", cor(a, n)^2, "\n")
mz <- feols(as.formula(paste("aewr_ppi ~ z_wage_seasonal_composition_real +", controls, "| county_fips + year")), est, vcov = ~aewr_region_id)
cat("z F(region):", coeftable(mz)[1, 3]^2, "\n")
for (y in spec$outcomes$outcome[1:4]) {
  o <- feols(as.formula(paste(y, "~ aewr_ppi +", controls, "| county_fips + year")), est, vcov = ~aewr_iv_cluster_id)
  iv <- feols(as.formula(paste(y, "~", controls, "| county_fips + year | aewr_ppi ~ z_wage_seasonal_composition_real")), est, vcov = ~aewr_iv_cluster_id)
  ivn <- feols(as.formula(paste(y, "~", controls, "| county_fips + year | aewr_ppi ~ oews_naive")), est, vcov = ~aewr_iv_cluster_id)
  cat(y, sprintf("OLS %.4g (%.3g)  IV %.4g (%.3g)  IV-naive %.4g (%.3g)  SE ratio %.2f\n",
    coef(o)[1], se(o)[1], coef(iv)[1], se(iv)[1], coef(ivn)[1], se(ivn)[1], se(iv)[1] / se(o)[1]))
}
