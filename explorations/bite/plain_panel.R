# Scratch diagnostic: plain TWFE regression of outcomes on the real AEWR, with
# the panel-IV controls, plus a lead test for regional shocks.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "specification.R"))
suppressMessages({
  library(arrow)
  library(dplyr)
  library(fixest)
  library(readr)
  library(fwildclusterboot)
})
out_dir <- commandArgs(trailingOnly = TRUE)[1]
spec <- panel_iv_read_specification(path_code("designs", "panel_iv", "estimation_spec.json"))
controls <- paste(spec$controls, collapse = " + ")
panel <- panel_iv_prepare_outcomes(
  read_parquet(path_processed("panel_iv_county_year.parquet")), spec
)
# AEWR(t+1) is the FLS wage measured in year t: contemporaneous regional
# conditions. Built from the region-year AEWR series in the panel.
aewr_lead <- panel %>%
  distinct(aewr_region_id, year, aewr_ppi) %>%
  transmute(aewr_region_id, year = year - 1L, aewr_ppi_f1 = aewr_ppi)
panel <- panel %>% left_join(aewr_lead, by = c("aewr_region_id", "year"))

base <- c("aewr_ppi", DISSIMILARITY_IV_CONTROL_COLUMNS, "z_wage_seasonal_composition_real")
rows <- list()
set.seed(20261022)
for (outcome in spec$outcomes$outcome[1:6]) {
  data <- panel %>%
    filter(year >= 2011, year <= DISSIMILARITY_IV_BEA_CONTROL_END_YEAR,
      if_all(all_of(c(outcome, base)), ~ is.finite(.x)),
      !is.na(aewr_iv_cluster_id))
  ols <- feols(as.formula(paste(outcome, "~ aewr_ppi +", controls, "| county_fips + year")),
    data = data, warn = FALSE, notes = FALSE)
  iv <- feols(as.formula(paste(outcome, "~", controls,
    "| county_fips + year | aewr_ppi ~ z_wage_seasonal_composition_real")),
    data = data, warn = FALSE, notes = FALSE)
  lead_data <- filter(data, is.finite(aewr_ppi_f1))
  lead <- feols(as.formula(paste(outcome, "~ aewr_ppi + aewr_ppi_f1 +", controls,
    "| county_fips + year")), data = lead_data, warn = FALSE, notes = FALSE)
  wild <- tryCatch(
    boottest(ols, param = "aewr_ppi", clustid = "aewr_region_id", B = 9999,
      type = "webb", fe = "county_fips"),
    error = function(e) NULL
  )
  term <- function(model, name, cluster) {
    t <- coeftable(summary(model, cluster = cluster))
    c(t[name, "Estimate"], t[name, "Std. Error"], t[name, "Pr(>|t|)"])
  }
  o_sub <- term(ols, "aewr_ppi", ~aewr_iv_cluster_id)
  o_reg <- term(ols, "aewr_ppi", ~aewr_region_id)
  i_sub <- term(iv, "fit_aewr_ppi", ~aewr_iv_cluster_id)
  l_now <- term(lead, "aewr_ppi", ~aewr_iv_cluster_id)
  l_lead <- term(lead, "aewr_ppi_f1", ~aewr_iv_cluster_id)
  rows[[outcome]] <- tibble::tibble(
    outcome = outcome, observations = nobs(ols), outcome_mean = mean(data[[outcome]]),
    ols = o_sub[1], ols_se_subregion = o_sub[2], ols_p_subregion = o_sub[3],
    ols_se_region = o_reg[2], ols_p_region = o_reg[3],
    ols_wild_p_region = if (is.null(wild)) NA_real_ else pval(wild),
    iv = i_sub[1], iv_se_subregion = i_sub[2], iv_p_subregion = i_sub[3],
    lead_obs = nobs(lead), lead_current = l_now[1], lead_current_se = l_now[2],
    lead_f1 = l_lead[1], lead_f1_se = l_lead[2], lead_f1_p = l_lead[3]
  )
}
results <- bind_rows(rows)
write_csv(results, file.path(out_dir, "plain_panel.csv"))
print(as.data.frame(results), digits = 3)
