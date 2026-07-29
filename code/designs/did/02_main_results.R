# Purpose: Estimate the four retained main DiD specifications.
# Output: outputs/tables/table_1_main_results.tex.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "did", "helpers.R"))
library(arrow)
library(dplyr)
library(fixest)

sample_full <- read_parquet(
  path_processed("did_county_year_panel.parquet")
) %>%
  did_sample()
sample_no_border <- sample_full %>%
  filter(!border_cz)

models <- list(
  did_model(sample_full, "h2a_cert_share_farm_workers_2011_start_year"),
  did_model(
    sample_full,
    "h2a_cert_share_farm_workers_2011_start_year",
    controls = TRUE
  ),
  did_model(
    sample_no_border,
    "h2a_cert_share_farm_workers_2011_start_year"
  ),
  did_model(
    sample_no_border,
    "h2a_cert_share_farm_workers_2011_start_year",
    controls = TRUE
  )
)

etable(
  models,
  tex = TRUE,
  title = "The Effect of the AEWR Wage Premium on H-2A Utilization",
  headers = did_table_headers,
  dict = did_table_dictionary,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_1_main_results.tex"),
  replace = TRUE
)
