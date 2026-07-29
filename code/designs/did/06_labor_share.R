# Purpose: Estimate the four retained farm-labor-share DiD specifications.
# Output: outputs/tables/table_laborshare_dd.tex.

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
  did_model(sample_full, "share_farm_laborexp_prodexp"),
  did_model(
    sample_full,
    "share_farm_laborexp_prodexp",
    controls = TRUE
  ),
  did_model(
    sample_no_border,
    "share_farm_laborexp_prodexp"
  ),
  did_model(
    sample_no_border,
    "share_farm_laborexp_prodexp",
    controls = TRUE
  )
)

etable(
  models,
  tex = TRUE,
  title = paste(
    "The Effect of the AEWR Wage Premium on Farm Labor Share",
    "of Production Expense"
  ),
  headers = did_table_headers,
  dict = c(
    did_table_dictionary,
    share_farm_laborexp_prodexp = "Labor share of farm production expense"
  ),
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_laborshare_dd.tex"),
  replace = TRUE
)
