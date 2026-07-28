# Purpose: Estimate the four retained Fisher-price DiD specifications.
# Output: outputs/tables/table_fisher_price_dd.tex.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "did", "helpers.R"))
library(arrow)
library(dplyr)
library(fixest)

sample_full <- read_parquet(
  path_processed("did_county_year_panel.parquet")
) |>
  did_sample()
sample_no_border <- sample_full |>
  filter(!border_cz)

models <- list(
  did_model(sample_full, "fisher_index_ppi"),
  did_model(sample_full, "fisher_index_ppi", controls = TRUE),
  did_model(sample_no_border, "fisher_index_ppi"),
  did_model(
    sample_no_border,
    "fisher_index_ppi",
    controls = TRUE
  )
)

etable(
  models,
  tex = TRUE,
  title = "The Effect of the AEWR Wage Premium on Agricultural Prices",
  headers = did_table_headers,
  dict = c(
    did_table_dictionary,
    fisher_index_ppi = "Fisher price index (real, 2012=100)"
  ),
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_fisher_price_dd.tex"),
  replace = TRUE
)
