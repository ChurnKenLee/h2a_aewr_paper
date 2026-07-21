# Purpose: Produce summary statistics for the main DD variables.
# Input: data/processed/county_df_analysis_year.parquet.
# Output: outputs/tables/table_sumstats_dd_variables.tex.
# Run after: code/c02_build/04_finalize_county_panel.R.

source(if (file.exists(file.path("code", "bootstrap_paths.R"))) {
  file.path("code", "bootstrap_paths.R")
} else {
  file.path("..", "bootstrap_paths.R")
})
source(path_code("c00_shared", "analysis_helpers.R"))
library(arrow)
library(tidyverse)

county_df <- read_parquet(path_processed("county_df_analysis_year.parquet"))
samp_base <- analysis_sample(county_df)

sumstats_vars <- list(
  "H-2A share of 2011 farm employment" = "h2a_cert_share_farm_workers_2011_start_year",
  "H-2A certified workers (start year)" = "nbr_workers_certified_start_year",
  "Farm employment 2011 (baseline)" = "emp_farm_2011",
  "AEWR p25 bite (2012 \\$)" = "aewr_cz_p25",
  "Log population" = "ln_pop_census",
  "Employment-to-population ratio" = "emp_pop_ratio"
)

sumstats_rows <- purrr::imap_dfr(sumstats_vars, function(col, label) {
  x <- samp_base[[col]]
  tibble(
    Variable = label,
    N = sum(!is.na(x)),
    Mean = mean(x, na.rm = TRUE),
    SD = sd(x, na.rm = TRUE),
    Min = min(x, na.rm = TRUE),
    Max = max(x, na.rm = TRUE)
  )
})

sumstats_tex <- c(
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{Summary Statistics: Difference-in-Differences Variables}",
  "\\label{tab:sumstats}",
  "\\begin{tabular}{lrrrrr}",
  "\\hline\\hline",
  "Variable & N & Mean & SD & Min & Max \\\\",
  "\\hline",
  apply(sumstats_rows, 1, function(r) {
    sprintf(
      "%s & %s & %.3f & %.3f & %.3f & %.3f \\\\",
      r["Variable"],
      format(as.integer(r["N"]), big.mark = ","),
      as.numeric(r["Mean"]),
      as.numeric(r["SD"]),
      as.numeric(r["Min"]),
      as.numeric(r["Max"])
    )
  }),
  "\\hline\\hline",
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(
  sumstats_tex,
  con = path_tables("table_sumstats_dd_variables.tex")
)
