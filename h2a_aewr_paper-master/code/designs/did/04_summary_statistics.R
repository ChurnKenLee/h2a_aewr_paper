# Purpose: Produce summary statistics for the retained DiD sample.
# Output: outputs/tables/table_sumstats_dd_variables.tex.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "did", "helpers.R"))
library(arrow)
library(dplyr)
library(purrr)
library(tibble)

sample <- read_parquet(
  path_processed("did_county_year_panel.parquet")
) %>%
  did_sample()

variables <- c(
  "H-2A share of 2011 farm employment" = "h2a_cert_share_farm_workers_2011_start_year",
  "H-2A certified workers (start year)" = "nbr_workers_certified_start_year",
  "Farm employment 2011 (baseline)" = "emp_farm_2011",
  "AEWR p25 bite (2012 \\$)" = "aewr_cz_p25",
  "Log population" = "ln_pop_census",
  "Employment-to-population ratio" = "emp_pop_ratio"
)

rows <- imap_dfr(variables, \(column, label) {
  values <- sample[[column]]
  tibble(
    Variable = label,
    N = sum(!is.na(values)),
    Mean = mean(values, na.rm = TRUE),
    SD = sd(values, na.rm = TRUE),
    Min = min(values, na.rm = TRUE),
    Max = max(values, na.rm = TRUE)
  )
})

table_lines <- c(
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{Summary Statistics: Difference-in-Differences Variables}",
  "\\label{tab:sumstats}",
  "\\begin{tabular}{lrrrrr}",
  "\\hline\\hline",
  "Variable & N & Mean & SD & Min & Max \\\\",
  "\\hline",
  apply(rows, 1, \(row) {
    sprintf(
      "%s & %s & %.3f & %.3f & %.3f & %.3f \\\\",
      row["Variable"],
      format(as.integer(row["N"]), big.mark = ","),
      as.numeric(row["Mean"]),
      as.numeric(row["SD"]),
      as.numeric(row["Min"]),
      as.numeric(row["Max"])
    )
  }),
  "\\hline\\hline",
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(
  table_lines,
  path_tables("table_sumstats_dd_variables.tex")
)
