library(arrow)

rm(list = ls())

ds <- open_dataset(here(
  "binaries",
  "acs_5year_2007_2012_2017_2022.parquet"
))
df <- ds %>%
  rename_with(tolower)

# New logic to find lowercase variables in your old scripts
find_lowercase_vars <- function(script_path, dataset_names) {
  script_text <- readLines(script_path)

  # Look for common IPUMS-style words in lowercase
  # (Matches words with at least one lowercase letter, 2-10 chars)
  matches <- unlist(regmatches(
    script_text,
    gregexpr("\\b[a-z0-9_]{2,10}\\b", script_text)
  ))

  # Compare against the lowercase version of your actual column names
  valid_vars <- intersect(unique(matches), tolower(dataset_names))
  return(valid_vars)
}

# Usage
file_vars <- find_lowercase_vars(
  here("code", "b03_01_acs_immigrant_imputation.R"),
  names(df)
)

file_vars <- toupper(file_vars)
file_vars
