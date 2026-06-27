# Test CZ-level AEWR IVs
rm(list = ls())
if (file.exists("paths.R")) {
  source("paths.R")
} else {
  source(file.path("code", "paths.R"))
}
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(foreign)
library(fixest)

county_df_iv <- read_parquet(path_processed(
  "county_df_analysis_year_iv.parquet"
)) %>%
  clean_names()

fs_sample <- county_df_iv %>%
  filter(
    any_cropland_2007 == 1,
    !is.na(aewr_ppi),
    !is.na(z_oews_agwage_l1),
    !is.na(aewr_region_fe),
    !is.na(year_fe),
    !is.na(cz_aewr_region_fe)
  )

cat("First-stage sample rows:", nrow(fs_sample), "\n")
cat("AEWR region FE:", n_distinct(fs_sample$aewr_region_fe), "\n")
cat("CZ x AEWR clusters:", n_distinct(fs_sample$cz_aewr_region_fe), "\n\n")

fs_1 <- feols(
  aewr_ppi ~ z_oews_agwage_l1 | aewr_region_fe + year_fe,
  data = fs_sample,
  vcov = ~cz_aewr_region_fe
)

delta <- coef(fs_1)["z_oews_agwage_l1"]
delta_se <- se(fs_1)["z_oews_agwage_l1"]
delta_t <- delta / delta_se
delta_f <- delta_t^2

cat("=== First stage: AEWR on donor wage IV ===\n")
cat("Outcome: aewr_ppi\n")
cat("Instrument: z_oews_agwage_l1\n")
cat("FE: AEWR region + year\n")
cat("Clustered SE: cz_aewr_region_fe\n\n")
cat("delta:", round(delta, 4), "\n")
cat("clustered SE:", round(delta_se, 4), "\n")
cat("t-stat:", round(delta_t, 3), "\n")
cat("single-instrument F:", round(delta_f, 2), "\n\n")

print(etable(fs_1))
