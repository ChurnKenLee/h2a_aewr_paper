# Compare different data sources for correlation with FLS
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

# Compare OEWS, QCEW, ACS in how closely they track FLS wage data
fls <- read_parquet(path_int("fls_region.parquet"))
fls_state <- read_parquet(path_int("fls_state.parquet"))

oews <- read_parquet(path_int("oews.parquet"))
