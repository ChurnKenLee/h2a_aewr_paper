# Harmonize county ANSI from disparate sources to 2010 standard
# List of sources we have as inputs:
# H-2A usage aggregated and predicted usage
# NASS quickstats
# OEWS
# QCEW
# CroplandCROS CDL
# Predicted H-2A usage
# ACS immigrant imputation
# ACS wage quantiles
# BEA farm employment
# NAWSPAD

library(here)
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(foreign)
library(haven)

rm(list = ls())

h2a <- read_parquet(here("binaries", "h2a_aggregated.parquet"))
h2a_predict <- read_parquet(here(
    "binaries",
    "h2a_prediction_using_elastic_net.parquet"
))
cz_wage_quantiles <- read_parquet(here(
    "binaries",
    "acs_czone_wage_quantiles.parquet"
))
qs_census <- read_parquet(here(
    "binaries",
    "qs_census_selected_obs.parquet"
))
cdl_acreage <- read_parquet(here(
    "binaries",
    "croplandcros_county_crop_acres.parquet"
))
qcs_qcew <- read_parquet(here("binaries", "acs_qcew.parquet"))
oews_agg <- read_parquet(here("binaries", "oews_county_aggregated.parquet"))
acs_immigrant_imputed <- read_parquet(here(
    "binaries",
    "acs_immigrant_imputed.parquet"
))
