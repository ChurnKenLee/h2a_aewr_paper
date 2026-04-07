# Harmonize county ANSI from disparate sources to 2010 standard
# List of sources we have as inputs:
# H-2A usage aggregated
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
