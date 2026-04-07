library(here)
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)

rm(list = ls())

cov <- read_parquet(here("binaries", "rma_sob_cov.parquet"))

crops <- cov %>% 
  distinct(commodity_name)

insurance_plans <- cov %>% 
  distinct(insurance_plan_name_abbreviation)

coverage_categories <- cov %>% 
  distinct(coverage_category)

delivery_types <- cov %>% 
  distinct(delivery_type)

quantity_types <- cov %>% 
  distinct(quantity_type)

tpu <- read_parquet(here("binaries", "rma_sob_tpu.parquet"))



