library(here)
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)

rm(list = ls())

rma_sob <- read_parquet(here("binaries", "rma_sob.parquet"))

crops <- rma_sob %>% 
  distinct(commodity_name)

insurance_plans <- rma_sob %>% 
  distinct(insurance_plan_name_abbreviation)

coverage_categories <- rma_sob %>% 
  distinct(coverage_category)

delivery_types <- rma_sob %>% 
  distinct(delivery_type)

quantity_types <- rma_sob %>% 
  distinct(quantity_type)
