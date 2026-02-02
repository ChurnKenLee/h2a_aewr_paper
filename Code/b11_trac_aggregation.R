library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(foreign)
library(tidycensus)
library(lubridate)
library(foreign)
library(haven)

rm(list = ls())

trac_df <- read_parquet(here("binaries", "trac.parquet")) %>% 
  clean_names() %>% 
  distinct(.keep_all = TRUE)

# Aggregate detainment counts
trac_df <- trac_df %>% 
  mutate(detainment_count = as.numeric(count)) %>% 
  mutate(year = fy)

# Aggregate to FIPS and year
agg_trac_df <- trac_df %>% 
  group_by(fips, year) %>% 
  summarise(ice_detainment_count = sum(detainment_count, na.rm = TRUE)) %>% 
  ungroup()

agg_trac_df <- agg_trac_df %>% 
  mutate(state_fips_code = substr(fips, 1, 2)) %>% 
  mutate(county_fips_code = substr(fips, 3, 5)) %>% 
  select(-fips)

# Export
agg_trac_df %>% 
  write_parquet(here("files_for_phil", "trac_aggregated.parquet")) %>% 
  write_parquet(here("binaries", "trac_aggregated.parquet"))


