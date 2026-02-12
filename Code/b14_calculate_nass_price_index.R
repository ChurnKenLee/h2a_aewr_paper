library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(tidycensus)

rm(list = ls())

#### Calculate Laspeyres and Paasche price indices ####
county_production_prices <- read_parquet(here("binaries", "county_price_output.parquet"))

county_production_prices <- county_production_prices %>% 
  filter(group_desc != "FIELD CROPS")

# Keep only census years
census_years_list <- c(2002, 2007, 2012, 2017, 2022)
county_production_prices <- county_production_prices %>% 
  filter(year %in% census_years_list)

# Generate product IDs; each crop is a different product;
# We want to calculate separate price indices for each county
county_production_prices <- county_production_prices %>%
  mutate(pt = price) %>% 
  mutate(qt = production) %>% 
  group_by(group_desc, commodity_desc, class_desc, prodn_practice_desc, util_practice_desc) %>%
  mutate(product_id = cur_group_id()) %>%
  ungroup() %>% 
  group_by(state_fips_code, county_code) %>%
  mutate(county_id = cur_group_id()) %>%
  ungroup()

# Get a full set of counties and years
counties <- county_production_prices %>% 
  distinct(state_name, state_fips_code, county_name, county_code, county_id)

years <- county_production_prices %>% 
  distinct(year)

county_years <- counties %>% cross_join(years) %>% 
  arrange(state_name, county_name, year)

prepped_df <- county_production_prices %>% 
  select(county_id, year, product_id, pt, qt) %>% 
  filter(qt != 0) %>% 
  filter(pt != 0)

# Calculate chained Laspeyres, Paasche, and Fisher price indices
index_list <- list()

for (y in census_years_list) {
  t0 <- y
  t1 <- t0 + 5
  
  t0_data <- prepped_df %>% 
    filter(year == t0) %>% 
    select(-year) %>% 
    rename(
      p0 = pt,
      q0 = qt
    )
  
  t1_data <- prepped_df %>% 
    filter(year == t1) %>% 
    select(-year) %>% 
    rename(
      p1 = pt,
      q1 = qt
    )
  
  index_data <- t0_data %>% 
    inner_join(t1_data) %>% 
    drop_na() %>% 
    mutate(
      p0q0 = p0*q0,
      p1q0 = p1*q0,
      p0q1 = p0*q1,
      p1q1 = p1*q1
    )
  
  index_data <- index_data %>% 
    group_by(county_id) %>% 
    summarize(
      sum_p0q0 = sum(p0q0),
      sum_p1q0 = sum(p1q0),
      sum_p0q1 = sum(p0q1),
      sum_p1q1 = sum(p1q1)
    ) %>% 
    ungroup()
  
  index_data <- index_data %>% 
    mutate(
      period_laspeyres = sum_p1q0/sum_p0q0,
      period_paasche = sum_p1q1/sum_p0q1,
      period_fisher = (period_laspeyres*period_paasche)**0.5
    )
  
  # Add calculated indices to index list to concat
  index_data <- index_data %>% 
    mutate(year = t0)
  
  index_list[[t0]] <- index_data
}
indices <- index_list %>% 
  bind_rows() %>% 
  select(-sum_p0q0, -sum_p1q0, -sum_p0q1, -sum_p1q1)

price_indices <- county_years %>% 
  left_join(indices)

# Export and share with Phil
price_indices <- price_indices %>% 
  select(-county_id)

test <- price_indices %>% 
  filter(!is.na(period_laspeyres))

price_indices %>%
  write_parquet(here("files_for_phil", "county_price_indices.parquet")) %>%
  write_parquet(here("binaries", "county_price_indices.parquet"))
