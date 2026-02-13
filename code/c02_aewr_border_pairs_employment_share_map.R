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
library(sf)
library(ggplot2)
library(ggnewscale)

library(tidycensus)
options(tigris_use_cache = TRUE)

rm(list = ls())

# AEWR
aewr <- read_csv(here("Data", "aewr", "state_year_aewr.csv")) %>% 
  rename(state_fips = state_fips_code) %>% 
  mutate(acs_year = case_when(
    between(year, 2000, 2004) ~ 2004,
    between(year, 2005, 2007) ~ 2007,
    between(year, 2008, 2012) ~ 2012,
    between(year, 2013, 2017) ~ 2017,
    between(year, 2018, 2022) ~ 2022
  )) %>% 
  group_by(state_fips, acs_year) %>% 
  summarize(aewr = mean(aewr)) %>% 
  ungroup() %>% 
  filter(!is.na(acs_year))

# Produce histogram with AEWR region border pairs instead of all borders within AEWR region
# Counties on AEWR region border
aewr_adjacent_counties <- read_csv(here("Data", "county_adjacency2010.csv"))

# Create 4 duplicate rows, one for each year, to merge with multi-year BEA data
aewr_adjacent_counties <- aewr_adjacent_counties %>% 
  mutate(acs_year = "2007,2012,2017,2022") %>% 
  separate_longer_delim(acs_year, delim = ",") %>% 
  mutate(acs_year = as.numeric(acs_year))

# Add AEWR
aewr_adjacent_counties <- aewr_adjacent_counties %>% 
  mutate(state_fips = substr(fipscounty, 1, 2)) %>% 
  mutate(neighbor_state_fips = substr(fipsneighbor, 1, 2)) %>% 
  left_join(aewr, by = c("state_fips", "acs_year")) %>% 
  rename(state_aewr = aewr) %>% 
  rename(county_fips = fipscounty)

aewr_adjacent_counties <- aewr_adjacent_counties %>% 
  left_join(aewr, by = c("neighbor_state_fips" = "state_fips", "acs_year")) %>% 
  rename(neighbor_state_aewr = aewr)

aewr_adjacent_counties <- aewr_adjacent_counties %>% 
  filter(!is.na(state_aewr) & !is.na(neighbor_state_aewr)) %>% 
  mutate(differing_aewr = state_aewr != neighbor_state_aewr) %>% 
  group_by(county_fips) %>% 
  mutate(is_aewr_border_county = any(differing_aewr == TRUE)) %>% 
  filter(county_fips != fipsneighbor) %>%  # Remove entries where county borders itself
  ungroup()

# Add BEA employment counts
bea <- read_parquet(here("files_for_phil", "bea_farm_nonfarm_emp.parquet"))

aewr_adjacent_counties <- aewr_adjacent_counties %>% 
  left_join(bea, by = c("county_fips", "acs_year"))

# Calculate group aggregates
# Total emp within AEWR region
aewr_region_totals <- aewr_adjacent_counties %>% 
  distinct(county_fips, acs_year, .keep_all = TRUE) %>% 
  group_by(state_aewr, acs_year) %>% 
  mutate(aewr_region_total_farm_emp = sum(bea_farm_emp, na.rm = TRUE)) %>% 
  ungroup() %>% 
  select(county_fips, acs_year, aewr_region_total_farm_emp)

# Total within AEWR region border pair
aewr_border_pair_totals <- aewr_adjacent_counties %>% 
  filter(is_aewr_border_county == TRUE) %>% 
  filter(state_aewr != neighbor_state_aewr) %>% # We keep only one observation per border county: the observation that identifies the neighboring county of another AEWR region
  distinct(county_fips, acs_year, .keep_all = TRUE) %>% # There are counties that border multiple counties in neighboring AEWR region, and so appear multiple times; this means that border counties that border multiple AEWR regions are assigned to only one border pair
  group_by(state_aewr, neighbor_state_aewr, acs_year) %>% 
  mutate(aewr_region_border_pair_total_farm_emp = sum(bea_farm_emp)) %>% 
  ungroup() %>% 
  select(county_fips, acs_year, aewr_region_border_pair_total_farm_emp)

# AEWR region interior non-border county totals
aewr_non_border_totals <- aewr_adjacent_counties %>% 
  filter(is_aewr_border_county == FALSE) %>% 
  distinct(county_fips, acs_year, .keep_all = TRUE) %>% 
  group_by(state_aewr, acs_year) %>% 
  mutate(aewr_region_nonborder_total_farm_emp = sum(bea_farm_emp, na.rm = TRUE)) %>% 
  select(county_fips, acs_year, aewr_region_nonborder_total_farm_emp) %>% 
  ungroup()

# Combine them all back together for plotting
aewr_map_df <- aewr_adjacent_counties %>% 
  left_join(aewr_border_pair_totals, by = c("county_fips", "acs_year")) %>% 
  left_join(aewr_non_border_totals, by = c("county_fips", "acs_year")) %>% 
  left_join(aewr_region_totals, by = c("county_fips", "acs_year")) %>% 
  mutate(numerator = aewr_region_nonborder_total_farm_emp) %>% 
  mutate(numerator = if_else(is.na(numerator), aewr_region_border_pair_total_farm_emp, numerator)) %>% 
  mutate(share = numerator/aewr_region_total_farm_emp) %>% 
  arrange(county_fips, acs_year)

aewr_map_df <- aewr_map_df %>% 
  distinct(county_fips, acs_year, .keep_all = TRUE) %>% 
  arrange(county_fips, acs_year) %>% 
  pivot_wider(
    id_cols = c("county_fips"),
    names_prefix = "share_",
    names_from = acs_year,
    values_from = share
  )

# Plot maps
county_sf <- get(data(county_laea)) %>% 
  mutate(county_fips = GEOID) %>% 
  left_join(aewr_map_df)

# 2007
bea_map <- ggplot() +
  geom_sf(data = county_sf, aes(fill = share_2007)) +
  scale_fill_viridis_c(name = "Share") +
  coord_sf(datum = NA) +
  ggtitle("Border pairs' share of total farm employment within each AEWR region, 2007")

ggsave("aewr_region_county_pairs_emp_share_map_2007.png", plot = bea_map, path = here("Output"))

# 2012
bea_map <- ggplot() +
  geom_sf(data = county_sf, aes(fill = share_2012)) +
  scale_fill_viridis_c(name = "Share") +
  coord_sf(datum = NA) +
  ggtitle("Border pairs' share of total farm employment within each AEWR region, 2012")

ggsave("aewr_region_county_pairs_emp_share_map_2012.png", plot = bea_map, path = here("Output"))

# 2017
bea_map <- ggplot() +
  geom_sf(data = county_sf, aes(fill = share_2017)) +
  scale_fill_viridis_c(name = "Share") +
  coord_sf(datum = NA) +
  ggtitle("Border pairs' share of total farm employment within each AEWR region, 2017")

ggsave("aewr_region_county_pairs_emp_share_map_2017.png", plot = bea_map, path = here("Output"))

# 2022
bea_map <- ggplot() +
  geom_sf(data = county_sf, aes(fill = share_2022)) +
  scale_fill_viridis_c(name = "Share") +
  coord_sf(datum = NA) +
  ggtitle("Border pairs' share of total farm employment within each AEWR region, 2022")

ggsave("aewr_region_county_pairs_emp_share_map_2022.png", plot = bea_map, path = here("Output"))

# Plot histograms
# Only want one observation per border pair-year
border_pair_farm_emp_share <- aewr_adjacent_counties %>% 
  left_join(aewr_border_pair_totals, by = c("county_fips", "acs_year")) %>% 
  left_join(aewr_region_totals, by = c("county_fips", "acs_year")) %>% 
  distinct(state_aewr, neighbor_state_aewr, acs_year, .keep_all = TRUE) %>% 
  mutate(share = aewr_region_border_pair_total_farm_emp/aewr_region_total_farm_emp) %>% 
  filter(!is.na(share))

# 2007
bea_histogram <- ggplot() +
  geom_histogram(data = border_pair_farm_emp_share %>% filter(acs_year == 2007), aes(x = share)) +
  xlab("Border pairs' share of total farm employment within each AEWR region, 2007")

ggsave("aewr_region_county_pairs_emp_share_hist_2007.png", plot = bea_histogram, path = here("Output"))

# 2012
bea_histogram <- ggplot() +
  geom_histogram(data = border_pair_farm_emp_share %>% filter(acs_year == 2012), aes(x = share)) +
  xlab("Border pairs' share of total farm employment within each AEWR region, 2012")

ggsave("aewr_region_county_pairs_emp_share_hist_2012.png", plot = bea_histogram, path = here("Output"))

# 2017
bea_histogram <- ggplot() +
  geom_histogram(data = border_pair_farm_emp_share %>% filter(acs_year == 2017), aes(x = share)) +
  xlab("Border pairs' share of total farm employment within each AEWR region, 2017")

ggsave("aewr_region_county_pairs_emp_share_hist_2017.png", plot = bea_histogram, path = here("Output"))

# 2022
bea_histogram <- ggplot() +
  geom_histogram(data = border_pair_farm_emp_share %>% filter(acs_year == 2022), aes(x = share)) +
  xlab("Border pairs' share of total farm employment within each AEWR region, 2022")

ggsave("aewr_region_county_pairs_emp_share_hist_2022.png", plot = bea_histogram, path = here("Output"))

# Check stability of treatment/control over time
treatment_counties <- aewr_adjacent_counties %>% 
  filter(is_aewr_border_county == TRUE) %>% 
  filter(state_aewr != neighbor_state_aewr) %>% # We keep only one observation per border county: the observation that identifies the neighboring county of another AEWR region
  distinct(county_fips, acs_year, .keep_all = TRUE) %>% # There are counties that border multiple counties in neighboring AEWR region, and so appear multiple times; this means that border counties that border multiple AEWR regions are assigned to only one border pair
  mutate(treatment_dummy = state_aewr > neighbor_state_aewr) %>% 
  pivot_wider(
    id_cols = c("county_fips"),
    names_prefix = "treatment_",
    names_from = acs_year,
    values_from = treatment_dummy
  )

treatment_counties <- treatment_counties %>% 
  mutate(all_treatment = (treatment_2007 == TRUE) & (treatment_2012 == TRUE) & (treatment_2017 == TRUE) & (treatment_2022 == TRUE)) %>% 
  mutate(all_control = (treatment_2007 == FALSE) & (treatment_2012 == FALSE) & (treatment_2017 == FALSE) & (treatment_2022 == FALSE)) %>% 
  mutate(all_same = all_treatment | all_control)

# Plot treatment/control maps over time
county_sf <- get(data(county_laea)) %>% 
  mutate(county_fips = GEOID) %>% 
  left_join(treatment_counties)

# Stable counties
bea_map <- ggplot() +
  geom_sf(data = county_sf, aes(fill = all_same)) +
  scale_fill_discrete(name = "Stable") +
  coord_sf(datum = NA) +
  ggtitle("Counties that have stable treatment/control status")

ggsave("counties_stable_treatment_map.png", plot = bea_map, path = here("Output"))

# 2007
bea_map <- ggplot() +
  geom_sf(data = county_sf, aes(fill = treatment_2007)) +
  scale_fill_discrete(name = "Treated") +
  coord_sf(datum = NA) +
  ggtitle("Treatment assignment, 2007")

ggsave("counties_treatment_2007_map.png", plot = bea_map, path = here("Output"))

# 2012
bea_map <- ggplot() +
  geom_sf(data = county_sf, aes(fill = treatment_2012)) +
  scale_fill_discrete(name = "Treated") +
  coord_sf(datum = NA) +
  ggtitle("Treatment assignment, 2012")

ggsave("counties_treatment_2012_map.png", plot = bea_map, path = here("Output"))

# 2017
bea_map <- ggplot() +
  geom_sf(data = county_sf, aes(fill = treatment_2017)) +
  scale_fill_discrete(name = "Treated") +
  coord_sf(datum = NA) +
  ggtitle("Treatment assignment, 2017")

ggsave("counties_treatment_2017_map.png", plot = bea_map, path = here("Output"))

# 2022
bea_map <- ggplot() +
  geom_sf(data = county_sf, aes(fill = treatment_2022)) +
  scale_fill_discrete(name = "Treated") +
  coord_sf(datum = NA) +
  ggtitle("Treatment assignment, 2022")

ggsave("counties_treatment_2022_map.png", plot = bea_map, path = here("Output"))
