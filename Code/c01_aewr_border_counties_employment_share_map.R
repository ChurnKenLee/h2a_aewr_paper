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
  filter(year == 2023) %>% 
  rename(state_fips = state_fips_code) %>% 
  select(state_fips, aewr)

# Counties on AEWR region border
aewr_adjacent_counties <- read_csv(here("Data", "county_adjacency2010.csv"))

# Add 2023 AEWR
aewr_adjacent_counties <- aewr_adjacent_counties %>% 
  mutate(state_fips = substr(fipscounty, 1, 2)) %>% 
  mutate(neighbor_state_fips = substr(fipsneighbor, 1, 2)) %>% 
  left_join(aewr, by = c("state_fips")) %>% 
  rename(state_aewr = aewr)

aewr_adjacent_counties <- aewr_adjacent_counties %>% 
  left_join(aewr, by = c("neighbor_state_fips" = "state_fips")) %>% 
  rename(neighbor_state_aewr = aewr)

aewr_adjacent_counties <- aewr_adjacent_counties %>% 
  filter(!is.na(state_aewr) & !is.na(neighbor_state_aewr)) %>% 
  mutate(is_aewr_border_county = state_aewr != neighbor_state_aewr)

# Keep only counties that border at least one other county with a different AEWR
aewr_adjacent_counties <- aewr_adjacent_counties %>% 
  group_by(fipscounty) %>% 
  summarize(on_aewr_region_border = any(is_aewr_border_county == TRUE)) %>% 
  rename(county_fips = fipscounty) %>% 
  ungroup()

# Add AEWR data to county shapefile
aewr_county_sf <- get(data(county_laea)) %>% 
  mutate(county_fips = GEOID) %>% 
  left_join(aewr_adjacent_counties) %>% 
  filter(on_aewr_region_border == TRUE)

aewr_state_sf <- get(data(state_laea)) %>% 
  mutate(state_fips = GEOID) %>% 
  left_join(aewr)

aewr_map <- ggplot() +
  geom_sf(data = aewr_state_sf, aes(fill = aewr)) +
  scale_fill_viridis_c(name = "AEWR") +
  new_scale_color() +
  geom_sf(data = aewr_county_sf, fill = "white", alpha = 0.3) + 
  coord_sf(datum = NA)

ggsave("aewr_region_border_counties_map_2023.png", plot = aewr_map, path = here("Output"))

# BEA
bea <- read_parquet(here("files_for_phil", "bea_farm_nonfarm_emp.parquet"))

# Calculate AEWR region totals and employment shares
# Add AEWR data and border county indicator
bea <- bea %>% 
  mutate(state_fips = substr(county_fips, 1, 2)) %>% 
  left_join(aewr) %>% 
  left_join(aewr_adjacent_counties) %>% 
  filter(!is.na(aewr)) %>% 
  filter(!is.na(on_aewr_region_border))

# Calculate county shares and AEWR border-county-only shares out of total AEWR farm employment
bea <- bea %>% 
  group_by(aewr, acs_year) %>% 
  mutate(aewr_region_total_farm_emp = sum(bea_farm_emp)) %>% 
  ungroup() %>% 
  group_by(aewr, acs_year, on_aewr_region_border) %>% 
  mutate(aewr_border_total_farm_emp = sum(bea_farm_emp)) %>% 
  ungroup() %>% 
  mutate(county_farm_emp_share = bea_farm_emp/aewr_region_total_farm_emp) %>% 
  mutate(aewr_border_farm_emp_share = aewr_border_total_farm_emp/aewr_region_total_farm_emp)

# Drop Hawaii
bea <- bea %>% filter(state_fips != "15")

# Need it in wide format for easier merging and mapping
bea <- bea %>% 
  pivot_wider(
    id_cols = county_fips,
    names_from = acs_year,
    values_from = c("county_farm_emp_share", "aewr_border_farm_emp_share", "on_aewr_region_border")
    )

# Add BEA data to county shapefile
bea_county_sf <- get(data(county_laea)) %>% 
  mutate(county_fips = GEOID) %>% 
  left_join(bea)

# 2007
bea_map <- ggplot() +
  geom_sf(data = bea_county_sf, aes(fill = county_farm_emp_share_2007)) +
  scale_fill_viridis_c(name = "Share") +
  geom_sf(data = aewr_county_sf, fill = "white", alpha = 0.3) + 
  coord_sf(datum = NA) +
  ggtitle("")

ggsave("aewr_region_individual_border_counties_emp_share_map_2007.png", plot = bea_map, path = here("Output"))

# 2012
bea_map <- ggplot() +
  geom_sf(data = bea_county_sf, aes(fill = county_farm_emp_share_2012)) +
  scale_fill_viridis_c(name = "Share") +
  geom_sf(data = aewr_county_sf, fill = "white", alpha = 0.3) + 
  coord_sf(datum = NA) +
  ggtitle("")

ggsave("aewr_region_individual_border_counties_emp_share_map_2012.png", plot = bea_map, path = here("Output"))

# 2017
bea_map <- ggplot() +
  geom_sf(data = bea_county_sf, aes(fill = county_farm_emp_share_2017)) +
  scale_fill_viridis_c(name = "Share") +
  geom_sf(data = aewr_county_sf, fill = "white", alpha = 0.3) + 
  coord_sf(datum = NA) +
  ggtitle("")

ggsave("aewr_region_individual_border_counties_emp_share_map_2027.png", plot = bea_map, path = here("Output"))

# 2022
bea_map <- ggplot() +
  geom_sf(data = bea_county_sf, aes(fill = county_farm_emp_share_2022)) +
  scale_fill_viridis_c(name = "Share") +
  geom_sf(data = aewr_county_sf, fill = "white", alpha = 0.3) + 
  coord_sf(datum = NA) +
  ggtitle("")

ggsave("aewr_region_individual_border_counties_emp_share_map_2022.png", plot = bea_map, path = here("Output"))

# Now do the same for border as a whole
# 2007
bea_map <- ggplot() +
  geom_sf(data = bea_county_sf, aes(fill = aewr_border_farm_emp_share_2007)) +
  scale_fill_viridis_c(name = "Share") +
  coord_sf(datum = NA) +
  ggtitle("")

ggsave("aewr_region_border_counties_emp_share_map_2007.png", plot = bea_map, path = here("Output"))

# 2012
bea_map <- ggplot() +
  geom_sf(data = bea_county_sf, aes(fill = aewr_border_farm_emp_share_2012)) +
  scale_fill_viridis_c(name = "Share") +
  coord_sf(datum = NA) +
  ggtitle("")

ggsave("aewr_region_border_counties_emp_share_map_2012.png", plot = bea_map, path = here("Output"))

# 2017
bea_map <- ggplot() +
  geom_sf(data = bea_county_sf, aes(fill = aewr_border_farm_emp_share_2017)) +
  scale_fill_viridis_c(name = "Share") +
  coord_sf(datum = NA) +
  ggtitle("")

ggsave("aewr_region_border_counties_emp_share_map_2017.png", plot = bea_map, path = here("Output"))

# 2022
bea_map <- ggplot() +
  geom_sf(data = bea_county_sf, aes(fill = aewr_border_farm_emp_share_2022)) +
  scale_fill_viridis_c(name = "Share") +
  coord_sf(datum = NA) +
  ggtitle("")

ggsave("aewr_region_border_counties_emp_share_map_2022.png", plot = bea_map, path = here("Output"))

# Produce histograms of distribution of farm employment fractions
bea_border_farm_emp_share <- bea %>% 
  distinct(aewr_border_farm_emp_share_2007, on_aewr_region_border_2007, .keep_all = TRUE) %>% 
  filter(on_aewr_region_border_2007 == TRUE)

# 2007
bea_histogram <- ggplot() +
  geom_histogram(data = bea_border_farm_emp_share, aes(x = aewr_border_farm_emp_share_2007)) +
  xlab("Share of farm employment within each AEWR region's border, 2007")

ggsave("aewr_region_border_counties_emp_share_hist_2007.png", plot = bea_histogram, path = here("Output"))

# 2012
bea_histogram <- ggplot() +
  geom_histogram(data = bea_border_farm_emp_share, aes(x = aewr_border_farm_emp_share_2012)) +
  xlab("Share of farm employment within each AEWR region's border, 2012")

ggsave("aewr_region_border_counties_emp_share_hist_2012.png", plot = bea_histogram, path = here("Output"))

# 2017
bea_histogram <- ggplot() +
  geom_histogram(data = bea_border_farm_emp_share, aes(x = aewr_border_farm_emp_share_2017)) +
  xlab("Share of farm employment within each AEWR region's border, 2017")

ggsave("aewr_region_border_counties_emp_share_hist_2017.png", plot = bea_histogram, path = here("Output"))

# 2022
bea_histogram <- ggplot() +
  geom_histogram(data = bea_border_farm_emp_share, aes(x = aewr_border_farm_emp_share_2022)) +
  xlab("Share of farm employment within each AEWR region's border, 2022")

ggsave("aewr_region_border_counties_emp_share_hist_2022.png", plot = bea_histogram, path = here("Output"))


