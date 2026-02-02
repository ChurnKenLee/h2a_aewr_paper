library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(tidycensus)
library(fixest)

rm(list = ls())

#### Load aggregated reports ####

terminal_market_agg <- read_parquet(here("binaries", "mymarketnews_terminal_market_aggregate.parquet"))
shipping_point_agg <- read_parquet(here("binaries", "mymarketnews_shipping_point_aggregate.parquet"))

# Get list of commodities
shipping_commodities <- shipping_point_agg %>% 
  distinct(commodity)
terminal_commodities <- terminal_market_agg %>% 
  distinct(commodity)
terminal_markets <- terminal_market_agg %>% 
  distinct(market_location_city)

# Map commodities to NASS commodities






