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

#### Load and aggregate reports ####
terminal_market_list <- list()
shipping_point_list <- list()

for (year in 2004:2004) {
  terminal_market_filename <- paste0("mymarketnews_terminal_market_reports_", year, ".parquet")
  print(terminal_market_filename)
  terminal_market <- read_parquet(here("binaries", terminal_market_filename))

  terminal_market <- terminal_market %>%
    mutate(year = substring(report_date, 7, 10))

  # Coerce prices to numeric
  terminal_market <- terminal_market %>%
    mutate_at(c('high_price', 'low_price', 'mostly_high_price', 'mostly_low_price'), as.numeric)

  terminal_market <- terminal_market %>% 
    mutate(
      price = case_when(
        !is.na(mostly_high_price) & !is.na(mostly_low_price) ~ (mostly_high_price + mostly_low_price)/2,
        (is.na(mostly_high_price) | is.na(mostly_low_price)) & (!is.na(high_price) & !is.na(low_price)) ~ (high_price + low_price)/2,
        .default = NA
      )
    )
  
  terminal_market <- terminal_market %>% 
    mutate(
      price = case_when(
        is.na(price) & !is.na(low_price) ~ low_price,
        is.na(price) & !is.na(high_price) ~ high_price,
        .default = price
      )
    )

  # Aggregate
  terminal_market_sum <- terminal_market %>%
    group_by(year, market_location_city, state_of_origin, commodity, variety, properties, organic, package, grade, item_size) %>%
    summarize(n = n(), price_mean = mean(price), price_sd = sd(price)) %>%
    arrange(market_location_city, commodity, origin) %>%
    ungroup()

  terminal_market_sum <- terminal_market_sum %>%
    mutate(year = year)

  terminal_market_list[[year]] <- terminal_market_sum

  # Repeat for shipping point report
  shipping_point_filename <- paste0("mymarketnews_shipping_point_reports_", year, ".parquet")
  print(shipping_point_filename)
  shipping_point <- read_parquet(here("binaries", shipping_point_filename))

  shipping_point <- shipping_point %>%
    mutate(year = substring(report_date, 7, 10))

  shipping_point <- shipping_point %>%
    mutate_at(c('high_price', 'low_price', 'mostly_high_price', 'mostly_low_price'), as.numeric)
  
  # Calculate price for each entry
  shipping_point <- shipping_point %>% 
    mutate(
      price = case_when(
        !is.na(mostly_high_price) & !is.na(mostly_low_price) ~ (mostly_high_price + mostly_low_price)/2,
        (is.na(mostly_high_price) | is.na(mostly_low_price)) & (!is.na(high_price) & !is.na(low_price)) ~ (high_price + low_price)/2,
        .default = NA
      )
    )
  
  shipping_point <- shipping_point %>% 
    mutate(
      price = case_when(
        is.na(price) & !is.na(low_price) ~ low_price,
        is.na(price) & !is.na(high_price) ~ high_price,
        .default = price
      )
    )

  shipping_point_sum <- shipping_point %>%
    group_by(year, state_of_origin, commodity, var, properties, organic, pkg, grade, item_size) %>%
    summarize(n = n(), price_mean = mean(price), price_sd = sd(price)) %>%
    ungroup()

  shipping_point_sum <- shipping_point_sum %>%
    mutate(year = year)

  shipping_point_list[[year]] <- shipping_point_sum
}

# Combine and save
terminal_market_agg <- bind_rows(terminal_market_list)
terminal_market_agg %>% write_parquet(here("binaries", "mymarketnews_terminal_market_aggregate.parquet"))

shipping_point_agg <- bind_rows(shipping_point_list)
shipping_point_agg %>% write_parquet(here("binaries", "mymarketnews_shipping_point_aggregate.parquet"))
