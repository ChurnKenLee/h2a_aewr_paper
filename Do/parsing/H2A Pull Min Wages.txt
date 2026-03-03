library(dplyr)
library(tidyverse)
library(purrr)
library(fredr)
library(arrow)

# You need your FRED API key
fredr_set_key("925af4f1c2cbc732cfb528ddab32861f")

# The FRED series ID for federal minimum wage (annual) 
federal_series <- "STTMINWGFG"

# folder_data <- "R:/Hoxie/H-2A Paper/Data Int/"

state_fips <- read_csv(paste0(folder_data, "fips_codes.csv"))

# A lookup of state FRED series id and FIPS
# You'll need to fill this out for all 50 states. As an example:

state_fred <- state_fips %>% 
  mutate(series_id = str_trim(paste0("STTMINWG", state_abbrev))) %>% 
  filter(fips <= 56) 

# state_fred <- tibble(
#   state = c("California", "Texas", "New York"),  # add other states
#   fips = c("06", "48", "36"),                    # add other FIPS
#   series_id = c("STTMINWGCA", "STTMINWGTX", "STTMINWGNY") # add other series
# )

# A function to fetch a series and return as a tidy tibble
get_min_wage <- function(series_id) {
  fredr(
    series_id = series_id,
    observation_start = as.Date("1968-01-01"),  # adjust start year as needed
    frequency = "a"  # annual
  ) %>%
    select(date, value) %>%
    mutate(year = lubridate::year(date)) %>%
    select(year, state_min_wage = value)
}

# Fetch the federal series
fed_df <- get_min_wage(federal_series) %>%
  rename(federal_min_wage = state_min_wage)

test <- get_min_wage(state_fred$series_id[11])

# # Fetch all states and bind
# state_dfs <- state_fred %>%
#   mutate(data = map(series_id, get_min_wage)) %>%
#   unnest(cols = c(data)) %>%
#   select(state, fips, year, state_min_wage)

results <- NULL
for (i in 1:length(state_fred$fips)) {
  series_id <- state_fred$series_id[i]
  fips <- state_fred$fips[i]
  state <- state_fred$state_abbrev[i]
  
  # Assign the result of tryCatch to temp
  temp <- tryCatch(
    { 
      get_min_wage(state_fred$series_id[i])
    },
    error = function(e) {
      print("error")
      return(tibble(
        year = 1968:2025,
        state_min_wage = NA,
        fips = fips,
        state = state))
    }
  )
  
  # Check if temp is NULL or contains all NAs
  if (is.null(temp) || all(is.na(temp$state_min_wage))) {
    temp <- tibble(
      year = 1968:2025,
      state_min_wage = NA,
      fips = fips,
      state = state)
  }
  
  temp <- temp %>% 
    mutate(fips = fips, 
           state = state)
  
  results <- bind_rows(results, temp)
  print(i)
}

# Cross join states with federal and compute prevailing
final_df <- results %>%
  left_join(fed_df, by = "year") %>%
  mutate(
    prevailing_min_wage = pmax(state_min_wage, federal_min_wage, na.rm = TRUE)
  ) %>%
  select(fips, year, state_min_wage, federal_min_wage, prevailing_min_wage)

# Inspect
head(final_df)

write_parquet(final_df, paste0(folder_data, "fred_state_minwages.parquet"))

# remove files -------------------

str_detect(ls(), "folder_")

objects <- data.frame(name = ls(), keep = str_detect(ls(), "folder_")) %>% 
  filter(keep == F)

rm(list = objects[,1])
gc()
