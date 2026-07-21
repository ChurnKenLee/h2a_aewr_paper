# Purpose: Refresh the external FRED state minimum-wage input.
# Inputs: raw/geographic_crosswalks/phil/fips_codes.csv and FRED_API_KEY_PHIL.
# Output: intermediate/fred_state_minwages.parquet.
# This utility is intentionally outside the manually ordered C workflow.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
source(path_code("c00_shared", "fips.R"))

library(tidyverse)
library(purrr)
library(fredr)
library(arrow)

# You need your FRED API key
fred_api_key <- Sys.getenv("FRED_API_KEY_PHIL")
if (!nzchar(fred_api_key)) {
  stop("FRED_API_KEY_PHIL must be set in .env or the environment.")
}
fredr_set_key(fred_api_key)
# The FRED series ID for federal minimum wage (annual)
federal_series <- "STTMINWGFG"

state_fips_lookup <- read_csv(path_raw(
  "geographic_crosswalks",
  "phil",
  "fips_codes.csv"
))

# A lookup of state FRED series id and FIPS
# You'll need to fill this out for all 50 states. As an example:

state_fred <- state_fips_lookup %>%
  mutate(
    fips = state_fips(fips),
    series_id = str_trim(paste0("STTMINWG", state_abbrev))
  ) %>%
  filter(as.integer(fips) <= 56)

# state_fred <- tibble(
#   state = c("California", "Texas", "New York"),  # add other states
#   fips = c("06", "48", "36"),                    # add other FIPS
#   series_id = c("STTMINWGCA", "STTMINWGTX", "STTMINWGNY") # add other series
# )

# A function to fetch a series and return as a tidy tibble
get_min_wage <- function(series_id) {
  fredr(
    series_id = series_id,
    observation_start = as.Date("1968-01-01"), # adjust start year as needed
    frequency = "a" # annual
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
        state = state
      ))
    }
  )

  # Check if temp is NULL or contains all NAs
  if (is.null(temp) || all(is.na(temp$state_min_wage))) {
    temp <- tibble(
      year = 1968:2025,
      state_min_wage = NA,
      fips = fips,
      state = state
    )
  }

  temp <- temp %>%
    mutate(fips = fips, state = state)

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

write_parquet(final_df, path_int("fred_state_minwages.parquet"))
