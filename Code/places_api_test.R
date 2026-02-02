library(googleway)
library(here)
library(readr)

api_key <- read_file(here("tools", "google_places_api_key.txt"))
set_key(key = api_key)

test_input <- "University of California, San Diego"

test_response <- google_find_place(
  input = test_input,
  inputtype = c("textquery"),
  fields = c("place_id")
  )
