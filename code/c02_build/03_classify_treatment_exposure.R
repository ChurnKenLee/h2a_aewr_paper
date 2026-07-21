# Purpose: Classify baseline H-2A exposure, treatment groups, borders, and periods.
# Input: data/intermediate/county_df_variable_cleaned_year.parquet.
# Output: data/intermediate/county_df_classified_year.parquet.
# Run after: 02_derive_analysis_variables.R.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
source(path_code("c00_shared", "fips.R"))
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)

county_df <- read_parquet(path_int("county_df_variable_cleaned_year.parquet"))

# new variables

# Cut using pred X actual use rates in 2008
true_share_cutoff <- 0.01
pred_share_cutoff <- 0.01

county_type_classification <- county_df %>%
  filter(year == 2008) %>%
  mutate(
    county_treatment_group_classification = case_when(
      (h2a_predicted_share_2011 > pred_share_cutoff) &
        (h2a_cert_share_farm_workers_2011_start_year >
          true_share_cutoff) ~ "always takers",
      (h2a_predicted_share_2011 > pred_share_cutoff) &
        (h2a_cert_share_farm_workers_2011_start_year <
          true_share_cutoff) ~ "adopters",
      (h2a_predicted_share_2011 < pred_share_cutoff) &
        (h2a_cert_share_farm_workers_2011_start_year >
          true_share_cutoff) ~ "defiers",
      (h2a_predicted_share_2011 < pred_share_cutoff) &
        (h2a_cert_share_farm_workers_2011_start_year <
          true_share_cutoff) ~ "never takers",
    ),
    county_simple_treatment_groups = case_when(
      (county_treatment_group_classification ==
        "always takers") ~ "always takers",
      (county_treatment_group_classification !=
        "always takers") ~ "exposed adoptors"
    )
  ) %>%
  select(
    countyfips,
    county_treatment_group_classification,
    county_simple_treatment_groups
  )


# cuts by 2008 h2a usage
h2a_use_df <- county_df %>%
  ungroup() %>%
  filter(year == 2008)


hist(h2a_use_df$nbr_workers_requested_start_year)
summary(h2a_use_df$nbr_workers_requested_start_year)
summary(h2a_use_df$h2a_req_share_farm_workers_start_year)

# cut by count

count_cuts <- quantile(
  h2a_use_df$nbr_workers_requested_start_year,
  probs = c(.5, .66, .75)
)

# cut by share

share_cuts <- quantile(
  h2a_use_df$h2a_cert_share_farm_workers_start_year,
  probs = c(.5, .66, .75),
  na.rm = T
)


h2a_use_df <- h2a_use_df %>%
  mutate(
    high_h2a_count_50 = ifelse(
      nbr_workers_certified_start_year > count_cuts[1],
      1,
      0
    ),
    high_h2a_count_66 = ifelse(
      nbr_workers_certified_start_year > count_cuts[2],
      1,
      0
    ),
    high_h2a_count_75 = ifelse(
      nbr_workers_certified_start_year > count_cuts[3],
      1,
      0
    ),
    high_h2a_share_50 = ifelse(
      h2a_cert_share_farm_workers_start_year > share_cuts[1] &
        !is.na(h2a_cert_share_farm_workers_start_year),
      1,
      0
    ),
    high_h2a_share_66 = ifelse(
      h2a_cert_share_farm_workers_start_year > share_cuts[2] &
        !is.na(h2a_cert_share_farm_workers_start_year),
      1,
      0
    ),
    high_h2a_share_75 = ifelse(
      h2a_cert_share_farm_workers_start_year > share_cuts[3] &
        !is.na(h2a_cert_share_farm_workers_start_year),
      1,
      0
    )
  )

h2a_use_df %>%
  group_by(high_h2a_count_50) %>%
  tally()

h2a_use_df <- h2a_use_df %>%
  select(
    countyfips,
    high_h2a_count_50,
    high_h2a_count_66,
    high_h2a_count_75,
    high_h2a_share_50,
    high_h2a_share_66,
    high_h2a_share_75
  )


county_df <- merge(
  x = county_df,
  y = h2a_use_df,
  by = "countyfips",
  all.x = T,
  all.y = F
)

county_type_classification <- county_type_classification %>%
  ungroup() %>%
  select(-any_of("county_fe"))

county_df <- county_df %>%
  left_join(county_type_classification, by = "countyfips")

# year dummys

summary(county_df$year)

county_df <- county_df %>%
  mutate(
    yeardummy_2008 = ifelse(year == 2008, 1, 0),
    yeardummy_2009 = ifelse(year == 2009, 1, 0),
    yeardummy_2010 = ifelse(year == 2010, 1, 0),
    yeardummy_2011 = ifelse(year == 2011, 1, 0),
    yeardummy_2012 = ifelse(year == 2012, 1, 0),
    yeardummy_2013 = ifelse(year == 2013, 1, 0),
    yeardummy_2014 = ifelse(year == 2014, 1, 0),
    yeardummy_2015 = ifelse(year == 2015, 1, 0),
    yeardummy_2016 = ifelse(year == 2016, 1, 0),
    yeardummy_2017 = ifelse(year == 2017, 1, 0),
    yeardummy_2018 = ifelse(year == 2018, 1, 0),
    yeardummy_2019 = ifelse(year == 2019, 1, 0),
    yeardummy_2020 = ifelse(year == 2020, 1, 0),
    yeardummy_2021 = ifelse(year == 2021, 1, 0),
    yeardummy_2022 = ifelse(year == 2022, 1, 0)
  )


# ID border CZs

cz_borders <- county_df %>%
  group_by(cz_out10) %>%
  summarise(
    AEWRregmin = min(aewr_region_num, na.rm = T),
    AEWRregmax = max(aewr_region_num, na.rm = T)
  )

cz_borders <- cz_borders %>%
  mutate(border_cz = ifelse(AEWRregmin != AEWRregmax, 1, 0))

cz_borders %>% group_by(border_cz) %>% tally()

county_df <- merge(
  x = county_df,
  y = cz_borders,
  by = "cz_out10",
  all.x = T,
  all.y = F
)


# pre post dummy

county_df <- county_df %>%
  mutate(postdummy = ifelse(year > 2011, 1, 0))

# low use dummy

county_df <- county_df %>%
  mutate(
    high_h2a_share_75_inverse = ifelse(high_h2a_share_75 == 0, 1, 0),
    high_h2a_share_66_inverse = ifelse(high_h2a_share_66 == 0, 1, 0),
    high_h2a_share_50_inverse = ifelse(high_h2a_share_50 == 0, 1, 0),
    high_h2a_count_75_inverse = ifelse(high_h2a_count_75 == 0, 1, 0),
    high_h2a_count_66_inverse = ifelse(high_h2a_count_66 == 0, 1, 0),
    high_h2a_count_50_inverse = ifelse(high_h2a_count_50 == 0, 1, 0)
  )

stopifnot(is.character(county_df$countyfips))
stopifnot(all(
  is.na(county_df$countyfips) | str_detect(county_df$countyfips, "^\\d{5}$")
))
stopifnot(is.character(county_df$statefips))
stopifnot(all(
  is.na(county_df$statefips) | str_detect(county_df$statefips, "^\\d{2}$")
))


write_parquet(county_df, path_int("county_df_classified_year.parquet"))
