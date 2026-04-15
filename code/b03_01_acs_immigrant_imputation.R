library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(foreign)
library(ipumsr)
library(haven)

rm(list = ls())

# Load ACS file
acs_df <- read_parquet(
  here("binaries", "acs_5year_for_immigrant_status_imputation.parquet"),
  col_select = c(
    "BPL",
    "CITIZEN",
    "HCOVPUB",
    "HINSIHS",
    "INCSS",
    "INCWELFR",
    "INCSUPP",
    "VETSTAT",
    "INDNAICS",
    "OCCSOC",
    "PERNUM",
    "SPLOC",
    "SAMPLE",
    "SERIAL",
    "YEAR",
    "MULTYEAR",
    "STATEFIP",
    "PUMA",
    "SEX",
    "AGE",
    "PERWT"
  )
) %>%
  clean_names()

# Add identifiers of legal status as in Borjas 2007
# Military status
military_occupations <- c(
  "551000",
  "551010",
  "552000",
  "552010",
  "553000",
  "553010",
  "559830"
)
military_sectors <- c(
  "9281101",
  "9281101P1",
  "9281102",
  "9281102P2",
  "9281103",
  "9281103P3",
  "9281104",
  "9281104P4",
  "9281105",
  "9281105P5",
  "9281106",
  "9281106P6",
  "9281107",
  "9281107P7"
)
# Public sector employee
public_sectors <- c(
  "9211MP",
  "92113",
  "92119",
  "92MP",
  "923",
  "92M1",
  "92M2",
  "928Z",
  "928P"
)
# Licensed occupations
licensed_occupations <- c(
  "171010",
  "171011",
  "171012", # Architects
  "395011", # Barbers
  "533020",
  "533051",
  "533052", # Bus drivers
  "291011", # Chiropractors
  "292021",
  "291292", # Dental hygienists
  "395012", # Hairdressers, hairstylists, cosmetologists
  "291020", # Dentists
  "292041",
  "292042",
  "292043", # EMTs and paramedics
  "172011",
  "1720XX",
  "172021",
  "172031",
  "172041",
  "172051",
  "172061",
  "172070",
  "172081",
  "172110",
  "172121",
  "172131",
  "172141",
  "172151",
  "172161",
  "172171",
  "1721XX",
  "1721YY", # Engineers
  "119061",
  "394031",
  "394031", # Funeral directors
  "413021", # Insurance agents
  "171020", # Surveyors, cartographers, photogrammetrists
  "131030",
  "Claims Adjusters, Appraisers, Examiners, and Investigators",
  "231011",
  "2310XX", # Lawyers
  "292061", # Licensed practical and licensed vocational nurses
  "119111", #	Medical and Health Services Managers
  "132070", # Credit Counselors and Loan Officers
  "291111",
  "291141", # Registered Nurses
  "291151", # Nurse Anesthetists
  "2911XX", # Nurse Practitioners and Nurse Midwives
  "291122", # Occupational Therapists
  "312010", # Occupational Therapy Assistants and Aides
  "291041", # Optometrists
  "291051", # Pharmacists
  "291123", # Physical Therapists
  "312020", # Physical Therapist Assistants and Aides
  "291060",
  "291210",
  "291240", # Physicians and Surgeons
  "291071", # Physician Assistants
  "291081", # Podiatrists
  "193030",
  "193033",
  "193034",
  "19303X", # Psychologists
  "419020", # Real Estate Brokers and Sales Agents
  "132021", # Appraisers and Assessors of Real Estate
  "132020", # Property appraisers and assessors
  "413031", # Securities, Commodities, and Financial Services Sales Agents
  "211020",
  "211021",
  "211022",
  "211023",
  "211029", # Social Workers
  "291127", # Speech-Language Pathologists
  "291131", # Veterinarians
  "251000",
  "252010",
  "252020",
  "252030",
  "252050" # Teachers
)

acs_df <- acs_df %>%
  mutate(
    is_citizen = (bpl < 150 | citizen %in% c(1, 2, 4)), # Born in US or naturalized
    public_health_insurance = (hcovpub == 2) | (hinsihs == 2), # Receives public health insurance
    social_security = between(incss, 1, 99998), # Social security
    public_assistance = between(incwelfr, 1, 99998), # AFDC
    ssi = between(incsupp, 1, 99998), # SSI
    born_in_cuba = (bpl == 250), # Born in Cuba
    military = (vetstat == 2) | # In military
      (indnaics %in% military_sectors) |
      (occsoc %in% military_occupations),
    public_sector = indnaics %in% public_sectors, # Public sector employee
    licensed_occupation = occsoc %in% licensed_occupations # Occupation is licensed
  )

# Identify spouses of citizens
# Create spouse lookup table
# Rename pernum to sploc for joining back
spouse_info <- acs_df %>%
  filter(sploc > 0) %>% # Has spouse = is spouse
  select(sample, serial, pernum, is_citizen) %>%
  rename(sploc = pernum, spouse_is_citizen_actual = is_citizen)

# Join it back to the main data
acs_df <- acs_df %>%
  left_join(spouse_info, by = c("sample", "serial", "sploc")) %>%
  mutate(spouse_is_citizen = replace_na(spouse_is_citizen_actual, FALSE)) %>%
  select(-spouse_is_citizen_actual)

# Add imputed immigrant type
acs_df <- acs_df %>%
  mutate(
    immigrant_type = case_when(
      # If they are foreign born and not a citizen...
      bpl > 120 & citizen != 1 ~ case_when(
        # ...but meet any of these criteria, they are documented
        is_citizen |
          public_health_insurance |
          social_security |
          public_assistance |
          ssi |
          born_in_cuba |
          military |
          public_sector |
          licensed_occupation |
          spouse_is_citizen ~ "documented_immigrant",
        # ...otherwise, undocumented
        .default = "undocumented_immigrant"
      ),
      # If born in US/territories or already a citizen
      .default = "non_immigrant"
    )
  ) %>%
  select(year, multyear, statefip, puma, sex, age, perwt, immigrant_type)

acs_df <- acs_df %>%
  select(year, multyear, statefip, puma, sex, age, perwt, immigrant_type)

# Pad state FIPS code and PUMA code for merging with GEOCORR
acs_df <- acs_df %>%
  mutate(
    statefip = str_pad(statefip, 2, side = "left", pad = "0"),
    puma = str_pad(puma, 5, side = "left", pad = "0")
  )

# Load GEOCORR crosswalk
# Have to skip the 2nd row
all_content <- readLines(here(
  "Data",
  "geocorr",
  "geocorr2014_puma2000_county2010.csv"
))
skip_second <- all_content[-2]
geocorr_2000_df <- read.csv(
  textConnection(skip_second),
  header = TRUE,
  stringsAsFactors = FALSE
)
geocorr_2000_df <- geocorr_2000_df %>%
  mutate(statefip = str_pad(state, 2, side = c("left"), pad = "0")) %>%
  mutate(puma = str_pad(puma2k, 5, side = c("left"), pad = "0")) %>%
  select(-c(state, puma2k))

all_content <- readLines(here(
  "Data",
  "geocorr",
  "geocorr2018_puma2010_county2010.csv"
))
skip_second <- all_content[-2]
geocorr_2012_df <- read.csv(
  textConnection(skip_second),
  header = TRUE,
  stringsAsFactors = FALSE
) %>%
  mutate(statefip = str_pad(state, 2, side = c("left"), pad = "0")) %>%
  mutate(puma = str_pad(puma12, 5, side = c("left"), pad = "0")) %>%
  select(-c(state, puma12))

# Split by survey year, then merge with GEOCORR
acs_puma_2000_df <- acs_df %>%
  filter(multyear < 2012) %>%
  left_join(geocorr_2000_df, by = c("statefip", "puma"))

acs_puma_2012_df <- acs_df %>%
  filter(multyear > 2011) %>%
  left_join(geocorr_2012_df, by = c("statefip", "puma"))

# Combine back into one df
acs_df <- bind_rows(acs_puma_2000_df, acs_puma_2012_df)

# Add group identifiers we care about
acs_df <- acs_df %>%
  mutate(prime_age = between(age, 25, 64)) %>%
  mutate(sex = if_else(sex == 1, "male", "female"))

# Aggregate to county level
acs_agg_df <- acs_df %>%
  mutate(county_perwt = perwt * afact) %>%
  group_by(year, county, sex, prime_age, immigrant_type) %>%
  summarise(pop = sum(county_perwt)) %>%
  ungroup()

# Harmonize variables with other dataset
acs_agg_df <- acs_agg_df %>%
  mutate(
    state_county_fips_code = str_pad(county, 5, side = c("left"), pad = "0")
  )

acs_agg_df <- acs_agg_df %>%
  mutate(state_fips_code = substr(state_county_fips_code, 1, 2)) %>%
  mutate(county_fips_code = substr(state_county_fips_code, 3, 5)) %>%
  select(-state_county_fips_code, -county)

# Export
acs_agg_df %>%
  write_parquet(here("binaries", "acs_immigrant_imputed.parquet"))
