library(here)
library(arrow)
library(tidyverse)
library(dplyr)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(readxl)
library(foreign)
library(ipumsr)

rm(list = ls())

# Read in ACS data and save as binary
# ddi <- read_ipums_ddi(here("Data", "acs", "usa_00028.xml"))
# data <- read_ipums_micro(ddi)
# write_parquet(data, here("binaries", "acs_5year_2007_2012_2017_2022.parquet"))

# Load ACS file
rm(list = ls())

acs_df <- read_parquet(here("binaries", "acs_5year_2007_2012_2017_2022.parquet")) %>% 
  clean_names()

# Add identifiers of legal status as in Borjas 2007
# Citizenship status
acs_df <- acs_df %>% 
  mutate(is_citizen = if_else(bpl < 150, TRUE, FALSE)) %>% # Born in the US + US territories
  mutate(is_citizen = if_else(citizen %in% c(1, 2, 4), TRUE, is_citizen)) # Naturalized

# Public health insurance recipient
acs_df <- acs_df %>% 
  mutate(public_health_insurance = case_when(
    hcovpub == 2 ~ TRUE, # any public health insurance coverage
    hinsihs == 2 ~ TRUE, # Indian Health Services insurance
    .default = FALSE
  ))

# SSI, AFDC, GA
acs_df <- acs_df %>% 
  mutate(social_security = FALSE) %>% 
  mutate(social_security = if_else(between(incss, 1, 99998), TRUE, social_security)) %>% # Social Security
  mutate(public_assistance = FALSE) %>% 
  mutate(public_assistance = if_else(between(incwelfr, 1, 99998), TRUE, public_assistance)) %>% # SSI, AFDC, GA
  mutate(ssi = FALSE) %>% 
  mutate(ssi = if_else(between(incsupp, 1, 99998), TRUE, ssi)) # SSI

# Born in Cuba
acs_df <- acs_df %>% 
  mutate(born_in_cuba = if_else(bpl == 250, TRUE, FALSE))

# Military status
military_occupations <- c("551000", "551010", "552000", "552010", "553000", "553010", "559830")
military_sectors <- c("9281101", "9281101P1", "9281102", "9281102P2", "9281103", "9281103P3", "9281104", "9281104P4", "9281105", "9281105P5", "9281106", "9281106P6", "9281107", "9281107P7")

acs_df <-  acs_df %>% 
  mutate(military = FALSE) %>% 
  mutate(military = if_else(vetstat == 2, TRUE, military)) %>% # Veteran
  mutate(military = indnaics %in% military_sectors) %>%  # Military sector
  mutate(military = occsoc %in% military_occupations) # Military occupation

# Public sector employee
public_sectors <- c("9211MP", "92113", "92119", "92MP", "923", "92M1", "92M2", "928Z", "928P")

acs_df <- acs_df %>% 
  mutate(public_sector = indnaics %in% public_sectors)

# Licensed occupations
licensed_occupations <- c(
  "171010", "171011", "171012", # Architects
  "395011", # Barbers
  "533020", "533051", "533052", # Bus drivers
  "291011", # Chiropractors
  "292021", "291292", # Dental hygienists
  "395012", # Hairdressers, hairstylists, cosmetologists
  "291020", # Dentists
  "292041", "292042", "292043", # EMTs and paramedics
  "172011", "1720XX", "172021", "172031", "172041", "172051", "172061", "172070", "172081", "172110", "172121", "172131" , "172141", "172151", "172161", "172171", "1721XX", "1721YY", # Engineers
  "119061", "394031", "394031", # Funeral directors
  "413021", # Insurance agents
  "171020", # Surveyors, cartographers, photogrammetrists
  "131030", "Claims Adjusters, Appraisers, Examiners, and Investigators",
  "231011", "2310XX", # Lawyers
  "292061", # Licensed practical and licensed vocational nurses
  "119111", #	Medical and Health Services Managers
  "132070", # Credit Counselors and Loan Officers
  "291111", "291141", # Registered Nurses
  "291151", # Nurse Anesthetists
  "2911XX", # Nurse Practitioners and Nurse Midwives
  "291122", # Occupational Therapists
  "312010", # Occupational Therapy Assistants and Aides
  "291041", # Optometrists
  "291051", # Pharmacists
  "291123", # Physical Therapists
  "312020", # Physical Therapist Assistants and Aides
  "291060", "291210", "291240", # Physicians and Surgeons
  "291071", # Physician Assistants
  "291081", # Podiatrists
  "193030", "193033", "193034", "19303X", # Psychologists
  "419020", # Real Estate Brokers and Sales Agents
  "132021", # Appraisers and Assessors of Real Estate
  "132020", # Property appraisers and assessors
  "413031", # Securities, Commodities, and Financial Services Sales Agents
  "211020", "211021", "211022", "211023", "211029", # Social Workers
  "291127", # Speech-Language Pathologists
  "291131", # Veterinarians
  "251000", "252010", "252020", "252030", "252050" # Teachers
)

acs_df <- acs_df %>% 
  mutate(licensed_occupation = occsoc %in% licensed_occupations)

# Identify spouses of citizens
# Iterate over pernum, getting citizen status of the spouse
acs_df <- acs_df %>% 
  mutate(spouse_is_citizen = FALSE)

acs_df_spousal <- acs_df %>% 
  filter(sploc != 0)

acs_df <- acs_df %>% 
  filter(sploc == 0)

for (i in 1:20) {
  print(paste("Currently on pernum=", i))
  acs_df_spousal <- acs_df_spousal %>% 
    mutate(pernum_is_citizen = (pernum == i) & is_citizen)
  
  acs_df_spousal <- acs_df_spousal %>% 
    group_by(sample, serial) %>% 
    mutate(hh_pernum_is_citizen = any(pernum_is_citizen == TRUE)) %>% 
    ungroup() %>% 
    mutate(spouse_is_citizen = if_else(sploc == i & hh_pernum_is_citizen, TRUE, spouse_is_citizen))
}

acs_df_spousal <- acs_df_spousal %>% 
  select(-pernum_is_citizen, -hh_pernum_is_citizen)

acs_df <- rbind(acs_df, acs_df_spousal)

# Add imputed immigrant type
acs_df <- acs_df %>% 
  mutate(immigrant_type = "non_immigrant") %>% 
  mutate(immigrant_type = if_else(bpl > 120 & citizen != 1, "undocumented_immigrant", immigrant_type)) %>% 
  mutate(immigrant_type = if_else(immigrant_type == "undocumented_immigrant" & (is_citizen | public_health_insurance | social_security | public_assistance | ssi | born_in_cuba | military | public_sector | licensed_occupation), "documented_immigrant", immigrant_type))

acs_df <- acs_df %>% 
  select(year, multyear, statefip, puma, sex, age, perwt, immigrant_type)

# Pad state FIPS code and PUMA code for merging with GEOCORR
acs_df <- acs_df %>% 
  mutate(statefip = str_pad(statefip, 2, side = c("left"), pad = "0")) %>% 
  mutate(puma = str_pad(puma, 5, side = c("left"), pad = "0"))

# Load GEOCORR crosswalk
# Have to skip the 2nd row
all_content <- readLines(here("Data", "geocorr", "geocorr2000_puma_county.csv"))
skip_second <- all_content[-2]
geocorr_2000_df <- read.csv(textConnection(skip_second), header = TRUE, stringsAsFactors = FALSE)
geocorr_2000_df <- geocorr_2000_df %>% 
  mutate(statefip = str_pad(state, 2, side = c("left"), pad = "0")) %>% 
  mutate(puma = str_pad(puma5, 5, side = c("left"), pad = "0")) %>% 
  select(-c(state, puma5))

all_content <- readLines(here("Data", "geocorr", "geocorr2018_puma_county.csv"))
skip_second <- all_content[-2]
geocorr_2012_df <- read.csv(textConnection(skip_second), header = TRUE, stringsAsFactors = FALSE) %>% 
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
  mutate(county_perwt = perwt*afact) %>% 
  group_by(year, county, sex, prime_age, immigrant_type) %>%
  summarise(pop = sum(county_perwt)) %>% 
  ungroup()

# Harmonize variables with other dataset
acs_agg_df <- acs_agg_df %>% 
  mutate(state_county_fips_code = str_pad(county, 5, side = c("left"), pad = "0"))

acs_agg_df <- acs_agg_df %>%
  mutate(state_fips_code = substr(state_county_fips_code, 1, 2)) %>% 
  mutate(county_fips_code = substr(state_county_fips_code, 3, 5)) %>% 
  select(-state_county_fips_code, -county)

# Export
acs_agg_df %>% 
  write_parquet(here("files_for_phil", "acs_immigrant_imputed.parquet")) %>% 
  write_parquet(here("binaries", "acs_immigrant_imputed.parquet"))
