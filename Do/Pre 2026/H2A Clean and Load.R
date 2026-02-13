## H2A: Load and clean datasets
## Phil Hoxie
## 1/11/24

## requires running master file first ##

## Two datasets, one by census period and one by year (calendar year)

## Fips Codes ------------------------------------------

fips_codes <- read.csv(file = paste0(folder_data, "fips_codes.csv"), stringsAsFactors = F)

## AEWR Regions ----------------------------------------

aewr_regions <- read.csv(file = paste0(folder_data, "aewr_regions.csv"), stringsAsFactors = F)

## Commuting Zones --------------------------------------

# source: https://sites.psu.edu/psucz/data/

cz_file <- read.csv(file = paste0(folder_data, "counties10-zqvz0r.csv"), stringsAsFactors = F)

cz_file <- cz_file %>% 
  rename(countyfips = FIPS,
         cz_out10 = OUT10)

saveRDS(cz_file, file = paste0(folder_data, "cz_file_2010.rds"))


cz_file_small <- cz_file %>% 
  select(countyfips, cz_out10) 
  
saveRDS(cz_file_small, file = paste0(folder_data, "cz_file_2010_small.rds"))


## PPI --------------------------------------------------
# Source: https://fred.stlouisfed.org/series/WPU01

ppi_data <- read.csv(file = paste0(folder_data, "WPU01.csv"), stringsAsFactors = F)

# average by year

ppi_data <- ppi_data %>% 
  mutate(year = as.numeric(substr(DATE, 1, 4)))

ppi_data <- ppi_data %>% 
  group_by(year) %>% 
  summarise(wpu01 = mean(WPU01))

# change base to 2012 

ppi_2012 <- ppi_data$wpu01[ppi_data$year == 2012]

ppi_data <- ppi_data %>% 
  mutate(ppi_2012 = wpu01 / ppi_2012)

ggplot(data = ppi_data, aes(y = ppi_2012, x = year))+
  geom_line()

ppi_data <- ppi_data %>% 
  select(year, ppi_2012)

saveRDS(ppi_data, file = paste0(folder_data, "ppi_2012.rds"))

## County Boundary ------------------------------------

# cleaned in the original stata file 
state_border_pairs <- read.csv(file = paste0(folder_data, "state_border_pairs.csv"), stringsAsFactors = F)
border_counties_allmatches <- read.csv(file = paste0(folder_data, "border_counties_allmatches.csv"), stringsAsFactors = F)

full_county_set <- read.csv(file = paste0(folder_data, "county_adjacency2010.csv"), stringsAsFactors = F)

## Census of Agriculture ------------------------------

census_of_agriculture <- read.csv(file = paste0(folder_data, "census_of_agriculture.csv"), stringsAsFactors = F)

# general cleaning #

census_of_agriculture <- census_of_agriculture %>% 
  select(-X) %>% 
  rename(val_str = value) %>% 
  mutate(value = as.numeric(str_replace_all(val_str, ",", "")))

head(census_of_agriculture)

ag_census_data_items <- census_of_agriculture %>% 
  group_by(commodity_desc) %>% 
  tally() 

# want: FARM OPERATIONS, AG LAND

census_of_agriculture_trim <- census_of_agriculture %>% 
  filter(commodity_desc == "FARM OPERATIONS" | commodity_desc == "AG LAND")

census_of_agriculture_trim_items <- census_of_agriculture_trim %>% 
  group_by(short_desc) %>% 
  tally() 

census_of_agriculture_trim <- census_of_agriculture_trim %>% 
  filter(short_desc == "AG LAND, CROPLAND - ACRES")

# fix fips 

census_of_agriculture_trim <- census_of_agriculture_trim %>% 
  mutate(countyfips = as.numeric(str_c(as.character(state_fips_code), 
                                       str_pad(as.character(county_code), width = 3, side = "left", pad = "0"))))

census_of_agriculture_trim <- census_of_agriculture_trim %>% 
  arrange(countyfips, year)

census_of_agriculture_trim <-census_of_agriculture_trim %>% 
  rename(census_period = year) %>% 
  mutate(label = "cropland_acr") %>% 
  select(census_period, countyfips, value, label) 

census_of_agriculture_trim <-census_of_agriculture_trim %>% 
  filter(!is.na(countyfips))

census_of_agriculture_cropland <-census_of_agriculture_trim %>% 
  pivot_wider(names_from = "label", values_from = "value")

head(census_of_agriculture_trim)

saveRDS(census_of_agriculture_cropland, file = paste0(folder_data, "census_ag_cropland.rds"))

rm(census_of_agriculture_trim)

census_of_agriculture_cropland_base <- census_of_agriculture_cropland %>% 
  filter(census_period == 2007) %>% 
  rename(cropland_acr_2007 = cropland_acr) %>% 
  select(-census_period)

saveRDS(census_of_agriculture_cropland_base, file = paste0(folder_data, "census_ag_cropland_2007.rds"))

rm(census_of_agriculture_cropland_base)

## H2A Data -------------------------------------------

# h2a_data <- read.csv(file = paste0(folder_data, "h2a.csv"), stringsAsFactors = F)
# 
# head(h2a_data)
# 
# h2a_data <- h2a_data %>% 
#   select(-X)
# 
# # census period 
# 
# h2a_data <- h2a_data %>% 
#   mutate(census_period = ifelse(year >= 2008 & year < 2012, 2012, 
#                                 ifelse(year >= 2012 & year < 2017, 2017, 
#                                        ifelse( year >= 2017 & year < 2022, 2022, NA))))
# 
# 
# # fix fips code 
# 
# h2a_data <- h2a_data %>% 
#   mutate(cnty_fips_string = str_pad(as.character(county_fips_code), width = 3, side = "left", pad = "0"),
#          st_fips_string = str_pad(as.character(state_fips_code), width = 2, side = "left", pad = "0"))
# 
# h2a_data <- h2a_data %>% 
#   mutate(countyfips = as.numeric(paste0(st_fips_string, cnty_fips_string)))
# 
# 
# # clean by dropping old codes # 
# 
# h2a_data <- h2a_data %>% 
#   filter(!is.na(census_period) & !is.na(county_fips_code)) %>% 
#   select(-cnty_fips_string, -st_fips_string, -state_fips_code, -county_fips_code)
# 
# # collapse by period, county (county and state fips)
# 
# h2a_data <- h2a_data %>% 
#   group_by(census_period, countyfips) %>% 
#   summarise(h2a_man_hours_certified = sum(h2a_man_hours_certified, na.rm = T), 
#             h2a_man_hours_requested = sum(h2a_man_hours_requested, na.rm = T), 
#             h2a_nbr_workers_certified = sum(h2a_nbr_workers_certified, na.rm = T), 
#             h2a_nbr_workers_requested = sum(h2a_nbr_workers_requested, na.rm = T), 
#             n_applications = sum(n_applications, na.rm = T))
# 
# 
# saveRDS(h2a_data, file = paste0(folder_data, "h2a_data.rds"))
# 
# # yearly, for TS 
# 
# h2a_data_ts <- read.csv(file = paste0(folder_data, "h2a.csv"), stringsAsFactors = F)
# h2a_data_ts <- h2a_data_ts %>% 
#   select(-X)
# 
# h2a_data_ts <- h2a_data_ts %>% 
#   group_by(case_year) %>% 
#   summarise(h2a_man_hours_certified = sum(h2a_man_hours_certified, na.rm = T), 
#             h2a_man_hours_requested = sum(h2a_man_hours_requested, na.rm = T), 
#             h2a_nbr_workers_certified = sum(h2a_nbr_workers_certified, na.rm = T), 
#             h2a_nbr_workers_requested = sum(h2a_nbr_workers_requested, na.rm = T), 
#             n_applications = sum(n_applications, na.rm = T))
# 
# saveRDS(h2a_data_ts, file = paste0(folder_data, "h2a_data_ts.rds"))

h2a_data <- read.csv(file = paste0(folder_data, "h2a_aggregated.csv"), stringsAsFactors = F)

head(h2a_data)

h2a_data <- h2a_data %>% 
  select(-X)

# census period 

h2a_data <- h2a_data %>% 
  mutate(census_period = ifelse(year >= 2008 & year < 2012, 2012, 
                                ifelse(year >= 2012 & year < 2017, 2017, 
                                       ifelse( year >= 2017 & year < 2022, 2022, NA))))


# fix fips code 

h2a_data <- h2a_data %>% 
  mutate(cnty_fips_string = str_pad(as.character(county_fips_code), width = 3, side = "left", pad = "0"),
         st_fips_string = str_pad(as.character(state_fips_code), width = 2, side = "left", pad = "0"))

h2a_data <- h2a_data %>% 
  mutate(countyfips = as.numeric(paste0(st_fips_string, cnty_fips_string)))


# clean by dropping old codes # 

h2a_data <- h2a_data %>% 
  filter(!is.na(census_period) & !is.na(county_fips_code)) %>% 
  select(-cnty_fips_string, -st_fips_string, -state_fips_code, -county_fips_code)

# collapse by period, county (county and state fips)

h2a_data <- h2a_data %>% 
  group_by(census_period, countyfips) %>% 
  summarise(nbr_workers_requested_all_years = sum(nbr_workers_requested_all_years, na.rm = T),
            nbr_workers_certified_all_years = sum(nbr_workers_certified_all_years, na.rm = T),
            man_hours_requested_all_years = sum(man_hours_requested_all_years, na.rm = T),
            man_hours_certified_all_years = sum(man_hours_certified_all_years, na.rm = T),
            nbr_applications_all_years = sum(nbr_applications_all_years, na.rm = T),
            nbr_workers_requested_start_year = sum(nbr_workers_requested_start_year, na.rm = T),
            nbr_workers_certified_start_year = sum(nbr_workers_certified_start_year, na.rm = T),
            man_hours_requested_start_year = sum(man_hours_requested_start_year, na.rm = T),
            man_hours_certified_start_year = sum(man_hours_certified_start_year, na.rm = T),
            nbr_applications_start_year = sum(nbr_applications_start_year, na.rm = T),
            nbr_workers_requested_fiscal_year = sum(nbr_workers_requested_fiscal_year, na.rm = T),
            nbr_workers_certified_fiscal_year = sum(nbr_workers_certified_fiscal_year, na.rm = T),
            man_hours_requested_fiscal_year = sum(man_hours_requested_fiscal_year, na.rm = T),
            man_hours_certified_fiscal_year = sum(man_hours_certified_fiscal_year, na.rm = T),
            nbr_applications_fiscal_year = sum(nbr_applications_fiscal_year, na.rm = T))


saveRDS(h2a_data, file = paste0(folder_data, "h2a_data.rds"))

# yearly, for TS 

h2a_data_ts <- read.csv(file = paste0(folder_data, "h2a_aggregated.csv"), stringsAsFactors = F)
h2a_data_ts <- h2a_data_ts %>% 
  select(-X)

h2a_data_ts <- h2a_data_ts %>% 
  group_by(year) %>% 
  summarise(h2a_man_hours_certified = sum(man_hours_certified_start_year, na.rm = T), 
            h2a_man_hours_requested = sum(man_hours_requested_start_year, na.rm = T), 
            h2a_nbr_workers_certified = sum(nbr_workers_certified_start_year, na.rm = T), 
            h2a_nbr_workers_requested = sum(nbr_workers_requested_start_year, na.rm = T), 
            n_applications = sum(nbr_applications_start_year, na.rm = T))


h2a_data_ts <- h2a_data_ts %>% 
  rename(case_year = year)

saveRDS(h2a_data_ts, file = paste0(folder_data, "h2a_data_ts.rds"))

## ACS & QCEW -------------------------------------------

acs_qcew_data <- read.csv(file = paste0(folder_data, "acs_qcew.csv"), stringsAsFactors = F)

acs_qcew_data <- acs_qcew_data %>% 
  select(-X)

names(acs_qcew_data)

# make some variables 
acs_qcew_data <- acs_qcew_data %>% 
  mutate(census_period = year, 
         acs_agemp_raw = acs_crop_emp + acs_animal_emp, 
         qcew_agemp_raw = qcew_crop_emp + qcew_animal_emp,
         acs_crop_emp_clean = ifelse(!is.na(acs_animal_emp) & !is.na(acs_crop_emp), acs_crop_emp,
                                     ifelse(!is.na(acs_animal_emp) & is.na(acs_crop_emp), 0, NA)),
         acs_animal_emp_clean = ifelse(!is.na(acs_animal_emp) & !is.na(acs_crop_emp), acs_animal_emp,
                                       ifelse(is.na(acs_animal_emp) & !is.na(acs_crop_emp), 0, NA)),
         qcew_crop_emp_clean = ifelse(!is.na(qcew_animal_emp) & !is.na(qcew_crop_emp), qcew_crop_emp,
                                      ifelse(!is.na(qcew_animal_emp) & is.na(qcew_crop_emp), 0, NA)),
         qcew_animal_emp_clean = ifelse(!is.na(qcew_animal_emp) & !is.na(qcew_crop_emp), qcew_animal_emp,
                                        ifelse(is.na(qcew_animal_emp) & !is.na(qcew_crop_emp), 0, NA)))
# make more variables, with the new variables 
acs_qcew_data <- acs_qcew_data %>% 
  mutate(empratio_crop_animal_acs = acs_crop_emp_clean / acs_animal_emp_clean, 
         empratio_crop_animal_qcew = qcew_crop_emp_clean / qcew_animal_emp_clean,
         ln_empratio_crop_animal_acs = log(acs_crop_emp_clean / acs_animal_emp_clean),
         ln_empratio_crop_animal_qcew = log(qcew_crop_emp_clean / qcew_animal_emp_clean),
         acs_agemp = acs_crop_emp_clean + acs_animal_emp_clean, 
         qcew_agemp = qcew_crop_emp_clean + qcew_animal_emp_clean)

acs_qcew_data %>% 
  group_by(year) %>% 
  tally()

acs_qcew_data <- acs_qcew_data %>% 
  mutate(countyfips = as.numeric(str_c(as.character(state_fips_code), 
                                     str_pad(as.character(county_fips_code), width = 3, side = "left", pad = "0"))))

acs_qcew_data <- acs_qcew_data %>% 
  select(-year, -state_fips_code, -county_fips_code)

saveRDS(acs_qcew_data, file = paste0(folder_data, "acs_qcew_data.rds"))

## NAWSPAD ---------------------------------------------
# state level data
nawspad_data <- read.csv(file = paste0(folder_data, "nawspad.csv"), stringsAsFactors = F)

nawspad_data %>% 
  group_by(crop_type) %>% 
  tally()


## OEWS ------------------------------------------------

oews_data <- read.csv(file = paste0(folder_data, "oews.csv"), stringsAsFactors = F)
# wages

## AEWR -------------------------------------------------

aewr_data <- read.csv(file = paste0(folder_data, "aewr.csv"), stringsAsFactors = F)

aewr_data <- aewr_data %>% 
  select(-X)

aewr_data <- merge(x = aewr_data, y = ppi_data, by = "year", all.x = T, all.y = F)

aewr_data <- aewr_data %>% 
  mutate(aewr_ppi = aewr / ppi_2012,
         census_period = ifelse(year > 2002 & year <= 2007, 2007, 
                                ifelse(year > 2007 & year <= 2012, 2012, 
                                       ifelse(year > 2012 & year <= 2017, 2017, 
                                              ifelse(year > 2017 & year <= 2022, 2022, NA)))))

aewr_data <- aewr_data %>% 
  filter(!is.na(census_period)) %>% 
  group_by(state_fips_code, census_period) %>% 
  summarise(aewr_ppi = mean(aewr_ppi, na.rm = T),
            aewr = mean(aewr, na.rm = T))

aewr_data <- merge(x = aewr_data, y = fips_codes, by.x = "state_fips_code", by.y = "fips", all.x = T, all.y = F)

aewr_data <- aewr_data %>% 
  select(aewr, aewr_ppi, census_period, state_abbrev)

saveRDS(aewr_data, file = paste0(folder_data, "aewr_data.rds"))

# Now I am deviating from the original stata file 

## BEA Data ---------------------------------------------

# job count data
bea_caemp25n_data <- read.csv(file = paste0(folder_data, "CAEMP25N/CAEMP25N__ALL_AREAS_2001_2022_trim.csv"), stringsAsFactors = F)
# save lines: 10 50 70 80 90

bea_caemp25n_data <- bea_caemp25n_data %>% 
  filter(LineCode == 10 | LineCode == 50 | LineCode == 70 | LineCode == 80 | LineCode == 90)

bea_caemp25n_data <- bea_caemp25n_data %>% 
  mutate(countyfips = as.numeric(str_trim(gsub("\"", "", GeoFIPS))), # remove qoutes
         category = ifelse(LineCode == 10, "emp_tot", 
                           ifelse(LineCode == 50, "emp_farm_propr", # farm proprietors
                                  ifelse(LineCode == 70, "emp_farm", 
                                         ifelse(LineCode == 80, "emp_nonfarm", 
                                                ifelse(LineCode == 90, "emp_privatenonfarm", 
                                                       NA))))))

bea_caemp25n_data <- bea_caemp25n_data %>% 
  select(9:32)

bea_caemp25n_data <- bea_caemp25n_data %>% # wow, that was easy
  pivot_longer(cols = starts_with("y"), names_to = "year", names_prefix = "y", values_to = "temp", values_drop_na = F)

bea_caemp25n_data <- bea_caemp25n_data %>% 
  mutate(emp = as.numeric(temp)) %>% 
  select(-temp)

# put into year - county rows, so, pivot again 

bea_caemp25n_data <- bea_caemp25n_data %>% # wow, that was easy
  pivot_wider(names_from = "category", values_from = "emp")

# collapse 

bea_caemp25n_data <- bea_caemp25n_data %>% 
  mutate(census_period = ifelse(year > 2002 & year <= 2007, 2007, 
                                ifelse(year > 2007 & year <= 2012, 2012, 
                                       ifelse(year > 2012 & year <= 2017, 2017, 
                                              ifelse(year > 2017 & year <= 2022, 2022, NA)))))

bea_caemp25n_data %>% 
  group_by(census_period) %>% 
  tally()

bea_caemp25n_data <- bea_caemp25n_data %>% 
  filter(!is.na(census_period)) %>% 
  group_by(census_period, countyfips) %>% 
  summarise(emp_tot = mean(emp_tot, na.rm = T),
            emp_farm_propr = mean(emp_farm_propr, na.rm = T),
            emp_farm = mean(emp_farm, na.rm = T),
            emp_nonfarm = mean(emp_nonfarm, na.rm = T),
            emp_privatenonfarm = mean(emp_privatenonfarm, na.rm = T))

# fix fips 
# from: https://www.economy.com/support/blog/getfile.asp?did=869A03D1-5D74-4376-A606-00A8C64DDB0B&fid=a18d6d4873834f749d42ded633850a5e.xlsx

bea_fips_xwalk <- read.csv(file = paste0(folder_data, "bea_fips_xwalk.csv"), stringsAsFactors = F)

county_list <- unique(select(full_county_set, fipscounty, countyname)) %>%  # all county fips and names
    mutate(indata = 1)

bea_fips_xwalk <- merge(x = bea_fips_xwalk, y = county_list, by.x = "realfips", by.y = "fipscounty", all.x = T, all.y = F)

# keep counties when conflict

bea_fips_xwalk <- bea_fips_xwalk %>% 
  filter(county == 1) %>% 
  select(realfips, beafips)

bea_caemp25n_data <- merge(x =  bea_caemp25n_data, y = bea_fips_xwalk, by.x = "countyfips", by.y = "beafips", all.x = T, all.y = F)

# fix fips 

bea_caemp25n_data <- bea_caemp25n_data %>% 
  rename(oldfips = countyfips) %>% 
  mutate(countyfips = ifelse(!is.na(realfips), realfips, oldfips))

bea_caemp25n_data <- bea_caemp25n_data %>% 
  select(-oldfips, -realfips)

saveRDS(bea_caemp25n_data, file = paste0(folder_data, "bea_caemp25n_data.rds"))

# farm finance
bea_cainc45_data <- read.csv(file = paste0(folder_data, "CAINC45/CAINC45__ALL_AREAS_1969_2022_trim.csv"), stringsAsFactors = F)
# save lines: 20 60 130 210 270 150

bea_cainc45_data <- bea_cainc45_data %>% 
  filter(LineCode == 20 | LineCode == 60 | LineCode == 130  | LineCode == 210 | LineCode == 270  | LineCode == 150)

bea_cainc45_data <- bea_cainc45_data %>% 
  mutate(countyfips = as.numeric(str_trim(gsub("\"", "", GeoFIPS))), # remove qoutes
         category = ifelse(LineCode == 60, "farm_cashcrops", # Cash receipts: Crops 
                           ifelse(LineCode == 20, "farm_cashanimal", #  Cash receipts: Livestock and products
                                  ifelse(LineCode == 130, "farm_govpayments", # Government payments
                                         ifelse(LineCode == 210, "farm_laborexpense", # Hired farm labor expenses
                                                ifelse(LineCode == 270, "farm_cashandinc", # cash receipts and other income
                                                       ifelse(LineCode == 150, "farm_prodexp", NA))))))) # Production expenses 

bea_cainc45_data %>% 
  group_by(category) %>% 
  tally()

names(bea_cainc45_data)

bea_cainc45_data <- bea_cainc45_data %>% 
  select(43:64) # be careful, this gets to be reall a lot of data. 

names(bea_cainc45_data)

bea_cainc45_data <- bea_cainc45_data %>% # wow, that was easy
  pivot_longer(cols = starts_with("y"), names_to = "year", 
               names_prefix = "y", values_to = "temp", 
               values_drop_na = T)

bea_cainc45_data <- bea_cainc45_data %>% 
  mutate(fin = as.numeric(temp)) %>% 
  select(-temp)

# put into year - county rows, so, pivot again 

bea_cainc45_data <- bea_cainc45_data %>% # wow, that was easy
  pivot_wider(names_from = "category", values_from = "fin")

# real 

bea_cainc45_data <- merge(bea_cainc45_data, ppi_data, by = "year", all.x = T, all.y = F)

bea_cainc45_data <- bea_cainc45_data %>% 
  mutate(farm_cashanimal_ppi = farm_cashanimal / ppi_2012,
         farm_cashcrops_ppi = farm_cashcrops / ppi_2012,
         farm_govpayments_ppi = farm_govpayments / ppi_2012,
         farm_prodexp_ppi = farm_prodexp / ppi_2012,
         farm_laborexpense_ppi = farm_laborexpense / ppi_2012,
         farm_cashandinc_ppi = farm_cashandinc / ppi_2012)

# collapse 

bea_cainc45_data <- bea_cainc45_data %>% 
  mutate(census_period = ifelse(year > 2002 & year <= 2007, 2007, 
                                ifelse(year > 2007 & year <= 2012, 2012, 
                                       ifelse(year > 2012 & year <= 2017, 2017, 
                                              ifelse(year > 2017 & year <= 2022, 2022, NA)))))

bea_cainc45_data %>% 
  group_by(census_period) %>% 
  tally()

names(bea_cainc45_data)

bea_cainc45_data <- bea_cainc45_data %>% 
  filter(!is.na(census_period)) %>% 
  group_by(census_period, countyfips) %>% 
  summarise(farm_cashanimal = mean(farm_cashanimal, na.rm = T),
            farm_cashcrops = mean(farm_cashcrops, na.rm = T),
            farm_govpayments = mean(farm_govpayments, na.rm = T),
            farm_prodexp = mean(farm_prodexp, na.rm = T),
            farm_laborexpense = mean(farm_laborexpense, na.rm = T),
            farm_cashandinc = mean(farm_cashandinc, na.rm = T),
            farm_cashanimal_ppi = mean(farm_cashanimal_ppi, na.rm = T),
            farm_cashcrops_ppi = mean(farm_cashcrops_ppi, na.rm = T),
            farm_govpayments_ppi = mean(farm_govpayments_ppi, na.rm = T),
            farm_prodexp_ppi = mean(farm_prodexp_ppi, na.rm = T),
            farm_laborexpense_ppi = mean(farm_laborexpense_ppi, na.rm = T),
            farm_cashandinc_ppi = mean(farm_cashandinc_ppi, na.rm = T))

bea_cainc45_data <- merge(x =  bea_cainc45_data, y = bea_fips_xwalk, by.x = "countyfips", by.y = "beafips", all.x = T, all.y = F)

# fix fips 

bea_cainc45_data <- bea_cainc45_data %>% 
  rename(oldfips = countyfips) %>% 
  mutate(countyfips = ifelse(!is.na(realfips), realfips, oldfips))

bea_cainc45_data <- bea_cainc45_data %>% 
  select(-oldfips, -realfips)

saveRDS(bea_cainc45_data, file = paste0(folder_data, "bea_cainc45_data.rds"))


## ACS Immigration imputation -----------------------------

acs_immigrant_imputed <- read.csv(file = paste0(folder_data, "acs_immigrant_imputed.csv"), stringsAsFactors = F)

acs_immigrant_imputed <- acs_immigrant_imputed %>% 
  select(-X)

# fix fips

acs_immigrant_imputed <- acs_immigrant_imputed %>% 
  mutate(countyfips = as.numeric(str_c(as.character(state_fips_code), 
                            str_pad(as.character(county_fips_code), width = 3, side = "left", pad = "0"))))

# make categories 

acs_immigrant_imputed <- acs_immigrant_imputed %>% 
  mutate(prime_cat = ifelse(prime_age == T, "p", "np"),
         sex_cat = ifelse(sex == "male", "m", "f"),
         im_cat = ifelse(immigrant_type == "documented_immigrant", "doc", 
                         ifelse(immigrant_type == "undocumented_immigrant", "undoc", "nat")))

acs_immigrant_imputed <- acs_immigrant_imputed %>% 
  mutate(category = str_c("pop_", sex_cat, "_", prime_cat, "_", im_cat))

# make wide

acs_immigrant_imputed <- acs_immigrant_imputed %>% 
  mutate(census_period = ifelse(year > 2002 & year <= 2007, 2007, 
                                ifelse(year > 2007 & year <= 2012, 2012, 
                                       ifelse(year > 2012 & year <= 2017, 2017, 
                                              ifelse(year > 2017 & year <= 2022, 2022, NA))))) %>% 
  select(census_period, pop, countyfips, category)

acs_immigrant_imputed <- acs_immigrant_imputed %>% 
  pivot_wider(names_from = "category", values_from = "pop")

# calc row totals
# total, prime aged, prime aged women, prime aged men, undocumented, prime aged undocumented 

acs_immigrant_imputed %>% 
  group_by(census_period) %>% 
  tally()

acs_immigrant_imputed[is.na(acs_immigrant_imputed)] <- 0

acs_immigrant_imputed <- acs_immigrant_imputed %>% 
  mutate(pop_total = pop_f_np_doc + pop_f_np_nat + pop_f_np_undoc + pop_f_p_doc + pop_f_p_nat + pop_f_p_undoc + pop_m_np_doc + pop_m_np_nat + pop_m_np_undoc + pop_m_p_doc + pop_m_p_nat + pop_m_p_undoc,
         pop_prime = pop_f_p_doc+ pop_f_p_nat + pop_f_p_undoc + pop_m_p_doc+ pop_m_p_nat + pop_m_p_undoc , 
         pop_prime_men = pop_m_p_doc + pop_m_p_nat + pop_m_p_undoc, 
         pop_prime_fem = pop_f_p_doc + pop_f_p_nat + pop_f_p_undoc , 
         pop_undoc = pop_f_np_undoc + pop_f_p_undoc + pop_m_np_undoc + pop_m_p_undoc, 
         pop_prime_undoc = pop_f_p_undoc + pop_m_p_undoc)

# calc shares 

acs_immigrant_imputed <- acs_immigrant_imputed %>% 
  mutate(share_undoc = pop_undoc / pop_total, 
         share_undoc_prime = pop_prime_undoc / pop_prime, 
         share_undoc_prime_men = pop_m_p_undoc / pop_prime_men)

saveRDS(acs_immigrant_imputed, file = paste0(folder_data, "acs_immigrant_imputed.rds"))

## County Pop estimates ---------------------------------
# documentation: https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2000-2009/co-est2009-alldata.pdf

census_pop_2000 <- read.csv(paste0(folder_data, "co-est2009-alldata.csv"), stringsAsFactors = F) 
census_pop_2010 <- read.csv(paste0(folder_data, "co-est2019-alldata.csv"), stringsAsFactors = F) 
census_pop_2020 <- read.csv(paste0(folder_data, "co-est2022-alldata.csv"), stringsAsFactors = F) 

census_pop_2000 <- census_pop_2000 %>% 
  filter(SUMLEV == 50) %>% 
  select(STATE, COUNTY, 10:19)
census_pop_2010 <- census_pop_2010 %>% 
  filter(SUMLEV == 50) %>% 
  select(STATE, COUNTY, 10:19)
census_pop_2020 <- census_pop_2020 %>% 
  filter(SUMLEV == 50) %>% 
  select(STATE, COUNTY, 9:11)

head(census_pop_2000)
head(census_pop_2010)
head(census_pop_2020)

census_pop_ests <- merge(x = census_pop_2000, y = census_pop_2010, by = c("STATE", "COUNTY"), all = T)
census_pop_ests <- merge(x = census_pop_ests, y = census_pop_2020, by = c("STATE", "COUNTY"), all = T)

rm(census_pop_2000, census_pop_2010, census_pop_2020)

census_pop_ests <- census_pop_ests %>% 
  mutate(countyfips = as.numeric(str_c(as.character(STATE), 
                                       str_pad(as.character(COUNTY), width = 3, side = "left", pad = "0"))))

census_pop_ests <- census_pop_ests %>% 
  select(-STATE, -COUNTY)

census_pop_ests <- census_pop_ests %>% 
  pivot_longer(cols = starts_with("POPESTIMATE"), names_to = "year", names_prefix = "POPESTIMATE", values_to = "pop_census", values_drop_na = F)

census_pop_ests <- census_pop_ests %>% 
  mutate(census_period = ifelse(year > 2002 & year <= 2007, 2007, 
                              ifelse(year > 2007 & year <= 2012, 2012, 
                                     ifelse(year > 2012 & year <= 2017, 2017, 
                                            ifelse(year > 2017 & year <= 2022, 2022, NA))))) %>% 
  select(-year)

census_pop_ests <- census_pop_ests %>% 
  filter(!is.na(census_period)) %>% 
  group_by(census_period, countyfips) %>% 
  summarise(pop_census = mean(pop_census, na.rm = T))

head(census_pop_ests)

saveRDS(census_pop_ests, file = paste0(folder_data, "census_pop_ests.rds"))

## Border Pair Data ------------------------------------

## Add CZ to each side ## 

head(border_counties_allmatches)
head(cz_file_small)

cz_file_small <- cz_file_small %>% 
  rename(fipscounty = countyfips) # side 1

border_counties_allmatches <- merge(x = border_counties_allmatches, y = cz_file_small, by = "fipscounty", all.x = T, all.y = F)

border_counties_allmatches %>% 
  filter(is.na(cz_out10)) %>% 
  tally() # it worked


cz_file_small <- cz_file_small %>% 
  rename(fipsneighbor = fipscounty,
         cz_out10_neighbor = cz_out10) # side 2

border_counties_allmatches <- merge(x = border_counties_allmatches, y = cz_file_small, by = "fipsneighbor", all.x = T, all.y = F)

head(border_counties_allmatches)

# stack and add years 

head(border_counties_allmatches)

years <- c(2007, 2012, 2017, 2022)

border_df <- NULL

for (i in 1:length(years)) {
  temp <- border_counties_allmatches %>% 
    mutate(census_period = years[i])
  border_df <- bind_rows(border_df, temp)
  rm(temp)
}

head(border_df)

border_df %>% 
  group_by(census_period) %>% 
  tally()

saveRDS(border_df, file = paste0(folder_data, "border_df.rds"))

# remove files 

rm(census_of_agriculture, border_counties_allmatches, acs_immigrant_imputed, acs_qcew_data, aewr_data, aewr_regions, bea_caemp25n_data, bea_cainc45_data, border_df, fips_codes, h2a_data, nawspad_data, oews_data, ppi_data, state_border_pairs)
gc()

## _____________________________________________________________________________
## Year by Year version --------------------------------------------------------
## _____________________________________________________________________________


## Commuting Zones --------------------------------------

# source: https://sites.psu.edu/psucz/data/

cz_file <- read.csv(file = paste0(folder_data, "counties10-zqvz0r.csv"), stringsAsFactors = F)

cz_file <- cz_file %>% 
  rename(countyfips = FIPS,
         cz_out10 = OUT10)

saveRDS(cz_file, file = paste0(folder_data, "cz_file_2010.rds"))


cz_file_small <- cz_file %>% 
  select(countyfips, cz_out10) 

saveRDS(cz_file_small, file = paste0(folder_data, "cz_file_2010_small.rds"))

## Fips Codes ------------------------------------------

fips_codes <- read.csv(file = paste0(folder_data, "fips_codes.csv"), stringsAsFactors = F)

## AEWR Regions ----------------------------------------

aewr_regions <- read.csv(file = paste0(folder_data, "aewr_regions.csv"), stringsAsFactors = F)

## PPI --------------------------------------------------
# Source: https://fred.stlouisfed.org/series/WPU01

ppi_data <- readRDS(file = paste0(folder_data, "ppi_2012.rds"))

## County Boundary ------------------------------------

# cleaned in the original stata file 
state_border_pairs <- read.csv(file = paste0(folder_data, "state_border_pairs.csv"), stringsAsFactors = F)
border_counties_allmatches <- read.csv(file = paste0(folder_data, "border_counties_allmatches.csv"), stringsAsFactors = F)

full_county_set <- read.csv(file = paste0(folder_data, "county_adjacency2010.csv"), stringsAsFactors = F)

dim(state_border_pairs)
dim(border_counties_allmatches)
dim(full_county_set)

## Census of Agriculture ------------------------------

census_of_agriculture <- read.csv(file = paste0(folder_data, "census_of_agriculture.csv"), stringsAsFactors = F)

# general cleaning #

census_of_agriculture <- census_of_agriculture %>% 
  select(-X) %>% 
  rename(val_str = value) %>% 
  mutate(value = as.numeric(str_replace_all(val_str, ",", "")))

head(census_of_agriculture)

ag_census_data_items <- census_of_agriculture %>% 
  group_by(commodity_desc) %>% 
  tally() 

# want: FARM OPERATIONS, AG LAND

census_of_agriculture_trim <- census_of_agriculture %>% 
  filter(commodity_desc == "FARM OPERATIONS" | commodity_desc == "AG LAND")

census_of_agriculture_trim_items <- census_of_agriculture_trim %>% 
  group_by(short_desc) %>% 
  tally() 

census_of_agriculture_trim <- census_of_agriculture_trim %>% 
  filter(short_desc == "AG LAND, CROPLAND - ACRES")

# fix fips 

census_of_agriculture_trim <- census_of_agriculture_trim %>% 
  mutate(countyfips = as.numeric(str_c(as.character(state_fips_code), 
                                       str_pad(as.character(county_code), width = 3, side = "left", pad = "0"))))

census_of_agriculture_trim <- census_of_agriculture_trim %>% 
  arrange(countyfips, year)

census_of_agriculture_trim <-census_of_agriculture_trim %>% 
  mutate(label = "cropland_acr") %>% 
  select(year, countyfips, value, label) 

census_of_agriculture_trim <-census_of_agriculture_trim %>% 
  filter(!is.na(countyfips))

census_of_agriculture_cropland <-census_of_agriculture_trim %>% 
  pivot_wider(names_from = "label", values_from = "value")

head(census_of_agriculture_trim)

saveRDS(census_of_agriculture_cropland, file = paste0(folder_data, "census_ag_cropland_year.rds"))

rm(census_of_agriculture_trim)

census_of_agriculture_cropland_base <- census_of_agriculture_cropland %>% 
  filter(year == 2007) %>% 
  rename(cropland_acr_2007 = cropland_acr) %>% 
  select(-year)

saveRDS(census_of_agriculture_cropland_base, file = paste0(folder_data, "census_ag_cropland_2007_year.rds"))

rm(census_of_agriculture_cropland_base)

## H2A Data -------------------------------------------

h2a_data <- read.csv(file = paste0(folder_data, "h2a_aggregated.csv"), stringsAsFactors = F)

head(h2a_data)

h2a_data <- h2a_data %>% 
  select(-X)


# fix fips code 

h2a_data <- h2a_data %>% 
  mutate(cnty_fips_string = str_pad(as.character(county_fips_code), width = 3, side = "left", pad = "0"),
         st_fips_string = str_pad(as.character(state_fips_code), width = 2, side = "left", pad = "0"))

h2a_data <- h2a_data %>% 
  mutate(countyfips = as.numeric(paste0(st_fips_string, cnty_fips_string)))


# clean by dropping old codes # 

h2a_data <- h2a_data %>% 
  filter(year <= 2022 & year > 2007 & !is.na(county_fips_code)) %>% 
  select(-cnty_fips_string, -st_fips_string, -state_fips_code, -county_fips_code)

# collapse by period, county (county and state fips)

## Handle NAs? 

head(h2a_data)

saveRDS(h2a_data, file = paste0(folder_data, "h2a_data_year.rds"))


## AEWR -------------------------------------------------

aewr_data <- read.csv(file = paste0(folder_data, "aewr.csv"), stringsAsFactors = F)

aewr_data <- aewr_data %>% 
  select(-X)

aewr_data <- merge(x = aewr_data, y = ppi_data, by = "year", all.x = T, all.y = F)

aewr_data <- aewr_data %>% 
  mutate(aewr_ppi = aewr / ppi_2012)

aewr_data <- aewr_data %>% 
  arrange(state_fips_code, year) %>% 
  group_by(state_fips_code) %>% 
  mutate(aewr_ppi_l1 = lag(aewr_ppi, n = 1, order_by = state_fips_code),
         aewr_l1 = lag(aewr, n = 1, order_by = state_fips_code),
         aewr_ppi_l2 = lag(aewr_ppi, n = 2, order_by = state_fips_code),
         aewr_l2 = lag(aewr, n = 2, order_by = state_fips_code))

aewr_data <- aewr_data %>% 
  filter(year > 2007 & year <= 2022)

aewr_data <- merge(x = aewr_data, y = fips_codes, by.x = "state_fips_code", by.y = "fips", all.x = T, all.y = F)

aewr_data <- aewr_data %>% 
  select(aewr, aewr_ppi, aewr_l1, aewr_ppi_l1, aewr_l2, aewr_ppi_l2, year, state_abbrev)

head(aewr_data)

saveRDS(aewr_data, file = paste0(folder_data, "aewr_data_year.rds"))

# full for TS

# pre 1995 source (CRS): https://www.everycrsreport.com/files/20080326_RL32861_9a486634f79a5f9c680cc5ba021e579cc67210a5.pdf

aewr_pre_1995 <- read.csv(file = paste0(folder_data, "aewr_crs_1990_2008.csv"), stringsAsFactors = F)

aewr_pre_1995 <- aewr_pre_1995 %>% 
  pivot_longer(cols = starts_with("X"), names_to = "year", names_prefix = "X", values_to = "aewr", values_drop_na = F)

aewr_pre_1995 <- merge(x = aewr_pre_1995, y = fips_codes, by.x = "State", by.y = "state", all.x = T, all.y = F)

aewr_pre_1995 <- aewr_pre_1995 %>% 
  select(fips, year, aewr) %>% 
  rename(state_fips_code = fips) %>% 
  transform(year = as.numeric(year)) %>% 
  filter(year < 1995)

aewr_data <- read.csv(file = paste0(folder_data, "aewr.csv"), stringsAsFactors = F)

aewr_data <- aewr_data %>% 
  select(-X)

# append dataseries 

aewr_data <- bind_rows(aewr_data, aewr_pre_1995) %>% arrange(state_fips_code, year)

aewr_data %>% group_by(year) %>% tally()

aewr_data <- merge(x = aewr_data, y = ppi_data, by = "year", all.x = T, all.y = F)

aewr_data <- aewr_data %>% 
  mutate(aewr_ppi = aewr / ppi_2012)



aewr_data <- merge(x = aewr_data, y = fips_codes, by.x = "state_fips_code", by.y = "fips", all.x = T, all.y = F)


head(aewr_data)

# collapse by regions 

aewr_data <- merge(x = aewr_data, y = aewr_regions, by = "state_abbrev", all.x = T)

aewr_data <- aewr_data %>% 
  group_by(aewr_region_num, year) %>% 
  summarise(aewr = mean(aewr, na.rm = T), 
            aewr_ppi = mean(aewr_ppi, na.rm = T))

head(aewr_data)

# make TS variables #
aewr_data <- aewr_data %>% 
  arrange(aewr_region_num, year) %>% 
  group_by(aewr_region_num) %>% 
  mutate(aewr_ppi_l1 = lag(aewr_ppi, n = 1, order_by = aewr_region_num),
         aewr_l1 = lag(aewr, n = 1, order_by = aewr_region_num),
         aewr_ppi_l2 = lag(aewr_ppi, n = 2, order_by = aewr_region_num),
         aewr_l2 = lag(aewr, n = 2, order_by = aewr_region_num))

aewr_data <- aewr_data %>% 
  arrange(aewr_region_num, year) %>% 
  mutate(ch_aewr = aewr - aewr_l1, 
         ch_aewr_ppi = aewr_ppi - aewr_ppi_l1, 
         pch_aewr = (aewr - aewr_l1) / aewr_l1 , 
         pch_aewr_ppi = (aewr_ppi - aewr_ppi_l1) / aewr_ppi_l1)

head(aewr_data)

aewr_data <- aewr_data %>% 
  filter(!is.na(aewr_region_num))

saveRDS(aewr_data, file = paste0(folder_data, "aewr_data_full.rds"))

# leav one out averages 

loo_averages <- NULL

for (i in 1:17) {
  temp <- aewr_data %>% 
    filter(aewr_region_num  != i) %>%  # leave one out, so not the one we want
    group_by(year) %>% 
    summarise(loo_aewr = mean(aewr, na.rm = T), 
              loo_aewr_ppi = mean(aewr_ppi, na.rm = T), 
              aewr_region_num = i)
  loo_averages <- bind_rows(loo_averages, temp)
  rm(temp)
}

aewr_data <- merge(x = aewr_data, y = loo_averages, by = c("year", "aewr_region_num"), all.x = T)

aewr_data <- aewr_data %>% 
  arrange(year, aewr_region_num) %>% 
  mutate(aewr_diff = aewr - loo_aewr,
         aewr_ppi_diff = aewr_ppi - loo_aewr_ppi,
         ln_aewr = log(aewr),
         ln_aewr_l1 = log(aewr / aewr_l1))

head(aewr_data)

## AEWR Region TS --------------------------------------------------------------

for(i in 1:17){
  plot <- ggplot(data = subset(aewr_data, aewr_region_num == i), aes(x = year, y = ln_aewr))+
    geom_line()+
    labs(title = paste0("AEWR Region Number: ", i))+
    xlab("Log AEWR (nominal)")
  plot
  ggsave(filename = paste0(folder_output, "AEWR TS/ts_ln_aewr_nominal_", i, ".png"), plot, device = "png")
  rm(plot)
  
  plot <- ggplot(data = subset(aewr_data, aewr_region_num == i), aes(x = year, y = ln_aewr_l1))+
    geom_line()+
    labs(title = paste0("AEWR Region Number: ", i))+
    xlab("Log Change AEWR (nominal)")
  plot
  ggsave(filename = paste0(folder_output, "AEWR TS/ts_ln_aewr_l1_nominal_", i, ".png"), plot, device = "png")
  rm(plot)
}

for(i in 1:17){
  plot <- ggplot(data = subset(aewr_data, aewr_region_num == i), aes(x = year, y = aewr_diff))+
    geom_line()+
    labs(title = paste0("AEWR Region Number: ", i))+
    xlab("AEWR (nominal) difference from LOO trend")
  plot
  ggsave(filename = paste0(folder_output, "AEWR TS/ts_aewr_diffloo_nominal_", i, ".png"), plot, device = "png")
  rm(plot)
  
  plot <- ggplot(data = subset(aewr_data, aewr_region_num == i), aes(x = year, y = aewr_ppi_diff))+
    geom_line()+
    labs(title = paste0("AEWR Region Number: ", i))+
    xlab("AEWR (real) difference from LOO trend")
  plot
  ggsave(filename = paste0(folder_output, "AEWR TS/ts_aewr_diffloo_real_", i, ".png"), plot, device = "png")
  rm(plot)
}


for (i in 1:17) {
  plot <- ggplot(data = subset(aewr_data, aewr_region_num == i), aes(x = year, y = aewr))+
    geom_line()+
    labs(title = paste0("AEWR Region Number: ", i))+
    xlab("AEWR (nominal)")
  plot
  ggsave(filename = paste0(folder_output, "AEWR TS/ts_aewr_nominal_", i, ".png"), plot, device = "png")
  rm(plot)

  plot <- ggplot(data = subset(aewr_data, aewr_region_num == i), aes(x = year, y = aewr_ppi))+
    geom_line()+
    labs(title = paste0("AEWR Region Number: ", i))+
    xlab("AEWR (real)")
  plot
  ggsave(filename = paste0(folder_output, "AEWR TS/ts_aewr_real_", i, ".png"), plot, device = "png")
  rm(plot)

  plot <- ggplot(data = subset(aewr_data, aewr_region_num == i), aes(x = year, y = ch_aewr ))+
    geom_line()+
    labs(title = paste0("AEWR Region Number: ", i))+
    xlab("Change AEWR (nominal)")
  plot
  ggsave(filename = paste0(folder_output, "AEWR TS/ts_change_aewr_nominal_", i, ".png"), plot, device = "png")
  rm(plot)

  plot <- ggplot(data = subset(aewr_data, aewr_region_num == i), aes(x = year, y = ch_aewr_ppi ))+
    geom_line()+
    labs(title = paste0("AEWR Region Number: ", i))+
    xlab("Change AEWR (real)")
  plot

  ggsave(filename = paste0(folder_output, "AEWR TS/ts_change_aewr_real_", i, ".png"), plot, device = "png")
  rm(plot)

  plot <- ggplot(data = subset(aewr_data, aewr_region_num == i), aes(x = year, y = pch_aewr ))+
    geom_line()+
    labs(title = paste0("AEWR Region Number: ", i))+
    xlab("Percent Change AEWR (nominal)")
  plot
  ggsave(filename = paste0(folder_output, "AEWR TS/ts_percentchange_aewr_nominal_", i, ".png"), plot, device = "png")
  rm(plot)

  plot <- ggplot(data = subset(aewr_data, aewr_region_num == i), aes(x = year, y = pch_aewr_ppi ))+
    geom_line()+
    labs(title = paste0("AEWR Region Number: ", i))+
    xlab("Percent Change AEWR (real)")
  plot

  ggsave(filename = paste0(folder_output, "AEWR TS/ts_percentchange_aewr_real_", i, ".png"), plot, device = "png")
  rm(plot)
}

combo_plot <- ggplot(data = aewr_data, aes(x = year, y = pch_aewr, group = as.factor(aewr_region_num), color = as.factor(aewr_region_num)))+
  geom_line()+
  geom_vline(xintercept = 2007)
combo_plot



## BEA Data ---------------------------------------------

# job count data
bea_caemp25n_data <- read.csv(file = paste0(folder_data, "CAEMP25N/CAEMP25N__ALL_AREAS_2001_2022_trim.csv"), stringsAsFactors = F)
# save lines: 10 50 70 80 90

bea_caemp25n_data <- bea_caemp25n_data %>% 
  filter(LineCode == 10 | LineCode == 50 | LineCode == 70 | LineCode == 80 | LineCode == 90)

bea_caemp25n_data <- bea_caemp25n_data %>% 
  mutate(countyfips = as.numeric(str_trim(gsub("\"", "", GeoFIPS))), # remove qoutes
         category = ifelse(LineCode == 10, "emp_tot", 
                           ifelse(LineCode == 50, "emp_farm_propr", # farm proprietors
                                  ifelse(LineCode == 70, "emp_farm", 
                                         ifelse(LineCode == 80, "emp_nonfarm", 
                                                ifelse(LineCode == 90, "emp_privatenonfarm", 
                                                       NA))))))

bea_caemp25n_data <- bea_caemp25n_data %>% 
  select(9:32)

bea_caemp25n_data <- bea_caemp25n_data %>% # wow, that was easy
  pivot_longer(cols = starts_with("y"), names_to = "year", names_prefix = "y", values_to = "temp", values_drop_na = F)

bea_caemp25n_data <- bea_caemp25n_data %>% 
  mutate(emp = as.numeric(temp)) %>% 
  select(-temp)

# put into year - county rows, so, pivot again 

bea_caemp25n_data <- bea_caemp25n_data %>% # wow, that was easy
  pivot_wider(names_from = "category", values_from = "emp")

bea_caemp25n_data %>% 
  group_by(year) %>% 
  tally()

bea_caemp25n_data <- bea_caemp25n_data %>% 
  filter(year > 2007 & year <= 2022) 

# fix fips 
# from: https://www.economy.com/support/blog/getfile.asp?did=869A03D1-5D74-4376-A606-00A8C64DDB0B&fid=a18d6d4873834f749d42ded633850a5e.xlsx

bea_fips_xwalk <- read.csv(file = paste0(folder_data, "bea_fips_xwalk.csv"), stringsAsFactors = F)

county_list <- unique(select(full_county_set, fipscounty, countyname)) %>%  # all county fips and names
  mutate(indata = 1)

bea_fips_xwalk <- merge(x = bea_fips_xwalk, y = county_list, by.x = "realfips", by.y = "fipscounty", all.x = T, all.y = F)

# keep counties when conflict

bea_fips_xwalk <- bea_fips_xwalk %>% 
  filter(county == 1) %>% 
  select(realfips, beafips)

bea_caemp25n_data <- merge(x =  bea_caemp25n_data, y = bea_fips_xwalk, by.x = "countyfips", by.y = "beafips", all.x = T, all.y = F)

# fix fips 

bea_caemp25n_data <- bea_caemp25n_data %>% 
  rename(oldfips = countyfips) %>% 
  mutate(countyfips = ifelse(!is.na(realfips), realfips, oldfips))

bea_caemp25n_data <- bea_caemp25n_data %>% 
  select(-oldfips, -realfips)

head(bea_caemp25n_data)

saveRDS(bea_caemp25n_data, file = paste0(folder_data, "bea_caemp25n_data_year.rds"))

# farm finance
bea_cainc45_data <- read.csv(file = paste0(folder_data, "CAINC45/CAINC45__ALL_AREAS_1969_2022_trim.csv"), stringsAsFactors = F)
# save lines: 20 60 130 210 270 150

bea_cainc45_data <- bea_cainc45_data %>% 
  filter(LineCode == 20 | LineCode == 60 | LineCode == 130  | LineCode == 210 | LineCode == 270  | LineCode == 150)

bea_cainc45_data <- bea_cainc45_data %>% 
  mutate(countyfips = as.numeric(str_trim(gsub("\"", "", GeoFIPS))), # remove qoutes
         category = ifelse(LineCode == 60, "farm_cashcrops", # Cash receipts: Crops 
                           ifelse(LineCode == 20, "farm_cashanimal", #  Cash receipts: Livestock and products
                                  ifelse(LineCode == 130, "farm_govpayments", # Government payments
                                         ifelse(LineCode == 210, "farm_laborexpense", # Hired farm labor expenses
                                                ifelse(LineCode == 270, "farm_cashandinc", # cash receipts and other income
                                                       ifelse(LineCode == 150, "farm_prodexp", NA))))))) # Production expenses 

bea_cainc45_data %>% 
  group_by(category) %>% 
  tally()

names(bea_cainc45_data)

bea_cainc45_data <- bea_cainc45_data %>% 
  select(43:64) # be careful, this gets to be reall a lot of data. 

names(bea_cainc45_data)

bea_cainc45_data <- bea_cainc45_data %>% # wow, that was easy
  pivot_longer(cols = starts_with("y"), names_to = "year", 
               names_prefix = "y", values_to = "temp", 
               values_drop_na = T)

bea_cainc45_data <- bea_cainc45_data %>% 
  mutate(fin = as.numeric(temp)) %>% 
  select(-temp)

# put into year - county rows, so, pivot again 

bea_cainc45_data <- bea_cainc45_data %>% # wow, that was easy
  pivot_wider(names_from = "category", values_from = "fin")

# real 

bea_cainc45_data <- merge(bea_cainc45_data, ppi_data, by = "year", all.x = T, all.y = F)

bea_cainc45_data <- bea_cainc45_data %>% 
  mutate(farm_cashanimal_ppi = farm_cashanimal / ppi_2012,
         farm_cashcrops_ppi = farm_cashcrops / ppi_2012,
         farm_govpayments_ppi = farm_govpayments / ppi_2012,
         farm_prodexp_ppi = farm_prodexp / ppi_2012,
         farm_laborexpense_ppi = farm_laborexpense / ppi_2012,
         farm_cashandinc_ppi = farm_cashandinc / ppi_2012)

bea_cainc45_data %>% 
  group_by(year) %>% 
  tally()

names(bea_cainc45_data)

bea_cainc45_data <- bea_cainc45_data %>% 
  filter(year > 2007 & year <= 2022) 

bea_cainc45_data <- merge(x =  bea_cainc45_data, y = bea_fips_xwalk, by.x = "countyfips", by.y = "beafips", all.x = T, all.y = F)

# fix fips 

bea_cainc45_data <- bea_cainc45_data %>% 
  rename(oldfips = countyfips) %>% 
  mutate(countyfips = ifelse(!is.na(realfips), realfips, oldfips))

bea_cainc45_data <- bea_cainc45_data %>% 
  select(-oldfips, -realfips)

saveRDS(bea_cainc45_data, file = paste0(folder_data, "bea_cainc45_data_year.rds"))


## County Pop estimates ---------------------------------
# documentation: https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2000-2009/co-est2009-alldata.pdf

census_pop_2000 <- read.csv(paste0(folder_data, "co-est2009-alldata.csv"), stringsAsFactors = F) 
census_pop_2010 <- read.csv(paste0(folder_data, "co-est2019-alldata.csv"), stringsAsFactors = F) 
census_pop_2020 <- read.csv(paste0(folder_data, "co-est2022-alldata.csv"), stringsAsFactors = F) 

ct_xwalk <- read.csv(paste0(paste0(folder_data, "ct_region_xwalk.csv")), stringsAsFactors = F)
ct_popgrth <- read.csv(paste0(paste0(folder_data, "ct_pop_grth.csv")), stringsAsFactors = F)

census_pop_2000 <- census_pop_2000 %>% 
  filter(SUMLEV == 50) %>% 
  select(STATE, COUNTY, 10:19)
census_pop_2010 <- census_pop_2010 %>% 
  filter(SUMLEV == 50) %>% 
  select(STATE, COUNTY, 10:19)
census_pop_2020 <- census_pop_2020 %>% 
  filter(SUMLEV == 50) %>% 
  select(STATE, COUNTY, 9:11)

head(census_pop_2000)
head(census_pop_2010)
head(census_pop_2020)

# fix CT county to region nonsense. 
# https://www.ctdata.org/blog/geographic-resources-for-connecticuts-new-county-equivalent-geography
# project using state growth rates

census_pop_ests <- merge(x = census_pop_2000, y = census_pop_2010, by = c("STATE", "COUNTY"), all = T)
census_pop_ests <- merge(x = census_pop_ests, y = census_pop_2020, by = c("STATE", "COUNTY"), all = T)

rm(census_pop_2000, census_pop_2010, census_pop_2020)

census_pop_ests <- census_pop_ests %>% 
  mutate(countyfips = as.numeric(str_c(as.character(STATE), 
                                       str_pad(as.character(COUNTY), width = 3, side = "left", pad = "0"))))

census_pop_ests <- census_pop_ests %>% 
  select(-STATE, -COUNTY)

census_pop_ests <- census_pop_ests %>% 
  pivot_longer(cols = starts_with("POPESTIMATE"), names_to = "year", names_prefix = "POPESTIMATE", values_to = "pop_census", values_drop_na = F)

census_pop_ests %>% 
  group_by(year) %>% 
  tally()

census_pop_ests <- census_pop_ests %>% 
  filter(year > 2007 & year <= 2022) 

head(census_pop_ests)

# fix CT

census_pop_ests <- census_pop_ests %>% 
  mutate(ct_region = ifelse(countyfips  >= 9000 & countyfips <= 9999 & year >= 2020, 1, 0)) # remove CT in 2020 on

ct_base <- census_pop_ests %>% 
  filter(countyfips  >= 9000 & countyfips <= 9999 & year==2019)

census_pop_ests <- census_pop_ests %>% 
  filter(ct_region == 0) %>% 
  select(-ct_region)

census_pop_ests <- census_pop_ests %>% 
  filter(countyfips < 9100 | countyfips > 9199) # get rid of the regions

ct_base <- ct_base %>% 
  filter(countyfips < 9100) %>% 
  select(countyfips, pop_census) %>% 
  mutate(pop_census2020 = pop_census * ct_popgrth[2,3],
         pop_census2021 = pop_census2020 * ct_popgrth[3,3],
         pop_census2022 = pop_census2021 * ct_popgrth[4,3])

ct_base <- ct_base %>% 
  select(-pop_census)

ct_fill <- ct_base %>% 
  pivot_longer(cols = starts_with("pop_census"), names_to = "year", names_prefix = "pop_census", values_to = "pop_census", values_drop_na = F)


census_pop_ests <- rbind(census_pop_ests, ct_fill)

# fix SD county change 

census_pop_ests$countyfips <-replace(census_pop_ests$countyfips, census_pop_ests$countyfips == 46102, 46113) 

census_pop_ests %>% filter(countyfips == 46111) %>% tally() # double

census_pop_ests <- census_pop_ests %>% 
  filter( !is.na(pop_census))

census_pop_ests %>% filter(countyfips == 46111) %>% tally() # fixed


saveRDS(census_pop_ests, file = paste0(folder_data, "census_pop_ests_year.rds"))

## Border Pair Data ------------------------------------

## Add CZ to each side ## 

head(border_counties_allmatches)
head(cz_file_small)

cz_file_small <- cz_file_small %>% 
  rename(fipscounty = countyfips) # side 1

border_counties_allmatches <- merge(x = border_counties_allmatches, y = cz_file_small, by = "fipscounty", all.x = T, all.y = F)

border_counties_allmatches %>% 
  filter(is.na(cz_out10)) %>% 
  tally() # it worked


cz_file_small <- cz_file_small %>% 
  rename(fipsneighbor = fipscounty,
         cz_out10_neighbor = cz_out10) # side 2

border_counties_allmatches <- merge(x = border_counties_allmatches, y = cz_file_small, by = "fipsneighbor", all.x = T, all.y = F)

head(border_counties_allmatches)

# stack and add years 

head(border_counties_allmatches)

years <- seq(2008, 2022)

border_df <- NULL

for (i in 1:length(years)) {
  temp <- border_counties_allmatches %>% 
    mutate(year = years[i])
  border_df <- bind_rows(border_df, temp)
  rm(temp)
}

head(border_df)

border_df %>% 
  group_by(year) %>% 
  tally()

saveRDS(border_df, file = paste0(folder_data, "border_df_year.rds"))

## all county sample ----------------------------------------------------------

head(full_county_set)

county_list_unique <- unique(select(full_county_set, fipscounty, countyname)) 

dim(county_list_unique)

years <- seq(2008, 2022)
county_df <- NULL

for (i in 1:length(years)) {
  temp <- county_list_unique %>% 
    mutate(year = years[i])
  county_df <- bind_rows(county_df, temp)
  rm(temp)
}

head(county_df)

county_df %>% 
  group_by(year) %>% 
  tally()

saveRDS(county_df, file = paste0(folder_data, "county_df_year.rds"))

# remove files -------------------

str_detect(ls(), "folder_")

objects <- data.frame(name = ls(), keep = str_detect(ls(), "folder_")) %>% 
  filter(keep == F)

rm(list = objects[,1])
gc()