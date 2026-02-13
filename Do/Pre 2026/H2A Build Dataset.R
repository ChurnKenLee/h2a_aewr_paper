## H2A Build Dataset
## Phil Hoxie 
## 1/31/24

## Run Master File First ##

## Load Data -------------------------------------------------------------------

# census period version #
acs_immigrant_imputed <- readRDS(paste0(folder_data, "acs_immigrant_imputed.rds"))
acs_qcew_data <- readRDS(paste0(folder_data, "acs_qcew_data.rds"))
aewr_data <- readRDS(paste0(folder_data, "aewr_data.rds"))
aewr_regions <- read.csv(file = paste0(folder_data, "aewr_regions.csv"), stringsAsFactors = F)
bea_caemp25n_data <- readRDS(paste0(folder_data, "bea_caemp25n_data.rds"))
bea_cainc45_data <- readRDS(paste0(folder_data, "bea_cainc45_data.rds"))
border_df <- readRDS(paste0(folder_data, "border_df.rds"))
fips_codes <- read.csv(file = paste0(folder_data, "fips_codes.csv"), stringsAsFactors = F)
h2a_data <- readRDS(paste0(folder_data, "h2a_data.rds"))
# nawspad_data
# oews_data
ppi_data <- read.csv(file = paste0(folder_data, "WPU01.csv"), stringsAsFactors = F)
state_border_pairs <- read.csv(file = paste0(folder_data, "state_border_pairs.csv"), stringsAsFactors = F)
border_counties_allmatches <- read.csv(file = paste0(folder_data, "border_counties_allmatches.csv"), stringsAsFactors = F)
census_of_agriculture_cropland <- readRDS(file = paste0(folder_data, "census_ag_cropland.rds"))

census_pop_ests <- readRDS(paste0(folder_data, "census_pop_ests.rds"))

census_of_agriculture_cropland_base <- readRDS(file = paste0(folder_data, "census_ag_cropland_2007.rds"))

## Build Main Paired Border County Dataset -------------------------------------

head(border_df) # this is the base, we will attache everything here. 
head(state_border_pairs) # this is the base, we will attache everything here. 
head(border_counties_allmatches) # this is the base, we will attache everything here. 

border_df %>% group_by(census_period) %>% tally()

# merge in for each side: 

# by state and year
head(aewr_data)

# by state
head(aewr_regions)

# merge for only one side 

# by 
# loop ? 

# by county and census_period
head(acs_immigrant_imputed)
head(acs_qcew_data) # year
head(bea_caemp25n_data)
head(bea_cainc45_data)
head(h2a_data)
head(census_pop_ests)

## a tad of prep ---------------------------------------------------------------

border_df <- border_df %>% 
  transform(state_abbrev = str_trim(state), 
            neighbor_abbrev = str_trim(neighbor_state))

head(border_df)

## both sides merge ----------------------------------------------------------

# first side

names(aewr_data)

dim(border_df)

border_df <- merge(x = border_df, y = aewr_data, by = c("census_period", "state_abbrev"), all.x = T, all.y = F)

dim(border_df) # DC missing

# no need to rename these

names(aewr_regions)
dim(border_df)

border_df <- merge(x = border_df, y = aewr_regions, by = c("state_abbrev"), all.x = T, all.y = F)

dim(border_df)

# second side (neighbr)

names(aewr_data)
names(border_df)
dim(border_df)

aewr_data_m <- aewr_data %>% 
  rename(aewr_neighbor = aewr, 
         aewr_ppi_neighbor = aewr_ppi,
         neighbor_abbrev =  state_abbrev)

border_df <- merge(x = border_df, y = aewr_data_m, 
                   by = c("census_period", "neighbor_abbrev"), 
                   all.x = T, all.y = F)
rm(aewr_data_m)

dim(border_df)

# no need to rename these

names(aewr_regions)
dim(border_df)

aewr_regions_m <- aewr_regions %>% 
  rename(aewr_region_num_neighbor = aewr_region_num,
         neighbor_abbrev =  state_abbrev)

border_df <- merge(x = border_df, y = aewr_regions_m, 
                   by = c("neighbor_abbrev"),
                   all.x = T, all.y = F)
rm(aewr_regions_m)

dim(border_df)
head(border_df)

# ID Pair "side" 
# This is arbitrary, but necessary

border_df %>% group_by(border_pair, orderpair) %>% tally()

border_df <- border_df %>% 
  mutate(border_side = ifelse(orderpair == "state first", 1, 0))

head(border_df)

# can we loop these 
# # by county and census_period
# head(acs_immigrant_imputed)
# head(acs_qcew_data) # year
# head(bea_caemp25n_data)
# head(bea_cainc45_data)
# head(h2a_data)

border_df <- border_df %>% 
  mutate(cz_same = ifelse(cz_out10 == cz_out10_neighbor, 1, 0))

border_df <- border_df %>% 
  rename(countyfips = fipscounty)

datasets <- c("acs_immigrant_imputed", "acs_qcew_data", "bea_caemp25n_data", "bea_cainc45_data", "h2a_data", "census_pop_ests", "census_of_agriculture_cropland")

for (i in 1:length(datasets)) {
  print(paste0("Rep ", i))
  temp <- get(datasets[i])
  print(dim(border_df))
  print(dim(temp))
  border_df <- merge(x = border_df, y = temp, 
                     by = c("census_period", "countyfips"),
                     all.x = T, all.y = F)
  rm(temp)
}
dim(border_df)
head(border_df)

# county only #

dim(census_of_agriculture_cropland_base)

border_df <- merge(x = border_df, y = census_of_agriculture_cropland_base, by = "countyfips", all.x = T, all.y = F) 

border_df %>% 
  group_by(census_period) %>% 
  tally()

border_df %>% 
  filter(census_period == 2007 & !is.na(cropland_acr_2007)) %>% 
  count()

## Variable cleaning ## -----------------

# H2A NAs to zero

border_df$nbr_workers_requested_all_years[is.na(border_df$nbr_workers_requested_all_years)] <- 0 
border_df$nbr_workers_certified_all_years[is.na(border_df$nbr_workers_certified_all_years)] <- 0 
border_df$man_hours_requested_all_years[is.na(border_df$man_hours_requested_all_years)] <- 0 
border_df$man_hours_certified_all_years[is.na(border_df$man_hours_certified_all_years)] <- 0 
border_df$nbr_applications_all_years[is.na(border_df$nbr_applications_all_years)] <- 0 

border_df$nbr_workers_requested_start_year[is.na(border_df$nbr_workers_requested_start_year)] <- 0 
border_df$nbr_workers_certified_start_year[is.na(border_df$nbr_workers_certified_start_year)] <- 0 
border_df$man_hours_requested_start_year[is.na(border_df$man_hours_requested_start_year)] <- 0 
border_df$man_hours_certified_start_year[is.na(border_df$man_hours_certified_start_year)] <- 0 
border_df$nbr_applications_start_year[is.na(border_df$nbr_applications_start_year)] <- 0 

border_df$nbr_workers_requested_fiscal_year[is.na(border_df$nbr_workers_requested_fiscal_year)] <- 0 
border_df$nbr_workers_certified_fiscal_year[is.na(border_df$nbr_workers_certified_fiscal_year)] <- 0 
border_df$man_hours_requested_fiscal_year[is.na(border_df$man_hours_requested_fiscal_year)] <- 0 
border_df$man_hours_certified_fiscal_year[is.na(border_df$man_hours_certified_fiscal_year)] <- 0 
border_df$nbr_applications_fiscal_year[is.na(border_df$nbr_applications_fiscal_year)] <- 0 

head(border_df)

# cropland zeros 

border_df$cropland_acr[is.na(border_df$cropland_acr)] <- 0 
border_df$cropland_acr_2007[is.na(border_df$cropland_acr_2007)] <- 0

# AEWR Diff 

border_df <- border_df %>% 
  mutate(aewr_diff = aewr - aewr_neighbor,
         aewr_ppi_diff = aewr_ppi - aewr_ppi_neighbor,
         aewr_diff_pct = (aewr - aewr_neighbor) / aewr_neighbor,
         aewr_ppi_diff_pct = (aewr_ppi - aewr_ppi_neighbor) / aewr_ppi_neighbor)

hist(border_df$aewr_diff)

# logs of some vars

border_df <- border_df %>% 
  mutate(ln_aewr = log(aewr), 
         ln_aewr_ppi = log(aewr_ppi), 
         ln_pop_total = log(pop_total), 
         ln_pop_prime = log(pop_prime), 
         ln_pop_undoc = log(pop_undoc), 
         ln_pop_prime_undoc = log(pop_prime_undoc), 
         ln_emp_farm_propr = log(emp_farm_propr), 
         ln_emp_farm = log(emp_farm), 
         ln_emp_nonfarm = log(emp_nonfarm), 
         ln_emp_privatenonfarm = log(emp_privatenonfarm),
         ln_pop_census = log(pop_census),
         ln_cropland_acr = log(cropland_acr),
         ln_cropland_acr_2007 = log(cropland_acr_2007),
         ln_nbr_workers_requested_start_years = log(nbr_workers_requested_start_year),
         ln_nbr_workers_requested_all_years = log(nbr_workers_requested_all_years),
         ln_nbr_applications_fiscal_year = log(nbr_applications_fiscal_year),
         ln1_nbr_workers_requested_start_years = log(nbr_workers_requested_start_year + 1),
         ln1_nbr_workers_requested_all_years = log(nbr_workers_requested_all_years + 1),
         ln1_nbr_applications_fiscal_year = log(nbr_applications_fiscal_year + 1))

# budget shares 

border_df <- border_df %>% 
  mutate(share_farm_laborexp_prodexp = farm_laborexpense / farm_prodexp,
        share_farm_crop_cashandinc = farm_cashcrops / farm_cashandinc,
        share_farm_animal_cashandinc = farm_cashanimal / farm_cashandinc,
        share_farm_govt_cashandinc = farm_govpayments / farm_cashandinc)

# emp-pop ratio 

border_df <- border_df %>% 
  mutate(emp_pop_ratio = emp_tot / pop_total,
         emp_farm_share = emp_farm / emp_tot)

# H2A outcome variables 

border_df <- border_df %>% 
  mutate(emp_pop_ratio = emp_tot / pop_total,
         emp_farm_share = emp_farm / emp_tot,
         h2a_req_share_farm_workers_start_year = nbr_workers_requested_start_year / emp_farm,
         h2a_cert_share_farm_workers_start_year = nbr_workers_certified_start_year / emp_farm) # add h2a apps per farm later

border_df$h2a_req_share_farm_workers_start_year[is.infinite(border_df$h2a_req_share_farm_workers_start_year)] <- NA
border_df$h2a_cert_share_farm_workers_start_year[is.infinite(border_df$h2a_cert_share_farm_workers_start_year)] <- NA

# AEWR border sample # 

border_df <- border_df %>% 
  mutate(aewr_border_sample = ifelse(aewr_region_num == aewr_region_num_neighbor, 0 , 1))

border_df %>% group_by(aewr_border_sample) %>% tally()

head(border_df)

border_df %>% 
  filter(is.na(aewr_border_sample)) %>% 
  group_by(state_abbrev) %>% 
  tally()

# identify lower of AEWR regions
border_df <- border_df %>% 
  mutate(lower_aewr_reg_num = ifelse(aewr_region_num < aewr_region_num_neighbor, aewr_region_num, 
                                     ifelse(aewr_region_num > aewr_region_num_neighbor, aewr_region_num_neighbor, NA)))

# make region border ID

border_df <- border_df %>% 
  mutate(aewr_region_border_id = ifelse(aewr_region_num == lower_aewr_reg_num, paste0(aewr_region_num,"_", aewr_region_num_neighbor),
                                        ifelse(aewr_region_num > lower_aewr_reg_num, paste0(aewr_region_num_neighbor,"_", aewr_region_num), NA)))

border_df %>% group_by(aewr_region_border_id) %>% tally()

unique(border_df$aewr_region_border_id) # 32 groups

# make pair ID

# identify lower of AEWR regions
border_df <- border_df %>% 
  mutate(lower_cnty_fips = ifelse(countyfips < fipsneighbor, countyfips, 
                                     ifelse(countyfips > fipsneighbor, fipsneighbor, NA)))

# make region border ID

border_df <- border_df %>% 
  mutate(pair_id = ifelse(countyfips == lower_cnty_fips, paste0(countyfips,"_", fipsneighbor),
                                        ifelse(countyfips > lower_cnty_fips, paste0(fipsneighbor,"_", countyfips), NA)))

border_df %>% group_by(pair_id) %>% tally()
border_df %>% group_by(aewr_region_border_id) %>% tally()

length(unique(border_df$pair_id))

# fixed effects

# period 

border_df$census_period_fe <- with(border_df, as.factor(census_period)) # treats counties in separate clusters as separate

# county 

border_df$county_unique_fe <- with(border_df, interaction(as.factor(countyfips),  aewr_region_border_id)) # treats counties in separate clusters as separate

border_df$county_fe <- with(border_df, as.factor(countyfips)) # treats counties in separate clusters as separate

# state

border_df$state_fe <- with(border_df, as.factor(state_abbrev)) # treats counties in separate clusters as separate

# AEWR Region

border_df$aewr_region_fe <- with(border_df, as.factor(aewr_region_num)) # treats counties in separate clusters as separate

# pair

border_df$pair_fe <- with(border_df, as.factor(pair_id)) # treats counties in separate clusters as separate

# pair x time

border_df$pairtime_fe <- with(border_df, interaction(as.factor(pair_id),  census_period))

levels(border_df$pairtime_fe) <- c(levels(border_df$pairtime_fe),"0.0") 

unique(border_df$census_period)

border_df$pairtime_fe[border_df$census_period == 2007] <- "0.0" # first period is 0

# aewr region pair x time

border_df$aewrbordertime_fe <- with(border_df, interaction(as.factor(aewr_region_border_id),  census_period))

levels(border_df$aewrbordertime_fe) <- c(levels(border_df$aewrbordertime_fe),"0.0") 

unique(border_df$census_period)

border_df$aewrbordertime_fe[border_df$census_period == 2007] <- "0.0" # first period is 0


# AEWR border side -------------------------------------------------------------

border_df <- border_df %>% 
  mutate(aewr_border_side = ifelse(aewr_region_num == lower_aewr_reg_num & aewr_border_sample == 1, 0, 
                                   ifelse(aewr_region_num != lower_aewr_reg_num & aewr_border_sample == 1, 1, NA)))

# sample restriction to counties with cropland ------------------------------------

# we want to drop counties with no cropland in 2007

border_df <- border_df %>% 
  mutate(any_cropland_2007 = ifelse(cropland_acr_2007 != 0 & !is.na(cropland_acr_2007), 1, 0))

border_df %>% 
  group_by(any_cropland_2007, census_period) %>% 
  tally()

# need to make sure both sides of the pair drop out #

no_crop_pairs <- border_df %>% 
  filter(any_cropland_2007 == 0 & census_period == 2007) %>% 
  select(pair_id)

no_crop_pairs <- no_crop_pairs %>% 
  mutate(no_cropland_2007_pair = 1)

dim(no_crop_pairs)

no_crop_pairs <- unique(no_crop_pairs)
dim(no_crop_pairs)

border_df <- merge(x = border_df, y = no_crop_pairs, by = "pair_id", all.x = T)

border_df$no_cropland_2007_pair[is.na(border_df$no_cropland_2007_pair)] <- 0


# check for NAs in key vars ------------------------------------

# use sum(is.na(x))

NA_df <- NULL 

dim(border_df)

border_df_samp <- border_df %>% 
  filter(!is.na(aewr_region_border_id))

for (i in 1:length(names(border_df))) {
  temp_na_cnt <- sum(is.na(border_df[,i]))
  temp_varname <- paste0(names(border_df)[i])
  temp_na_cnt_samp <- sum(is.na(border_df_samp[,i]))

  temp <- data.frame(var = temp_varname, 
                     full_na_cnt = temp_na_cnt, 
                     border_samp_na_cnt = temp_na_cnt_samp)
  
  NA_df <- bind_rows(NA_df, temp)
  rm(temp, temp_na_cnt, temp_varname, temp_na_cnt_samp)
}

# easiest / most important to fix
fixcounty <- border_df %>% 
  filter(is.na(emp_tot) & !is.na(aewr_region_border_id))

fixcounty %>% 
  group_by(state_abbrev) %>% 
  tally() # BEA issue for VA counties. We need to drop bristol city. It will be ok. 

saveRDS(border_df, file = paste0(folder_data, "border_df_analysis.rds"))

## Remove files ## -------------------------------------------------------------

str_detect(ls(), "folder_")

objects <- data.frame(name = ls(), keep = str_detect(ls(), "folder_")) %>% 
  filter(keep == F)

rm(list = objects[,1])
gc()

rm(objects)

## -----------------------------------------------------------------------------
## Yearly Dataset ## -----------------------------------------------------------
## -----------------------------------------------------------------------------

## Load Data -------------------------------------------------------------------

# yearly versions #
aewr_data <- readRDS(paste0(folder_data, "aewr_data_year.rds"))
aewr_regions <- read.csv(file = paste0(folder_data, "aewr_regions.csv"), stringsAsFactors = F)
bea_caemp25n_data <- readRDS(paste0(folder_data, "bea_caemp25n_data_year.rds"))
bea_cainc45_data <- readRDS(paste0(folder_data, "bea_cainc45_data_year.rds"))
border_df <- readRDS(paste0(folder_data, "border_df_year.rds"))
fips_codes <- read.csv(file = paste0(folder_data, "fips_codes.csv"), stringsAsFactors = F)
h2a_data <- readRDS(paste0(folder_data, "h2a_data_year.rds"))
# nawspad_data
# oews_data
ppi_data <- read.csv(file = paste0(folder_data, "WPU01.csv"), stringsAsFactors = F)
state_border_pairs <- read.csv(file = paste0(folder_data, "state_border_pairs.csv"), stringsAsFactors = F)
border_counties_allmatches <- read.csv(file = paste0(folder_data, "border_counties_allmatches.csv"), stringsAsFactors = F)
census_of_agriculture_cropland <- readRDS(file = paste0(folder_data, "census_ag_cropland_year.rds"))

census_pop_ests <- readRDS(paste0(folder_data, "census_pop_ests_year.rds"))

census_of_agriculture_cropland_base <- readRDS(file = paste0(folder_data, "census_ag_cropland_2007_year.rds"))

## Build Main Paired Border County Dataset -------------------------------------

head(border_df) # this is the base, we will attache everything here. 
head(state_border_pairs) # this is the base, we will attache everything here. 
head(border_counties_allmatches) # this is the base, we will attache everything here. 

border_df %>% group_by(year) %>% tally()

# merge in for each side: 

# by state and year
head(aewr_data)

# by state
head(aewr_regions)

# merge for only one side 

# by 
# loop ? 

# by county and census_period

head(bea_caemp25n_data)
head(bea_cainc45_data)
head(h2a_data)
head(census_pop_ests)

## a tad of prep ---------------------------------------------------------------

border_df <- border_df %>% 
  transform(state_abbrev = str_trim(state), 
            neighbor_abbrev = str_trim(neighbor_state))

head(border_df)

## both sides merge ----------------------------------------------------------

# first side

names(aewr_data)

dim(border_df)

border_df <- merge(x = border_df, y = aewr_data, by = c("year", "state_abbrev"), all.x = T, all.y = F)

dim(border_df) # DC missing

head(border_df)
# no need to rename these

names(aewr_regions)
dim(border_df)

border_df <- merge(x = border_df, y = aewr_regions, by = c("state_abbrev"), all.x = T, all.y = F)

dim(border_df)

# second side (neighbr)

names(aewr_data)
names(border_df)
dim(border_df)

aewr_data_m <- aewr_data %>% 
  rename(aewr_neighbor = aewr, 
         aewr_ppi_neighbor = aewr_ppi,
         aewr_neighbor_l1 = aewr_l1, 
         aewr_ppi_neighbor_l1 = aewr_ppi_l1,
         aewr_neighbor_l2 = aewr_l2, 
         aewr_ppi_neighbor_l2 = aewr_ppi_l2,
         neighbor_abbrev =  state_abbrev)

border_df <- merge(x = border_df, y = aewr_data_m, 
                   by = c("year", "neighbor_abbrev"), 
                   all.x = T, all.y = F)
rm(aewr_data_m)

dim(border_df)

# no need to rename these

names(aewr_regions)
dim(border_df)

aewr_regions_m <- aewr_regions %>% 
  rename(aewr_region_num_neighbor = aewr_region_num,
         neighbor_abbrev =  state_abbrev)

border_df <- merge(x = border_df, y = aewr_regions_m, 
                   by = c("neighbor_abbrev"),
                   all.x = T, all.y = F)
rm(aewr_regions_m)

dim(border_df)
head(border_df)

# ID Pair "side" 
# This is arbitrary, but necessary

border_df %>% group_by(border_pair, orderpair) %>% tally()

border_df <- border_df %>% 
  mutate(border_side = ifelse(orderpair == "state first", 1, 0))

head(border_df)

# can we loop these 
# # by county and census_period
# head(acs_immigrant_imputed)
# head(acs_qcew_data) # year
# head(bea_caemp25n_data)
# head(bea_cainc45_data)
# head(h2a_data)

border_df <- border_df %>% 
  mutate(cz_same = ifelse(cz_out10 == cz_out10_neighbor, 1, 0))

border_df <- border_df %>% 
  rename(countyfips = fipscounty)

datasets <- c( "bea_caemp25n_data", "bea_cainc45_data", "h2a_data", "census_pop_ests", "census_of_agriculture_cropland")

for (i in 1:length(datasets)) {
  print(paste0("Rep ", i))
  temp <- get(datasets[i])
  print(dim(border_df))
  print(dim(temp))
  border_df <- merge(x = border_df, y = temp, 
                     by = c("year", "countyfips"),
                     all.x = T, all.y = F)
  rm(temp)
}
dim(border_df)
head(border_df)

# county only #

dim(census_of_agriculture_cropland_base)

border_df <- merge(x = border_df, y = census_of_agriculture_cropland_base, by = "countyfips", all.x = T, all.y = F) 

border_df %>% 
  group_by(year) %>% 
  tally()

border_df %>% 
  filter(year == 2008 & !is.na(cropland_acr_2007)) %>% 
  count()

## Variable cleaning ## -----------------

# H2A NAs to zero

border_df$nbr_workers_requested_all_years[is.na(border_df$nbr_workers_requested_all_years)] <- 0 
border_df$nbr_workers_certified_all_years[is.na(border_df$nbr_workers_certified_all_years)] <- 0 
border_df$man_hours_requested_all_years[is.na(border_df$man_hours_requested_all_years)] <- 0 
border_df$man_hours_certified_all_years[is.na(border_df$man_hours_certified_all_years)] <- 0 
border_df$nbr_applications_all_years[is.na(border_df$nbr_applications_all_years)] <- 0 

border_df$nbr_workers_requested_start_year[is.na(border_df$nbr_workers_requested_start_year)] <- 0 
border_df$nbr_workers_certified_start_year[is.na(border_df$nbr_workers_certified_start_year)] <- 0 
border_df$man_hours_requested_start_year[is.na(border_df$man_hours_requested_start_year)] <- 0 
border_df$man_hours_certified_start_year[is.na(border_df$man_hours_certified_start_year)] <- 0 
border_df$nbr_applications_start_year[is.na(border_df$nbr_applications_start_year)] <- 0 

border_df$nbr_workers_requested_fiscal_year[is.na(border_df$nbr_workers_requested_fiscal_year)] <- 0 
border_df$nbr_workers_certified_fiscal_year[is.na(border_df$nbr_workers_certified_fiscal_year)] <- 0 
border_df$man_hours_requested_fiscal_year[is.na(border_df$man_hours_requested_fiscal_year)] <- 0 
border_df$man_hours_certified_fiscal_year[is.na(border_df$man_hours_certified_fiscal_year)] <- 0 
border_df$nbr_applications_fiscal_year[is.na(border_df$nbr_applications_fiscal_year)] <- 0 

head(border_df)

# cropland zeros 

border_df$cropland_acr[is.na(border_df$cropland_acr)] <- 0 
border_df$cropland_acr_2007[is.na(border_df$cropland_acr_2007)] <- 0

# AEWR Diff 

border_df <- border_df %>% 
  mutate(aewr_diff = aewr - aewr_neighbor,
         aewr_ppi_diff = aewr_ppi - aewr_ppi_neighbor,
         aewr_diff_pct = (aewr - aewr_neighbor) / aewr_neighbor,
         aewr_ppi_diff_pct = (aewr_ppi - aewr_ppi_neighbor) / aewr_ppi_neighbor,
         aewr_diff_l1 = aewr_l1 - aewr_neighbor_l1,
         aewr_ppi_diff_l1 = aewr_ppi_l1 - aewr_ppi_neighbor_l1,
         aewr_diff_pct_l1 = (aewr_l1 - aewr_neighbor_l1) / aewr_neighbor_l1,
         aewr_ppi_diff_pct_l1 = (aewr_ppi_l1 - aewr_ppi_neighbor_l1) / aewr_ppi_neighbor_l1,
         aewr_diff_l2 = aewr_l2 - aewr_neighbor_l2,
         aewr_ppi_diff_l2 = aewr_ppi_l2 - aewr_ppi_neighbor_l2,
         aewr_diff_pct_l2 = (aewr_l2 - aewr_neighbor_l2) / aewr_neighbor_l2,
         aewr_ppi_diff_pct_l2 = (aewr_ppi_l2 - aewr_ppi_neighbor_l2) / aewr_ppi_neighbor_l2)

hist(border_df$aewr_diff)

border_df <- border_df %>% 
  mutate(emp_pop_ratio = emp_tot / pop_census)

# logs of some vars

border_df <- border_df %>% 
  mutate(ln_aewr = log(aewr), 
         ln_aewr_ppi = log(aewr_ppi), 
         ln_aewr_l1 = log(aewr_l1), 
         ln_aewr_ppi_l1 = log(aewr_ppi_l1), 
         ln_aewr_l2 = log(aewr_l2), 
         ln_aewr_ppi_l2 = log(aewr_ppi_l2), 
         ln_emp_farm_propr = log(emp_farm_propr), 
         ln_emp_farm = log(emp_farm), 
         ln_emp_nonfarm = log(emp_nonfarm), 
         ln_emp_privatenonfarm = log(emp_privatenonfarm),
         ln_pop_census = log(pop_census),
         ln_cropland_acr = log(cropland_acr),
         ln_cropland_acr_2007 = log(cropland_acr_2007),
         ln_nbr_workers_requested_start_years = log(nbr_workers_requested_start_year),
         ln_nbr_workers_requested_all_years = log(nbr_workers_requested_all_years),
         ln_nbr_applications_fiscal_year = log(nbr_applications_fiscal_year),
         ln1_nbr_workers_requested_start_years = log(nbr_workers_requested_start_year + 1),
         ln1_nbr_workers_requested_all_years = log(nbr_workers_requested_all_years + 1),
         ln1_nbr_applications_fiscal_year = log(nbr_applications_fiscal_year + 1))

# budget shares 

border_df <- border_df %>% 
  mutate(share_farm_laborexp_prodexp = farm_laborexpense / farm_prodexp,
         share_farm_crop_cashandinc = farm_cashcrops / farm_cashandinc,
         share_farm_animal_cashandinc = farm_cashanimal / farm_cashandinc,
         share_farm_govt_cashandinc = farm_govpayments / farm_cashandinc)

# H2A outcome variables 

border_df <- border_df %>% 
  mutate(h2a_req_share_farm_workers_start_year = nbr_workers_requested_start_year / emp_farm,
         h2a_cert_share_farm_workers_start_year = nbr_workers_certified_start_year / emp_farm) # add h2a apps per farm later

border_df$h2a_req_share_farm_workers_start_year[is.infinite(border_df$h2a_req_share_farm_workers_start_year)] <- NA
border_df$h2a_cert_share_farm_workers_start_year[is.infinite(border_df$h2a_cert_share_farm_workers_start_year)] <- NA

# AEWR border sample # 

border_df <- border_df %>% 
  mutate(aewr_border_sample = ifelse(aewr_region_num == aewr_region_num_neighbor, 0 , 1))

border_df %>% group_by(aewr_border_sample) %>% tally()

head(border_df)

border_df %>% 
  filter(is.na(aewr_border_sample)) %>% 
  group_by(state_abbrev) %>% 
  tally()

# identify lower of AEWR regions
border_df <- border_df %>% 
  mutate(lower_aewr_reg_num = ifelse(aewr_region_num < aewr_region_num_neighbor, aewr_region_num, 
                                     ifelse(aewr_region_num > aewr_region_num_neighbor, aewr_region_num_neighbor, NA)))

# make region border ID

border_df <- border_df %>% 
  mutate(aewr_region_border_id = ifelse(aewr_region_num == lower_aewr_reg_num, paste0(aewr_region_num,"_", aewr_region_num_neighbor),
                                        ifelse(aewr_region_num > lower_aewr_reg_num, paste0(aewr_region_num_neighbor,"_", aewr_region_num), NA)))

border_df %>% group_by(aewr_region_border_id) %>% tally()

unique(border_df$aewr_region_border_id) # 32 groups

# make pair ID

# identify lower of AEWR regions
border_df <- border_df %>% 
  mutate(lower_cnty_fips = ifelse(countyfips < fipsneighbor, countyfips, 
                                  ifelse(countyfips > fipsneighbor, fipsneighbor, NA)))

# make region border ID

border_df <- border_df %>% 
  mutate(pair_id = ifelse(countyfips == lower_cnty_fips, paste0(countyfips,"_", fipsneighbor),
                          ifelse(countyfips > lower_cnty_fips, paste0(fipsneighbor,"_", countyfips), NA)))

border_df %>% group_by(pair_id) %>% tally()
border_df %>% group_by(aewr_region_border_id) %>% tally()

length(unique(border_df$pair_id))

# fixed effects

# period 

border_df$year_fe <- with(border_df, as.factor(year)) # treats counties in separate clusters as separate

# county 

border_df$county_unique_fe <- with(border_df, interaction(as.factor(countyfips),  aewr_region_border_id)) # treats counties in separate clusters as separate

border_df$county_fe <- with(border_df, as.factor(countyfips)) # treats counties in separate clusters as separate

# state

border_df$state_fe <- with(border_df, as.factor(state_abbrev)) # treats counties in separate clusters as separate

# AEWR Region

border_df$aewr_region_fe <- with(border_df, as.factor(aewr_region_num)) # treats counties in separate clusters as separate

# pair

border_df$pair_fe <- with(border_df, as.factor(pair_id)) # treats counties in separate clusters as separate

# pair x time

border_df$pairtime_fe <- with(border_df, interaction(as.factor(pair_id),  year))

levels(border_df$pairtime_fe) <- c(levels(border_df$pairtime_fe),"0.0") 

unique(border_df$year)

border_df$pairtime_fe[border_df$year == 2008] <- "0.0" # first period is 0

# aewr region pair x time

border_df$aewrbordertime_fe <- with(border_df, interaction(as.factor(aewr_region_border_id),  year))

levels(border_df$aewrbordertime_fe) <- c(levels(border_df$aewrbordertime_fe),"0.0") 

unique(border_df$year)

border_df$aewrbordertime_fe[border_df$year == 2008] <- "0.0" # first period is 0


# AEWR border side -------------------------------------------------------------

border_df <- border_df %>% 
  mutate(aewr_border_side = ifelse(aewr_region_num == lower_aewr_reg_num & aewr_border_sample == 1, 0, 
                                   ifelse(aewr_region_num != lower_aewr_reg_num & aewr_border_sample == 1, 1, NA)))

# sample restriction to counties with cropland ------------------------------------

# we want to drop counties with no cropland in 2007

border_df <- border_df %>% 
  mutate(any_cropland_2007 = ifelse(cropland_acr_2007 != 0 & !is.na(cropland_acr_2007), 1, 0))

border_df %>% 
  group_by(any_cropland_2007, year) %>% 
  tally()

# need to make sure both sides of the pair drop out #

no_crop_pairs <- border_df %>% 
  filter(any_cropland_2007 == 0 & year == 2008) %>% 
  select(pair_id)

no_crop_pairs <- no_crop_pairs %>% 
  mutate(no_cropland_2007_pair = 1)

dim(no_crop_pairs)

no_crop_pairs <- unique(no_crop_pairs)
dim(no_crop_pairs)

border_df <- merge(x = border_df, y = no_crop_pairs, by = "pair_id", all.x = T)

border_df$no_cropland_2007_pair[is.na(border_df$no_cropland_2007_pair)] <- 0


# check for NAs in key vars ------------------------------------

# use sum(is.na(x))

NA_df <- NULL 

dim(border_df)

border_df_samp <- border_df %>% 
  filter(!is.na(aewr_region_border_id))

for (i in 1:length(names(border_df))) {
  temp_na_cnt <- sum(is.na(border_df[,i]))
  temp_varname <- paste0(names(border_df)[i])
  temp_na_cnt_samp <- sum(is.na(border_df_samp[,i]))
  
  temp <- data.frame(var = temp_varname, 
                     full_na_cnt = temp_na_cnt, 
                     border_samp_na_cnt = temp_na_cnt_samp)
  
  NA_df <- bind_rows(NA_df, temp)
  rm(temp, temp_na_cnt, temp_varname, temp_na_cnt_samp)
}

# easiest / most important to fix
fixcounty <- border_df %>% 
  filter(is.na(emp_tot) & !is.na(aewr_region_border_id))

fixcounty %>% 
  group_by(state_abbrev) %>% 
  tally() # BEA issue for VA counties. We need to drop bristol city. It will be ok. 

## lags of h2a variables


border_df <- border_df %>%
  arrange(county_unique_fe, year) %>%
  group_by(county_unique_fe) %>%
  mutate(nbr_workers_requested_all_years_l1  = lag( nbr_workers_requested_all_years, n = 1, order_by = county_unique_fe),
         nbr_workers_requested_all_years_l2   = lag( nbr_workers_requested_all_years, n = 2, order_by = county_unique_fe),
         nbr_workers_certified_all_years_l1  = lag(nbr_workers_certified_all_years , n = 1, order_by = county_unique_fe),
         nbr_workers_certified_all_years_l2  = lag(nbr_workers_certified_all_years , n = 2, order_by = county_unique_fe),
         man_hours_requested_all_years_l1  = lag(man_hours_requested_all_years , n = 1, order_by = county_unique_fe),
         man_hours_requested_all_years_l2  = lag(man_hours_requested_all_years , n = 2, order_by = county_unique_fe),
         man_hours_certified_all_years_l1  = lag(man_hours_certified_all_years , n = 1, order_by = county_unique_fe),
         man_hours_certified_all_years_l2  = lag(man_hours_certified_all_years , n = 2, order_by = county_unique_fe),
         nbr_applications_all_years_l1 = lag(nbr_applications_all_years , n = 1, order_by = county_unique_fe),
         nbr_applications_all_years_l2 = lag(nbr_applications_all_years , n = 2, order_by = county_unique_fe),
         nbr_workers_requested_start_year_l1  = lag(nbr_workers_requested_start_year , n = 1, order_by = county_unique_fe),
         nbr_workers_requested_start_year_l2  = lag(nbr_workers_requested_start_year , n = 2, order_by = county_unique_fe),
         nbr_workers_certified_start_year_l1  = lag(nbr_workers_certified_start_year , n = 1, order_by = county_unique_fe),
         nbr_workers_certified_start_year_l2  = lag(nbr_workers_certified_start_year , n = 2, order_by = county_unique_fe),
         man_hours_requested_start_year_l1  = lag(man_hours_requested_start_year , n = 1, order_by = county_unique_fe),
         man_hours_requested_start_year_l2  = lag(man_hours_requested_start_year , n = 2, order_by = county_unique_fe),
         man_hours_certified_start_year_l1  = lag(man_hours_certified_start_year , n = 1, order_by = county_unique_fe),
         man_hours_certified_start_year_l2  = lag(man_hours_certified_start_year , n = 2, order_by = county_unique_fe),
         nbr_applications_start_year_l1 = lag(nbr_applications_start_year , n = 1, order_by = county_unique_fe),
         nbr_applications_start_year_l2 = lag(nbr_applications_start_year , n = 2, order_by = county_unique_fe),
         nbr_workers_requested_fiscal_year_l1  = lag(nbr_workers_requested_fiscal_year , n = 1, order_by = county_unique_fe),
         nbr_workers_requested_fiscal_year_l2  = lag(nbr_workers_requested_fiscal_year , n = 2, order_by = county_unique_fe),
         nbr_workers_certified_fiscal_year_l1  = lag(nbr_workers_certified_fiscal_year , n = 1, order_by = county_unique_fe),
         nbr_workers_certified_fiscal_year_l2  = lag(nbr_workers_certified_fiscal_year , n = 2, order_by = county_unique_fe),
         man_hours_requested_fiscal_year_l1  = lag(man_hours_requested_fiscal_year , n = 1, order_by = county_unique_fe),
         man_hours_requested_fiscal_year_l2  = lag(man_hours_requested_fiscal_year , n = 2, order_by = county_unique_fe),
         man_hours_certified_fiscal_year_l1  = lag(man_hours_certified_fiscal_year , n = 1, order_by = county_unique_fe),
         man_hours_certified_fiscal_year_l2  = lag(man_hours_certified_fiscal_year , n = 2, order_by = county_unique_fe),
         nbr_applications_fiscal_year_l1 = lag(nbr_applications_fiscal_year , n = 1, order_by = county_unique_fe),
         nbr_applications_fiscal_year_l2 = lag(nbr_applications_fiscal_year , n = 2, order_by = county_unique_fe))


saveRDS(border_df, file = paste0(folder_data, "border_df_analysis_year.rds"))

## Remove files ## -------------------------------------------------------------

str_detect(ls(), "folder_")

objects <- data.frame(name = ls(), keep = str_detect(ls(), "folder_")) %>% 
  filter(keep == F)

rm(list = objects[,1])
gc()
rm(objects)

## Yearly Full County Dataset ------------------------------------------------

## ------- Full County ----------------------------------------------------------------------
## Yearly Dataset ## -----------------------------------------------------------
## -------- Full County -----------------------------------------------------------

## Load Data -------------------------------------------------------------------

# yearly versions #
aewr_data <- readRDS(paste0(folder_data, "aewr_data_year.rds"))
aewr_regions <- read.csv(file = paste0(folder_data, "aewr_regions.csv"), stringsAsFactors = F)
bea_caemp25n_data <- readRDS(paste0(folder_data, "bea_caemp25n_data_year.rds"))
bea_cainc45_data <- readRDS(paste0(folder_data, "bea_cainc45_data_year.rds"))
border_df <- readRDS(paste0(folder_data, "border_df_year.rds"))
fips_codes <- read.csv(file = paste0(folder_data, "fips_codes.csv"), stringsAsFactors = F)
h2a_data <- readRDS(paste0(folder_data, "h2a_data_year.rds"))
# nawspad_data
# oews_data
ppi_data <- read.csv(file = paste0(folder_data, "WPU01.csv"), stringsAsFactors = F)
state_border_pairs <- read.csv(file = paste0(folder_data, "state_border_pairs.csv"), stringsAsFactors = F)
border_counties_allmatches <- read.csv(file = paste0(folder_data, "border_counties_allmatches.csv"), stringsAsFactors = F)
census_of_agriculture_cropland <- readRDS(file = paste0(folder_data, "census_ag_cropland_year.rds"))

census_pop_ests <- readRDS(paste0(folder_data, "census_pop_ests_year.rds"))

census_of_agriculture_cropland_base <- readRDS(file = paste0(folder_data, "census_ag_cropland_2007_year.rds"))

# base for full county dataset 

county_df <- readRDS(paste0(folder_data, "county_df_year.rds"))

# CZ

cz_file_small <- readRDS(file = paste0(folder_data, "cz_file_2010_small.rds"))

head(county_df)
head(border_df) # this is the base, we will attache everything here.
head(state_border_pairs) # this is the base, we will attache everything here.
head(border_counties_allmatches) # this is the base, we will attache everything here.

county_df %>% group_by(year) %>% tally()

# merge in for each side:

# by state and year
head(aewr_data)

# by state
head(aewr_regions)

# merge for only one side

# by
# loop ?

# by county and census_period

head(bea_caemp25n_data)
head(bea_cainc45_data)
head(h2a_data)
head(census_pop_ests)

## a tad of prep ---------------------------------------------------------------

class(county_df$fipscounty) 

# make state fips 

county_df <- county_df %>% 
  mutate(statefips = floor(fipscounty/1000))

hist(county_df$statefips) # it worked! 

## both sides merge ----------------------------------------------------------

# fips first 

county_df <- merge(x = county_df, y = fips_codes, by.x = "statefips", by.y = "fips", all.x = T, all.y = F)

county_df <- county_df %>% 
  filter(statefips <= 56) # only states

# first side

names(aewr_data)

dim(county_df)

county_df <- merge(x = county_df, y = aewr_data, by = c("year", "state_abbrev"), all.x = T, all.y = F)

dim(county_df) # DC missing

head(county_df)
# no need to rename these

names(aewr_regions)
dim(county_df)

county_df <- merge(x = county_df, y = aewr_regions, by = c("state_abbrev"), all.x = T, all.y = F)

dim(county_df)

county_df <- county_df %>% 
  rename(countyfips = fipscounty)

datasets <- c( "bea_caemp25n_data", "bea_cainc45_data", "h2a_data", "census_pop_ests", "census_of_agriculture_cropland")

for (i in 1:length(datasets)) {
  print(paste0("Rep ", i))
  temp <- get(datasets[i])
  print(dim(county_df))
  print(dim(temp))
  county_df <- merge(x = county_df, y = temp,
                     by = c("year", "countyfips"),
                     all.x = T, all.y = F)
  rm(temp)
}
dim(county_df)
head(county_df)

# county only #

dim(census_of_agriculture_cropland_base)

county_df <- merge(x = county_df, y = census_of_agriculture_cropland_base, by = "countyfips", all.x = T, all.y = F)

county_df %>%
  group_by(year) %>%
  tally()

county_df %>%
  filter(year == 2008 & !is.na(cropland_acr_2007)) %>%
  count()

## Variable cleaning ## -----------------

# H2A NAs to zero

county_df$nbr_workers_requested_all_years[is.na(county_df$nbr_workers_requested_all_years)] <- 0
county_df$nbr_workers_certified_all_years[is.na(county_df$nbr_workers_certified_all_years)] <- 0
county_df$man_hours_requested_all_years[is.na(county_df$man_hours_requested_all_years)] <- 0
county_df$man_hours_certified_all_years[is.na(county_df$man_hours_certified_all_years)] <- 0
county_df$nbr_applications_all_years[is.na(county_df$nbr_applications_all_years)] <- 0

county_df$nbr_workers_requested_start_year[is.na(county_df$nbr_workers_requested_start_year)] <- 0
county_df$nbr_workers_certified_start_year[is.na(county_df$nbr_workers_certified_start_year)] <- 0
county_df$man_hours_requested_start_year[is.na(county_df$man_hours_requested_start_year)] <- 0
county_df$man_hours_certified_start_year[is.na(county_df$man_hours_certified_start_year)] <- 0
county_df$nbr_applications_start_year[is.na(county_df$nbr_applications_start_year)] <- 0

county_df$nbr_workers_requested_fiscal_year[is.na(county_df$nbr_workers_requested_fiscal_year)] <- 0
county_df$nbr_workers_certified_fiscal_year[is.na(county_df$nbr_workers_certified_fiscal_year)] <- 0
county_df$man_hours_requested_fiscal_year[is.na(county_df$man_hours_requested_fiscal_year)] <- 0
county_df$man_hours_certified_fiscal_year[is.na(county_df$man_hours_certified_fiscal_year)] <- 0
county_df$nbr_applications_fiscal_year[is.na(county_df$nbr_applications_fiscal_year)] <- 0

head(county_df)

# cropland zeros

county_df$cropland_acr[is.na(county_df$cropland_acr)] <- 0
county_df$cropland_acr_2007[is.na(county_df$cropland_acr_2007)] <- 0

# emp pop ratio

county_df <- county_df %>%
  mutate(emp_pop_ratio = emp_tot / pop_census)

# logs of some vars

county_df <- county_df %>%
  mutate(ln_aewr = log(aewr),
         ln_aewr_ppi = log(aewr_ppi),
         ln_aewr_l1 = log(aewr_l1),
         ln_aewr_ppi_l1 = log(aewr_ppi_l1),
         ln_aewr_l2 = log(aewr_l2),
         ln_aewr_ppi_l2 = log(aewr_ppi_l2),
         ln_emp_farm_propr = log(emp_farm_propr),
         ln_emp_farm = log(emp_farm),
         ln_emp_nonfarm = log(emp_nonfarm),
         ln_emp_privatenonfarm = log(emp_privatenonfarm),
         ln_pop_census = log(pop_census),
         ln_cropland_acr = log(cropland_acr),
         ln_cropland_acr_2007 = log(cropland_acr_2007),
         ln_nbr_workers_requested_start_years = log(nbr_workers_requested_start_year),
         ln_nbr_workers_requested_all_years = log(nbr_workers_requested_all_years),
         ln_nbr_applications_fiscal_year = log(nbr_applications_fiscal_year),
         ln1_nbr_workers_requested_start_years = log(nbr_workers_requested_start_year + 1),
         ln1_nbr_workers_requested_all_years = log(nbr_workers_requested_all_years + 1),
         ln1_nbr_applications_fiscal_year = log(nbr_applications_fiscal_year + 1))

# budget shares

county_df <- county_df %>%
  mutate(share_farm_laborexp_prodexp = farm_laborexpense / farm_prodexp,
         share_farm_crop_cashandinc = farm_cashcrops / farm_cashandinc,
         share_farm_animal_cashandinc = farm_cashanimal / farm_cashandinc,
         share_farm_govt_cashandinc = farm_govpayments / farm_cashandinc)

# H2A outcome variables

county_df <- county_df %>%
  mutate(h2a_req_share_farm_workers_start_year = nbr_workers_requested_start_year / emp_farm,
         h2a_cert_share_farm_workers_start_year = nbr_workers_certified_start_year / emp_farm) # add h2a apps per farm later

county_df$h2a_req_share_farm_workers_start_year[is.infinite(county_df$h2a_req_share_farm_workers_start_year)] <- NA
county_df$h2a_cert_share_farm_workers_start_year[is.infinite(county_df$h2a_cert_share_farm_workers_start_year)] <- NA

# add in CZs 

county_df <- merge(x = county_df, y = cz_file_small, by = "countyfips", all.x = T, all.y = F)

# fixed effects

# period

county_df$year_fe <- with(county_df, as.factor(year)) # treats counties in separate clusters as separate

# county

county_df$county_fe <- with(county_df, as.factor(countyfips)) # treats counties in separate clusters as separate

# state

county_df$state_fe <- with(county_df, as.factor(state_abbrev)) # treats counties in separate clusters as separate

# AEWR Region

county_df$aewr_region_fe <- with(county_df, as.factor(aewr_region_num)) # treats counties in separate clusters as separate

# cZ 
county_df$cz_fe <- with(county_df, as.factor(cz_out10)) # treats counties in separate clusters as separate


# CZ x time

county_df$cztime_fe <- with(county_df, interaction(as.factor(cz_out10),  year))

levels(county_df$cztime_fe) <- c(levels(county_df$cztime_fe),"0.0")

unique(county_df$year)

county_df$cztime_fe[county_df$year == 2008] <- "0.0" # first period is 0

# aewr region  x time

county_df$aewrregtime_fe <- with(county_df, interaction(as.factor(aewr_region_num),  year))

levels(county_df$aewrregtime_fe) <- c(levels(county_df$aewrregtime_fe),"0.0")

unique(county_df$year)

county_df$aewrregtime_fe[county_df$year == 2008] <- "0.0" # first period is 0


# sample restriction to counties with cropland ------------------------------------

# we want to drop counties with no cropland in 2007

county_df <- county_df %>%
  mutate(any_cropland_2007 = ifelse(cropland_acr_2007 != 0 & !is.na(cropland_acr_2007), 1, 0))

county_df %>%
  group_by(any_cropland_2007, year) %>%
  tally()

# remove HI, AK, and DC 

county_df <- county_df %>% 
  filter(!is.na(aewr) & !is.na(aewr_region_num))

# check for NAs in key vars ------------------------------------

# use sum(is.na(x))

NA_df <- NULL

dim(county_df)

for (i in 1:length(names(county_df))) {
  temp_na_cnt <- sum(is.na(county_df[,i]))
  temp_varname <- paste0(names(county_df)[i])


  temp <- data.frame(var = temp_varname,
                     full_na_cnt = temp_na_cnt)

  NA_df <- bind_rows(NA_df, temp)
  rm(temp, temp_na_cnt, temp_varname)
}

# easiest / most important to fix
fixcounty <- county_df %>%
  filter(is.na(pop_census))

fixcounty %>%
  group_by(state_abbrev) %>%
  tally() # BEA issue for VA counties. We need to drop bristol city. It will be ok.

## lags of h2a variables


county_df <- county_df %>%
  arrange(county_fe, year) %>%
  group_by(county_fe) %>%
  mutate(nbr_workers_requested_all_years_l1  = lag( nbr_workers_requested_all_years, n = 1, order_by = county_fe),
         nbr_workers_requested_all_years_l2   = lag( nbr_workers_requested_all_years, n = 2, order_by = county_fe),
         nbr_workers_certified_all_years_l1  = lag(nbr_workers_certified_all_years , n = 1, order_by = county_fe),
         nbr_workers_certified_all_years_l2  = lag(nbr_workers_certified_all_years , n = 2, order_by = county_fe),
         man_hours_requested_all_years_l1  = lag(man_hours_requested_all_years , n = 1, order_by = county_fe),
         man_hours_requested_all_years_l2  = lag(man_hours_requested_all_years , n = 2, order_by = county_fe),
         man_hours_certified_all_years_l1  = lag(man_hours_certified_all_years , n = 1, order_by = county_fe),
         man_hours_certified_all_years_l2  = lag(man_hours_certified_all_years , n = 2, order_by = county_fe),
         nbr_applications_all_years_l1 = lag(nbr_applications_all_years , n = 1, order_by = county_fe),
         nbr_applications_all_years_l2 = lag(nbr_applications_all_years , n = 2, order_by = county_fe),
         nbr_workers_requested_start_year_l1  = lag(nbr_workers_requested_start_year , n = 1, order_by = county_fe),
         nbr_workers_requested_start_year_l2  = lag(nbr_workers_requested_start_year , n = 2, order_by = county_fe),
         nbr_workers_certified_start_year_l1  = lag(nbr_workers_certified_start_year , n = 1, order_by = county_fe),
         nbr_workers_certified_start_year_l2  = lag(nbr_workers_certified_start_year , n = 2, order_by = county_fe),
         man_hours_requested_start_year_l1  = lag(man_hours_requested_start_year , n = 1, order_by = county_fe),
         man_hours_requested_start_year_l2  = lag(man_hours_requested_start_year , n = 2, order_by = county_fe),
         man_hours_certified_start_year_l1  = lag(man_hours_certified_start_year , n = 1, order_by = county_fe),
         man_hours_certified_start_year_l2  = lag(man_hours_certified_start_year , n = 2, order_by = county_fe),
         nbr_applications_start_year_l1 = lag(nbr_applications_start_year , n = 1, order_by = county_fe),
         nbr_applications_start_year_l2 = lag(nbr_applications_start_year , n = 2, order_by = county_fe),
         nbr_workers_requested_fiscal_year_l1  = lag(nbr_workers_requested_fiscal_year , n = 1, order_by = county_fe),
         nbr_workers_requested_fiscal_year_l2  = lag(nbr_workers_requested_fiscal_year , n = 2, order_by = county_fe),
         nbr_workers_certified_fiscal_year_l1  = lag(nbr_workers_certified_fiscal_year , n = 1, order_by = county_fe),
         nbr_workers_certified_fiscal_year_l2  = lag(nbr_workers_certified_fiscal_year , n = 2, order_by = county_fe),
         man_hours_requested_fiscal_year_l1  = lag(man_hours_requested_fiscal_year , n = 1, order_by = county_fe),
         man_hours_requested_fiscal_year_l2  = lag(man_hours_requested_fiscal_year , n = 2, order_by = county_fe),
         man_hours_certified_fiscal_year_l1  = lag(man_hours_certified_fiscal_year , n = 1, order_by = county_fe),
         man_hours_certified_fiscal_year_l2  = lag(man_hours_certified_fiscal_year , n = 2, order_by = county_fe),
         nbr_applications_fiscal_year_l1 = lag(nbr_applications_fiscal_year , n = 1, order_by = county_fe),
         nbr_applications_fiscal_year_l2 = lag(nbr_applications_fiscal_year , n = 2, order_by = county_fe))

# new variables 

names(county_df)

# cuts by 2008 h2a usage 

h2a_use_df <- county_df %>% 
  ungroup() %>% 
  filter(year == 2008)

dim(h2a_use_df)

hist(h2a_use_df$nbr_workers_requested_start_year)
summary(h2a_use_df$nbr_workers_requested_start_year)
summary(h2a_use_df$h2a_req_share_farm_workers_start_year)

# cut by count

count_cuts <- quantile(h2a_use_df$nbr_workers_requested_start_year, probs = c(.5, .66, .75))

# cut by share 

share_cuts <- quantile(h2a_use_df$h2a_req_share_farm_workers_start_year, probs = c(.5, .66, .75), na.rm = T)


h2a_use_df <- h2a_use_df %>% 
  mutate(high_h2a_count_50 = ifelse(nbr_workers_requested_start_year > count_cuts[1], 1, 0),
         high_h2a_count_66 = ifelse(nbr_workers_requested_start_year > count_cuts[2], 1, 0),
         high_h2a_count_75 = ifelse(nbr_workers_requested_start_year > count_cuts[3], 1, 0),
         high_h2a_share_50 = ifelse(h2a_req_share_farm_workers_start_year > share_cuts[1] & !is.na(h2a_req_share_farm_workers_start_year), 1, 0),
         high_h2a_share_66 = ifelse(h2a_req_share_farm_workers_start_year > share_cuts[2] & !is.na(h2a_req_share_farm_workers_start_year), 1, 0),
         high_h2a_share_75 = ifelse(h2a_req_share_farm_workers_start_year > share_cuts[3] & !is.na(h2a_req_share_farm_workers_start_year), 1, 0))

h2a_use_df %>% 
  group_by(high_h2a_count_50) %>% 
  tally()

h2a_use_df <- h2a_use_df %>% 
  select(countyfips, high_h2a_count_50, high_h2a_count_66, high_h2a_count_75, high_h2a_share_50, high_h2a_share_66, high_h2a_share_75)

head(h2a_use_df)

dim(h2a_use_df)

county_df <- merge(x = county_df, y = h2a_use_df, by = "countyfips", all.x = T, all.y = F)

names(county_df) #check
# year dummys

summary(county_df$year)

county_df <- county_df  %>% 
  mutate(yeardummy_2008 = ifelse(year == 2008, 1, 0),
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
         yeardummy_2022 = ifelse(year == 2022, 1, 0))


# ID border CZs

cz_borders <- county_df %>% 
  group_by(cz_out10) %>% 
  summarise(AEWRregmin = min(aewr_region_num, na.rm = T),
            AEWRregmax = max(aewr_region_num, na.rm = T))

cz_borders <- cz_borders %>% 
  mutate(border_cz = ifelse(AEWRregmin != AEWRregmax, 1, 0))

cz_borders %>% group_by(border_cz) %>% tally()

county_df <- merge(x = county_df, y = cz_borders, by = "cz_out10", all.x = T, all.y = F)

names(county_df) #check


# pre post dummy 

county_df <- county_df %>% 
  mutate(postdummy = ifelse(year > 2012, 1, 0))

saveRDS(county_df, file = paste0(folder_data, "county_df_analysis_year.rds"))

## Remove files ## -------------------------------------------------------------

str_detect(ls(), "folder_")

objects <- data.frame(name = ls(), keep = str_detect(ls(), "folder_")) %>%
  filter(keep == F)

rm(list = objects[,1])
gc()
rm(objects)

