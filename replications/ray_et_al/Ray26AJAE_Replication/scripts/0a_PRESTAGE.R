# ===========================================================================
#  This script:
#   1: Defines various crosswalks, 
#       reads in GRIDs from SIMPLE-G database, 
#       FIPS, Commuting Zones (CZs) & State names and Production regions,
#   2: Pulls data from various data sources (listed in Workflow.txt)
#   3: Output files are saved in Replication/temp/
#  All input files in folder Replication/in/<name_of_source>/<file_name>
# ===========================================================================

cat("Starting 0a_PRESTAGE.R...\n")

# 0.   Libraries ----
cat("[0] Loading libraries...\n")
library(readxl)
library(tidyr)
library(writexl)      
library(sf)
library(ggplot2)
library(RColorBrewer)
library(viridis)
library(scales)
library(cowplot)
library(ggrepel)
library(dplyr) 
library(ipumsr)
library(readstata13)
library(stringr) #str_sub
library(patchwork)
library(tidyr)
library(HARr)
library(doParallel)
library(doRNG)
options(warn = -1)
options(dplyr.summarise.inform = FALSE)

cat("Libraries loaded.\n")

# 1.1: Read in crosswalks ----
cat("[1.1] Reading in crosswalks (FIPS, CZ, State, Production Region mappings)...\n")
GRID2FIPS <- read_excel(file.path("in/Crosswalk/GRID_FIPS_STATE.xlsx"))
FIPSinSIMPLE    <- GRID2FIPS %>%
  dplyr::select(FIPS) %>%
  unique() %>%
  mutate(FIPS_ed = sprintf("%05d", FIPS))

STATE     <- GRID2FIPS %>% 
  dplyr::select(FIPS, STATE_CODE) %>%
  unique()

STATENAME   <- read_excel(file.path("in/Crosswalk/STATEFIPS2NAME.xlsx")) %>%
  filter(!STATEFIPS %in% c(02,15))

FIPS2PR   <- read_excel(file.path("in/Crosswalk/FIPS2PR_071024.xlsx")) %>%
  dplyr:: select(FIPS, PR) %>%
  rename(PR9 = PR) %>% 
  rbind(c(FIPS = 8014, PR9 = 4)) %>%
  filter(FIPS!=12025) %>% # 12086, updated FIPS for 12025 included
  unique()

FIPS2CZ   <- read_excel(file.path("in/Crosswalk/FIPS2CZ_251024.xlsx"))

CZ2PR <- read_excel(file.path("in/Crosswalk/CZ2PR.xlsx")) %>%
  rename(PR9 = PR) %>%
  filter(CZ %in% c(1:605))

CZ_names <- read_excel(file.path("in/Crosswalk/CZ_names.xlsx"))
CZ_names <- CZ_names[1:605, ]
cz = c(CZ_names$CZ)

# Production change at PR level
GRID2PR     <- read_excel(file.path("in/Crosswalk/GRID2PR.xlsx"))
GRID2PR     <- subset(GRID2PR, select=c("SNO", "PR"))
GRID2PR     <- merge(GRID2PR, subset(GRID2FIPS, select=c("SNO", "FIPS")), by=c("SNO"))

FIPS_names <- paste0("FP", as.character(FIPSinSIMPLE$FIPS_ed))
FIPS_reg = c(FIPS_names, 'E_Euro',  'N_Afr', 'SSA',     'S_Amer', 'AUS_NZ',  'EU',     'S_Asia', 
             'CC_Amer', 'S_Afr', 'SE_Asia', 'CAN',    'CHN_MNG', 'M_East', 'JPN_KR', 'C_Asia')

cz_reg = c(CZ_names$CZ, 'E_Euro',  'N_Afr', 'SSA',     'S_Amer', 'AUS_NZ',  'EU',     'S_Asia', 
           'CC_Amer', 'S_Afr', 'SE_Asia', 'CAN',    'CHN_MNG', 'M_East', 'JPN_KR', 'C_Asia')

numfips <- length(FIPS_names)

GRID2LS <- read_excel(file.path("in/Crosswalk/GRID2FIPS2LS.xlsx"))
GRID2CZ <- read_excel(file.path("in/Crosswalk/GRID2FIPS2CZ.xlsx"))

# Creating a data frame with numeric and character columns directly
PR <- data.frame(
  PR9 = as.numeric(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
  pr = c("Heartland", "Northern Crescent", "Northern Great Plain", "Prairie Gateway", 
         "Eastern Uplands", "Southern Seaboard", "Fruitful Rim", 
         "Basin & Range", "Mississippi Portal", "US"))

pr_names = c("Heartland", "NorthCrescen", "NorthGreatPl", "PrairieGatew", "EastUplands", "SouthSeabr", "FruitfulRim", "BasinaRange", "MississippiP",
             "E_Euro", "N_Afr", "SSA", "S_Amer", "AUS_NZ", " EU", "S_Asia", "CC_Amer", "S_Afr", "SE_Asia", "CAN", "CHN_MNG", "M_East", "JPN_KR", "C_Asia")

# Display the data frame
# print(PR)

# 1.2: Estimate Input Cost-Shares ----
cat("[1.2] Estimating input cost shares from SIMPLE-G GRID/CZ data...\n")
GRIDDATA2010 <- read_har(file.path("in/GRIDDATA.har"),                  toLowerCase = F)

data2010     <- read_har(file.path("in/CZDATA.har"), toLowerCase = F)
shares2010   <- cbind(CZ=1:605, VLND = data2010$VLND[1:605], VWTR = data2010$VWTR[1:605], VWOR = data2010$VWOR[1:605], 
                              VLAB = data2010$VLAB[1:605], VNLB = data2010$VNLB[1:605], VNIT = data2010$VNIT[1:605],
                              QNIT = data2010$QNIT[1:605], QLAB = data2010$QLAB[1:605], QLND = data2010$QLND[1:605]) %>%
  as.data.frame() %>%
  mutate(VLW = VLND + VWTR + VWOR) %>%
  mutate(VW  = VWTR + VWOR) %>%
  mutate(VCROP= VLW + VLAB + VNLB) %>%
  dplyr:: select(CZ, VCROP, VLAB, VNLB, VLW, VW, VNIT, QLAB, QLND) %>%
  left_join(CZ2PR) %>%
  group_by(PR9) %>%
  mutate(VCROPa = sum(VCROP, na.rm = TRUE),
         VLABa = sum(VLAB, na.rm = TRUE),
         VLWa  = sum(VLW,  na.rm = TRUE),
         VNLBa = sum(VNLB, na.rm = TRUE),
         VNITa = sum(VNIT, na.rm = TRUE),
         QLABa = sum(QLAB, na.rm = TRUE),
         QLNDa = sum(QLND,  na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(vshr_crp = VCROP / VCROPa,
         vshr_lab = VLAB / VLABa,
         vshr_nlb = VNLB / VNLBa,
         vshr_lw  = VLW  / VLWa,
         qshr_lab = QLAB / QLABa,
         qshr_lnd = QLND / QLNDa) %>%
  mutate(vshr_nit = VNIT / rowSums(across(c(VLW, VLAB, VNLB, VNIT)))) 

# 1.3: Price indices ----
cat("[1.3] Building price indices (CPI adjustments, crop price changes)...\n")
# USDA data on Commodity Prices, recent reports:
# https://usda.library.cornell.edu/concern/publications/c821gj76b?locale=en&page=2&release_end_date=2012-12-31&release_start_date=2010-01-06#release-items
# Historical data from USDA QUICKSTAT
# adjust for inflation from 2011 dollars 
price_ind_quickst <- read_excel(file.path("in/USDA_QUICKSTAT/prices_data_allgroups.xlsx")) %>%
  mutate(Value = as.numeric(gsub(",","",Value))) %>%
  mutate(cat = ifelse(Commodity=="CROP TOTALS", 1,
                      ifelse(Commodity == "FEED GRAINS", 2,
                             ifelse(Commodity == "FOOD GRAINS", 3,
                                    ifelse(Commodity == "OIL-BEARING CROPS", 4,
                                           ifelse(Commodity=="FRUIT & TREE NUT TOTALS", 5, 6)))))) %>%
  subset( select=c("Year", "cat", "Value")) %>%
  pivot_wider(id_cols = Year, names_from = cat, values_from = Value, names_prefix = "group_")

crop_longterm_avg02 <- price_ind_quickst %>%
  filter(Year %in% 2000:2004) %>%  # Select rows for years 1997-2007
  dplyr::select(-Year) %>%  
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)))  %>% # Calculate average for numeric columns
  mutate(Year = 2002)

crop_longterm_avg17 <- price_ind_quickst %>%
  filter(Year %in% 2015:2019) %>%  
  dplyr::select(-Year) %>%  # Exclude Year column
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)))  %>% # Calculate average for numeric columns
  mutate(Year = 2017)

prices_crops <- bind_rows(crop_longterm_avg02, crop_longterm_avg17) %>%
  pivot_wider(names_from = "Year", values_from = everything()) 

p.crops0217  = (prices_crops$group_1_2017 - prices_crops$group_1_2002) / prices_crops$group_1_2002*100
p.feedgr0217 = (prices_crops$group_2_2017 - prices_crops$group_2_2002) / prices_crops$group_2_2002*100
p.foodgr0217 = (prices_crops$group_3_2017 - prices_crops$group_3_2002) / prices_crops$group_3_2002*100
p.oilgr0217  = (prices_crops$group_4_2017 - prices_crops$group_4_2002) / prices_crops$group_4_2002*100
p.fn0217     = (prices_crops$group_5_2017 - prices_crops$group_5_2002) / prices_crops$group_5_2002*100
p.vg0217     = (prices_crops$group_6_2017 - prices_crops$group_6_2002) / prices_crops$group_6_2002*100

# https://www.bls.gov/data/inflation_calculator.htm
# updated 10/13/25
cpi9702 <- 112.73 
cpi0702 <- 86.73  
cpi0710 <- 113.95
cpi1012 <- 106.75
cpi1217 <- 106.99 
cpi2217 <- 82.86  
cpi0217 <- 137.12 

cpi_table <- tibble(Year       = c(1997,      2002,    2007,       2012,        2017,    2022),
                    CPI_to2002 = c(cpi9702/100, 100/100, cpi0702/100, NA,         NA,      NA),   # 1997→2002, 2007→2002
                    CPI_to2017 = c(NA,          NA,      NA,         cpi1217/100, 100/100, cpi2217/100))    # 2012→2017, 2022→2017

# cpi for wages (FLS)
cpi9597 <- 95.26
cpi9697 <- 98.01
cpi9897 <- 101.81
cpi9997 <- 104.11

cpi0002 <- 95.95
cpi0102 <- 98.56
cpi0302 <- 102.11
cpi0402 <- 105.16

cpi0507 <- 94.29
cpi0607 <- 97.89
cpi0807 <- 105.18
cpi0907 <- 103.62

# cpi1012 <- defined earlier
cpi1112 <- 98.88
cpi1312 <- 102.08
cpi1412 <- 103.82

cpi1517 <- 97.49
cpi1617 <- 98.3
cpi1817 <- 102.95
cpi1917 <- 104.81

cpi2022 <- 87.73
cpi2122 <- 92.34
cpi2322 <- 103.63
cpi2422 <- 106.25

# retail gasoline price: https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=pet&s=emm_epm0_pte_nus_dpg&f=m
PFUEL2002 = mean(c(1.148,	1.155,	1.289,	1.439,	1.434,	1.424,	1.438,	1.438,	1.441,	1.486,	1.461,	1.429))
PFUEL2017 = mean(c(2.458,	2.416,	2.437,	2.528,	2.503,	2.460,	2.414,	2.494,	2.761,	2.621,	2.678,	2.594))
p_gas = (PFUEL2017 - (PFUEL2002*cpi0217/100))/(PFUEL2002*cpi0217/100)*100

# 2:   Input and Output data for 2002 and 2017 ----
cat("[2] Building input and output data for 2002 and 2017...\n")
# 2.1: OUTPUT data ----
cat("[2.1] Processing OUTPUT data (crop values and quantities, Ag Census)...\n")
#     i.  Read in VCROP data: Ag Census ----
#           using national feed, food and oilseeds share to calculate food group prices ----
# VCROP reported for foll subgroups:
# BEDDING PLANT TOTALS, BERRY TOTALS, CROP TOTALS, FLORICULTURE TOTALS, FRUIT & TREE NUT TOTALS, HORTICULTURE TOTALS, NURSERY TOTALS, VEGETABLE TOTALS
# using Grains, CROP TOTALS, F&N TOTAL and VEG TOTAL  
# Grains include Feed + Food + Oilseeds

GRAINS_SUB <- read_excel(file.path("in/USDA_QUICKSTAT/grain_subgroups_nat.xlsx")) #2017
t_feed = (GRAINS_SUB$Value[GRAINS_SUB$Commodity == "BARLEY"] +
            GRAINS_SUB$Value[GRAINS_SUB$Commodity == "SORGHUM"] +
            GRAINS_SUB$Value[GRAINS_SUB$Commodity == "CORN"]) / GRAINS_SUB$Value[GRAINS_SUB$Commodity == "GRAIN"] 
t_food = (GRAINS_SUB$Value[GRAINS_SUB$Commodity == "WHEAT"] +
            GRAINS_SUB$Value[GRAINS_SUB$Commodity == "RICE"] ) / GRAINS_SUB$Value[GRAINS_SUB$Commodity == "GRAIN"] 
t_oilseeds = 1 - t_feed - t_food
p.gr.cal = (t_feed * p.feedgr0217) + (t_food * p.foodgr0217) + (t_oilseeds * p.oilgr0217)

# food group prices defined
p_crops = p.crops0217
p_gr = p.gr.cal
p_fn = p.fn0217
p_vg = p.vg0217
#p_ot 
# Other food group price changes - calculated after VCROP

#    ii.  Combine all VCROP data at FIPS level ----
cat("[2.1.2] Start Processing VCROP data from CoA...\n")
VCRP1 <- read_excel(file.path("in/USDA_QUICKSTAT/VCROP_SALES1.xlsx")) %>%
  as.data.frame()

VCRP2 <- read_excel(file.path("in/USDA_QUICKSTAT/VCROP_SALES2.xlsx")) %>%
  as.data.frame()

# Census years 1997 - 2022
VCRP <- rbind(VCRP1, VCRP2)

VCROP_long <- VCRP %>%
  rename(State.ANSI  = `State ANSI`, 
         County.ANSI = `County ANSI`,
         data        = `Data Item`) %>%
  mutate(Value_current_USD = as.numeric(Value)) %>%
  mutate(cat = ifelse(data=="CROP TOTALS - SALES, MEASURED IN $", "VCROP", 
                      ifelse(data=="FRUIT & TREE NUT TOTALS - SALES, MEASURED IN $", "VFN", 
                             ifelse(data=="VEGETABLE TOTALS, INCL SEEDS & TRANSPLANTS, IN THE OPEN - SALES, MEASURED IN $", "VVG", "VGR")))) %>%
  mutate(FIPS = as.numeric(sprintf("%d%03d", State.ANSI,  County.ANSI))) %>%
  subset( select=c("FIPS", "Year", "cat", "Value_current_USD")) %>%
  filter(!is.na(FIPS)) %>%
  #convert Value col to 2002 or 2017 dollars dollars
  left_join(cpi_table) %>%
  filter(Year %in% c(1997,2002,2007, 2012, 2017, 2022)) %>%
  #everything to 02 or 17 prices
  mutate(Value = case_when(Year >=1997 & Year <= 2007 ~ Value_current_USD * CPI_to2002,
                           Year >=2012 & Year <= 2022 ~ Value_current_USD * CPI_to2017)) %>%
  dplyr::select(-starts_with("CPI_to"), -Value_current_USD)

# df <- VCROP_long %>% # id duplicates
#   filter(cat=="VCROP") %>%
#   dplyr::group_by(FIPS, cat, Year) %>%
#   dplyr::summarise(n = dplyr::n(), .groups = "drop") %>%
#   dplyr::filter(n > 1L) 

VCROP_wide_old <- merge( pivot_wider(subset(VCROP_long, cat==1), names_from = Year, values_from = Value, names_prefix = "VCROP") ,
                         pivot_wider(subset(VCROP_long, cat==2), names_from = Year, values_from = Value, names_prefix = "VFN"  ), by=c("FIPS")) %>%
  left_join(         pivot_wider(subset(VCROP_long, cat==3), names_from = Year, values_from = Value, names_prefix = "VVG"  ), by=c("FIPS")) %>%
  left_join(         pivot_wider(subset(VCROP_long, cat==4), names_from = Year, values_from = Value, names_prefix = "VGR"  ), by=c("FIPS"))

VCROP_wide_old <- VCROP_long %>%
  pivot_wider(names_from = c(cat, Year),
              values_from = Value,
              names_glue = "{cat}{Year}")

VCROP <- VCROP_long %>%
  pivot_wider(names_from = c(cat, Year),
              values_from = Value,
              names_glue = "{cat}{Year}") %>%
  as.data.frame() %>%
  filter(!FIPS %in% c(2000:3000,15000:16000)) %>%
  # calculate the longterm average from raw data or assign a small number, 
  mutate(VCROP2002_ltavg_old = coalesce( rowMeans(across(c(VCROP1997, VCROP2002, VCROP2007)), na.rm=T), 50), 
         VGR2002_ltavg       = coalesce( rowMeans(across(c(           VGR2002,   VGR2007))  , na.rm=T), 50), 
         VFN2002_ltavg       = coalesce( rowMeans(across(c(VFN1997,   VFN2002,   VFN2007))  , na.rm=T), 50), 
         VVG2002_ltavg       = coalesce( rowMeans(across(c(           VVG2002,   VVG2007))  , na.rm=T), 50)) %>%
  # calculate other crops as diff, or sum of known crops 
  mutate(VOT2002_ltavg      = ifelse( (VCROP2002_ltavg_old - rowSums(across(c(VGR2002_ltavg, VFN2002_ltavg, VVG2002_ltavg)))) < 0, 50, 
                                      (VCROP2002_ltavg_old - rowSums(across(c(VGR2002_ltavg, VFN2002_ltavg, VVG2002_ltavg)))) ) ) %>%
  
  mutate(VCROP2017_ltavg_old = coalesce( rowMeans(across(c(VCROP2012, VCROP2017, VCROP2022)), na.rm=T),   50), 
         VGR2017_ltavg       = coalesce( rowMeans(across(c(VGR2012,   VGR2017,   VGR2022))    , na.rm=T), 50), 
         VFN2017_ltavg       = coalesce( rowMeans(across(c(VFN2012,   VFN2017,   VFN2022))    , na.rm=T), 50), 
         VVG2017_ltavg       = coalesce( rowMeans(across(c(VVG2012,   VVG2017,   VVG2022))    , na.rm=T), 50)) %>%
  # calculate other crops as diff, or sum of known crops 
  mutate(VOT2017_ltavg      = ifelse( (VCROP2017_ltavg_old - rowSums(across(c(VGR2017_ltavg, VFN2017_ltavg, VVG2017_ltavg)), na.rm=T)) < 0, 50, 
                                      (VCROP2017_ltavg_old - rowSums(across(c(VGR2017_ltavg, VFN2017_ltavg, VVG2017_ltavg)), na.rm=T)) ) ) %>%
  
  # check whether VCROP captures all sub group production data?
  mutate(check2002 = VCROP2002_ltavg_old - rowSums(across(c(VGR2002_ltavg, VFN2002_ltavg, VVG2002_ltavg, VOT2002_ltavg))),
         check2017 = VCROP2017_ltavg_old - rowSums(across(c(VGR2017_ltavg, VFN2017_ltavg, VVG2017_ltavg, VOT2017_ltavg)))) %>%
  # some counties have info of sub-group prod but VCROP=NA, calculate VCROP as sum of subgroups
  mutate(VCROP2002_ltavg = rowSums(across(c(VGR2002_ltavg, VFN2002_ltavg, VVG2002_ltavg, VOT2002_ltavg))),
         VCROP2017_ltavg = rowSums(across(c(VGR2017_ltavg, VFN2017_ltavg, VVG2017_ltavg, VOT2017_ltavg)))) %>%
  ungroup() %>%
  mutate(tgr02    = round(VGR2002_ltavg/VCROP2002_ltavg * 100,1),
         tgr17    = round(VGR2017_ltavg/VCROP2017_ltavg * 100,1),
         # used to aggregate quantity changes
         tgr      = round(rowMeans(across(c(tgr02, tgr17)), na.rm=TRUE), 1),
         tfn02    = round(VFN2002_ltavg/VCROP2002_ltavg * 100,1),
         tfn17    = round(VFN2017_ltavg/VCROP2017_ltavg * 100,1),
         tfn      = round(rowMeans(across(c(tfn02, tfn17)), na.rm=TRUE), 1),
         tvg02    = round(VVG2002_ltavg/VCROP2002_ltavg * 100,1),
         tvg17    = round(VVG2017_ltavg/VCROP2017_ltavg * 100,1),
         tvg      = round(rowMeans(across(c(tvg02, tvg17)), na.rm=TRUE), 1),
         tot02    = round(VOT2002_ltavg/VCROP2002_ltavg * 100,1),
         tot17    = round(VOT2017_ltavg/VCROP2017_ltavg * 100,1),
         tot      = round(rowMeans(across(c(tot02, tot17)), na.rm=TRUE), 1)) 

#           National changes, check and validate against empirical estimates ----
#q_gr = v_gr - p_gr
v_gr = ( sum(VCROP$VGR2017_ltavg, na.rm=T) - (sum(VCROP$VGR2002_ltavg, na.rm=T) * cpi0217/100) ) / (sum(VCROP$VGR2002_ltavg, na.rm=T) * cpi0217/100) * 100
q_gr = v_gr - p.gr.cal

v_fn = ( sum(VCROP$VFN2017_ltavg, na.rm=T) - (sum(VCROP$VFN2002_ltavg, na.rm=T) * cpi0217/100) ) / (sum(VCROP$VFN2002_ltavg, na.rm=T) * cpi0217/100) * 100
q_fn = v_fn - p.fn0217

v_vg = ( sum(VCROP$VVG2017_ltavg, na.rm=T) - (sum(VCROP$VVG2002_ltavg, na.rm=T) * cpi0217/100) ) / (sum(VCROP$VVG2002_ltavg, na.rm=T) * cpi0217/100) * 100
q_vg = v_vg - p.vg0217

v_ot = ( sum(VCROP$VGR2017_ltavg, na.rm=T) - (sum(VCROP$VGR2002_ltavg, na.rm=T) * cpi0217/100) ) / (sum(VCROP$VGR2002_ltavg, na.rm=T) * cpi0217/100) * 100
q_ot = v_ot - p.gr.cal

# national shares
tgr17_us  = sum(VCROP$VGR2017_ltavg, na.rm=T) / sum(VCROP$VCROP2017_ltavg, na.rm=T)
tfn17_us  = sum(VCROP$VFN2017_ltavg, na.rm=T) / sum(VCROP$VCROP2017_ltavg, na.rm=T)
tvg17_us  = sum(VCROP$VVG2017_ltavg, na.rm=T) / sum(VCROP$VCROP2017_ltavg, na.rm=T)
tot17_us  = sum(VCROP$VOT2017_ltavg, na.rm=T) / sum(VCROP$VCROP2017_ltavg, na.rm=T)

# Price changes of other food groups
# calculate price index for residual group of other crops
p_ot = ( p.crops0217 - ( (tgr17_us * p.gr.cal) + (tfn17_us * p.fn0217) + (tvg17_us * p.vg0217) ) ) / tot17_us
p_ot # negative, TFP report estimates 8.8%, quality adjusted?

# changes in q_fn too large relative to national estimates; use tfn to adjust such that weighted sum = 5.2
VFN <- VCROP %>%
  dplyr::select(FIPS, VFN2002_ltavg, VFN2017_ltavg) %>%
  group_by(FIPS) %>%
  summarize(across(everything(), sum, na.rm=T)) %>%
  group_by(FIPS) %>%
  summarize(VFN = rowMeans(across(c(VFN2002_ltavg, VFN2017_ltavg), sum, na.rm=T))) %>%
  ungroup() %>%
  mutate(VFN_tot = sum(VFN)) %>%
  mutate(shr_fn = VFN / VFN_tot)

#           Aggregating to CZ and PR ----
cat("[2.1.2] Aggregate to CZ and PR \n")
# define function (aggregate_VFN) based on the aggregation below
# VFN_LS <- VCROP %>%
#   left_join(FIPS2LS) %>%
#   dplyr::select(LS10, VFN2002_ltavg, VFN2017_ltavg) %>%
#   group_by(LS10) %>%
#   summarize(across(everything(), sum, na.rm=T)) %>%
#   group_by(LS10) %>%
#   summarize(VFN = rowMeans(across(c(VFN2002_ltavg, VFN2017_ltavg), sum, na.rm=T))) %>%
#   ungroup() %>%
#   mutate(VFN_tot = sum(VFN)) %>%
#   mutate(shr_fn = VFN / VFN_tot) %>%
#   filter(!is.na(LS10))

aggregate_VFN <- function(data, join_table, group_var) {
  data %>%
    left_join(join_table, by = "FIPS") %>%
    dplyr::select(all_of(group_var), VFN2002_ltavg, VFN2017_ltavg) %>%
    group_by(across(all_of(group_var))) %>%
    summarize(across(c(VFN2002_ltavg, VFN2017_ltavg), sum, na.rm = TRUE), .groups = "drop") %>%
    mutate(VFN = rowMeans(across(c(VFN2002_ltavg, VFN2017_ltavg)), na.rm = TRUE)) %>%
    mutate(VFN_tot = sum(VFN, na.rm = TRUE)) %>%
    mutate(shr_fn = VFN / VFN_tot) %>%
    filter(!is.na(!!sym(group_var)))  # Drop rows where the group variable is NA
}

# Apply function for different aggregation levels
VFN_CZ <- aggregate_VFN(VCROP, FIPS2CZ, "CZ")
VFN_PR <- aggregate_VFN(VCROP, FIPS2PR, "PR9")

#   iii.  QCROP ----
cat("[2.1.3] Calculate qcrop using qc = vc - pc \n")
# incorporate the FN adjustment to QCROP
QCROP <- VCROP %>%
  mutate(vc   = round(((VCROP2017_ltavg - (VCROP2002_ltavg * cpi0217/100))  / (VCROP2002_ltavg * cpi0217/100) * 100),1), # adjusting 2002 for inflation by 37%
         vgr  = round(((VGR2017_ltavg   - (VGR2002_ltavg   * cpi0217/100))  / (VGR2002_ltavg   * cpi0217/100) * 100),1),
         vfn  = round(((VFN2017_ltavg   - (VFN2002_ltavg   * cpi0217/100))  / (VFN2002_ltavg   * cpi0217/100) * 100),1),
         vvg  = round(((VVG2017_ltavg   - (VVG2002_ltavg   * cpi0217/100))  / (VVG2002_ltavg   * cpi0217/100) * 100),1),
         vot  = round(((VOT2017_ltavg   - (VOT2002_ltavg   * cpi0217/100))  / (VOT2002_ltavg   * cpi0217/100) * 100),1))  %>%
  
  # Trim  outliers (-85 and 200)
  mutate(qc    = round(vc  - p_crops,   1), # change in quantity index without crop specic price changes
         q_gr  = ifelse( vgr==0, 0, ifelse( (vgr - p_gr) > 200, 200, ifelse((vgr - p_gr) < -85, -85, round(vgr - p_gr,1)))),
         q_fn  = ifelse( vfn==0, 0, ifelse( (vfn - p_fn) > 200, 200, ifelse((vfn - p_fn) < -85, -85, round(vfn - p_fn,1)))),
         q_vg  = ifelse( vvg==0, 0, ifelse( (vvg - p_vg) > 200, 200, ifelse((vvg - p_vg) < -85, -85, round(vvg - p_vg,1)))),
         q_ot  = ifelse( vot==0, 0, ifelse( (vot - p_ot) > 200, 200, ifelse((vot - p_ot) < -85, -85, round(vot - p_ot,1)))))

# adjusting national avg weighted by cost share of VFN in a CZ relative to national (shr_fn) to aggregate to 5.2 instead of -61.5 as per vfn - p_fn calculation
target.qfn = 5.2
df.QFN <- QCROP %>%
  dplyr::select(FIPS, q_fn) %>%
  left_join(subset(VFN, select=c(FIPS, shr_fn))) %>%
  mutate(shr_fn = ifelse(is.na(shr_fn), 0, shr_fn))

# find a k, such that RHS = target.qfn
# target.qfn = sum{fips, ( (q_fn + (k*shr_fn)) * shr_fn)}
# target.qfn = sum{fips, ( q_fn* shr_fn + k*shr_fn^2)}

k = ( target.qfn+30 - sum(df.QFN$q_fn * df.QFN$shr_fn) ) / (sum((df.QFN$shr_fn)^2))

df.QFN <- df.QFN %>%
  mutate(adj = ifelse(k*shr_fn > 80, 80, k*shr_fn)) %>%
  rename(q_fn_old = q_fn) %>%
  mutate(q_fn = round( ifelse( q_fn_old + adj < -85, -85, ifelse(q_fn_old + adj > 200, 200, q_fn_old + adj))) ,1)

QCROP <- QCROP %>%
  dplyr::select(-q_fn) %>%
  left_join(subset(df.QFN, select=c(FIPS, q_fn_old, q_fn))) %>%
  mutate(q_agg_old = round((tgr/100 * q_gr) + (tfn/100 * q_fn_old) + (tvg/100 * q_vg) + (tot/100 * q_ot),1),
         q_agg_raw = round((tgr/100 * q_gr) + (tfn/100 * q_fn) + (tvg/100 * q_vg) + (tot/100 * q_ot),1),
         # here replacing >500 with 500
         q_agg = ifelse(q_agg_raw < -85, -85, ifelse(q_agg_raw > 500, 500, q_agg_raw))) %>% # retaining large numbers (200 - 500) here, later trimmed to 200
  # old refs to quality unadjusted
  mutate(vfn_old = vfn,
         vfn = q_fn + p_fn,
         vc_old = vc,
         vc     = (tgr/100*vgr) + (tfn/100*vfn) + (tvg/100*vvg) + (tot/100*vot),
         p_agg  = vc - q_agg) %>%
  # calculate share of each crop type in overall vc and qc
  mutate(t_vgr = tgr/100*vgr,
         t_vfn = tfn/100*vfn,
         t_vvg = tvg/100*vvg,
         t_qot = tot/100*vot,
         t_qgr = tgr/100*q_gr,
         t_qfn = tfn/100*q_fn,
         t_qvg = tvg/100*q_vg,
         t_qot = tot/100*q_ot)  %>%
  filter(!is.na(FIPS))  %>%
  left_join(FIPS2CZ) %>%
  left_join(subset(FIPS2PR, select=c("FIPS", "PR9"))) %>%
  mutate(PR9 = ifelse(is.na(PR9) & FIPS == 46102, 3, PR9))  # 1 county in S.Dakota

#    iv.  Aggregating to CZ and PR ----
#         a.   QC_CZ ----
QC_CZ <- data.frame(CZ = 1:605) %>%
  left_join(VCROP %>% left_join(FIPS2CZ)) %>%
  group_by(CZ) %>%
  summarize(across(c(ends_with("2"), ends_with("7"), ends_with("_ltavg")), sum, na.rm=T)) %>%
  # calculate the longterm average from raw data or assign a small number, 
  mutate(tgr02    = round(VGR2002_ltavg/VCROP2002_ltavg * 100,1),
         tgr17    = round(VGR2017_ltavg/VCROP2017_ltavg * 100,1),
         # used to aggregate quantity changes
         tgr      = round(rowMeans(across(c(tgr02, tgr17)), na.rm=TRUE), 1),
         tfn02    = round(VFN2002_ltavg/VCROP2002_ltavg * 100,1),
         tfn17    = round(VFN2017_ltavg/VCROP2017_ltavg * 100,1),
         tfn      = round(rowMeans(across(c(tfn02, tfn17)), na.rm=TRUE), 1),
         tvg02    = round(VVG2002_ltavg/VCROP2002_ltavg * 100,1),
         tvg17    = round(VVG2017_ltavg/VCROP2017_ltavg * 100,1),
         tvg      = round(rowMeans(across(c(tvg02, tvg17)), na.rm=TRUE), 1),
         tot02    = round(VOT2002_ltavg/VCROP2002_ltavg * 100,1),
         tot17    = round(VOT2017_ltavg/VCROP2017_ltavg * 100,1),
         tot      = round(rowMeans(across(c(tot02, tot17)), na.rm=TRUE), 1)) %>%
  # if this crop was not grown in 2002 & grown in 2017
  mutate(vc   = round(((VCROP2017_ltavg - (VCROP2002_ltavg * cpi0217/100))  / (VCROP2002_ltavg * cpi0217/100) * 100),1), # adjusting 2002 for inflation by 37%
         vgr  = round(((VGR2017_ltavg   - (VGR2002_ltavg   * cpi0217/100))  / (VGR2002_ltavg   * cpi0217/100) * 100),1),
         vfn  = round(((VFN2017_ltavg   - (VFN2002_ltavg   * cpi0217/100))  / (VFN2002_ltavg   * cpi0217/100) * 100),1),
         vvg  = round(((VVG2017_ltavg   - (VVG2002_ltavg   * cpi0217/100))  / (VVG2002_ltavg   * cpi0217/100) * 100),1),
         vot  = round(((VOT2017_ltavg   - (VOT2002_ltavg   * cpi0217/100))  / (VOT2002_ltavg   * cpi0217/100) * 100),1))  %>%
  # when no change in prod or is.na (i.e. for 10 CZs with no production data)
  mutate(qc_old    = ifelse( vc==0  | is.na(vc),  0, round(vc  - p_crops, 1)), # change in quantity index without crop specic price changes
         q_gr_old  = ifelse( vgr==0 | is.na(vgr), 0, round(vgr - p_gr,    1)),
         q_fn_old  = ifelse( vfn==0 | is.na(vfn), 0, round(vfn - p_fn,    1)),
         q_vg_old  = ifelse( vvg==0 | is.na(vvg), 0, round(vvg - p_vg,    1)),
         q_ot_old  = ifelse( vot==0 | is.na(vot), 0, round(vot - p_ot,    1)))

# adjusting FN for quality
df.QFN.CZ <- QC_CZ %>%
  dplyr::select(CZ, q_fn_old) %>%
  left_join(subset(VFN_CZ, select=c(CZ, shr_fn))) %>%
  mutate(q_fn_chch = ifelse(shr_fn < 0.01 & abs(q_fn_old) > 500, 1, 0)) %>%
  # very large, replace the  them to be 200 max
  mutate(q_fn_old = ifelse(q_fn_chch==1, 200, q_fn_old)) %>%
  mutate(shr_fn = ifelse(is.na(shr_fn), 0, shr_fn))

# setting a larger target so that the max adj=k*shr_fn can be capped 
k      = ( target.qfn+25 - sum(df.QFN.CZ$q_fn_old * df.QFN.CZ$shr_fn) ) / (sum((df.QFN.CZ$shr_fn)^2))

df.QFN.CZ <- df.QFN.CZ %>%
  mutate(adj = ifelse(k*shr_fn > 100, 100, k*shr_fn)) %>%
  mutate(q_fn_ed = round( ifelse( q_fn_old + adj < -85, -85, ifelse(q_fn_old + adj > 200, 200, q_fn_old + adj))) ,1)

QC_CZ <- QC_CZ %>%
  left_join(subset(df.QFN.CZ, select=c(CZ, q_fn_ed))) %>%
  # trimming for cases where 0 in 2002 and some in 2017 leading to large q_XX_old
  mutate(q_gr = case_when(q_gr_old > 250 ~ 250, TRUE ~ q_gr_old),
         q_fn = case_when(q_fn_ed  > 250 ~ 250, TRUE ~ q_fn_ed),
         q_vg = case_when(q_vg_old > 250 ~ 250, TRUE ~ q_vg_old),
         q_ot = case_when(q_ot_old > 250 ~ 250, TRUE ~ q_ot_old)) %>%
  mutate(q_agg_old = round((tgr/100 * q_gr_old) + (tfn/100 * q_fn_ed) + (tvg/100 * q_vg_old) + (tot/100 * q_ot_old),1),
         q_agg_raw = round((tgr/100 * q_gr) + (tfn/100 * q_fn) + (tvg/100 * q_vg) + (tot/100 * q_ot),1),
         # here replacing >500 with 500
         q_agg = ifelse(q_agg_raw < -85, -85, ifelse(q_agg_raw > 250, 250, q_agg_raw))) %>%
  # old refs to quality unadjusted
  mutate(vfn_old = vfn,
         vfn = q_fn + p_fn,
         vc_old = vc,
         vc     = (tgr/100*(p_gr + q_gr)) + (tfn/100*(p_fn + q_fn)) + (tvg/100*(p_vg + q_vg)) + (tot/100*(p_ot + q_ot)),
         p_agg  = vc - q_agg) %>%
  # calculate share of each crop type in overall vc and qc
  mutate(t_vgr = tgr/100*vgr,
         t_vfn = tfn/100*vfn,
         t_vvg = tvg/100*vvg,
         t_qot = tot/100*vot,
         t_qgr = tgr/100*q_gr,
         t_qfn = tfn/100*q_fn,
         t_qvg = tvg/100*q_vg,
         t_qot = tot/100*q_ot)  %>%
  filter(!is.na(CZ)) 

#         c.   QC_PR ----
QC_PR <- VCROP %>%
  left_join(FIPS2PR) %>%
  dplyr::select(c(PR9, ends_with("_ltavg"), starts_with("V"), ends_with("old"))) %>%
  group_by(PR9) %>%
  # VCROP for 2002 and 2017
  summarize(across(c(ends_with("2"), ends_with("7"), ends_with("_ltavg"), ends_with("old")), sum, na.rm=T)) %>%
  # check whether VCROP captures all sub group production data?
  mutate(check2002 = VCROP2002_ltavg_old - rowSums(across(c(VGR2002_ltavg, VFN2002_ltavg, VVG2002_ltavg, VOT2002_ltavg))),
         check2017 = VCROP2017_ltavg_old - rowSums(across(c(VGR2017_ltavg, VFN2017_ltavg, VVG2017_ltavg, VOT2017_ltavg)))) %>%
  # some counties have info of sub-group prod but VCROP=NA, calculate VCROP as sum of subgroups
  mutate(VCROP2002_ltavg = rowSums(across(c(VGR2002_ltavg, VFN2002_ltavg, VVG2002_ltavg, VOT2002_ltavg))),
         VCROP2017_ltavg = rowSums(across(c(VGR2017_ltavg, VFN2017_ltavg, VVG2017_ltavg, VOT2017_ltavg)))) %>%
  mutate(tgr02    = round(VGR2002_ltavg/VCROP2002_ltavg * 100,1),
         tgr07    = round(VGR2007/VCROP2007 * 100,1),
         tgr12    = round(VGR2012/VCROP2012 * 100,1),
         tgr17    = round(VGR2017_ltavg/VCROP2017_ltavg * 100,1),
         tgr22    = round(VGR2022/VCROP2022 * 100,1),
         
         tfn02    = round(VFN2002_ltavg/VCROP2002_ltavg * 100,1),
         tfn07    = round(VFN2007/VCROP2007 * 100,1),
         tfn12    = round(VFN2012/VCROP2012 * 100,1),
         tfn17    = round(VFN2017_ltavg/VCROP2017_ltavg * 100,1),
         tfn22    = round(VFN2022/VCROP2022 * 100,1),
         
         tvg02    = round(VVG2002_ltavg/VCROP2002_ltavg * 100,1),
         tvg07    = round(VVG2007/VCROP2007 * 100,1),
         tvg12    = round(VVG2012/VCROP2012 * 100,1),
         tvg17    = round(VVG2017/VCROP2017 * 100,1),
         tvg22    = round(VVG2022/VCROP2022 * 100,1),
         
         tot02    = round(VOT2002_ltavg/VCROP2002_ltavg * 100,1),
         tot07    = round(100 - rowSums(across(c(tgr07, tfn07, tvg07)))),
         tot12    = round(100 - rowSums(across(c(tgr12, tfn12, tvg12)))),
         tot17    = round(VOT2017_ltavg/VCROP2017_ltavg * 100,1),
         tot22    = round(100 - rowSums(across(c(tgr22, tfn22, tvg22)))),
         
         tgr      = round(rowMeans(across(c(tgr02, tgr17)), na.rm=TRUE), 1),
         tfn      = round(rowMeans(across(c(tfn02, tfn17)), na.rm=TRUE), 1),
         tvg      = round(rowMeans(across(c(tvg02, tvg17)), na.rm=TRUE), 1),
         tvg      = round(rowMeans(across(c(tvg02, tvg17)), na.rm=TRUE), 1),
         tot      = round(rowMeans(across(c(tot02, tot17)), na.rm=TRUE), 1)) %>%
  # if this crop was not grown in 2002 & grown in 2017
  mutate(vc   = round(((VCROP2017_ltavg - (VCROP2002_ltavg * cpi0217/100))  / (VCROP2002_ltavg * cpi0217/100) * 100),1), # adjusting 2002 for inflation by 37%
         vgr  = round(((VGR2017_ltavg   - (VGR2002_ltavg   * cpi0217/100))  / (VGR2002_ltavg   * cpi0217/100) * 100),1),
         vfn  = round(((VFN2017_ltavg   - (VFN2002_ltavg   * cpi0217/100))  / (VFN2002_ltavg   * cpi0217/100) * 100),1),
         vvg  = round(((VVG2017_ltavg   - (VVG2002_ltavg   * cpi0217/100))  / (VVG2002_ltavg   * cpi0217/100) * 100),1),
         vot  = round(((VOT2017_ltavg   - (VOT2002_ltavg   * cpi0217/100))  / (VOT2002_ltavg   * cpi0217/100) * 100),1))  %>%
  # when no change in prod or is.na (i.e. for 10 CZs with no production data)
  mutate(qc_old    = ifelse( vc==0  | is.na(vc),  0, round(vc  - p_crops, 1)), # change in quantity index without crop specic price changes
         q_gr_old  = ifelse( vgr==0 | is.na(vgr), 0, round(vgr - p_gr,    1)),
         q_fn_old  = ifelse( vfn==0 | is.na(vfn), 0, round(vfn - p_fn,    1)),
         q_vg_old  = ifelse( vvg==0 | is.na(vvg), 0, round(vvg - p_vg,    1)),
         q_ot_old  = ifelse( vot==0 | is.na(vot), 0, round(vot - p_ot,    1)))

# adjusting FN for quality
df.QFN.PR <- QC_PR %>%
  dplyr::select(PR9, q_fn_old) %>%
  left_join(subset(VFN_PR, select=c(PR9, shr_fn))) %>%
  mutate(shr_fn = ifelse(is.na(shr_fn), 0, shr_fn))

k = ( target.qfn - sum(df.QFN.PR$q_fn_old * df.QFN.PR$shr_fn) ) / (sum((df.QFN.PR$shr_fn)^2))
cat("Solved adjustment factor k (PR-level FN quality target):", round(k, 3), "\n")

df.QFN.PR <- df.QFN.PR %>%
  mutate(adj = k*shr_fn) %>%
  #dplyr::select(-q_fn_old) %>%
  mutate(q_fn_ed = round( ifelse( q_fn_old + adj < -85, -85, ifelse(q_fn_old + adj > 200, 200, q_fn_old + adj))) ,1)

QC_PR <- QC_PR %>%
  #dplyr::select(-q_fn) %>%
  left_join(subset(df.QFN.PR, select=c(PR9, q_fn_old, q_fn_ed))) %>%
  # trimming for cases where 0 in 2002 and some in 2017 leading to large q_XX_old
  mutate(q_gr = case_when(q_gr_old > 250 ~ 250, TRUE ~ q_gr_old),
         q_fn = case_when(q_fn_ed  > 250 ~ 250, TRUE ~ q_fn_ed),
         q_vg = case_when(q_vg_old > 250 ~ 250, TRUE ~ q_vg_old),
         q_ot = case_when(q_ot_old > 250 ~ 250, TRUE ~ q_ot_old)) %>%
  mutate(q_agg_old = round((tgr/100 * q_gr_old) + (tfn/100 * q_fn_ed) + (tvg/100 * q_vg_old) + (tot/100 * q_ot_old),1),
         q_agg_raw = round((tgr/100 * q_gr) + (tfn/100 * q_fn) + (tvg/100 * q_vg) + (tot/100 * q_ot),1),
         # here replacing >500 with 500
         q_agg = ifelse(q_agg_raw < -85, -85, ifelse(q_agg_raw > 250, 250, q_agg_raw))) %>%
  # old refs to quality unadjusted
  mutate(vfn_old = vfn,
         vfn = q_fn + p_fn,
         vc_old = vc,
         vc     = (tgr/100*(p_gr + q_gr)) + (tfn/100*(p_fn + q_fn)) + (tvg/100*(p_vg + q_vg)) + (tot/100*(p_ot + q_ot)),
         p_agg  = vc - q_agg) %>%
  # calculate share of each crop type in overall vc and qc
  mutate(t_vgr = tgr/100*vgr,
         t_vfn = tfn/100*vfn,
         t_vvg = tvg/100*vvg,
         t_qot = tot/100*vot,
         t_qgr = tgr/100*q_gr,
         t_qfn = tfn/100*q_fn,
         t_qvg = tvg/100*q_vg,
         t_qot = tot/100*q_ot)  %>%
  filter(PR9 %in% c(1:9)) %>%
  # removing these two to be consistent with exports to excel and Tables.lsx
  dplyr::select(-q_fn_old, -vfn_old, -q_agg_old, -q_agg_raw, -vfn_old, -vc_old) %>%
  left_join(PR)


# 2.2:  LABOR data ----
cat("[2.2] Processing LABOR data (wages, H-2A, hours: FLS & NAWS)...\n")
LAB_CROP <- read_excel(file.path("in/USDA_QUICKSTAT/coa_state_pct_crop_animal.xlsx")) %>%
  mutate(State.ANSI = as.numeric(stateansi))

#     i.  a.Reading in Wages - FLS ----
cat("FLR: Farm Labor Regions for FLS wage estimates \n")
ST2FLR     <- read_excel(file.path("in/FarmLaborSurvey/ST2FLR.xlsx")) %>%
  rename(FLR = FARM_LABOR_REGION) %>%
  dplyr::select(FLR, STATE_NAME) %>%
  full_join(STATENAME) %>%
  rename(STATE_CODE = STATEFIPS) %>%
  dplyr::select(STATE_CODE, FLR)

HWAGE_ST97 <- read_excel(file.path("in/FarmLaborSurvey/Wage_FLS97.xlsx")) %>% # in 2002$
  rename(STATE_CODE = `State ANSI`,
         HWage = Value) %>%
  dplyr::select(STATE_CODE, Year, HWage) %>%
  # adjusting to current dollars
  mutate(HWage =  ifelse(Year==1995, HWage * (100/cpi9597),
                         ifelse(Year==1996, HWage * (100/cpi9697),
                                ifelse(Year==1998, HWage * (cpi9897/100),
                                       ifelse(Year==1999, HWage * (cpi9997/100), HWage))))) %>%
  group_by(STATE_CODE) %>%
  summarize(HWage97 = mean(HWage)) 

HWAGE_ST02 <- read_excel(file.path("in/FarmLaborSurvey/Wage_FLS.xlsx")) %>% # in 2002$
  rename(STATE_CODE = `State ANSI`,
         HWage = Value) %>%
  dplyr::select(STATE_CODE, Year, HWage) %>%
  filter(Year<=2006) %>% # FLS does not report state level averages from 2007, only CA, HI and FL there on out
  # adjusting to current dollars
  mutate(HWage =  ifelse(Year==2000, HWage * (100/cpi0002),
                         ifelse(Year==2001, HWage * (100/cpi0102),
                                ifelse(Year==2003, HWage * (cpi0302/100),
                                       ifelse(Year==2004, HWage * (cpi0402/100), HWage))))) %>%
  group_by(STATE_CODE) %>%
  summarize(HWage02 = mean(HWage)) 

HWAGE_ST07121722 <- read_excel(file.path("in/FarmLaborSurvey/Wage_FLS2005_2024.xlsx")) %>% # in 2017$
  rename(HWage = Value,
         FLR = Region) %>%
  dplyr::select(FLR, Year, HWage) %>%
  # adjusting to current dollars
  mutate(HWage07 = ifelse(Year==2005, HWage * (100/cpi0507),
                          ifelse(Year==2006, HWage * (100/cpi0607),
                                 ifelse(Year==2008, HWage * (cpi0807/100),
                                        ifelse(Year==2009, HWage * (cpi0907/100), HWage)))),
         HWage12 = ifelse(Year==2010, HWage * (100/cpi1012),
                          ifelse(Year==2011, HWage * (100/cpi1112),
                                 ifelse(Year==2013, HWage * (cpi1312/100),
                                        ifelse(Year==2014, HWage * (cpi1412/100), HWage)))),
         HWage17 = ifelse(Year==2015, HWage * (100/cpi1517),
                          ifelse(Year==2016, HWage * (100/cpi1617),
                                 ifelse(Year==2018, HWage * (cpi1817/100),
                                        ifelse(Year==2019, HWage * (cpi1917/100), HWage)))),
         HWage22 = ifelse(Year==2020, HWage * (100/cpi2022),
                          ifelse(Year==2021, HWage * (100/cpi2122),
                                 ifelse(Year==2023, HWage * (cpi2322/100),
                                        ifelse(Year==2024, HWage * (cpi2422/100), HWage))))) 

HWAGE_ST07 <- HWAGE_ST07121722 %>%
  dplyr::select(FLR, Year, HWage07) %>%
  filter(Year<=2009) %>%
  full_join(ST2FLR) %>%
  group_by(STATE_CODE) %>%
  summarize(HWage07 = mean(HWage07, na.rm=T)) 

HWAGE_ST12 <- HWAGE_ST07121722 %>%
  dplyr::select(FLR, Year, HWage12) %>%
  filter(Year>2009 & Year<=2014) %>%
  full_join(ST2FLR) %>%
  group_by(STATE_CODE) %>%
  summarize(HWage12 = mean(HWage12, na.rm=T)) 

HWAGE_ST17 <- HWAGE_ST07121722 %>%
  dplyr::select(FLR, Year, HWage17) %>%
  filter(Year>2014 & Year<=2019) %>%
  full_join(ST2FLR) %>%
  group_by(STATE_CODE) %>%
  summarize(HWage17 = mean(HWage17, na.rm=T)) 

HWAGE_ST22 <- HWAGE_ST07121722 %>%
  dplyr::select(FLR, Year, HWage22) %>%
  filter(Year>2019 & Year<=2024) %>%
  full_join(ST2FLR) %>%
  group_by(STATE_CODE) %>%
  summarize(HWage22 = mean(HWage22, na.rm=T)) 

HWAGE_ST_FLS <- HWAGE_ST97 %>%
  full_join(HWAGE_ST02) %>%
  full_join(HWAGE_ST07) %>%
  full_join(ST2FLR) %>%
  group_by(FLR) %>%
  mutate(W97_flr = mean(HWage97, na.rm=T),
         W02_flr = mean(HWage02, na.rm=T)) %>%
  ungroup() %>%
  mutate(HWage97 = ifelse(is.na(HWage97), W97_flr, HWage97),
         HWage02 = ifelse(is.na(HWage02), W02_flr, HWage02)) %>%
  dplyr::select(-W97_flr, -W02_flr) %>%
  full_join(HWAGE_ST12) %>%
  full_join(HWAGE_ST17) %>%
  full_join(HWAGE_ST22) %>%
  filter(!STATE_CODE %in% c(02,15, NA))

#         b.Reading in Wages - NAWS ----
cat("NAWS wage estimates \n")
HWAGE_PR_wide <- read_xlsx(file.path("in/NAWS/fwage_PRlevel_pooled.xlsx")) %>% # in 2017$
  rename(PR9 = PR,
         mean_wage = waget1_mean,
         se_wage   = waget1_se) %>%
  # adjusting to current dollars
  mutate(mean_wage = ifelse(year==2002, mean_wage/1.37,
                            ifelse(year==2010, mean_wage*.8898, mean_wage))) %>%
  left_join(PR) %>%
  dplyr::select(-PR_name) %>%
  arrange(PR9, year) %>%
  filter(!is.na(mean_wage))

HWAGE_PR_long <- HWAGE_PR_wide %>%
  mutate(CI95_low_wage  = round(mean_wage - (1.96*se_wage),1),
         CI95_high_wage = round(mean_wage + (1.96*se_wage),1)) %>%
  dplyr::select(-c(se_wage,N, pr))

HWAGE_PR_NAWS <- HWAGE_PR_long %>%
  pivot_wider(names_from = year, values_from = c(mean_wage, CI95_low_wage, CI95_high_wage)) %>%
  rename(HWage02       = mean_wage_2002,
         HWage17       = mean_wage_2017,
         CI95L_HWage02 = CI95_low_wage_2002,
         CI95L_HWage17 = CI95_low_wage_2017,
         CI95H_HWage02 = CI95_high_wage_2002,
         CI95H_HWage17 = CI95_high_wage_2017) %>%
  arrange(PR9) 
# save(HWAGE_PR_NAWS, file = "out/HWAGE_PR_wide.RData")

#         Compare Wage_PR between NAWS and FLS ---- 
HWAGE_PR <- STATE %>%
  left_join(HWAGE_ST02) %>%
  left_join(HWAGE_ST17) %>%
  left_join(FIPS2PR) %>%
  group_by(PR9) %>%
  summarize(HWage02_fls = mean(HWage02, na.rm=T),
            HWage17_fls = mean(HWage17, na.rm=T)) %>%
  filter(!is.na(PR9)) %>%
  left_join(HWAGE_PR_NAWS %>% 
              dplyr::select(PR9, HWage02, HWage17) %>% 
              rename(HWage02_naws = HWage02,
                     HWage17_naws = HWage17)) %>%
  mutate(pl_h_fls  = (HWage17_fls  - (HWage02_fls *1.37)) /(HWage02_fls *1.37) * 100,
         pl_h_naws = (HWage17_naws - (HWage02_naws*1.37)) /(HWage02_naws*1.37) * 100) %>%
  left_join(PR) %>%
  dplyr::select(PR9, pr, HWage02_fls, HWage02_naws, HWage17_fls, HWage17_naws, pl_h_fls, pl_h_naws)

#    ii.  Housing rents from ACS ----
cat("Non-wage costs\n")
#          get FMR for 2002 & 2017 from HUD

# MSA: Micropolitan Statistical Areas
MSA2FIPS <- read_excel(file.path("in/HUD/MSA2FIPS.xlsx")) %>%
  data.frame()

FMR02 <- read_excel(file.path("in/HUD/FMR2002F.xls")) %>%
  data.frame()

FMR02_MSA <- FMR02 %>%
  filter(startsWith(ID_AGIS2, "MSA")) %>%
  rename(MSA = ID_AGIS2) %>%
  full_join(MSA2FIPS) %>%
  dplyr::select(FIPS, FMR4BR)

FMR02_df <- FMR02 %>%
  filter(startsWith(ID_AGIS2, "CNTY")) %>%
  mutate(FIPS_ed = stringr::str_sub(ID_AGIS2, 5, 9),
         FIPS = as.numeric(FIPS_ed)) %>%
  dplyr::select(FIPS, FMR4BR) %>%
  rbind(FMR02_MSA) %>%
  group_by(FIPS) %>%
  summarize(FMR4BR_02 = mean(FMR4BR, na.rm=T))

FMR17 <- read_excel(file.path("in/HUD/FY2017-4050-County-Level_Data.xlsx")) %>%
  data.frame() %>%
  mutate(FIPS_ed = stringr::str_sub(fips2010, 1, 5),
         FIPS = as.numeric(FIPS_ed),
         FMR4_17 = fmr4) %>%
  dplyr::select(FIPS, FMR4_17) %>%
  group_by(FIPS) %>%
  summarize(FMR4BR_17 = mean(FMR4_17, na.rm=T))

FMR <- FIPS2CZ %>%
  left_join(FMR02_df) %>%
  left_join(FMR17) 

# repeated from data081925.R 
persons_all <- read_excel(file.path("in/QCEW/qcew_h2a_migrant_county.xlsx")) 
persons <- persons_all %>%
  filter(year %in% c(2002, 2015:2019)) %>% # centered around 2017
  mutate(y_grp = ifelse(year == 2002, "02", "17"),
         type = case_when(worker_typology == "Direct Hire H2A"             ~ "MDHLAB_persons",
                          worker_typology == "FLC Hire H2A"                ~ "MCHLAB_persons")) %>%
  dplyr::select(area_fips, y_grp , type, emp) %>%
  mutate(FIPS = as.numeric(area_fips)) %>%
  filter(!(FIPS) %in% c(2000:2999, 15000:15999)) %>%
  group_by(FIPS, y_grp, type) %>%
  summarise(avg_emp = mean(emp, na.rm = TRUE), .groups = "drop") %>%  # Average over 5 years
  filter(!is.na(type)) %>%
  pivot_wider(id_cols = FIPS,                       # keep FIPS as rows
              names_from = c(type, y_grp),          # use both y_grp and type for column names
              values_from = avg_emp,                # fill with avg_emp values
              names_sep = "_")                        # separator in column names

h2a_data <- read_excel(file.path("in/H2A/h2a_hours_wage_weeks_county.xlsx")) %>%
  rename(FIPS = county) 

h2a_fips <- FIPS2CZ %>%
  full_join(h2a_data) %>%
  #dplyr::select(-CZ) %>%
  rename(H2A_direct_hrs   = avg_weekly_hours_direct,
         H2A_direct_weeks = avg_annual_weeks_direct,
         H2A_direct_wage  = avg_hourly_wage_direct,
         H2A_flc_hrs      = avg_weekly_hours_flc,
         H2A_flc_weeks    = avg_annual_weeks_flc,
         H2A_flc_wage     = avg_hourly_wage_flc) %>%
  filter(!(FIPS) %in% c(2000:2999, 15000:15999)) %>%
  filter(!is.na(FIPS)) %>%
  filter(!is.na(CZ)) %>%
  group_by(CZ) %>%
  mutate(across(c(H2A_direct_hrs, H2A_direct_weeks, H2A_direct_wage, H2A_flc_hrs, H2A_flc_weeks, H2A_flc_wage), 
                ~ mean(.x, na.rm = TRUE), .names = "CZ_{.col}")) %>%
  ungroup() %>%
  
  # Join LS info and compute means at LS level
  left_join(CZ2PR) %>%
  group_by(PR9) %>%
  mutate(across(c(H2A_direct_hrs, H2A_direct_weeks, H2A_direct_wage, H2A_flc_hrs, H2A_flc_weeks, H2A_flc_wage), 
                ~ mean(.x, na.rm = TRUE), .names = "LS_{.col}")) %>%
  ungroup() %>%
  
  # Replace missing values using coalesce (CZ -> LS -> Original)
  mutate(
    H2A_direct_hrs   = coalesce(H2A_direct_hrs, CZ_H2A_direct_hrs, LS_H2A_direct_hrs),
    H2A_direct_weeks = coalesce(H2A_direct_weeks, CZ_H2A_direct_weeks, LS_H2A_direct_weeks),
    H2A_direct_wage  = coalesce(H2A_direct_wage, CZ_H2A_direct_wage, LS_H2A_direct_wage),
    H2A_flc_hrs      = coalesce(H2A_flc_hrs, CZ_H2A_flc_hrs, LS_H2A_flc_hrs),
    H2A_flc_weeks    = coalesce(H2A_flc_weeks, CZ_H2A_flc_weeks, LS_H2A_flc_weeks),
    H2A_flc_wage     = coalesce(H2A_flc_wage, CZ_H2A_flc_wage, LS_H2A_flc_wage)) %>%
  group_by(FIPS) %>%
  summarize(dweeks = mean(H2A_direct_weeks, na.rm=T),
            cweeks = mean(H2A_flc_weeks, na.rm=T),
            dhrs = mean(H2A_direct_hrs, na.rm=T),
            chrs = mean(H2A_flc_hrs, na.rm=T),
            dwage = mean(H2A_direct_wage, na.rm=T),
            cwage = mean(H2A_flc_wage, na.rm=T)) %>%
  dplyr::select(FIPS, dweeks, dhrs, dwage, cweeks, chrs, cwage) 

# at CZ level
h2a <- FIPS2CZ %>%
  full_join(h2a_data) %>%
  rename(H2A_direct_hrs   = avg_weekly_hours_direct,
         H2A_direct_weeks = avg_annual_weeks_direct,
         H2A_direct_wage  = avg_hourly_wage_direct,
         H2A_flc_hrs      = avg_weekly_hours_flc,
         H2A_flc_weeks    = avg_annual_weeks_flc,
         H2A_flc_wage     = avg_hourly_wage_flc) %>%
  filter(!(FIPS) %in% c(2000:2999, 15000:15999)) %>%
  filter(!is.na(FIPS)) %>%
  filter(!is.na(CZ)) %>%
  group_by(CZ) %>%
  mutate(across(c(H2A_direct_hrs, H2A_direct_weeks, H2A_direct_wage, H2A_flc_hrs, H2A_flc_weeks, H2A_flc_wage), 
                ~ mean(.x, na.rm = TRUE), .names = "CZ_{.col}")) %>%
  ungroup() %>%
  
  # Join LS info and compute means at LS level
  left_join(CZ2PR) %>%
  group_by(PR9) %>%
  mutate(across(c(H2A_direct_hrs, H2A_direct_weeks, H2A_direct_wage, H2A_flc_hrs, H2A_flc_weeks, H2A_flc_wage), 
                ~ mean(.x, na.rm = TRUE), .names = "LS_{.col}")) %>%
  ungroup() %>%
  
  # Replace missing values using coalesce (CZ -> LS -> Original)
  mutate(
    H2A_direct_hrs   = coalesce(H2A_direct_hrs, CZ_H2A_direct_hrs, LS_H2A_direct_hrs),
    H2A_direct_weeks = coalesce(H2A_direct_weeks, CZ_H2A_direct_weeks, LS_H2A_direct_weeks),
    H2A_direct_wage  = coalesce(H2A_direct_wage, CZ_H2A_direct_wage, LS_H2A_direct_wage),
    H2A_flc_hrs      = coalesce(H2A_flc_hrs, CZ_H2A_flc_hrs, LS_H2A_flc_hrs),
    H2A_flc_weeks    = coalesce(H2A_flc_weeks, CZ_H2A_flc_weeks, LS_H2A_flc_weeks),
    H2A_flc_wage     = coalesce(H2A_flc_wage, CZ_H2A_flc_wage, LS_H2A_flc_wage)) %>%
  #filter(LS10==7)
  # Drop temporary columns
  dplyr::select(-starts_with("CZ_"), -starts_with("LS_")) %>%
  # get CZ levels wage cost and add housing cost
  group_by(CZ) %>%
  summarize(PR9   = mean(PR9, na.rm=T),
            dweeks = mean(H2A_direct_weeks, na.rm=T),
            cweeks = mean(H2A_flc_weeks, na.rm=T),
            dhrs = mean(H2A_direct_hrs, na.rm=T),
            chrs = mean(H2A_flc_hrs, na.rm=T),
            dwage = mean(H2A_direct_wage, na.rm=T),
            cwage = mean(H2A_flc_wage, na.rm=T)) %>%
  dplyr::select(CZ, PR9, dweeks, dhrs, dwage, cweeks, chrs, cwage) 

housing <- persons %>%
  left_join(FMR) %>%
  left_join(FIPS2CZ) %>%
  group_by(CZ) %>%
  summarize(hcost02 = mean(FMR4BR_02 / 8, na.rm=T),
            hcost17 = mean(FMR4BR_17 / 8, na.rm=T),
            dper02  = sum(MDHLAB_persons_02, na.rm=T),
            dper17  = sum(MDHLAB_persons_17, na.rm=T),
            cper02  = sum(MCHLAB_persons_02, na.rm=T),
            cper17  = sum(MCHLAB_persons_17, na.rm=T)) %>%
  dplyr::select(CZ, hcost02, hcost17, dper02, dper17, cper02, cper17) %>% # Estimate!!Median gross rent --!!Total!!
  full_join(h2a) %>%
  filter(!is.na(CZ)) %>%
  mutate(wage_per_hr_cost = ((cper17 * cweeks * chrs * cwage) + (dper17 * dweeks * dhrs * dwage)) / ((cper17 * cweeks * chrs) + (dper17 * dweeks * dhrs)),
         hous_per_hr_cost = ((cweeks * hcost17/4.35) + (dweeks * hcost17/4.24)) / ((cweeks * chrs) + (dweeks * dhrs)),
         hous_prem_percent = round(hous_per_hr_cost / wage_per_hr_cost * 100, 2)) %>%
  left_join(CZ2PR) %>%
  group_by(PR9) %>%
  mutate(pr_lev = mean(hous_prem_percent, na.rm=T),
         hous_prem_percent = coalesce(hous_prem_percent, pr_lev)) %>%
  ungroup() %>%
  dplyr::select(CZ, PR9, hous_prem_percent, wage_per_hr_cost, hous_per_hr_cost)

housing_pr <- housing %>%
  dplyr::select(-CZ) %>%
  group_by(PR9) %>%
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE))) %>%
  #mutate(markup = 30/housing_pr$hous_prem_percent[7] * hous_prem_percent) %>%
  dplyr::select(PR9, wage_per_hr_cost, hous_per_hr_cost, hous_prem_percent) %>%
  left_join(PR)

c_markup   <- c(20, 30, 40) # max contract premium in california
markup     <- c_markup /housing_pr$hous_prem_percent[7] # markup * housing premium = contract premium
max_markup <- c(1.45, 1.55, 1.65) # max for the contract premium

#   iii.  Reading VLAB from CoA ----
cat("VLABOR from CoA \n")
# VLABOR HIRED + VLABOR CONTRACT
VLABOR <- read_excel(file.path("in/USDA_QUICKSTAT/VLABOR9702071722.xlsx"))
VLAB_long <- VLABOR %>%
  rename(State.ANSI = `State ANSI`,
         County.ANSI = `County ANSI`,
         type = `Data Item`) %>%
  mutate(FIPS = as.numeric(sprintf("%d%03d", State.ANSI,  County.ANSI))) %>%
  left_join(LAB_CROP %>% dplyr::select(pct_crop, State.ANSI), by=c("State.ANSI")) %>%
  mutate(Value_current_USD = as.numeric(Value) * pct_crop) %>% # adjusting for % of total employment in crops vs crops+livestock %>%
  mutate(hired = ifelse(type == "LABOR, HIRED - EXPENSE, MEASURED IN $",1,0)) %>%
  subset(select = c("FIPS", "Year", "hired", "Value_current_USD")) %>%
  filter(!is.na(FIPS)) %>%
  #convert Value col to 2002 or 2017 dollars dollars
  left_join(cpi_table) %>%
  filter(Year %in% c(1997,2002,2007, 2012, 2017, 2022)) %>%
  mutate(Value = case_when(Year >=1997 & Year <= 2007 ~ Value_current_USD * CPI_to2002,
                           Year >=2012 & Year <= 2022 ~ Value_current_USD * CPI_to2017)) %>%
  dplyr::select(-starts_with("CPI_to"), -Value_current_USD)

VLAB_wide_all <- merge(pivot_wider(subset(VLAB_long, hired==0), names_from = Year, values_from = Value, names_prefix = "VLAB_C") ,
                       pivot_wider(subset(VLAB_long, hired==1), names_from = Year, values_from = Value, names_prefix = "VLAB_H"  ), by=c("FIPS"))

VLAB_wide <- VLAB_long %>%
  mutate(hired = ifelse(hired == 1, "H", "C")) %>%
  pivot_wider(
    names_from = c(hired, Year),
    values_from = Value,
    names_glue = "VLAB_{hired}{Year}")

VLAB <- VLAB_wide %>%
  rename(VLAB_H2002_old = VLAB_H2002,
         VLAB_H2017_old = VLAB_H2017,
         
         VLAB_C2002_old = VLAB_C2002,
         VLAB_C2017_old = VLAB_C2017) %>%
  full_join(FIPS2PR, by=c("FIPS")) %>%
  full_join(FIPS2CZ, by=c("FIPS")) %>%
  # if missing, take avg of before and after, while replacing with 0 both
  mutate(VLAB_C2002_ltavg = rowMeans(across(c(VLAB_C1997, VLAB_C2002_old, VLAB_C2007)), na.rm=TRUE),
         VLAB_C2017_ltavg = rowMeans(across(c(VLAB_C2012, VLAB_C2017_old, VLAB_C2022)), na.rm=TRUE),
         VLAB_H2002_ltavg = rowMeans(across(c(VLAB_H1997, VLAB_H2002_old, VLAB_H2007)), na.rm=TRUE),
         VLAB_H2017_ltavg = rowMeans(across(c(VLAB_H2012, VLAB_H2017_old, VLAB_H2022)), na.rm=TRUE)) %>%
  
  mutate(VLAB2002_ltavg     = rowSums(across(c(VLAB_C2002_ltavg, VLAB_H2002_ltavg)), na.rm=T),
         VLAB2017_ltavg     = rowSums(across(c(VLAB_C2017_ltavg, VLAB_H2017_ltavg)), na.rm=T)) %>%
  filter(!is.na(CZ))

VLAB_CZ <- VLAB %>%
  group_by(CZ) %>%
  summarize(across(c(VLAB2002_ltavg, VLAB2017_ltavg), ~ sum(.x, na.rm = TRUE))) %>%
  filter(!is.na(CZ))

VLAB_PR <- VLAB %>%
  group_by(PR9) %>%
  summarize(across(c(VLAB2002_ltavg, VLAB2017_ltavg), ~ sum(.x, na.rm = TRUE))) %>%
  filter(!is.na(PR9))

#    iv.  QLAB using  FLS & NAWS with markups----
cat("Estimate QLABOR \n")
# uses input VLAB
# NAWS & FLS

# inflated contract wages with housing premium and then peggs FR to 30% and other regions inflated accordingly
gen_QLAB <- function(source, markup, max_markup) {
  df <- VLAB %>%
    dplyr::select(FIPS, VLAB_C2002_ltavg, VLAB_C2017_ltavg, VLAB_H2002_ltavg, VLAB_H2017_ltavg, VLAB2002_ltavg, VLAB2017_ltavg) %>%
    left_join(FIPS2CZ) %>%
    left_join(FIPS2PR) %>%
    left_join(housing %>% dplyr::select(CZ, hous_prem_percent)) %>%
    mutate(STATE_CODE = as.numeric(if_else(nchar(FIPS) == 4, stringr::str_sub(FIPS, 1, 1),     # 1st digit if 4-digit FIPS
                                           stringr::str_sub(FIPS, 1, 2)))) %>%      # 1st two digits if 5-digit FIPS
    left_join(source) %>% # by=STATE_CODE or PR9
    # Use markup to calculate contract hours, total hours, for each year
    mutate(contract_markup    = (1 + (hous_prem_percent* markup)/100),
           contract_markup_ed = ifelse( (1 + (hous_prem_percent* markup)/100) > max_markup, max_markup, (1 + (hous_prem_percent* markup)/100)),
           
           WAGE_CHourly2002        = round( HWage02 * contract_markup, 1),
           QLAB_CHours2002         = VLAB_C2002_ltavg / WAGE_CHourly2002,
           QLAB_HHours2002         = VLAB_H2002_ltavg / HWage02,
           QLAB_Hours2002          = rowSums(across(c(QLAB_HHours2002, QLAB_CHours2002)), na.rm=T),
           
           WAGE_CHourly2017        = round(HWage17 * contract_markup, 1),
           QLAB_CHours2017         = VLAB_C2017_ltavg / WAGE_CHourly2017,
           QLAB_HHours2017         = VLAB_H2017_ltavg / HWage17,
           QLAB_Hours2017          = rowSums(across(c(QLAB_HHours2017, QLAB_CHours2017)), na.rm=T)) %>%
    
    # long terms averages of wages in current $
    mutate(WAGE02 = ifelse(!is.na(QLAB_Hours2002), VLAB2002_ltavg / QLAB_Hours2002, NA),
           WAGE17 = ifelse(!is.na(QLAB_Hours2017), VLAB2017_ltavg / QLAB_Hours2017, NA)) %>%
    
    # filling in gaps with CZ averages, also repeated for CZ and PR functions
    left_join(FIPS2CZ) %>%
    group_by(CZ) %>%
    mutate(VLAB2002_ltavg_cz = sum(VLAB2002_ltavg, na.rm=T),
           VLAB2017_ltavg_cz = sum(VLAB2017_ltavg, na.rm=T),
           QLAB_Hours2002_cz = sum(QLAB_Hours2002, na.rm=T),
           QLAB_Hours2017_cz = sum(QLAB_Hours2017, na.rm=T)) %>%
    mutate(WAGE02z = VLAB2002_ltavg_cz / QLAB_Hours2002_cz,
           WAGE17z = VLAB2017_ltavg_cz / QLAB_Hours2017_cz) %>%
    ungroup() %>%
    filter(!is.na(CZ)) %>%
    filter(!is.na(PR9)) %>%
    dplyr::select(-VLAB2002_ltavg_cz, -VLAB2017_ltavg_cz, -QLAB_Hours2002_cz, -QLAB_Hours2017_cz) %>%
    
    mutate(WAGE02 = ifelse(is.na(WAGE02), WAGE02z, WAGE02),
           WAGE17 = ifelse(is.na(WAGE17), WAGE17z, WAGE17)) %>%
    
    # overall wages including markup 
    mutate(c_markup17 = round(WAGE17 / HWage17,2)) %>%
    
    # calculate shares
    mutate(shr_hired02 = case_when(VLAB2002_ltavg == 0 ~ 0,
                                   VLAB2002_ltavg > 0 ~ VLAB_H2002_ltavg  / VLAB2002_ltavg),
           shr_hired17 = case_when(VLAB2017_ltavg == 0 ~ 0,
                                   VLAB2017_ltavg > 0 ~ VLAB_H2017_ltavg  / VLAB2017_ltavg),
           shr_hired   = rowMeans(across(c(shr_hired02, shr_hired17)), na.rm=T)) %>%
    
    mutate(vl   = round((VLAB2017_ltavg   - (VLAB2002_ltavg  *cpi0217/100)) / (VLAB2002_ltavg  *cpi0217/100) *100, 1),
           vl   = ifelse(is.nan(vl), 0, vl),
           vl_h = round((VLAB_H2017_ltavg - (VLAB_H2002_ltavg*cpi0217/100)) / (VLAB_H2002_ltavg*cpi0217/100) *100, 1),
           vl_c = round((VLAB_C2017_ltavg - (VLAB_C2002_ltavg*cpi0217/100)) / (VLAB_C2002_ltavg*cpi0217/100) *100, 1),
           t_vl_h = (shr_hired    * vl_h), 
           t_vl_c = ((1-shr_hired)* vl_c),
           ql   = round((QLAB_Hours2017  - QLAB_Hours2002 ) / QLAB_Hours2002   * 100, 1),
           ql   = ifelse(QLAB_Hours2002 == 0 & QLAB_Hours2002==0, 0, ql),
           ql_h = round((QLAB_HHours2017 - QLAB_HHours2002) / QLAB_HHours2002  * 100, 1),
           ql_h = ifelse(QLAB_HHours2002 == 0 & QLAB_HHours2002 ==0, 0, ql_h),
           ql_c = round((QLAB_CHours2017 - QLAB_CHours2002) / QLAB_CHours2002  * 100, 1),
           ql_c = ifelse(QLAB_CHours2002 == 0 & QLAB_CHours2002 ==0, 0, ql_c),
           t_ql_h = (rowMeans(across(c(QLAB_HHours2002, QLAB_HHours2017)), na.rm=T) / rowMeans(across(c(QLAB_Hours2002, QLAB_Hours2017)), na.rm=T)     * ql_h), 
           t_ql_c = ((1-(rowMeans(across(c(QLAB_HHours2002, QLAB_HHours2017)), na.rm=T)) / rowMeans(across(c(QLAB_Hours2002, QLAB_Hours2017)), na.rm=T)) * ql_c),
           
           pl    = round((WAGE17 - (WAGE02*cpi0217/100)) / (WAGE02*cpi0217/100) * 100, 1), 
           
           pl_h = round(vl_h - ql_h, 1),
           pl_c = round(vl_c - ql_c, 1),
           pl1 =  round(vl   - ql,   1))
  return(df)
}

# QLAB_NAWS/FLS_ContractPremLow/Med/High - use NAWS+Med (_NAWS_CM) to report changes in labor
QLAB_NAWS_CL <- gen_QLAB(HWAGE_PR_NAWS, markup[1], max_markup[1])
QLAB_NAWS_CM <- gen_QLAB(HWAGE_PR_NAWS, markup[2], max_markup[1]) # saved later
QLAB_NAWS_CH <- gen_QLAB(HWAGE_PR_NAWS, markup[3], max_markup[1])

QLAB_FLS_CL <- gen_QLAB(HWAGE_ST_FLS, markup[1], max_markup[1])
QLAB_FLS_CM <- gen_QLAB(HWAGE_ST_FLS, markup[2], max_markup[1])
QLAB_FLS_CH <- gen_QLAB(HWAGE_ST_FLS, markup[3], max_markup[1])

#         a. combine to see FLS & NAWS QLAB -- FIPS ----
QLAB_F_raw <- QLAB_NAWS_CL %>%
  dplyr::select(FIPS, QLAB_Hours2002, QLAB_Hours2017, WAGE02, WAGE17,  pl, ql) %>%
  rename(QLAB02_NAWS_high = QLAB_Hours2002,
         QLAB17_NAWS_high = QLAB_Hours2017,
         WAGE02_NAWS_low = WAGE02,
         WAGE17_NAWS_low = WAGE17,
         pl_NAWS_low = pl,
         ql_NAWS_low = ql) %>%
  left_join(QLAB_NAWS_CM %>% dplyr::select(FIPS, QLAB_Hours2002, QLAB_Hours2017, WAGE02, WAGE17,  pl, ql)) %>%
  rename(QLAB02_NAWS_med = QLAB_Hours2002,
         QLAB17_NAWS_med = QLAB_Hours2017,
         WAGE02_NAWS_med = WAGE02,
         WAGE17_NAWS_med = WAGE17,
         pl_NAWS_med = pl,
         ql_NAWS_med = ql) %>%
  left_join(QLAB_NAWS_CH %>% dplyr::select(FIPS, QLAB_Hours2002, QLAB_Hours2017, WAGE02, WAGE17,  pl, ql)) %>%
  rename(QLAB02_NAWS_low = QLAB_Hours2002,
         QLAB17_NAWS_low = QLAB_Hours2017,
         WAGE02_NAWS_high = WAGE02,
         WAGE17_NAWS_high = WAGE17,
         pl_NAWS_high = pl,
         ql_NAWS_high = ql) %>%
  left_join(QLAB_FLS_CL %>% dplyr::select(FIPS, QLAB_Hours2002, QLAB_Hours2017, WAGE02, WAGE17,  pl, ql)) %>%
  rename(QLAB02_FLS_high = QLAB_Hours2002,
         QLAB17_FLS_high = QLAB_Hours2017,
         WAGE02_FLS_low = WAGE02,
         WAGE17_FLS_low = WAGE17,
         pl_FLS_low = pl,
         ql_FLS_low = ql) %>%
  left_join(QLAB_FLS_CM %>% dplyr::select(FIPS, QLAB_Hours2002, QLAB_Hours2017, WAGE02, WAGE17,  pl, ql)) %>%
  rename(QLAB02_FLS_med = QLAB_Hours2002,
         QLAB17_FLS_med = QLAB_Hours2017,
         WAGE02_FLS_med = WAGE02,
         WAGE17_FLS_med = WAGE17,
         pl_FLS_med = pl,
         ql_FLS_med = ql) %>%
  left_join(QLAB_FLS_CH %>% dplyr::select(FIPS, QLAB_Hours2002, QLAB_Hours2017, WAGE02, WAGE17,  pl, ql)) %>%
  rename(QLAB02_FLS_low = QLAB_Hours2002,
         QLAB17_FLS_low = QLAB_Hours2017,
         WAGE02_FLS_high = WAGE02,
         WAGE17_FLS_high = WAGE17,
         pl_FLS_high = pl,
         ql_FLS_high = ql)%>%
  dplyr::select(FIPS, QLAB02_NAWS_low, QLAB02_NAWS_med, QLAB02_NAWS_high, QLAB17_NAWS_low, QLAB17_NAWS_med, QLAB17_NAWS_high, 
                      WAGE02_NAWS_low, WAGE02_NAWS_med, WAGE02_NAWS_high, WAGE17_NAWS_low, WAGE17_NAWS_med, WAGE17_NAWS_high,
                      pl_NAWS_low,     pl_NAWS_med,     pl_NAWS_high, 
                      ql_NAWS_low,     ql_NAWS_med,     ql_NAWS_high, 
                QLAB02_FLS_low, QLAB02_FLS_med, QLAB02_FLS_high, QLAB17_FLS_low, QLAB17_FLS_med, QLAB17_FLS_high, 
                WAGE02_FLS_low, WAGE02_FLS_med, WAGE02_FLS_high, WAGE17_FLS_low, WAGE17_FLS_med, WAGE17_FLS_high,
                pl_FLS_low,     pl_FLS_med,     pl_FLS_high, 
                ql_FLS_low,     ql_FLS_med,     ql_FLS_high) %>%
  left_join(VLAB %>% dplyr::select(FIPS, VLAB2002_ltavg, VLAB2017_ltavg))

#        a1. some counties have large increases (>200%)  ----
qlab.ed.fips <- QLAB_F_raw %>%
  filter(ql_NAWS_med >200) %>%
  pull(FIPS)

qlab.ed.cz <- FIPS2CZ %>% filter(FIPS %in% qlab.ed.fips) %>%  pull(CZ) %>% unique()

QLAB_F_ed <- QLAB_F_raw %>%
  filter(FIPS %in% qlab.ed.fips) %>%
  # rename raw values for reference
  rename(ql_NAWS_med_raw      = ql_NAWS_med,
         ql_NAWS_low_raw      = ql_NAWS_low,
         ql_NAWS_high_raw     = ql_NAWS_high,
         QLAB17_NAWS_med_raw  = QLAB17_NAWS_med,
         QLAB17_NAWS_low_raw  = QLAB17_NAWS_low,
         QLAB17_NAWS_high_raw = QLAB17_NAWS_high,
         
         ql_FLS_med_raw       = ql_FLS_med,
         ql_FLS_low_raw       = ql_FLS_low,
         ql_FLS_high_raw      = ql_FLS_high,
         QLAB17_FLS_med_raw   = QLAB17_FLS_med,
         QLAB17_FLS_low_raw   = QLAB17_FLS_low,
         QLAB17_FLS_high_raw  = QLAB17_FLS_high,
         
         VLAB2017_ltavg_raw   = VLAB2017_ltavg) %>%
  # cap ql and recompute 17-levels and value variables
  mutate(
    ## --- NAWS ---
    ql_NAWS_med      = pmin(ql_NAWS_med_raw, 200), 
    QLAB17_NAWS_med  = QLAB02_NAWS_med * (1 + ql_NAWS_med / 100),
    VLAB2017_NAWS_ltavg   = QLAB17_NAWS_med * WAGE17_NAWS_med,
    
    ## --- FLS ---
    ql_FLS_med      = pmin(ql_FLS_med_raw, 200),
    QLAB17_FLS_med  = QLAB02_FLS_med * (1 + ql_FLS_med / 100),
    VLAB2017_FLS_ltavg   = QLAB17_FLS_med * WAGE17_FLS_med) %>%
  rowwise() %>%
  mutate(VLAB2017_ltavg = rowMeans(across(c(VLAB2017_NAWS_ltavg, VLAB2017_FLS_ltavg)))) %>%
  dplyr::select(-VLAB2017_NAWS_ltavg, -VLAB2017_FLS_ltavg, -QLAB17_NAWS_med, -QLAB17_FLS_med) %>%
  mutate(
    # recompute high/low assuming constant value of labor (VLAB)
    QLAB17_NAWS_low  = VLAB2017_ltavg / WAGE17_NAWS_high,
    QLAB17_NAWS_med  = VLAB2017_ltavg / WAGE17_NAWS_med, # recalculating here, just to be consistent with low-med-high
    QLAB17_NAWS_high = VLAB2017_ltavg / WAGE17_NAWS_low,
    ql_NAWS_low      = (QLAB17_NAWS_low  - QLAB02_NAWS_low)  / QLAB02_NAWS_low  * 100,
    ql_NAWS_high     = (QLAB17_NAWS_high - QLAB02_NAWS_high) / QLAB02_NAWS_high * 100,
    
    # recompute high/low assuming constant value of labor (VLAB)
    QLAB17_FLS_low  = VLAB2017_ltavg / WAGE17_FLS_high,
    QLAB17_FLS_med  = VLAB2017_ltavg / WAGE17_FLS_med, # recalculating here, just to be consistent with low-med-high
    QLAB17_FLS_high = VLAB2017_ltavg / WAGE17_FLS_low,
    ql_FLS_low      = (QLAB17_FLS_low  - QLAB02_FLS_low)  / QLAB02_FLS_low  * 100,
    ql_FLS_high     = (QLAB17_FLS_high - QLAB02_FLS_high) / QLAB02_FLS_high * 100,
    
    vl              = (VLAB2017_ltavg  - (VLAB2002_ltavg*cpi0217/100))  / (VLAB2002_ltavg*cpi0217/100)  * 100) %>%
  dplyr::select(names(QLAB_F_raw))
  
QLAB_F <- QLAB_F_raw %>%
  filter(!FIPS %in% qlab.ed.fips) %>%
  rbind(QLAB_F_ed) %>%
  dplyr::select(FIPS, starts_with("VLAB"), starts_with("QLAB"), starts_with("ql"), starts_with("WAGE"), starts_with("pl")) %>%
  mutate(
    # Round ql and pl vars to 2 decimals
    across(matches("^(ql|pl|WAGE)"), ~ round(.x, 1)),
    # Round all other numeric vars (except FIPS) to 0
    across(!matches("^(FIPS|ql|pl|WAGE)"), ~ round(.x, 0)))

#         b. combine to see FLS & NAWS QLAB -- CZ ----
QLAB_CZ <- QLAB_F %>%
  left_join(FIPS2CZ) %>%
  group_by(CZ) %>%
  summarize(
    # sum all variables that start with QLAB or VLAB
    across(starts_with(c("QLAB", "VLAB2")), ~ sum(.x, na.rm = TRUE)),
    # take mean of all variables that start with WAGE
    across(starts_with("WAGE"), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  mutate(ql_NAWS_med      = (QLAB17_NAWS_med  - QLAB02_NAWS_med)            / QLAB02_NAWS_med            * 100,
         ql_NAWS_low      = (QLAB17_NAWS_low  - QLAB02_NAWS_low)            / QLAB02_NAWS_low            * 100,
         ql_NAWS_high     = (QLAB17_NAWS_high - QLAB02_NAWS_high)           / QLAB02_NAWS_high           * 100,
         pl_NAWS_med      = (WAGE17_NAWS_med  - (WAGE02_NAWS_med*cpi0217/100))  / (WAGE02_NAWS_med*cpi0217/100)  * 100,
         pl_NAWS_low      = (WAGE17_NAWS_low  - (WAGE02_NAWS_low*cpi0217/100))  / (WAGE02_NAWS_low*cpi0217/100)  * 100,
         pl_NAWS_high     = (WAGE17_NAWS_high - (WAGE02_NAWS_high*cpi0217/100)) / (WAGE02_NAWS_high*cpi0217/100) * 100,
         
         ql_FLS_med      = (QLAB17_FLS_med  - QLAB02_FLS_med)  / QLAB02_FLS_med                      * 100,
         ql_FLS_low      = (QLAB17_FLS_low  - QLAB02_FLS_low)  / QLAB02_FLS_low                      * 100,
         ql_FLS_high     = (QLAB17_FLS_high - QLAB02_FLS_high) / QLAB02_FLS_high                     * 100,
         pl_FLS_med      = (WAGE17_FLS_med  - (WAGE02_FLS_med*cpi0217/100))  / (WAGE02_FLS_med*cpi0217/100)  * 100,
         pl_FLS_low      = (WAGE17_FLS_low  - (WAGE02_FLS_low*cpi0217/100))  / (WAGE02_FLS_low*cpi0217/100)  * 100,
         pl_FLS_high     = (WAGE17_FLS_high - (WAGE02_FLS_high*cpi0217/100)) / (WAGE02_FLS_high*cpi0217/100) * 100,
         
         vl              = (VLAB2017_ltavg  - (VLAB2002_ltavg*cpi0217/100))  / (VLAB2002_ltavg*cpi0217/100)  * 100) %>%
  dplyr::select(CZ, starts_with("VLAB"), vl, starts_with("QLAB"), starts_with("ql"), starts_with("WAGE"), starts_with("pl")) %>%
  mutate(
    # Round ql and pl vars to 2 decimals
    across(matches("^(ql|pl|WAGE)"), ~ round(.x, 1)),
    # Round all other numeric vars (except FIPS) to 0
    across(!matches("^(CZ|ql|pl|WAGE)"), ~ round(.x, 0)))

#         c. combine to see FLS & NAWS QLAB -- PR9 ----
QLAB_PR <- QLAB_F %>%
  left_join(FIPS2PR) %>%
  group_by(PR9) %>%
  summarize(
    # sum all variables that start with QLAB or VLAB
    across(starts_with(c("QLAB", "VLAB")), ~ sum(.x, na.rm = TRUE)),
    # take mean of all variables that start with WAGE
    across(starts_with("WAGE"), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  mutate(ql_NAWS_med      = (QLAB17_NAWS_med  - QLAB02_NAWS_med)            / QLAB02_NAWS_med            * 100,
         ql_NAWS_low      = (QLAB17_NAWS_low  - QLAB02_NAWS_low)            / QLAB02_NAWS_low            * 100,
         ql_NAWS_high     = (QLAB17_NAWS_high - QLAB02_NAWS_high)           / QLAB02_NAWS_high           * 100,
         pl_NAWS_med      = (WAGE17_NAWS_med  - (WAGE02_NAWS_med*cpi0217/100))  / (WAGE02_NAWS_med*cpi0217/100)  * 100,
         pl_NAWS_low      = (WAGE17_NAWS_low  - (WAGE02_NAWS_low*cpi0217/100))  / (WAGE02_NAWS_low*cpi0217/100)  * 100,
         pl_NAWS_high     = (WAGE17_NAWS_high - (WAGE02_NAWS_high*cpi0217/100)) / (WAGE02_NAWS_high*cpi0217/100) * 100,
         
         ql_FLS_med      = (QLAB17_FLS_med  - QLAB02_FLS_med)  / QLAB02_FLS_med                      * 100,
         ql_FLS_low      = (QLAB17_FLS_low  - QLAB02_FLS_low)  / QLAB02_FLS_low                      * 100,
         ql_FLS_high     = (QLAB17_FLS_high - QLAB02_FLS_high) / QLAB02_FLS_high                     * 100,
         pl_FLS_med      = (WAGE17_FLS_med  - (WAGE02_FLS_med*cpi0217/100))  / (WAGE02_FLS_med*cpi0217/100)  * 100,
         pl_FLS_low      = (WAGE17_FLS_low  - (WAGE02_FLS_low*cpi0217/100))  / (WAGE02_FLS_low*cpi0217/100)  * 100,
         pl_FLS_high     = (WAGE17_FLS_high - (WAGE02_FLS_high*cpi0217/100)) / (WAGE02_FLS_high*cpi0217/100) * 100,
         
         vl              = (VLAB2017_ltavg  - (VLAB2002_ltavg*cpi0217/100))  / (VLAB2002_ltavg*cpi0217/100)  * 100) %>%
  dplyr::select(PR9, starts_with("VLAB"), vl, starts_with("QLAB"), starts_with("ql"), starts_with("WAGE"), starts_with("pl")) %>%
  mutate(
    # Round ql and pl vars to 2 decimals
    across(matches("^(ql|pl|WAGE)"), ~ round(.x, 1)),
    # Round all other numeric vars (except FIPS) to 0
    across(!matches("^(PR9|ql|pl|WAGE)"), ~ round(.x, 0)))

#         d. combine to see FLS & NAWS QLAB -- US ----
QLAB_US <- QLAB_F %>%
  summarize(
    # sum all variables that start with QLAB or VLAB
    across(starts_with(c("QLAB", "VLAB")), ~ sum(.x, na.rm = TRUE))) %>%
  mutate(WAGE02_NAWS_low  = VLAB2002_ltavg / QLAB02_NAWS_low,
         WAGE02_NAWS_med  = VLAB2002_ltavg / QLAB02_NAWS_med,
         WAGE02_NAWS_high = VLAB2002_ltavg / QLAB02_NAWS_high,
         
         WAGE02_FLS_low   = VLAB2002_ltavg / QLAB02_FLS_low,
         WAGE02_FLS_med   = VLAB2002_ltavg / QLAB02_FLS_med,
         WAGE02_FLS_high  = VLAB2002_ltavg / QLAB02_FLS_high,
         
         WAGE17_NAWS_low  = VLAB2017_ltavg / QLAB17_NAWS_low,
         WAGE17_NAWS_med  = VLAB2017_ltavg / QLAB17_NAWS_med,
         WAGE17_NAWS_high = VLAB2017_ltavg / QLAB17_NAWS_high,
         
         WAGE17_FLS_low   = VLAB2017_ltavg / QLAB17_FLS_low,
         WAGE17_FLS_med   = VLAB2017_ltavg / QLAB17_FLS_med,
         WAGE17_FLS_high  = VLAB2017_ltavg / QLAB17_FLS_high) %>%
  
  mutate(ql_NAWS_med      = (QLAB17_NAWS_med  - QLAB02_NAWS_med)            / QLAB02_NAWS_med            * 100,
         ql_NAWS_low      = (QLAB17_NAWS_low  - QLAB02_NAWS_low)            / QLAB02_NAWS_low            * 100,
         ql_NAWS_high     = (QLAB17_NAWS_high - QLAB02_NAWS_high)           / QLAB02_NAWS_high           * 100, 
         
         pl_NAWS_med      = (WAGE17_NAWS_med  - (WAGE02_NAWS_med*cpi0217/100))  / (WAGE02_NAWS_med*cpi0217/100)  * 100,
         pl_NAWS_low      = (WAGE17_NAWS_low  - (WAGE02_NAWS_low*cpi0217/100))  / (WAGE02_NAWS_low*cpi0217/100)  * 100,
         pl_NAWS_high     = (WAGE17_NAWS_high - (WAGE02_NAWS_high*cpi0217/100)) / (WAGE02_NAWS_high*cpi0217/100) * 100,
         
         ql_FLS_med      = (QLAB17_FLS_med  - QLAB02_FLS_med)  / QLAB02_FLS_med                      * 100,
         ql_FLS_low      = (QLAB17_FLS_low  - QLAB02_FLS_low)  / QLAB02_FLS_low                      * 100,
         ql_FLS_high     = (QLAB17_FLS_high - QLAB02_FLS_high) / QLAB02_FLS_high                     * 100,
         pl_FLS_med      = (WAGE17_FLS_med  - (WAGE02_FLS_med*cpi0217/100))  / (WAGE02_FLS_med*cpi0217/100)  * 100,
         pl_FLS_low      = (WAGE17_FLS_low  - (WAGE02_FLS_low*cpi0217/100))  / (WAGE02_FLS_low*cpi0217/100)  * 100,
         pl_FLS_high     = (WAGE17_FLS_high - (WAGE02_FLS_high*cpi0217/100)) / (WAGE02_FLS_high*cpi0217/100) * 100,
         
         vl              = (VLAB2017_ltavg  - (VLAB2002_ltavg*cpi0217/100))  / (VLAB2002_ltavg*cpi0217/100)  * 100) %>%
  dplyr::select(starts_with("VLAB"), vl, starts_with("QLAB"), starts_with("ql"), starts_with("WAGE"), starts_with("pl")) %>%
  mutate(
    # Round ql and pl vars to 2 decimals
    across(matches("^(ql|pl|WAGE)"), ~ round(.x, 1)),
    # Round all other numeric vars (except FIPS) to 0
    across(!matches("^(ql|pl|WAGE)"), ~ round(.x, 0)))

QLAB_PR <- QLAB_PR %>%
  rbind(QLAB_US %>% mutate(PR9=10))

# 2.3:  INTERMEDIATE INPUTS ----
cat("[2.3] Processing intermediate input data...\n")
quality.unadj.chem = 0.75 # Fig4 USDA report pp 16
quality.unadj.pfert = 0.54 # Fig2 USDA report pp 14
quality.unadj.qfert = 0.5  # Fig2 USDA report pp 14

#     i.  Prices ----
pz = read_excel(file.path("in/USDA_tfp/table01a.xlsx"), skip=2) %>%
  mutate(across(everything(), as.numeric)) %>%
  rename(Seed_P = "Intermediate inputs: Feed and seed: Price (index, 2015=1)",
         Seed_Q = "Intermediate inputs: Feed and seed: Quantity (million, $2015)",
         Chem_P = "Intermediate inputs: Pesticides: Price (index, 2015=1)",
         Chem_Q = "Intermediate inputs: Pesticides: Quantity (million, $2015)",
         # I think the Census data on fuel/gass is something diff from energy which has very high price increase
         Energy_P = "Intermediate inputs: Energy: Price (index, 2015=1)",
         Energy_Q = "Intermediate inputs: Energy: Quantity (million, $2015)",
         Fert_P = "Intermediate inputs: Fertilizer and lime: Price (index, 2015=1)",
         Fert_Q = "Intermediate inputs: Fertilizer and lime: Quantity (million, $2015)",
         Mach_P  = "Capital: Durable equipment: Price (index, 2015=1)",
         Mach_Q  = "Capital: Durable equipment: Quantity (million, $2015)",
         Oth_P  = "Intermediate inputs: Other intermediate inputs: Price (index, 2015=1)",
         Oth_Q  = "Intermediate inputs: Other intermediate inputs: Quantity (million, $2015)") %>%
  dplyr::select(Year, Seed_P, Seed_Q, Chem_P, Chem_Q, Energy_P, Energy_Q, Fert_P, Fert_Q, Mach_Q, Mach_P, Oth_P, Oth_Q)

pz_longterm_avg <- pz %>%
  filter(Year %in% 1997:2007) %>%  # Select rows for years 1997-2007
  dplyr::select(-Year) %>%  # Exclude Year column
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)))  %>% # Calculate average for numeric columns
  mutate(Year = 2002)

pz_longterm_avg2 <- pz %>%
  filter(Year %in% 2012:2020) %>%  # Select rows for years 2012-2020
  dplyr::select(-Year) %>%  # Exclude Year column
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)))  %>% # Calculate average for numeric columns
  mutate(Year = 2017)

prices_z <- bind_rows(pz_longterm_avg, pz_longterm_avg2) %>%
  pivot_wider(names_from = "Year", values_from = everything())

p_seed     = (prices_z$Seed_P_2017 - prices_z$Seed_P_2002)/ prices_z$Seed_P_2002 * 100
q_seed     = (prices_z$Seed_Q_2017 - prices_z$Seed_Q_2002)/ prices_z$Seed_Q_2002 * 100

p_chem     = (prices_z$Chem_P_2017 - prices_z$Chem_P_2002)/ prices_z$Chem_P_2002 * 100
p_chem_unadj = (prices_z$Chem_P_2017 - (prices_z$Chem_P_2002/quality.unadj.chem))/ (prices_z$Chem_P_2002/quality.unadj.chem) * 100
q_chem     = (prices_z$Chem_Q_2017 - prices_z$Chem_Q_2002)/ prices_z$Chem_Q_2002 * 100

#p_energy  = (prices_z$Energy_P_2017 - prices_z$Energy_P_2002)/ prices_z$Energy_P_2002 * 100
p_fert     = (prices_z$Fert_P_2017 - prices_z$Fert_P_2002)/ prices_z$Fert_P_2002 * 100
p_fert_unadj = (prices_z$Fert_P_2017 - (prices_z$Fert_P_2002/quality.unadj.pfert))/ (prices_z$Fert_P_2002/quality.unadj.pfert) * 100
q_fert     = (prices_z$Fert_Q_2017 - prices_z$Fert_Q_2002)/ prices_z$Fert_Q_2002 * 100

p_mach     = (prices_z$Mach_P_2017 - prices_z$Mach_P_2002)/ prices_z$Mach_P_2002 * 100
q_mach     = (prices_z$Mach_Q_2017 - prices_z$Mach_Q_2002)/ prices_z$Mach_Q_2002 * 100

p_oth      = (prices_z$Oth_P_2017 - prices_z$Oth_P_2002)/ prices_z$Oth_P_2002 * 100
q_oth      = (prices_z$Oth_Q_2017 - prices_z$Oth_Q_2002)/ prices_z$Oth_Q_2002 * 100

#    ii.  Values from CoA ----
# 
load(file.path("in/VNLBz.RData")) 
load(file.path("in/VNITz.RData")) 

VNLBr <- VNLBz %>%
  left_join(CZ2PR) %>%
  group_by(PR9) %>%
  summarize(VNLB = sum(VNLB))

VNITr <- VNITz %>%
  left_join(CZ2PR) %>%
  group_by(PR9) %>%
  summarize(Total = sum(Total))

VZ2022 <- read_excel(file.path("in/USDA_QUICKSTAT/VZ2022.xlsx"))
VZ2017 <- read_excel(file.path("in/USDA_QUICKSTAT/VZ2017.xlsx"))
VZ2012 <- read_excel(file.path("in/USDA_QUICKSTAT/VZ2012.xlsx"))
VZ2007 <- read_excel(file.path("in/USDA_QUICKSTAT/VZ2007.xlsx"))
VZ2002 <- read_excel(file.path("in/USDA_QUICKSTAT/VZ2002.xlsx"))
VZ1997 <- read_excel(file.path("in/USDA_QUICKSTAT/VZ1997.xlsx"))
VSEED  <- read_excel(file.path("in/USDA_QUICKSTAT/VSEED_allyears.xlsx")) # missed this col in the selection of above file
VFERT  <- read_excel(file.path("in/USDA_QUICKSTAT/VFERT_allyears.xlsx")) # missed this col in the selection of above file

VZ_long1 <- VZ1997 %>%
  rbind(VZ2002) %>%
  rbind(VZ2007) %>%
  rbind(VZ2012) %>%
  rbind(VZ2017) %>%
  rbind(VZ2022) %>%
  rbind(VSEED)  %>%
  rbind(VFERT)

VZ_long    <- VZ_long1 %>%
  rename(State.ANSI = `State ANSI`,
         County.ANSI = `County ANSI`,
         z_type = `Data Item`) %>%
  dplyr::select(-County) %>%
  mutate(FIPS = as.numeric(sprintf("%d%03d", State.ANSI,  County.ANSI))) %>%
  subset(FIPS <= 14999 | FIPS >= 15999) %>%
  subset(FIPS <= 1999 | FIPS >= 2999) %>%
  mutate(Value_current_USD = as.numeric(Value)) %>%
  filter(z_type!="SEEDS, FOR COVER CROPS - EXPENSE, MEASURED IN $") %>%
  mutate(z_type = ifelse(z_type == "CHEMICAL TOTALS - EXPENSE, MEASURED IN $", "VC",
                         ifelse(z_type == "FERTILIZER TOTALS, INCL LIME & SOIL CONDITIONERS - EXPENSE, MEASURED IN $", "VF",
                                ifelse(z_type == "DEPRECIATION - EXPENSE, MEASURED IN $", "VD",
                                       ifelse(z_type == "FUELS, INCL LUBRICANTS - EXPENSE, MEASURED IN $", "VG",
                                              ifelse(z_type == "SEEDS & PLANTS TOTALS - EXPENSE, MEASURED IN $", "VS",
                                                     ifelse(z_type == "LABOR, CONTRACT - EXPENSE, MEASURED IN $", "VLC",
                                                            ifelse(z_type == "LABOR, HIRED - EXPENSE, MEASURED IN $", "VLH", 
                                                                   ifelse(z_type == "RENT, CASH, LAND & BUILDINGS - EXPENSE, MEASURED IN $", "VRENT",
                                                                          ifelse(z_type == "SUPPLIES & REPAIRS, (EXCL LUBRICANTS) - EXPENSE, MEASURED IN $", "VR",
                                                                                 ifelse(z_type == "TAXES, PROPERTY, REAL ESTATE & NON-REAL ESTATE, (EXCL PAID BY LANDLORD) - EXPENSE, MEASURED IN $", "VT",
                                                                                        ifelse(z_type == "INTEREST - EXPENSE, MEASURED IN $", "VI", NA)))))))))))) %>%
  dplyr::select(FIPS, Year, z_type, Value_current_USD)%>%
  filter(!is.na(FIPS)) %>%
  #convert Value col to 2002 or 2017 dollars dollars
  left_join(cpi_table) %>%
  filter(Year %in% c(1997,2002,2007, 2012, 2017, 2022)) %>%
  mutate(Value = case_when(Year >=1997 & Year <= 2007 ~ Value_current_USD * CPI_to2002,
                           Year >=2012 & Year <= 2022 ~ Value_current_USD * CPI_to2017)) %>%
  dplyr::select(-starts_with("CPI_to"), -Value_current_USD)

# calculate VZ
VZ_wide_old <- VZ_long %>%
  #retain chems, fert, seed, gas, depre, # excluding labor, and rents
  filter(z_type %in% c("VC", "VF", "VS", "VG", "VD", "VR", "VI", "VT")) %>%
  group_by(FIPS, Year) %>%
  summarise(Value = sum(Value, na.rm = TRUE), .groups = "drop") %>%
  mutate(z_type = "VZ") %>%
  mutate(shr = 1)

VZ_oth <- VZ_long %>%
  filter(z_type %in% c("VR", "VI", "VT")) %>%
  group_by(FIPS, Year) %>%
  summarise(Value = sum(Value, na.rm = TRUE), .groups = "drop") %>%
  mutate(z_type = "VO") 

#add VZ to intermediate input values
VZ_wide <- VZ_long %>%
  #retain chems, fert, seed, gas, depre, # excluding labor, and rents
  filter(z_type %in% c("VC", "VF", "VS", "VG", "VD")) %>%
  rbind(VZ_oth) %>%
  group_by(FIPS, Year) %>%
  mutate(shr = Value / sum(Value, na.rm=T)) %>%
  ungroup() %>%
  rbind(VZ_wide_old) %>%
  arrange(FIPS, Year, z_type) 

VZ_wide_ltavg <- VZ_wide %>%
  mutate(z_year = paste(z_type, Year, sep = "_")) %>%
  dplyr::select(FIPS, z_year, Value) %>%
  pivot_wider(names_from = z_year, values_from = Value) %>%
  
  mutate(VC_2002_ltavg = rowMeans(across(c(VC_1997, VC_2002, VC_2007)), na.rm=T),
         VC_2017_ltavg = rowMeans(across(c(VC_2012, VC_2017, VC_2022)), na.rm=T),
         vchem = round( (VC_2017_ltavg - (VC_2002_ltavg * cpi0217/100))/(VC_2002_ltavg * cpi0217/100)*100, 2),
         vchem = ifelse(is.infinite(vchem), 0, ifelse(is.na(vchem), 0, vchem)),
         vchem_adj = round( (VC_2017_ltavg - (VC_2002_ltavg * 0.97 * cpi0217/100))/(VC_2002_ltavg * 0.97 * cpi0217/100)*100, 2),
         vchem_adj = ifelse(is.infinite(vchem_adj), 0, 
                            ifelse(is.na(vchem_adj), 0, vchem_adj)),
         qchem     = round(vchem     - p_chem     , 2),
         qchem     = round(vchem_adj - p_chem     , 2),
         qchem_unadj = round(vchem - p_chem_unadj , 2)) %>%
  mutate(VF_2002_ltavg = rowMeans(across(c(VF_1997, VF_2002, VF_2007)), na.rm=T),
         VF_2017_ltavg = rowMeans(across(c(VF_2012, VF_2017, VF_2022)), na.rm=T),
         vfert = round( (VF_2017_ltavg - (VF_2002_ltavg * cpi0217/100))/(VF_2002_ltavg * cpi0217/100)*100, 2),
         vfert = ifelse(is.infinite(vfert), 0, ifelse(is.na(vfert), 0, vfert)),
         vfert_adj = round( (VF_2017_ltavg - (VF_2002_ltavg * 0.76 * cpi0217/100))/(VF_2002_ltavg * 0.76 * cpi0217/100)*100, 2),
         vfert_adj = ifelse(is.infinite(vfert_adj), 0,                             ifelse(is.na(vfert_adj), 0, vfert_adj)),
         qfert     = round(vfert     - p_fert     , 2),
         qfert     = round(vfert_adj - p_fert     , 2),
         qfert_unadj = round(vfert - p_fert_unadj , 2)) %>%
  mutate(VS_2002_ltavg = rowMeans(across(c(VS_1997, VS_2002, VS_2007)), na.rm=T),
         VS_2017_ltavg = rowMeans(across(c(VS_2012, VS_2017, VS_2022)), na.rm=T),
         vseed = round( (VS_2017_ltavg - (VS_2002_ltavg * cpi0217/100))/(VS_2002_ltavg * cpi0217/100)*100, 2),
         vseed = ifelse(is.infinite(vseed), 0, ifelse(is.na(vseed), 0, vseed)),
         qseed = round(vseed - p_seed , 2)) %>%
  mutate(VG_2002_ltavg = rowMeans(across(c(VG_1997, VG_2002, VG_2007)), na.rm=T),
         VG_2017_ltavg = rowMeans(across(c(VG_2012, VG_2017, VG_2022)), na.rm=T),
         vgas = round( (VG_2017_ltavg - (VG_2002_ltavg * cpi0217/100))/(VG_2002_ltavg * cpi0217/100)*100, 2),
         vgas = ifelse(is.infinite(vgas), 0, ifelse(is.na(vgas), 0, vgas)),
         qgas = round(vgas - p_gas , 2)) %>%
  mutate(VD_2002_ltavg = rowMeans(across(c(VD_2002, VD_2007)), na.rm=T),
         VD_2017_ltavg = rowMeans(across(c(VD_2012, VD_2017, VD_2022)), na.rm=T),
         vmach = round( (VD_2017_ltavg - (VD_2002_ltavg * cpi0217/100))/(VD_2002_ltavg * cpi0217/100)*100, 2),
         vmach = ifelse(is.infinite(vmach), 0, ifelse(is.na(vmach), 0, vmach)),
         qmach = round(vmach - (p_mach), 2))  %>%
  mutate(VO_2002_ltavg = rowMeans(across(c(VO_1997, VO_2002, VO_2007)), na.rm=T),
         VO_2017_ltavg = rowMeans(across(c(VO_2012, VO_2017, VO_2022)), na.rm=T),
         voth = round( (VO_2017_ltavg - (VO_2002_ltavg * cpi0217/100))/(VO_2002_ltavg * cpi0217/100)*100, 2),
         voth = ifelse(is.infinite(voth), 0, ifelse(is.na(voth), 0, voth)),
         qoth = voth) %>%
  mutate(VZ_2002_ltavg = rowSums(across(c(VC_2002_ltavg, VS_2002_ltavg, VG_2002_ltavg, VD_2002_ltavg, VO_2002_ltavg)), na.rm=T),
         VZ_2017_ltavg = rowSums(across(c(VC_2017_ltavg, VS_2017_ltavg, VG_2017_ltavg, VD_2017_ltavg, VO_2017_ltavg)), na.rm=T))

VZ_US_wide_ltavg <- VZ_wide %>%
  mutate(z_year = paste(z_type, Year, sep = "_")) %>%
  dplyr::select(FIPS, z_year, Value) %>%
  pivot_wider(names_from = z_year, values_from = Value) %>%
  
  dplyr::select(-FIPS) %>%
  summarize(across(everything(), sum, na.rm=T)) %>%
  mutate(VC_2002_ltavg = rowMeans(across(c(VC_1997, VC_2002, VC_2007)), na.rm=T),
         VC_2017_ltavg = rowMeans(across(c(VC_2012, VC_2017, VC_2022)), na.rm=T),
         vchem = round( (VC_2017_ltavg - (VC_2002_ltavg * cpi0217/100))/(VC_2002_ltavg * cpi0217/100)*100, 2),
         vchem = ifelse(is.infinite(vchem), 0, ifelse(is.na(vchem), 0, vchem)),
         vchem_adj = round( (VC_2017_ltavg - (VC_2002_ltavg * 0.97 * cpi0217/100))/(VC_2002_ltavg * 0.97 * cpi0217/100)*100, 2),
         vchem_adj = ifelse(is.infinite(vchem_adj), 0, 
                            ifelse(is.na(vchem_adj), 0, vchem_adj)),
         qchem     = round(vchem - p_chem     , 2),
         qchem_adj = round(vchem_adj - p_chem     , 2),
         qchem_unadj = round(vchem - p_chem_unadj , 2)) %>%
  mutate(VF_2002_ltavg = rowMeans(across(c(VF_1997, VF_2002, VF_2007)), na.rm=T),
         VF_2017_ltavg = rowMeans(across(c(VF_2012, VF_2017, VF_2022)), na.rm=T),
         vfert = round( (VF_2017_ltavg - (VF_2002_ltavg * cpi0217/100))/(VF_2002_ltavg * cpi0217/100)*100, 2),
         vfert = ifelse(is.infinite(vfert), 0, ifelse(is.na(vfert), 0, vfert)),
         vfert_adj = round( (VF_2017_ltavg - (VF_2002_ltavg * 0.76 * cpi0217/100))/(VF_2002_ltavg * 0.76 * cpi0217/100)*100, 2),
         vfert_adj = ifelse(is.infinite(vfert_adj), 0,                             ifelse(is.na(vfert_adj), 0, vfert_adj)),
         qfert     = round(vfert - p_fert     , 2),
         qfert_adj = round(vfert_adj - p_fert     , 2),
         qfert_unadj = round(vfert - p_fert_unadj , 2)) %>%
  mutate(VS_2002_ltavg = rowMeans(across(c(VS_1997, VS_2002, VS_2007)), na.rm=T),
         VS_2017_ltavg = rowMeans(across(c(VS_2012, VS_2017, VS_2022)), na.rm=T),
         vseed = round( (VS_2017_ltavg - (VS_2002_ltavg * cpi0217/100))/(VS_2002_ltavg * cpi0217/100)*100, 2),
         vseed = ifelse(is.infinite(vseed), 0, ifelse(is.na(vseed), 0, vseed)),
         qseed = round(vseed - p_seed , 2)) %>%
  mutate(VG_2002_ltavg = rowMeans(across(c(VG_1997, VG_2002, VG_2007)), na.rm=T),
         VG_2017_ltavg = rowMeans(across(c(VG_2012, VG_2017, VG_2022)), na.rm=T),
         vgas = round( (VG_2017_ltavg - (VG_2002_ltavg * cpi0217/100))/(VG_2002_ltavg * cpi0217/100)*100, 2),
         vgas = ifelse(is.infinite(vgas), 0, ifelse(is.na(vgas), 0, vgas)),
         qgas = round(vgas - p_gas , 2)) %>%
  mutate(VD_2002_ltavg = rowMeans(across(c(VD_2002, VD_2007)), na.rm=T),
         VD_2017_ltavg = rowMeans(across(c(VD_2012, VD_2017, VD_2022)), na.rm=T),
         vmach = round( (VD_2017_ltavg - (VD_2002_ltavg * cpi0217/100))/(VD_2002_ltavg * cpi0217/100)*100, 2),
         vmach = ifelse(is.infinite(vmach), 0, ifelse(is.na(vmach), 0, vmach)),
         qmach = round(vmach - (p_mach), 2))  %>%
  mutate(VO_2002_ltavg = rowMeans(across(c(VO_1997, VO_2002, VO_2007)), na.rm=T),
         VO_2017_ltavg = rowMeans(across(c(VO_2012, VO_2017, VO_2022)), na.rm=T),
         voth = round( (VO_2017_ltavg - (VO_2002_ltavg * cpi0217/100))/(VO_2002_ltavg * cpi0217/100)*100, 2),
         voth = ifelse(is.infinite(voth), 0, ifelse(is.na(voth), 0, voth)),
         qoth = voth) %>%
  mutate(VZ_2002_ltavg = rowSums(across(c(VC_2002_ltavg, VS_2002_ltavg, VG_2002_ltavg, VD_2002_ltavg, VO_2002_ltavg)), na.rm=T),
         VZ_2017_ltavg = rowSums(across(c(VC_2017_ltavg, VS_2017_ltavg, VG_2017_ltavg, VD_2017_ltavg, VO_2017_ltavg)), na.rm=T))

VZ_CZ_wide_ltavg <- VZ_wide %>%
  mutate(z_year = paste(z_type, Year, sep = "_")) %>%
  dplyr::select(FIPS, z_year, Value) %>%
  pivot_wider(names_from = z_year, values_from = Value) %>%
  
  left_join(FIPS2CZ) %>%
  dplyr::select(-FIPS) %>%
  group_by(CZ) %>%
  summarize(across(everything(), sum, na.rm=T)) %>%
  mutate(VC_2002_ltavg = rowMeans(across(c(VC_1997, VC_2002, VC_2007)), na.rm=T),
         VC_2017_ltavg = rowMeans(across(c(VC_2012, VC_2017, VC_2022)), na.rm=T),
         vchem = round( (VC_2017_ltavg - (VC_2002_ltavg * cpi0217/100))/(VC_2002_ltavg * cpi0217/100)*100, 2),
         vchem = ifelse(is.infinite(vchem), 0, ifelse(is.na(vchem), 0, vchem)),
         vchem_adj = round( (VC_2017_ltavg - (VC_2002_ltavg * 0.97 * cpi0217/100))/(VC_2002_ltavg * 0.97 * cpi0217/100)*100, 2),
         vchem_adj = ifelse(is.infinite(vchem_adj), 0, 
                            ifelse(is.na(vchem_adj), 0, vchem_adj)),
         qchem     = round(vchem     - p_chem     , 2),
         qchem_adj = round(vchem_adj - p_chem     , 2),
         qchem_unadj = round(vchem - p_chem_unadj , 2)) %>%
  mutate(VF_2002_ltavg = rowMeans(across(c(VF_1997, VF_2002, VF_2007)), na.rm=T),
         VF_2017_ltavg = rowMeans(across(c(VF_2012, VF_2017, VF_2022)), na.rm=T),
         vfert = round( (VF_2017_ltavg - (VF_2002_ltavg * cpi0217/100))/(VF_2002_ltavg * cpi0217/100)*100, 2),
         vfert = ifelse(is.infinite(vfert), 0, ifelse(is.na(vfert), 0, vfert)),
         vfert_adj = round( (VF_2017_ltavg - (VF_2002_ltavg * 0.76 * cpi0217/100))/(VF_2002_ltavg * 0.76 * cpi0217/100)*100, 2),
         vfert_adj = ifelse(is.infinite(vfert_adj), 0,                             ifelse(is.na(vfert_adj), 0, vfert_adj)),
         qfert     = round(vfert - p_fert     , 2),
         qfert_adj = round(vfert_adj - p_fert     , 2),
         qfert_unadj = round(vfert - p_fert_unadj , 2)) %>%
  mutate(VS_2002_ltavg = rowMeans(across(c(VS_1997, VS_2002, VS_2007)), na.rm=T),
         VS_2017_ltavg = rowMeans(across(c(VS_2012, VS_2017, VS_2022)), na.rm=T),
         vseed = round( (VS_2017_ltavg - (VS_2002_ltavg * cpi0217/100))/(VS_2002_ltavg * cpi0217/100)*100, 2),
         vseed = ifelse(is.infinite(vseed), 0, ifelse(is.na(vseed), 0, vseed)),
         qseed = round(vseed - p_seed , 2)) %>%
  mutate(VG_2002_ltavg = rowMeans(across(c(VG_1997, VG_2002, VG_2007)), na.rm=T),
         VG_2017_ltavg = rowMeans(across(c(VG_2012, VG_2017, VG_2022)), na.rm=T),
         vgas = round( (VG_2017_ltavg - (VG_2002_ltavg * cpi0217/100))/(VG_2002_ltavg * cpi0217/100)*100, 2),
         vgas = ifelse(is.infinite(vgas), 0, ifelse(is.na(vgas), 0, vgas)),
         qgas = round(vgas - p_gas , 2)) %>%
  mutate(VD_2002_ltavg = rowMeans(across(c(VD_2002, VD_2007)), na.rm=T),
         VD_2017_ltavg = rowMeans(across(c(VD_2012, VD_2017, VD_2022)), na.rm=T),
         vmach = round( (VD_2017_ltavg - (VD_2002_ltavg * cpi0217/100))/(VD_2002_ltavg * cpi0217/100)*100, 2),
         vmach = ifelse(is.infinite(vmach), 0, ifelse(is.na(vmach), 0, vmach)),
         qmach = round(vmach - (p_mach), 2))  %>%
  mutate(VO_2002_ltavg = rowMeans(across(c(VO_1997, VO_2002, VO_2007)), na.rm=T),
         VO_2017_ltavg = rowMeans(across(c(VO_2012, VO_2017, VO_2022)), na.rm=T),
         voth = round( (VO_2017_ltavg - (VO_2002_ltavg * cpi0217/100))/(VO_2002_ltavg * cpi0217/100)*100, 2),
         voth = ifelse(is.infinite(voth), 0, ifelse(is.na(voth), 0, voth)),
         qoth = voth) %>%
  mutate(VZ_2002_ltavg = rowSums(across(c(VC_2002_ltavg, VS_2002_ltavg, VG_2002_ltavg, VD_2002_ltavg, VO_2002_ltavg)), na.rm=T),
         VZ_2017_ltavg = rowSums(across(c(VC_2017_ltavg, VS_2017_ltavg, VG_2017_ltavg, VD_2017_ltavg, VO_2017_ltavg)), na.rm=T)) %>%
  filter(CZ %in% c(1:605)) %>%
  rbind(cbind(CZ = 606, VZ_US_wide_ltavg))

VZ_PR_wide_ltavg <- VZ_wide %>%
  mutate(z_year = paste(z_type, Year, sep = "_")) %>%
  dplyr::select(FIPS, z_year, Value) %>%
  pivot_wider(names_from = z_year, values_from = Value) %>%
  
  left_join(FIPS2PR %>% dplyr::select(FIPS, PR9)) %>%
  dplyr::select(-FIPS) %>%
  group_by(PR9) %>%
  summarize(across(everything(), sum, na.rm=T)) %>%
  mutate(VC_2002_ltavg = rowMeans(across(c(VC_1997, VC_2002, VC_2007)), na.rm=T),
         VC_2017_ltavg = rowMeans(across(c(VC_2012, VC_2017, VC_2022)), na.rm=T),
         vchem = round( (VC_2017_ltavg - (VC_2002_ltavg * cpi0217/100))/(VC_2002_ltavg * cpi0217/100)*100, 2),
         vchem = ifelse(is.infinite(vchem), 0, ifelse(is.na(vchem), 0, vchem)),
         vchem_adj = round( (VC_2017_ltavg - (VC_2002_ltavg * 0.97 * cpi0217/100))/(VC_2002_ltavg * 0.97 * cpi0217/100)*100, 2),
         vchem_adj = ifelse(is.infinite(vchem_adj), 0, 
                            ifelse(is.na(vchem_adj), 0, vchem_adj)),
         qchem      = round(vchem - p_chem     , 2),
         qchem_adj = round(vchem_adj - p_chem     , 2),
         qchem_unadj = round(vchem - p_chem_unadj , 2)) %>%
  mutate(VF_2002_ltavg = rowMeans(across(c(VF_1997, VF_2002, VF_2007)), na.rm=T),
         VF_2017_ltavg = rowMeans(across(c(VF_2012, VF_2017, VF_2022)), na.rm=T),
         vfert = round( (VF_2017_ltavg - (VF_2002_ltavg * cpi0217/100))/(VF_2002_ltavg * cpi0217/100)*100, 2),
         vfert = ifelse(is.infinite(vfert), 0, ifelse(is.na(vfert), 0, vfert)),
         vfert_adj = round( (VF_2017_ltavg - (VF_2002_ltavg * 0.76 * cpi0217/100))/(VF_2002_ltavg * 0.76 * cpi0217/100)*100, 2),
         vfert_adj = ifelse(is.infinite(vfert_adj), 0,                             ifelse(is.na(vfert_adj), 0, vfert_adj)),
         qfert     = round(vfert - p_fert     , 2),
         qfert_adj = round(vfert_adj - p_fert     , 2),
         qfert_unadj = round(vfert - p_fert_unadj , 2)) %>%
  mutate(VS_2002_ltavg = rowMeans(across(c(VS_1997, VS_2002, VS_2007)), na.rm=T),
         VS_2017_ltavg = rowMeans(across(c(VS_2012, VS_2017, VS_2022)), na.rm=T),
         vseed = round( (VS_2017_ltavg - (VS_2002_ltavg * cpi0217/100))/(VS_2002_ltavg * cpi0217/100)*100, 2),
         vseed = ifelse(is.infinite(vseed), 0, ifelse(is.na(vseed), 0, vseed)),
         qseed = round(vseed - p_seed , 2)) %>%
  mutate(VG_2002_ltavg = rowMeans(across(c(VG_1997, VG_2002, VG_2007)), na.rm=T),
         VG_2017_ltavg = rowMeans(across(c(VG_2012, VG_2017, VG_2022)), na.rm=T),
         vgas = round( (VG_2017_ltavg - (VG_2002_ltavg * cpi0217/100))/(VG_2002_ltavg * cpi0217/100)*100, 2),
         vgas = ifelse(is.infinite(vgas), 0, ifelse(is.na(vgas), 0, vgas)),
         qgas = round(vgas - p_gas , 2)) %>%
  mutate(VD_2002_ltavg = rowMeans(across(c(VD_2002, VD_2007)), na.rm=T),
         VD_2017_ltavg = rowMeans(across(c(VD_2012, VD_2017, VD_2022)), na.rm=T),
         vmach = round( (VD_2017_ltavg - (VD_2002_ltavg * cpi0217/100))/(VD_2002_ltavg * cpi0217/100)*100, 2),
         vmach = ifelse(is.infinite(vmach), 0, ifelse(is.na(vmach), 0, vmach)),
         qmach = round(vmach - (p_mach), 2))  %>%
  mutate(VO_2002_ltavg = rowMeans(across(c(VO_1997, VO_2002, VO_2007)), na.rm=T),
         VO_2017_ltavg = rowMeans(across(c(VO_2012, VO_2017, VO_2022)), na.rm=T),
         voth = round( (VO_2017_ltavg - (VO_2002_ltavg * cpi0217/100))/(VO_2002_ltavg * cpi0217/100)*100, 2),
         voth = ifelse(is.infinite(voth), 0, ifelse(is.na(voth), 0, voth)),
         qoth = voth) %>%
  mutate(VZ_2002_ltavg = rowSums(across(c(VC_2002_ltavg, VS_2002_ltavg, VG_2002_ltavg, VD_2002_ltavg, VO_2002_ltavg)), na.rm=T),
         VZ_2017_ltavg = rowSums(across(c(VC_2017_ltavg, VS_2017_ltavg, VG_2017_ltavg, VD_2017_ltavg, VO_2017_ltavg)), na.rm=T)) %>%
  filter(PR9 %in% c(1:9)) %>%
  rbind(cbind(PR9 = 10, VZ_US_wide_ltavg))

#   iii.  Shares of int input in VZ ----
Shares.in.z <- VZ_wide %>%
  # drop Value col
  dplyr::select(-Value) %>%
  # drop VZ row
  filter(shr != 1) %>%
  mutate(z_year = paste(z_type, Year, sep = "_")) %>%
  dplyr::select(FIPS, z_year, shr) %>%
  # extract the z_type (first letter after 'V') and last 2 digits of year
  mutate(z_short = paste0("t_", substr(z_year, 2, 2)),  # e.g., VC -> t_C
         yr_short = substr(z_year, nchar(z_year) - 1, nchar(z_year)), # e.g., 1997 -> 97
         colname = paste0(z_short, yr_short) ) %>%  # e.g., tC97
  select(FIPS, colname, shr) %>%
  pivot_wider(names_from = colname,values_from = shr) %>%
  mutate(t_C    = round(rowMeans(across(starts_with("t_C")), na.rm=T), 2),
         t_F    = round(rowMeans(across(starts_with("t_F")), na.rm=T), 2),
         t_G    = round(rowMeans(across(starts_with("t_G")), na.rm=T), 2),
         t_S    = round(rowMeans(across(starts_with("t_S")), na.rm=T), 2),
         t_D    = round(rowMeans(across(starts_with("t_D")), na.rm=T), 2),
         t_O    = round(rowMeans(across(starts_with("t_O")), na.rm=T), 2)) 
# calculating pz in QZ_XXX

Shares.US.z <- VZ_wide %>%
  filter(shr != 1) %>%                          # drop total (VZ) rows
  dplyr::select(-shr) %>%
  group_by(Year, z_type) %>%                # summing over all FIPS
  summarize(Value = sum(Value, na.rm = TRUE)) %>%
  ungroup() %>%
  group_by(Year) %>%           
  mutate(shr = Value / sum(Value, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(z_year = paste(z_type, Year, sep = "_")) %>%  # still safe—Year was in grouped data
  dplyr::select(z_year, shr) %>%
  mutate(z_short = paste0("t_", substr(z_year, 2, 2)),      # e.g., VC → t_C
         yr_short = substr(z_year, nchar(z_year) - 1, nchar(z_year)),  # e.g., 1997 → 97
         colname = paste0(z_short, yr_short)) %>%                # e.g., t_C97
  dplyr::select(colname, shr) %>%
  pivot_wider(names_from = colname,values_from = shr) %>%
  mutate(t_C    = round(rowMeans(across(starts_with("t_C")), na.rm=T), 2),
         t_F    = round(rowMeans(across(starts_with("t_F")), na.rm=T), 2),
         t_G    = round(rowMeans(across(starts_with("t_G")), na.rm=T), 2),
         t_S    = round(rowMeans(across(starts_with("t_S")), na.rm=T), 2),
         t_D    = round(rowMeans(across(starts_with("t_D")), na.rm=T), 2),
         t_O    = round(rowMeans(across(starts_with("t_O")), na.rm=T), 2)) %>%
  mutate(adj_fac=1) # just adding to match Shares.in.CZ

Shares.CZ.z <- VZ_wide %>%
  filter(shr != 1) %>%                          # drop total (VZ) rows
  dplyr::select(-shr) %>%
  left_join(FIPS2CZ, by = "FIPS") %>%           # map each FIPS to CZ
  group_by(CZ, Year, z_type) %>%                
  summarize(Value = sum(Value, na.rm = TRUE)) %>%
  ungroup() %>%
  group_by(CZ, Year) %>%           
  mutate(shr = Value / sum(Value, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(z_year = paste(z_type, Year, sep = "_")) %>%  # still safe—Year was in grouped data
  dplyr::select(CZ, z_year, shr) %>%
  mutate(z_short = paste0("t_", substr(z_year, 2, 2)),      # e.g., VC → t_C
         yr_short = substr(z_year, nchar(z_year) - 1, nchar(z_year)),  # e.g., 1997 → 97
         colname = paste0(z_short, yr_short)) %>%                # e.g., t_C97
  dplyr::select(CZ, colname, shr) %>%
  pivot_wider(names_from = colname,values_from = shr) %>%
  left_join(VNLBz) %>%
  left_join(VNITz) %>%
  rename(VNIT = Total) %>%
  filter(CZ %in% c(1:605)) %>%
  mutate(adj_fac = (VNLB + VNIT) / (VNLB)) %>%
  # using adj to adjust input shares
  mutate(t_C    = round(rowMeans(across(starts_with("t_C")), na.rm=T), 2) * adj_fac,
         t_F    = round(rowMeans(across(starts_with("t_F")), na.rm=T), 2) * adj_fac,
         t_G    = round(rowMeans(across(starts_with("t_G")), na.rm=T), 2) * adj_fac,
         t_S    = round(rowMeans(across(starts_with("t_S")), na.rm=T), 2) * adj_fac,
         t_D    = round(rowMeans(across(starts_with("t_D")), na.rm=T), 2) * adj_fac,
         t_O    = round(rowMeans(across(starts_with("t_O")), na.rm=T), 2) * adj_fac) %>%
  dplyr::select(-VNLB, -VNIT) %>%
  filter(CZ %in% c(1:605)) %>%
  rbind(c(CZ = 606, Shares.US.z)) 
  
# calculating pz in QZ_CZ
Shares.PR.z <- VZ_wide %>%
  filter(shr != 1) %>%                          # drop total (VZ) rows
  dplyr::select(-shr) %>%
  left_join(FIPS2PR %>% dplyr::select(FIPS, PR9)) %>%
  group_by(PR9, Year, z_type) %>%                
  summarize(Value = sum(Value, na.rm = TRUE)) %>%
  ungroup() %>%
  group_by(PR9, Year) %>%
  mutate(shr = Value / sum(Value, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(z_year = paste(z_type, Year, sep = "_")) %>%  # still safe—Year was in grouped data
  dplyr::select(PR9, z_year, shr) %>%
  mutate(z_short = paste0("t_", substr(z_year, 2, 2)),      # e.g., VC → t_C
         yr_short = substr(z_year, nchar(z_year) - 1, nchar(z_year)),  # e.g., 1997 → 97
         colname = paste0(z_short, yr_short)) %>%                # e.g., t_C97
  dplyr::select(PR9, colname, shr) %>%
  pivot_wider(names_from = colname,values_from = shr) %>%
  left_join(VNLBr) %>%
  left_join(VNITr) %>%
  rename(VNIT = Total) %>%
  filter(PR9 %in% c(1:9)) %>%
  # using adj to adjust input shares
  mutate(adj_fac = (VNLB + VNIT) / (VNLB)) %>%
  mutate(t_C    = round(rowMeans(across(starts_with("t_C")), na.rm=T), 2) * adj_fac,
         t_F    = round(rowMeans(across(starts_with("t_F")), na.rm=T), 2) * adj_fac,
         t_G    = round(rowMeans(across(starts_with("t_G")), na.rm=T), 2) * adj_fac,
         t_S    = round(rowMeans(across(starts_with("t_S")), na.rm=T), 2) * adj_fac,
         t_D    = round(rowMeans(across(starts_with("t_D")), na.rm=T), 2) * adj_fac,
         t_O    = round(rowMeans(across(starts_with("t_O")), na.rm=T), 2)  * adj_fac) %>%
  dplyr::select(-VNLB, -VNIT) %>%
  rbind(c(PR9 = 10, Shares.US.z))

#    iv.  QZ ----
QZ <- Shares.in.z %>%
  dplyr::select(FIPS, ends_with(c("C", "D", "G", "S", "O"))) %>%
  left_join(VZ_wide_ltavg %>% dplyr::select(FIPS, matches("^v[cfsgmo]"), matches("^q[cfsgmo]"))) %>%
  # not using adj_factor in reporting changes in vz
  mutate(vz       = round( (t_C * vchem)   + (t_G * vgas) + (t_S * vseed) +(t_D * vmach) + (t_O * voth), 2),
         qz       = round( (t_C * qchem)   + (t_G * qgas) + (t_S * qseed) +(t_D * qmach) + (t_O * qoth), 2)) %>%
  mutate(qz       = ifelse(is.na(qz), 0, qz),
         vz = ifelse(is.na(vz), 0, vz),
         # calculating pz here: including quality adjustments for chem, other only inflation adjustment
         pz       = vz - qz) %>%
  right_join(FIPS2CZ) %>%
  group_by(CZ) %>%
  mutate(pz_cz = mean(pz, na.rm=T)) %>%
  ungroup() %>%
  mutate(pz = ifelse(is.na(pz), pz_cz, pz)) %>%
  dplyr::select(FIPS, vchem, qchem, vseed, qseed, vgas, qgas, vmach, qmach, voth, qoth, vz,qz, pz)

QZ_US <- Shares.US.z %>%
  dplyr::select(ends_with(c("C", "D", "G", "S", "O"))) %>%
  cbind(VZ_US_wide_ltavg %>% dplyr::select(matches("^v[cfsgmo]"), matches("^q[cfsgmo]"))) %>%
  # adjusted for z without N fert
  mutate(vz       = round( (t_C * vchem_adj)   + (t_G * vgas) + (t_S * vseed) +(t_D * vmach) + (t_O * voth), 2),
         qz       = round( (t_C * qchem_adj)   + (t_G * qgas) + (t_S * qseed) +(t_D * qmach) + (t_O * qoth), 2)) %>%
  mutate(qz       = ifelse(is.na(qz), 0, qz),
         vz = ifelse(is.na(vz), 0, vz), 
         pz       = vz - qz) %>%
  dplyr::select(vchem, qchem, vseed, qseed, vgas, qgas, vmach, qmach, voth, qoth, vz,qz, pz)

QZ_CZ <- Shares.CZ.z %>%
  dplyr::select(CZ, ends_with(c("C", "D", "G", "S", "O"))) %>%
  left_join(VZ_CZ_wide_ltavg %>% dplyr::select(CZ, vchem, qchem, vseed, qseed, vgas, qgas, vmach, qmach, voth, qoth)) %>%
  dplyr::select(-ends_with("7"), -ends_with("2"), -ends_with("_ltavg")) %>%
  # using t_ adjusted by adj_fac and not q/vfert
  mutate(vz       = round( (t_C * vchem)   + (t_G * vgas) + (t_S * vseed) +(t_D * vmach) + (t_O * voth), 2),
         qz       = round( (t_C * qchem)   + (t_G * qgas) + (t_S * qseed) +(t_D * qmach) + (t_O * qoth), 2)) %>%
  mutate(qz       = ifelse(is.na(qz), 0, qz),
         vz = ifelse(is.na(vz), 0, vz),
         pz       = vz - qz) %>%
  right_join(CZ2PR) %>%
  group_by(PR9) %>%
  mutate(pz_pr = mean(pz, na.rm=T)) %>%
  ungroup() %>%
  mutate(pz = ifelse(is.na(pz), pz_pr, pz)) %>%
  dplyr::select(CZ, vchem, qchem, vseed, qseed, vgas, qgas, vmach, qmach, voth, qoth, vz,qz, pz) %>%
  filter(CZ %in% c(1:605)) %>%
  rbind(c(CZ=606, QZ_US))

QZ_PR <- Shares.PR.z %>%
  dplyr::select(PR9, ends_with(c("C", "D", "F", "G", "S", "O"))) %>%
  left_join(VZ_PR_wide_ltavg %>% dplyr::select(PR9, vchem, qchem, vseed, qseed, vgas, qgas, vmach, qmach, voth, qoth)) %>%
  # not using adj_factor in reporting changes in vz
  mutate(vz       = round( (t_C * vchem)   + (t_G * vgas) + (t_S * vseed) +(t_D * vmach) + (t_O * voth), 2),
         qz       = round( (t_C * qchem)   + (t_G * qgas) + (t_S * qseed) +(t_D * qmach) + (t_O * qoth), 2)) %>%
  mutate(qz       = ifelse(is.na(qz), 0, qz),
         vz = ifelse(is.na(vz), 0, vz),
         pz       = vz - qz) %>%
  dplyr::select(PR9, vchem, qchem, vseed, qseed, vgas, qgas, vmach, qmach, voth, qoth, vz,qz, pz) %>%
  filter(PR9 %in% c(1:9)) %>%
  rbind(c(PR9=10, QZ_US))

# 2.4:  Resources: Land + Water ----
cat("[2.4] Processing land and water resource data...\n")
#     i.  Land ----
acres2ha = 0.404686
QLAND      <- read_excel(file.path("in/USDA_QUICKSTAT/CROPLAND.xlsx")) %>%
  rename(State.ANSI  = `State ANSI`,
         County.ANSI = `County ANSI`,
         type        = `Data Item`,
         HLAND       =  Value) %>%
  filter(type == "AG LAND, CROPLAND, HARVESTED - ACRES") %>%
  mutate(FIPS = as.numeric(sprintf("%d%03d", State.ANSI,  County.ANSI))) %>%
  mutate(HLAND = as.numeric(HLAND) * acres2ha,
         County.ANSI = ifelse(County.ANSI==102 & State.ANSI==46, 113, County.ANSI),      # 46113 Shannon County, renamed to Oglala lakota County 46102
         County.ANSI = ifelse(County.ANSI==86  & State.ANSI==12, 25,  County.ANSI)) %>%  #  Dade county (FIPS 12025) is renamed as Miami-Dade
  subset(select = c("FIPS", "Year", "HLAND")) %>%
  filter(!is.na(FIPS)) %>%
  pivot_wider(names_from = Year, values_from = c(HLAND), names_prefix = "HLAND") %>%
  group_by(FIPS) %>%
  mutate(HLAND2002_ltavg = rowMeans(across(c(HLAND1997, HLAND2002, HLAND2007)), na.rm=T),
         HLAND2017_ltavg = rowMeans(across(c(HLAND2012, HLAND2017, HLAND2022)), na.rm=T)) %>%
  ungroup() %>%
  mutate(qc  = (HLAND2017_ltavg  - HLAND2002_ltavg)  / HLAND2002_ltavg  * 100)

QLAND_CZ <- QLAND %>%
  left_join(FIPS2CZ) %>%
  dplyr::select(-FIPS) %>%
  group_by(CZ) %>%
  summarize(across(everything(), sum, na.rm=T)) %>%
  filter(!is.na(CZ)) %>%
  mutate(qc  = (HLAND2017_ltavg  - HLAND2002_ltavg)  / HLAND2002_ltavg  * 100)

QLAND_PR <- QLAND %>%
  left_join(subset(FIPS2PR, select=c(FIPS, PR9))) %>%
  left_join(FIPS2CZ) %>%
  dplyr::select(-FIPS) %>%
  group_by(PR9) %>%
  summarize(across(everything(), sum, na.rm=T)) %>%
  filter(!is.na(PR9)) %>%
  mutate(qc  = (HLAND2017_ltavg  - HLAND2002_ltavg)  / HLAND2002_ltavg  * 100)

#    ii.  Cropland rents ----
PLAND_wide      <- read_excel(file.path("in/USDA_QUICKSTAT/rent9717_allyears.xlsx"))

PLAND_long <- PLAND_wide %>%
  rename(State.ANSI = `State ANSI`,
         County.ANSI = `County ANSI`) %>%
  mutate(FIPS = as.numeric(sprintf("%d%03d", State.ANSI,  County.ANSI))) %>%
  mutate(rent_tot_current_USD = as.numeric(Value)) %>%
  mutate(County.ANSI = ifelse(County.ANSI==102 & State.ANSI==46, 113, County.ANSI),      # 46113 Shannon County, renamed to Oglala lakota County 46102
         County.ANSI = ifelse(County.ANSI==86  & State.ANSI==12, 25,  County.ANSI)) %>%  #  Dade county (FIPS 12025) is renamed as Miami-Dade
  subset(select = c("FIPS", "Year", "rent_tot_current_USD")) %>%
  filter(!is.na(FIPS)) %>%
  left_join(cpi_table) %>%
  filter(Year %in% c(1997,2002,2007, 2012, 2017, 2022)) %>%
  mutate(rent_tot = case_when(Year >=1997 & Year <= 2007 ~ rent_tot_current_USD * CPI_to2002,
                              Year >=2012 & Year <= 2022 ~ rent_tot_current_USD * CPI_to2017)) %>%
  dplyr::select(-starts_with("CPI_to"), -rent_tot_current_USD) %>%
  pivot_wider(names_from = Year, values_from = c(rent_tot), names_prefix = "rent_tot") %>%
  group_by(FIPS) %>%
  summarize(RENT_TOT97 = sum(rent_tot1997, na.rm=T),
            RENT_TOT02 = sum(rent_tot2002, na.rm=T),
            RENT_TOT07 = sum(rent_tot2007, na.rm=T),
            RENT_TOT12 = sum(rent_tot2012, na.rm=T),
            RENT_TOT17 = sum(rent_tot2017, na.rm=T),
            RENT_TOT22 = sum(rent_tot2022, na.rm=T)) %>%
  left_join(subset(QLAND)) %>%
  mutate(RENT97_old = RENT_TOT97 / HLAND1997,
         RENT02_old = RENT_TOT02 / HLAND2002,
         RENT07_old = RENT_TOT07 / HLAND2007,
         RENT12_old = RENT_TOT12 / HLAND2012,
         RENT17_old = RENT_TOT17 / HLAND2017,
         RENT22_old = RENT_TOT22 / HLAND2022) %>%
  mutate(RENT02_ltavg_old = rowMeans(across(c(RENT97_old, RENT02_old, RENT07_old)), na.rm = TRUE),
         RENT17_ltavg_old = rowMeans(across(c(RENT12_old, RENT17_old, RENT22_old)), na.rm = TRUE))

r97_low  = quantile(PLAND_long$RENT97_old,        na.rm=T, 0.05)
r97_high = quantile(PLAND_long$RENT97_old,        na.rm=T, 0.95)
r02_low  = quantile(PLAND_long$RENT02_ltavg_old,  na.rm=T, 0.05)
r02_high = quantile(PLAND_long$RENT02_ltavg_old,  na.rm=T, 0.95)
r07_low  = quantile(PLAND_long$RENT07_old,        na.rm=T, 0.05)
r07_high = quantile(PLAND_long$RENT07_old,        na.rm=T, 0.95)
r12_low  = quantile(PLAND_long$RENT12_old,        na.rm=T, 0.05)
r12_high = quantile(PLAND_long$RENT12_old,        na.rm=T, 0.95)
r17_low  = quantile(PLAND_long$RENT17_ltavg_old,  na.rm=T, 0.05)
r17_high = quantile(PLAND_long$RENT17_ltavg_old,  na.rm=T, 0.95)
r22_low  = quantile(PLAND_long$RENT22_old,        na.rm=T, 0.05)
r22_high = quantile(PLAND_long$RENT22_old,        na.rm=T, 0.95)

PLAND <- PLAND_long %>%
  mutate(RENT97 = ifelse(RENT97_old              < r97_low,  r97_low,
                         ifelse(RENT97_old       > r97_high, r97_high, RENT97_old)),
         RENT02 = ifelse(RENT02_ltavg_old        < r02_low,  r02_low,
                         ifelse(RENT02_ltavg_old > r02_high, r02_high, RENT02_ltavg_old)),
         RENT07 = ifelse(RENT07_old              < r07_low,  r07_low,
                         ifelse(RENT07_old       > r07_high, r07_high, RENT07_old)),
         RENT12 = ifelse(RENT12_old              < r12_low,  r12_low,
                         ifelse(RENT12_old       > r12_high, r12_high, RENT12_old)),
         RENT17 = ifelse(RENT17_ltavg_old        < r17_low,  r17_low,
                         ifelse(RENT17_ltavg_old > r17_high, r17_high, RENT17_ltavg_old)),
         RENT22 = ifelse(RENT22_old              < r22_low,  r22_low,
                         ifelse(RENT22_old       > r22_high, r22_high, RENT22_old))) %>%
  mutate(pc_old  = (RENT17_old - (RENT02_old * 1.37))/(RENT02_old * 1.37) * 100,
         pc      = ifelse( (RENT17     - (RENT02     * 1.37))/(RENT02     * 1.37)     * 100 <= -85, -85,
                           (RENT17     - (RENT02     * 1.37))/(RENT02     * 1.37)     * 100) ) %>%
  dplyr::select(FIPS, RENT_TOT97, RENT_TOT02, RENT_TOT07, RENT_TOT12, RENT_TOT17, RENT_TOT22,
                RENT97,     RENT02,     RENT07,     RENT12,     RENT17,     RENT22, pc_old, pc) %>%
  filter(!is.na(FIPS)) %>%
  mutate(RENT02 = ifelse(is.infinite(RENT02), 0 , RENT02),
         RENT17 = ifelse(is.infinite(RENT17), 0 , RENT17)) %>%
  right_join(FIPS2CZ) %>%
  group_by(CZ) %>%
  mutate(pc_cz = mean(pc, na.rm=T)) %>%
  ungroup() %>%
  mutate(pc = ifelse(is.na(pc), pc_cz, pc)) 

PLAND_CZ <- PLAND %>%
  left_join(FIPS2CZ) %>%
  group_by(CZ) %>%
  summarize(RENT_TOT97 = sum(RENT_TOT97, na.rm=T),
            RENT_TOT02 = sum(RENT_TOT02, na.rm=T),
            RENT_TOT07 = sum(RENT_TOT07, na.rm=T),
            RENT_TOT12 = sum(RENT_TOT12, na.rm=T),
            RENT_TOT17 = sum(RENT_TOT17, na.rm=T),
            RENT_TOT22 = sum(RENT_TOT22, na.rm=T)) %>%
  dplyr::filter(!is.na(CZ)) %>%
  left_join(QLAND_CZ) %>%
  mutate(RENT97   = ifelse(HLAND1997 ==0, NA, RENT_TOT97 / HLAND1997),
         RENT02_y = ifelse(HLAND2002 ==0, NA, RENT_TOT02 / HLAND2002),
         RENT07   = ifelse(HLAND2007 ==0, NA, RENT_TOT07 / HLAND2007),
         RENT12   = ifelse(HLAND2012 ==0, NA, RENT_TOT12 / HLAND2012),
         RENT17_y = ifelse(HLAND2017 ==0, NA, RENT_TOT17 / HLAND2017),
         RENT22   = ifelse(HLAND2022 ==0, NA, RENT_TOT22 / HLAND2022)) %>%
  mutate(RENT02 = rowMeans(across(c(RENT97, RENT02_y, RENT07)), na.rm = TRUE),
         RENT17 = rowMeans(across(c(RENT12, RENT17_y, RENT22)), na.rm = TRUE)) %>%
  mutate(pc      = ifelse( (RENT17     - (RENT02     * 1.37))/(RENT02 * 1.37)     * 100 >= 400, 400,
                           (RENT17     - (RENT02     * 1.37))/(RENT02 * 1.37)     * 100) ) %>%
  dplyr::select(CZ, RENT_TOT97, RENT_TOT02, RENT_TOT07, RENT_TOT12, RENT_TOT17, RENT_TOT22,
                RENT97,     RENT02,     RENT07,     RENT12,     RENT17,     RENT22, pc) 

PLAND_PR <- PLAND %>%
  left_join(FIPS2PR) %>%
  group_by(PR9) %>%
  summarize(RENT_TOT97 = sum(RENT_TOT97, na.rm=T),
            RENT_TOT02 = sum(RENT_TOT02, na.rm=T),
            RENT_TOT07 = sum(RENT_TOT07, na.rm=T),
            RENT_TOT12 = sum(RENT_TOT12, na.rm=T),
            RENT_TOT17 = sum(RENT_TOT17, na.rm=T),
            RENT_TOT22 = sum(RENT_TOT22, na.rm=T)) %>%
  dplyr::filter(!is.na(PR9)) %>%
  left_join(QLAND_PR) %>%
  mutate(RENT97   = ifelse(HLAND1997 ==0, NA, RENT_TOT97 / HLAND1997),
         RENT02_y = ifelse(HLAND2002 ==0, NA, RENT_TOT02 / HLAND2002),
         RENT07   = ifelse(HLAND2007 ==0, NA, RENT_TOT07 / HLAND2007),
         RENT12   = ifelse(HLAND2012 ==0, NA, RENT_TOT12 / HLAND2012),
         RENT17_y = ifelse(HLAND2017 ==0, NA, RENT_TOT17 / HLAND2017),
         RENT22   = ifelse(HLAND2022 ==0, NA, RENT_TOT22 / HLAND2022)) %>%
  mutate(RENT02 = rowMeans(across(c(RENT97, RENT02_y, RENT07)), na.rm = TRUE),
         RENT17 = rowMeans(across(c(RENT12, RENT17_y, RENT22)), na.rm = TRUE)) %>%
  mutate(pc      = ifelse( (RENT17     - (RENT02     * 1.37))/(RENT02 * 1.37)     * 100 >= 400, 400,
                           (RENT17     - (RENT02     * 1.37))/(RENT02 * 1.37)     * 100) ) %>%
  dplyr::select(PR9, RENT_TOT97, RENT_TOT02, RENT_TOT07, RENT_TOT12, RENT_TOT17, RENT_TOT22,
                RENT97,     RENT02,     RENT07,     RENT12,     RENT17,     RENT22, pc) %>%
  mutate(RENT02 = ifelse(is.infinite(RENT02), 0 , RENT02),
         RENT17 = ifelse(is.infinite(RENT17), 0 , RENT17))

#   iii.  Change in water use ----
# using USGS data on total irrigation water
water00_st <- read_excel(file.path("in/USGS/water_use/water_state_2000.xlsx"), skip=2) %>%
  dplyr::select(STATE, `Total...8`) %>%
  filter(!is.na(STATE)) %>%
  rename(WAT00_st =`Total...8`) %>%
  mutate(WAT00_st = as.numeric(WAT00_st)) %>%
  mutate(STATE_NAME = toupper(STATE)) %>%  # Convert STATE_NAME to uppercase
  left_join(STATENAME) %>%
  filter(!is.na(STATEFIPS))

water05 <- read_excel(file.path("in/USGS/water_use/usco2005.xls")) %>%
  as.data.frame() %>%
  filter(! STATEFIPS %in% c("02", "15")) %>%
  subset(select=c("FIPS", "IR-WFrTo")) %>%
  rename(FIPS_ed = FIPS,
         WAT05   = "IR-WFrTo") %>%
  mutate(FIPS = as.numeric(sub("^0+", "", FIPS_ed))) %>%
  left_join(STATE) %>%
  rename(STATEFIPS = STATE_CODE) %>%
  dplyr::select(FIPS, STATEFIPS, WAT05) %>%
  group_by(STATEFIPS) %>%
  mutate(WAT05_st = sum(WAT05, na.rm=T)) %>%
  ungroup() %>%
  mutate(shr = WAT05/WAT05_st) %>%
  filter(!is.na(FIPS))

water15 <- read_excel(file.path("in/USGS/water_use/usco2015v2.0.xlsx"), skip=1) %>%
  as.data.frame() %>%
  filter(! STATEFIPS %in% c(02, 15)) %>%
  subset(select=c("FIPS", "STATEFIPS", "IR-WFrTo")) %>%
  rename(WAT15   = "IR-WFrTo") %>%
  dplyr::select(FIPS, WAT15)

water00 <-  subset(water05, select=c(STATEFIPS, FIPS, shr)) %>%
  left_join(subset(water00_st, select=c(STATEFIPS, WAT00_st))) %>%
  mutate(WAT00 = ifelse(is.nan(WAT00_st * shr), 0.01, (WAT00_st * shr))) %>%
  dplyr::select(FIPS, WAT00)

WATER <- water15 %>%
  full_join(water05) %>%
  full_join(water00) %>%
  mutate(WAT02 = rowMeans(across(c(WAT00, WAT05)))) %>%
  dplyr::select(FIPS, WAT02, WAT15) %>%
  mutate(qw0215 = (WAT15 - WAT02) / WAT02 *100,
         qw0215_annual = qw0215 / (2015-2002),
         qw = qw0215_annual * (2017-2002)) %>%
  mutate(qw = ifelse(qw>200, 200, qw)) %>%
  mutate(qw = ifelse(is.na(qw), 0, qw))

WATER_US <- WATER %>%
  summarize(across(c(WAT02, WAT15), sum, na.rm=T)) %>%
  mutate(qw0215 = (WAT15 - WAT02) / WAT02 *100,
         qw0215_annual = qw0215 / (2015-2002),
         qw = qw0215_annual * (2017-2002))%>%
  mutate(qw = ifelse(qw>200, 200, qw)) %>%
  mutate(qw = ifelse(is.na(qw), 0, qw))

WATER_PR <- WATER %>%
  left_join(subset(FIPS2PR, select=c(FIPS, PR9))) %>%
  group_by(PR9) %>%
  summarize(across(c(WAT02, WAT15), sum, na.rm=T)) %>%
  mutate(qw0215 = (WAT15 - WAT02) / WAT02 *100,
         qw0215_annual = qw0215 / (2015-2002),
         qw = qw0215_annual * (2017-2002)) %>%
  filter(!is.na(PR9)) %>%
  rbind(c(PR9=10, WATER_US)) %>%
  left_join(PR) %>%
  mutate(qw = ifelse(qw>200, 200, qw)) %>%
  mutate(qw = ifelse(is.na(qw), 0, qw))

WATER_CZ <- WATER %>%
  left_join(FIPS2CZ) %>%
  group_by(CZ) %>%
  summarize(across(c(WAT02, WAT15), sum, na.rm=T)) %>%
  mutate(qw0215 = (WAT15 - WAT02) / WAT02 *100,
         qw0215_annual = qw0215 / (2015-2002),
         qw = qw0215_annual * (2017-2002)) %>%
  mutate(qw = ifelse(qw>200, 200, qw)) %>%
  mutate(qw = ifelse(is.na(qw), 0, qw)) %>%
  filter(!is.na(CZ)) %>%
  rbind(c(CZ=606, WATER_US)) 

# 2.5:  SS Shocks ----
cat("[2.5] Building steady-state (SS) shocks...\n")
#     i.  LABOR(DOMESTIC + IMMIGRANT) ----
#    ii.  DOMESTIC, Non-Ag Wages (Cons+Service, Non-Ag overall) ----
#' Source: from the Quarterly Census of Employment and Wages
#' Use Types of Non-Crop work reported in NAWS: QCEW codes 
#' Crop prodn:                      111
#' Non-Crop Ag - animal production: 112
#' Manufacturing:                  31-33
#' Construction:                    23
#' Extraction services:             21
#' Mechanic/Repair/Maintenance:    811
#' Food Industry services:         722
#' Service to buildings and dwellings: 5617
# 
# phi_dom_us: change in wages of in jobs that compete from the same pool of ag workers
# https://www.bls.gov/cew/downloadable-data-files.htm
# using more detailed industry specific wage data
qcew02_df <- read_excel("in/QCEW/qcew02.xlsx", sheet = 1) %>%
  as.data.frame()
qcew17_df <- read_excel("in/QCEW/qcew17.xlsx", sheet=1) %>%
  as.data.frame()

DOM_WAGEf <- qcew02_df %>%
  rename(annual_avg_emplvl02  = annual_avg_emplvl,
         total_annual_wages02 = total_annual_wages) %>%
  full_join(qcew17_df, by=c("FIPS", "Group")) %>%
  rename(annual_avg_emplvl17  = annual_avg_emplvl,
         total_annual_wages17 = total_annual_wages) %>%
  mutate(p_wage      = round(((Wage17 - (Wage02 * 1.37))/ (Wage02 * 1.37)*100),1))

DOM_WAGE_cs  <- DOM_WAGEf %>%
  filter(Group %in% c("cons", "services")) %>%
  dplyr::select(-Group, -p_wage, -Wage02, -Wage17) %>%
  group_by(FIPS) %>%
  summarize(across(everything(), sum, na.rm=T)) %>%
  mutate(Wage02 = total_annual_wages02 / annual_avg_emplvl02,
         Wage17 = total_annual_wages17 / annual_avg_emplvl17) %>%
  mutate(p_wage      = round(((Wage17 - (Wage02 * 1.37))/ (Wage02 * 1.37)*100),1)) %>%
  mutate(Group = "cs") %>%
  dplyr::select(FIPS, Group, p_wage) 

DOM_WAGE_agg <- DOM_WAGEf %>%
  # weighted avg of other jobs wage change
  filter(Group != "Ag") %>%
  group_by(FIPS) %>%
  # total employment in FIPS
  mutate(tot02 = sum(annual_avg_emplvl02, na.rm=T),
         tot17 = sum(annual_avg_emplvl17, na.rm=T)) %>%
  ungroup() %>%
  # share of employment in the sector/Group
  mutate(shr02 = annual_avg_emplvl02 / tot02,
         shr17 = annual_avg_emplvl17 / tot17) %>%
  mutate(shr = rowMeans(across(c(shr02, shr17)))) %>%
  group_by(FIPS) %>%
  summarize(p_wage = sum(shr*p_wage, na.rm=T)) %>%
  mutate(Group = "avg_nonag") %>%
  dplyr::select(FIPS, Group, p_wage) 

DOM_WAGE <- DOM_WAGEf %>%
  dplyr::select(FIPS, Group, p_wage) %>%
  rbind(DOM_WAGE_cs) %>%
  rbind(DOM_WAGE_agg) %>%
  pivot_wider(names_from = Group, values_from = p_wage, names_prefix = "p_") %>%
  mutate(across(everything(), ~ ifelse(is.na(.) | is.nan(.), 0, .)))

#CZ
DOM_WAGEz <- DOM_WAGEf %>%
  left_join(FIPS2CZ) %>%
  dplyr::select(-p_wage, -Wage02, -Wage17) %>%
  group_by(CZ, Group) %>%
  summarize(across(everything(), sum, na.rm=T)) %>%
  filter(!is.na(CZ)) %>%
  mutate(Wage02 = total_annual_wages02 / annual_avg_emplvl02,
         Wage17 = total_annual_wages17 / annual_avg_emplvl17,
         p_wage = round(((Wage17 - (Wage02 * 1.37))/ (Wage02 * 1.37)*100),1)) 

DOM_WAGEz_cs  <- DOM_WAGEz %>%
  filter(Group %in% c("cons", "services")) %>%
  dplyr::select(-Group, -p_wage, -Wage02, -Wage17) %>%
  group_by(CZ) %>%
  summarize(across(everything(), sum, na.rm=T)) %>%
  mutate(Wage02 = total_annual_wages02 / annual_avg_emplvl02,
         Wage17 = total_annual_wages17 / annual_avg_emplvl17) %>%
  mutate(p_wage      = round(((Wage17 - (Wage02 * 1.37))/ (Wage02 * 1.37)*100),1)) %>%
  mutate(Group = "cs") %>%
  dplyr::select(CZ, Group, p_wage) 

DOM_WAGEz_agg <- DOM_WAGEz %>%
  # weighted avg of other jobs wage change
  filter(Group != "Ag") %>%
  group_by(CZ) %>%
  mutate(tot02 = sum(annual_avg_emplvl02, na.rm=T),
         tot17 = sum(annual_avg_emplvl17, na.rm=T)) %>%
  ungroup() %>%
  mutate(shr02 = annual_avg_emplvl02 / tot02,
         shr17 = annual_avg_emplvl17 / tot17) %>%
  mutate(shr = rowMeans(across(c(shr02, shr17)))) %>%
  group_by(CZ) %>%
  summarize(p_wage = sum(shr*p_wage, na.rm=T)) %>%
  mutate(Group = "avg_nonag") %>%
  dplyr::select(CZ, Group, p_wage) 

DOM_WAGE_CZ <- DOM_WAGEz %>%
  dplyr::select(CZ, Group, p_wage) %>%
  rbind(DOM_WAGEz_cs) %>%
  rbind(DOM_WAGEz_agg) %>%
  pivot_wider(names_from = Group, values_from = p_wage, names_prefix = "p_") %>%
  mutate(across(everything(), ~ ifelse(is.na(.) | is.nan(.), 0, .)))

#PR
DOM_WAGEa <- DOM_WAGEf %>%
  left_join(FIPS2PR %>% dplyr::select(FIPS, PR9)) %>%
  dplyr::select(-p_wage, -Wage02, -Wage17) %>%
  group_by(PR9, Group) %>%
  summarize(across(everything(), sum, na.rm=T)) %>%
  filter(!is.na(PR9)) %>%
  mutate(Wage02 = total_annual_wages02 / annual_avg_emplvl02,
         Wage17 = total_annual_wages17 / annual_avg_emplvl17,
         p_wage = round(((Wage17 - (Wage02 * 1.37))/ (Wage02 * 1.37)*100),1)) 

DOM_WAGEa_cs  <- DOM_WAGEa %>%
  filter(Group %in% c("cons", "services")) %>%
  dplyr::select(-Group, -p_wage, -Wage02, -Wage17) %>%
  group_by(PR9) %>%
  summarize(across(everything(), sum, na.rm=T)) %>%
  mutate(Wage02 = total_annual_wages02 / annual_avg_emplvl02,
         Wage17 = total_annual_wages17 / annual_avg_emplvl17) %>%
  mutate(p_wage      = round(((Wage17 - (Wage02 * 1.37))/ (Wage02 * 1.37)*100),1)) %>%
  mutate(Group = "cs") %>%
  dplyr::select(PR9, Group, p_wage) 

DOM_WAGEa_agg <- DOM_WAGEa %>%
  # weighted avg of other jobs wage change
  filter(Group != "Ag") %>%
  group_by(PR9) %>%
  mutate(tot02 = sum(annual_avg_emplvl02, na.rm=T),
         tot17 = sum(annual_avg_emplvl17, na.rm=T)) %>%
  ungroup() %>%
  mutate(shr02 = annual_avg_emplvl02 / tot02,
         shr17 = annual_avg_emplvl17 / tot17) %>%
  mutate(shr = rowMeans(across(c(shr02, shr17)))) %>%
  group_by(PR9) %>%
  summarize(p_wage = sum(shr*p_wage, na.rm=T)) %>%
  mutate(Group = "avg_nonag") %>%
  dplyr::select(PR9, Group, p_wage) 

DOM_WAGE_PR <- DOM_WAGEa %>%
  dplyr::select(PR9, Group, p_wage) %>%
  rbind(DOM_WAGEa_cs) %>%
  rbind(DOM_WAGEa_agg) %>%
  pivot_wider(names_from = Group, values_from = p_wage, names_prefix = "p_") %>%
  mutate(across(everything(), ~ ifelse(is.na(.) | is.nan(.), 0, .)))

#   iii.  IMMIGRANT workers ----
load("in/CPS/acs02_df.RData")
load("in/CPS/acs17_df.RData")

immigrant02_CZ <- acs02.df %>%
  full_join(FIPS2CZ) %>%
  group_by(CZ) %>%
  summarise(total_weight02   = sum(total_weight02, na.rm = TRUE),
            immigrant_wt02  = sum(immigrant_wt02, na.rm = TRUE),
            s_immi02 = immigrant_wt02 / total_weight02) %>% # weighted (no. of immigrants vs no. of workers in ag)
  filter(!is.na(CZ))

immigrant02_PR <- acs02.df %>%
  full_join(FIPS2PR %>% dplyr::select(FIPS, PR9)) %>%
  group_by(PR9) %>%
  summarise(total_weight02   = sum(total_weight02, na.rm = TRUE),
            immigrant_wt02  = sum(immigrant_wt02, na.rm = TRUE),
            s_immi02 = immigrant_wt02 / total_weight02) %>%
  filter(!is.na(PR9))

immigrant17_CZ <- acs17.df %>%
  full_join(FIPS2CZ) %>%
  group_by(CZ) %>%
  summarise(total_weight17   = sum(total_weight17, na.rm = TRUE),
            immigrant_wt17  = sum(immigrant_wt17, na.rm = TRUE),
            s_immi17 = immigrant_wt17 / total_weight17) %>%
  filter(!is.na(CZ))

immigrant17_PR <- acs17.df %>%
  full_join(FIPS2PR %>% dplyr::select(FIPS, PR9)) %>%
  group_by(PR9) %>%
  summarise(total_weight17   = sum(total_weight17, na.rm = TRUE),
            immigrant_wt17  = sum(immigrant_wt17, na.rm = TRUE),
            s_immi17 = immigrant_wt17 / total_weight17) %>%
  filter(!is.na(PR9))

immigrant_CZ <- immigrant02_CZ %>%
  full_join(immigrant17_CZ) %>%
  mutate(s_immi = rowMeans(across(c(s_immi02, s_immi17)), na.rm=T)) %>%
  mutate(across(everything(), ~ ifelse(is.na(.) | is.nan(.), 0, .)))

immigrant_PR <- immigrant02_PR %>%
  full_join(immigrant17_PR) %>%
  mutate(s_immi = rowMeans(across(c(s_immi02, s_immi17)), na.rm=T)) %>%
  mutate(across(everything(), ~ ifelse(is.na(.) | is.nan(.), 0, .)))

#         phi_immi:     Shock to immigrant ag workers: Charlton & Taylor (2016), from annual to strudy period ----
phi_immi_annual <- c(0.76, 1, 1.17) # using 99% CI from CT2016
phi_immi <- 100 - (100 * (1 - phi_immi_annual/100) ^ (2017 - 2002) ) # + sign matches the labor ss curve in theory
cat("Immigrant labor supply shock (phi_immi), low/med/high:", paste(round(phi_immi, 2), collapse=", "), "\n")

#    iv.  phi_(DOM + IMMI), with Scenarios (sectors: Non_ag/CS) + (markup: LMH) ----
# combos and names
sector   <- c("p_avg_nonag", "p_cs")
combos   <- expand.grid(phi_val = phi_immi, sector = sector, stringsAsFactors = FALSE)
comboz.name <- c("PHILz_NL", "PHILz_NM", "PHILz_NH",
                 "PHILz_CL", "PHILz_CM", "PHILz_CH")
comboa.name <- c("PHILa_NL", "PHILa_NM", "PHILa_NH",
                 "PHILa_CL", "PHILa_CM", "PHILa_CH")

# Function, combines 
PHIL <- function(input_df, sec_col, phi_val, immi_df, unit_col = "CZ") {
  input_df %>%
    dplyr::select(all_of(unit_col), all_of(sec_col)) %>%
    left_join(immi_df %>% dplyr::select(all_of(unit_col), s_immi), by = unit_col) %>%
    #(ss shock to MX farm workers * share of immigrants) + (1-share)*competing sector of employment )
    mutate(phil = phi_val * s_immi + .data[[sec_col]] * (1 - s_immi)) }

# Run PHIL for each combination and assign names dynamically
PHILz_list <- vector("list", length(comboz.name))
PHILa_list <- vector("list", length(comboa.name))

for (i in seq_len(nrow(combos))) {
  df_out <- PHIL(input_df = DOM_WAGE_CZ,
                 # edited 11-03-26
                 # input_df = DOM_EMP_CZ,
                 sec_col  = combos$sector[i],
                 phi_val  = combos$phi_val[i],
                 immi_df  = immigrant_CZ,
                 unit_col = "CZ") %>%
    mutate(sector  = combos$sector[i],
           phi_val = combos$phi_val[i])
  
  # Save into list and environment with name
  PHILz_list[[i]] <- df_out
  assign(comboz.name[i], df_out, envir = .GlobalEnv)
}

for (i in seq_len(nrow(combos))) {
  df_out <- PHIL(input_df = DOM_WAGE_PR,
                 # edited 11-03-26
                 # input_df = DOM_EMP_PR,
                 sec_col  = combos$sector[i],
                 phi_val  = combos$phi_val[i],
                 immi_df  = immigrant_PR,
                 unit_col = "PR9") %>%
    mutate(sector  = combos$sector[i],
           phi_val = combos$phi_val[i])
  
  # Save into list and environment with name
  PHILa_list[[i]] <- df_out
  assign(comboa.name[i], df_out, envir = .GlobalEnv)
}

# Name the list elements for clarity
names(PHILz_list) <- comboz.name
names(PHILa_list) <- comboa.name
names(PHILz_list)
names(PHILa_list)


# 2.6   Resource = land+water ----
cat("[2.6] Combining land + water resource shocks...\n")
#     i.  land that was Pasture on Cropland in 02 but not in 2017 ----
# From USDA documentation: https://www.nass.usda.gov/Publications/AgCensus/2017/Full_Report/Volume_1%2C_Chapter_1_US/usappxb.pdf?utm_source=chatgpt.com
# AG LAND = CROPLAND + HARVESTED LAND + PASTURE LAND + Other
# Cropland = Harvested + Cropland used as pasture + failed crops + summer fallow + idle
# idle includes CRP + fallow + cover crops
# calculating change in supply of cropland, exogenous to my analysis, 
# as the absolute change in IPF land between 2002 - 2017 as share of CROPLAND in 2002

SCLAND9702 <- read_excel(file.path("in/USDA_QUICKSTAT/CROPLAND9702.xlsx"))
SCLAND0712 <- read_excel(file.path("in/USDA_QUICKSTAT/CROPLAND0712.xlsx")) 
SCLAND1722 <- read_excel(file.path("in/USDA_QUICKSTAT/CROPLAND1722.xlsx"))

SCLAND_long     <- SCLAND9702 %>%
  rbind(SCLAND0712) %>%
  rbind(SCLAND1722) %>%
  rename(State.ANSI = `State ANSI`,
         County.ANSI = `County ANSI`,
         type = `Data Item`) %>%
  dplyr::select(-County) %>%
  mutate(FIPS = as.numeric(sprintf("%d%03d", State.ANSI,  County.ANSI))) %>%
  subset(FIPS <= 14999 | FIPS >= 15999) %>%
  subset(FIPS <= 1999 | FIPS >= 2999) %>%
  mutate(type = ifelse(type == "AG LAND, CROPLAND - ACRES", "CROPLAND",
                       ifelse(type == "AG LAND, CROPLAND, (EXCL HARVESTED & PASTURED), ALL CROPS FAILED - ACRES", "FAILED",
                              ifelse(type == "AG LAND, CROPLAND, (EXCL HARVESTED & PASTURED), CULTIVATED SUMMER FALLOW - ACRES", "FALLOW",
                                     ifelse(type == "AG LAND, CROPLAND, (EXCL HARVESTED & PASTURED), IDLE - ACRES", "IDLE", 
                                            ifelse(type == "AG LAND, CROPLAND, PASTURED ONLY - ACRES", "PASTURE", "HARVESTED")))))) %>%
  mutate(Value = as.numeric(Value) * acres2ha) %>%
  dplyr::select(FIPS, Year, type, Value) %>%
  arrange(FIPS, Year)

SCLAND_wide <- SCLAND_long %>%
  mutate(l_year = paste(type, Year, sep = "_")) %>%
  dplyr::select(FIPS, l_year, Value) %>%
  pivot_wider(names_from = l_year, values_from = Value) %>%
  mutate(CROPLAND_2002_ltavg  = coalesce( rowMeans(across(c(CROPLAND_1997,  CROPLAND_2002,  CROPLAND_2007)), na.rm=T), 100),
         PAST_2002_ltavg      = coalesce( rowMeans(across(c(PASTURE_1997,   PASTURE_2002,  PASTURE_2007)), na.rm=T), 0),
         FAILED_2002_ltavg    = coalesce( rowMeans(across(c(FAILED_1997,    FAILED_2002,    FAILED_2007)), na.rm=T), 0),
         FALLOW_2002_ltavg    = coalesce( rowMeans(across(c(FALLOW_1997,    FALLOW_2002,    FALLOW_2007)), na.rm=T), 0),
         IDLE_2002_ltavg      = coalesce( rowMeans(across(c(                IDLE_2002,      IDLE_2007)), na.rm=T), 0),
         HARV_2002_ltavg      = coalesce( rowMeans(across(c(HARVESTED_1997, HARVESTED_2002, HARVESTED_2007)), na.rm=T), 0),
         
         CROPLAND_2017_ltavg  = coalesce( rowMeans(across(c(CROPLAND_2012,  CROPLAND_2017,  CROPLAND_2022)), na.rm=T), 100),
         PAST_2017_ltavg      = coalesce( rowMeans(across(c(PASTURE_2012,   PASTURE_2017,   PASTURE_2022)), na.rm=T), 0),
         FAILED_2017_ltavg    = coalesce( rowMeans(across(c(FAILED_2012,    FAILED_2017,    FAILED_2022)), na.rm=T), 0),
         FALLOW_2017_ltavg    = coalesce( rowMeans(across(c(FALLOW_2012,    FALLOW_2017,    FALLOW_2022)), na.rm=T), 0),
         IDLE_2017_ltavg      = coalesce( rowMeans(across(c(IDLE_2012,      IDLE_2017,      IDLE_2022)), na.rm=T), 0),
         HARV_2017_ltavg      = coalesce( rowMeans(across(c(HARVESTED_2012, HARVESTED_2017, HARVESTED_2022)), na.rm=T), 0)) %>%
  
  mutate(FAL_2002_ltavg = rowSums(across(c(FALLOW_2002_ltavg, FAILED_2002_ltavg, FALLOW_2002_ltavg)), na.rm=T),
         FAL_2017_ltavg = rowSums(across(c(FAILED_2017_ltavg, FALLOW_2017_ltavg)), na.rm=T)) %>%
  dplyr::select(-starts_with("FALLOW")) %>%
  dplyr::select(-starts_with("FAILED")) %>%
  dplyr::select(FIPS, ends_with("_ltavg"))

SCLAND_US <- SCLAND_wide %>%
  summarize(across(c(ends_with("_ltavg")), sum, na.rm=T)) %>%
  
  mutate(p_id02     = IDLE_2002_ltavg / CROPLAND_2002_ltavg * 100,
         p_id17     = IDLE_2017_ltavg / CROPLAND_2017_ltavg * 100,
         
         p_past02   = PAST_2002_ltavg / CROPLAND_2002_ltavg * 100,
         p_past17   = PAST_2017_ltavg / CROPLAND_2017_ltavg * 100,
         
         p_fal02  = FAL_2002_ltavg / CROPLAND_2002_ltavg * 100,
         p_fal17  = FAL_2017_ltavg / CROPLAND_2017_ltavg * 100,
         
         p_ncrp02  = p_id02 + p_past02 + p_fal02,
         p_ncrp17  = p_id17 + p_past17 + p_fal17,
         ipf_lev     = rowSums(across(c(IDLE_2017_ltavg, PAST_2017_ltavg, FAL_2017_ltavg))) - rowSums(across(c(IDLE_2002_ltavg, PAST_2002_ltavg, FAL_2002_ltavg))) )

SCLAND <- SCLAND_wide %>%
  group_by(FIPS) %>%
  summarize(across(c(ends_with("_ltavg")), sum, na.rm=T)) %>%
  
  mutate(p_id02     = IDLE_2002_ltavg / CROPLAND_2002_ltavg * 100,
         p_id17     = IDLE_2017_ltavg / CROPLAND_2017_ltavg * 100,
         
         p_past02   = PAST_2002_ltavg / CROPLAND_2002_ltavg * 100,
         p_past17   = PAST_2017_ltavg / CROPLAND_2017_ltavg * 100,
         
         p_fal02  = FAL_2002_ltavg / CROPLAND_2002_ltavg * 100,
         p_fal17  = FAL_2017_ltavg / CROPLAND_2017_ltavg * 100,
         
         p_ncrp02  = p_id02 + p_past02 + p_fal02,
         p_ncrp17  = p_id17 + p_past17 + p_fal17,
         ipf_lev     = rowSums(across(c(IDLE_2017_ltavg, PAST_2017_ltavg, FAL_2017_ltavg))) - rowSums(across(c(IDLE_2002_ltavg, PAST_2002_ltavg, FAL_2002_ltavg))) ) %>%
  rbind(c(FIPS=99999, SCLAND_US))

SCLAND_CZ <- SCLAND_wide %>%
  left_join(FIPS2CZ) %>%
  filter(!is.na(CZ)) %>%
  group_by(CZ) %>%
  summarize(across(c(ends_with("_ltavg")), sum, na.rm=T)) %>%
  
  mutate(p_id02     = IDLE_2002_ltavg / CROPLAND_2002_ltavg * 100,
         p_id17     = IDLE_2017_ltavg / CROPLAND_2017_ltavg * 100,
         
         p_past02   = PAST_2002_ltavg / CROPLAND_2002_ltavg * 100,
         p_past17   = PAST_2017_ltavg / CROPLAND_2017_ltavg * 100,
         
         p_fal02  = FAL_2002_ltavg / CROPLAND_2002_ltavg * 100,
         p_fal17  = FAL_2017_ltavg / CROPLAND_2017_ltavg * 100,
         
         p_ncrp02  = p_id02 + p_past02 + p_fal02,
         p_ncrp17  = p_id17 + p_past17 + p_fal17,
         ipf_lev     = rowSums(across(c(IDLE_2017_ltavg, PAST_2017_ltavg, FAL_2017_ltavg))) - rowSums(across(c(IDLE_2002_ltavg, PAST_2002_ltavg, FAL_2002_ltavg))) ) %>%
  rbind(c(CZ=606, SCLAND_US))

SCLAND_PR <- SCLAND_wide %>%
  left_join(FIPS2PR) %>%
  group_by(PR9) %>%
  summarize(across(c(ends_with("_ltavg")), sum, na.rm=T)) %>%
  
  mutate(p_id02     = IDLE_2002_ltavg / CROPLAND_2002_ltavg * 100,
         p_id17     = IDLE_2017_ltavg / CROPLAND_2017_ltavg * 100,
         
         p_past02   = PAST_2002_ltavg / CROPLAND_2002_ltavg * 100,
         p_past17   = PAST_2017_ltavg / CROPLAND_2017_ltavg * 100,
         
         p_fal02  = FAL_2002_ltavg / CROPLAND_2002_ltavg * 100,
         p_fal17  = FAL_2017_ltavg / CROPLAND_2017_ltavg * 100,
         
         p_ncrp02  = p_id02 + p_past02 + p_fal02,
         p_ncrp17  = p_id17 + p_past17 + p_fal17,
         ipf_lev     = rowSums(across(c(IDLE_2017_ltavg, PAST_2017_ltavg, FAL_2017_ltavg))) - rowSums(across(c(IDLE_2002_ltavg, PAST_2002_ltavg, FAL_2002_ltavg))) ) %>%
  rbind(c(PR9=10, SCLAND_US)) %>%
  left_join(PR)

# if ifp_lev  >0, decrease in cropland supply in 2017
# if ipf_lev  <0, increase in cropland supply in 2017

#    ii.  land that was idle, fallow or pasture in 2002 and not in 2017 ----
# could have been converted to cropland or pastureland 
# adjust p_idf with share pastureland/(crop + pastureland)
PASLAND9722 <- read_excel(file.path("in/USDA_QUICKSTAT/PASTURELAND_allyrs.xlsx"))

PASLAND_long     <- PASLAND9722 %>%
  rename(State.ANSI = `State ANSI`,
         County.ANSI = `County ANSI`,
         type = `Data Item`) %>%
  dplyr::select(-County) %>%
  mutate(FIPS = as.numeric(sprintf("%d%03d", State.ANSI,  County.ANSI))) %>%
  subset(FIPS <= 14999 | FIPS >= 15999) %>%
  subset(FIPS <= 1999 | FIPS >= 2999) %>%
  mutate(Value = as.numeric(Value) * acres2ha) %>%
  dplyr::select(FIPS, Year, Value) %>%
  arrange(FIPS, Year)

PASLAND_wide <- PASLAND_long %>%
  pivot_wider(names_from = Year, values_from = Value, names_prefix = "PASLAND_") %>%
  mutate(PASLAND2002_ltavg  = coalesce( rowMeans(across(c(PASLAND_1997,  PASLAND_2002,  PASLAND_2007)), na.rm=T), 100),
         PASLAND2017_ltavg  = coalesce( rowMeans(across(c(PASLAND_2012,  PASLAND_2017,  PASLAND_2022)), na.rm=T), 100)) %>%
  left_join(subset(SCLAND, select=c(FIPS, CROPLAND_2002_ltavg, CROPLAND_2017_ltavg)))

PASLAND <- PASLAND_wide %>%
  mutate(shr_past02 = round( PASLAND2002_ltavg / (PASLAND2002_ltavg + CROPLAND_2002_ltavg), 3),
         shr_past17 = round( PASLAND2017_ltavg / (PASLAND2017_ltavg + CROPLAND_2017_ltavg), 3)) %>%
  mutate(shr_past   = rowMeans(across(c(shr_past02, shr_past02)))) %>%
  dplyr::select(FIPS, shr_past)

PASLAND_US <- PASLAND_wide %>%
  summarize(PASLAND2002_ltavg   = sum(PASLAND2002_ltavg, na.rm=T),
            PASLAND2017_ltavg   = sum(PASLAND2002_ltavg, na.rm=T),
            CROPLAND_2002_ltavg = sum(CROPLAND_2002_ltavg, na.rm=T),
            CROPLAND_2017_ltavg = sum(CROPLAND_2017_ltavg, na.rm=T)) %>%
  mutate(shr_past02 = round( PASLAND2002_ltavg / (PASLAND2002_ltavg + CROPLAND_2002_ltavg), 3),
         shr_past17 = round( PASLAND2017_ltavg / (PASLAND2017_ltavg + CROPLAND_2017_ltavg), 3)) %>%
  mutate(shr_past   = rowMeans(across(c(shr_past02, shr_past02)))) %>%
  dplyr::select(shr_past)

PASLAND_CZ <- PASLAND_wide %>%
  left_join(FIPS2CZ) %>%
  group_by(CZ) %>%
  summarize(PASLAND2002_ltavg   = sum(PASLAND2002_ltavg, na.rm=T),
            PASLAND2017_ltavg   = sum(PASLAND2002_ltavg, na.rm=T),
            CROPLAND_2002_ltavg = sum(CROPLAND_2002_ltavg, na.rm=T),
            CROPLAND_2017_ltavg = sum(CROPLAND_2017_ltavg, na.rm=T)) %>%
  mutate(shr_past02 = round( PASLAND2002_ltavg / (PASLAND2002_ltavg + CROPLAND_2002_ltavg), 3),
         shr_past17 = round( PASLAND2017_ltavg / (PASLAND2017_ltavg + CROPLAND_2017_ltavg), 3)) %>%
  mutate(shr_past   = rowMeans(across(c(shr_past02, shr_past02)))) %>%
  dplyr::select(CZ, shr_past) %>%
  filter(!is.na(CZ)) %>%
  rbind(c(CZ =606, PASLAND_US))

PASLAND_PR <- PASLAND_wide %>%
  left_join(subset(FIPS2PR, select=c(FIPS, PR9))) %>%
  group_by(PR9) %>%
  summarize(PASLAND2002_ltavg   = sum(PASLAND2002_ltavg, na.rm=T),
            PASLAND2017_ltavg   = sum(PASLAND2002_ltavg, na.rm=T),
            CROPLAND_2002_ltavg = sum(CROPLAND_2002_ltavg, na.rm=T),
            CROPLAND_2017_ltavg = sum(CROPLAND_2017_ltavg, na.rm=T)) %>%
  mutate(shr_past02 = round( PASLAND2002_ltavg / (PASLAND2002_ltavg + CROPLAND_2002_ltavg), 3),
         shr_past17 = round( PASLAND2017_ltavg / (PASLAND2017_ltavg + CROPLAND_2017_ltavg), 3)) %>%
  mutate(shr_past   = rowMeans(across(c(shr_past02, shr_past02)))) %>%
  dplyr::select(PR9, shr_past) %>%
  filter(!is.na(PR9)) %>%
  rbind(c(PR9 = 10, PASLAND_US))

#   iii.  Urbanization shock  ----
# Initialize an empty list to store data frames
urban_data_list <- list()

# Loop through PR1 to PR9
for (pr in 1:9) {
  file_path <- paste0("in/USGS/USGS_processing0315/USGS_PR", pr, ".xlsx")  # Dynamic file path
  
  temp_data <- data.frame(read_excel(file.path(file_path))) 
  df <- data.frame(FIPS = temp_data$FIPS) %>%
    mutate(cland02  = temp_data[, 3],
           cpland02 = temp_data[, 5],
           c2uland  = temp_data[, 7],
           cp2uland = temp_data[, 9]) 
  urban_data_list[[paste0("PR", pr)]] <- df  }

# pizel size to area conversion
p2h <- (30 * 30) / (100 * 100) # 1 30x30mpixel = 0.09 ha

# Bind all data frames into a single data frame
URBAN <- bind_rows(urban_data_list) %>%
  mutate(cland02    = replace_na(cland02  * p2h,0),
         cpland02   = replace_na(cpland02 * p2h,0),
         # >0 c2uland number means backward shift in cropland ss
         c2uland    = replace_na(c2uland  * p2h,0),
         cp2uland   = replace_na(cp2uland * p2h,0),
         p_c2uland  = ifelse(cland02  ==0,0, round( c2uland  / cland02  *100, 2)),
         p_cp2uland = ifelse(cpland02 ==0,0, round( cp2uland / cpland02 *100, 2)))

URBAN_US <- URBAN %>%
  summarize(across(c(cland02, cpland02, c2uland, cp2uland), sum, na.rm = TRUE), .groups = "drop") %>%
  mutate(p_c2uland  = ifelse(cland02 ==0,  0, round( cp2uland / cpland02 *100, 2)),
         p_cp2uland = ifelse(cpland02 ==0, 0, round( cp2uland / cpland02 *100, 2))) 

# changes in q_fn too large!! use tfn to adjust such that weighted sum = 5.2
aggregate_URBAN <- function(data, join_table, group_var) {
  data %>%
    left_join(join_table, by = "FIPS") %>%
    dplyr::select(all_of(group_var), cland02, cpland02, c2uland, cp2uland) %>%
    group_by(across(all_of(group_var))) %>%
    summarize(across(c(cland02, cpland02, c2uland, cp2uland), sum, na.rm = TRUE), .groups = "drop") %>%
    mutate(p_c2uland  = ifelse(cland02 ==0,  0, round( c2uland / cland02 *100, 2)),
           p_cp2uland = ifelse(cpland02 ==0, 0, round( cp2uland / cpland02 *100, 2)))  %>%
    filter(!is.na(!!sym(group_var)))  # Drop rows where the group variable is NA
}

# Apply function for different aggregation levels
URBAN_CZ <- aggregate_URBAN(URBAN, FIPS2CZ, "CZ") %>%
  rbind(c(CZ=606, URBAN_US))
URBAN_PR <- aggregate_URBAN(URBAN, FIPS2PR, "PR9") %>%
  rbind(c(PR9=10, URBAN_US))

#     v.  combine land ss shock (non-harvested land 2 cland and urbanlan2cland) ----
PHIC <- SCLAND %>%
  dplyr::select(FIPS, CROPLAND_2002_ltavg, ipf_lev) %>%
  left_join(subset(PASLAND, select=c(FIPS, shr_past))) %>%
  # adj land moved out of crop(pasture, idle,fal) with (1-shr_past)
  mutate(ipf_lev_adj = ipf_lev * (1-shr_past)) %>%
  left_join(URBAN) %>%
  mutate(p_ipf    = ifelse(CROPLAND_2002_ltavg==0,0, ipf_lev_adj / CROPLAND_2002_ltavg * 100),
         p_c2u    = ifelse(CROPLAND_2002_ltavg==0,0, c2uland / CROPLAND_2002_ltavg * 100),
         p_c2u_nlcd   = ifelse(cland02==0,0, c2uland / cland02 * 100),
         p_cp2u_nlcd   = ifelse(cpland02==0,0, cp2uland / cpland02 * 100),
         phic_lev = rowSums(across(c(ipf_lev_adj,c2uland)), na.rm=T),
         phic     = ifelse(CROPLAND_2002_ltavg==0,0, phic_lev / CROPLAND_2002_ltavg * 100)) %>%
  mutate(phic  = ifelse(phic>90,  90, phic))

aggregate_PHIC <- function(data_SCLAND, data_URBAN, data_PAST, group_var) {
  data_SCLAND %>%
    dplyr::select(group_var, CROPLAND_2002_ltavg, ipf_lev) %>%
    left_join(data_PAST, by = group_var) %>%
    # adj land moved out of crop(pasture, idle,fal) with (1-shr_past)
    mutate(ipf_lev_adj = ipf_lev * (1-shr_past)) %>%
    left_join(data_URBAN, by = group_var) %>%
    dplyr::select(all_of(group_var),  CROPLAND_2002_ltavg, cland02, cpland02, shr_past, ipf_lev, ipf_lev_adj, c2uland, cp2uland) %>%
    mutate(p_pas    = round((cpland02 - cland02)/cland02*100, 2),
           p_ipf    = ifelse(CROPLAND_2002_ltavg==0,0, ipf_lev_adj / CROPLAND_2002_ltavg * 100),
           p_c2u    = ifelse(CROPLAND_2002_ltavg==0,0, c2uland / CROPLAND_2002_ltavg * 100),
           p_c2u_nlcd    = ifelse(cland02==0,0,  c2uland / cland02 * 100),
           phic_lev = rowSums(across(c(ipf_lev_adj,c2uland)), na.rm=T),
           phic      = ifelse(CROPLAND_2002_ltavg==0,0, phic_lev / CROPLAND_2002_ltavg * 100)) %>%
    mutate(phic  = ifelse(phic>90,  90, phic)) %>%
    filter(!is.na(!!sym(group_var)))  # Drop rows where the group variable is NA
}

PHIC_CZ <- aggregate_PHIC(SCLAND_CZ, URBAN_CZ, PASLAND_CZ, "CZ")
PHIC_PR <- aggregate_PHIC(SCLAND_PR, URBAN_PR, PASLAND_PR, "PR9") 

#    vi.  investigating cases where phic<0 (land ss increase) AND qc<0 & pc>0 ----
inv <- merge(PHIC_CZ, QLAND_CZ, by=c("CZ")) %>%
  left_join(subset(PASLAND_CZ, select=c(CZ, shr_past))) %>%
  left_join(PLAND_CZ) %>%
  left_join(CZ2PR) %>%
  dplyr::select(CZ, PR9, shr_past, CROPLAND_2002_ltavg, ipf_lev, ipf_lev_adj, p_ipf, cland02, cpland02, c2uland, p_c2u, phic, phic_lev, 
                HLAND1997, HLAND2002, HLAND2007, HLAND2002_ltavg, HLAND2012, HLAND2017, HLAND2022, HLAND2017_ltavg, qc,
                RENT97, RENT02, RENT07, RENT12, RENT17, RENT22, pc) %>%
  filter(phic<0 & qc<0 & pc>0) %>%
  left_join(subset(QC_CZ, select=c(CZ,q_agg)))

inv.SCLAND_CZ <- SCLAND_CZ %>% 
  filter(CZ %in% inv$CZ) %>%
  left_join(QLAND_CZ)


# # CZs in FR where phic <0 in inv
inv.cz <- inv$CZ

# 
# # such CZs removing increase in land ss, retain shock due to urbanization
PHIC_CZ <- PHIC_CZ %>%
  left_join(CZ2PR) %>%
  left_join(subset(QLAND_CZ, select=c(CZ, qc))) %>%
  left_join(subset(PLAND_CZ, select=c(CZ,pc))) %>%
  #
  mutate(phic = ifelse(CZ %in% inv.cz & PR9!=3, p_c2u, phic)) %>%
  mutate(phic = ifelse(PR9==2 & qc<0 & pc<0 & phic<0, p_c2u, phic)) %>%
  dplyr::select(-PR9) %>%
  dplyr::select(-phic_lev) %>%
  # use these to plot in levels terms
  mutate(uland_lev = p_c2u/100*CROPLAND_2002_ltavg,
         ipf_lev   = p_ipf/100*CROPLAND_2002_ltavg,
         phic_lev =  phic/100*CROPLAND_2002_ltavg)

# recreate PHIC_PR
PHIC_PR <- PHIC_CZ %>%
  left_join(CZ2PR) %>%
  dplyr::select(CZ, PR9, CROPLAND_2002_ltavg, phic, p_ipf, p_c2u) %>%
  mutate(p_ipf = ifelse(CZ %in% inv.cz & PR9!=3, 0, p_ipf)) %>%
  mutate(uland_lev = p_c2u/100*CROPLAND_2002_ltavg,
         ipf_lev   = p_ipf/100*CROPLAND_2002_ltavg,
         phic_lev =  phic/100*CROPLAND_2002_ltavg) %>%
  group_by(PR9) %>%
  summarize(CROPLAND_2002_ltavg = sum(CROPLAND_2002_ltavg),
            uland_lev = sum(uland_lev),
            ipf_lev   = sum(ipf_lev),
            phic_lev            = sum(phic_lev)) %>%
  mutate(p_c2u = uland_lev/CROPLAND_2002_ltavg*100*-1,
         p_ipf = ipf_lev/CROPLAND_2002_ltavg*100*-1,
         phic = phic_lev / CROPLAND_2002_ltavg*100*-1)

# 2.7.  TFP change ----
cat("[2.7] Computing TFP change for initial values...\n")
GRIDDATA2010 <- read_har(file.path("in/GRIDDATA.har"),                  toLowerCase = F)
VCROPg <- as.data.frame(GRIDDATA2010$VCRP) %>%
  mutate(SNO=1:75666) %>%
  mutate(VCROPg = Irrigated + Rainfed)

# gridcell VCROP in SIMPLE-G
VC <- VCROPg[(1:75651), ] %>%
  mutate(VCROP = `Irrigated` + `Rainfed`) %>%
  dplyr::select(SNO, VCROP) %>%
  left_join(subset(GRID2CZ, select=c("SNO", "CZ")), by=c("SNO"))   %>%
  left_join(subset(GRID2FIPS, select=c("SNO", "STATE_CODE")), by=c("SNO"))

AOCROPg  <- read_excel(file.path("in/validation_aocrop.xlsx"))
AOCROPg <- AOCROPg %>%
  left_join(subset(VC, select=c("SNO", "VCROP")), by=c("SNO")) %>%
  rename(VCROPg = VCROP) %>%
  left_join(GRID2CZ, by=c("SNO")) %>%
  left_join(subset(GRID2PR, select=c("SNO", "PR")), by=c("SNO")) %>%
  group_by(PR) %>%
  mutate(VCROPm = sum(VCROPg), na.rm=T) %>%
  ungroup() %>%
  group_by(CZ) %>%
  mutate(VCROPz = sum(VCROPg), na.rm=T) %>%
  ungroup() %>%
  group_by(FIPS) %>%
  mutate(VCROP_fips = sum(VCROPg), na.rm=T) %>%
  ungroup() 

AOCROP_US <- AOCROPg %>%
  mutate(VCROP_US = sum(VCROPg), na.rm=T) %>%
  summarize(ao_us = sum(VCROPg/VCROP_US*p_ACROPg)) %>%
  as.numeric()

AOCROP_PR <- AOCROPg %>%
  rename(PR9 = PR) %>%
  group_by(PR9) %>%
  summarize(ao_pr = sum(VCROPg/VCROPm*p_ACROPg)) %>%
  rbind(c(10, AOCROP_US)) %>%
  mutate(ao_pr = round(ao_pr, 1))

# 3:    END OF DATA PREP: Create dfs for first stage estimation - data_fips, data_cz and data_pr----
cat("[3] Assembling final data frames for first-stage estimation ...\n")
#     i.  NON_LABOR: Collecting variables for FIPS, CZ and PR ----
#         a. data_fips ----
data_fips <- FIPSinSIMPLE %>%
  dplyr::select(FIPS) %>%
  full_join(subset(QCROP, select=c(FIPS,  VCROP2002_ltavg, VCROP2017_ltavg, vc, q_agg, p_agg, qc, q_gr, q_fn, q_vg, q_ot))) %>% # only q_agg and qaug before 111124
  rename(qcrop = qc,
         q = q_agg,
         vcrop = vc) %>%
  mutate(vcrop = ifelse(vcrop>250, 250, vcrop)) %>%
  left_join(VLAB %>% dplyr::select(FIPS, VLAB2002_ltavg, VLAB2017_ltavg)) %>%
  full_join(QZ %>% dplyr::select(FIPS, vchem, qchem, vseed, qseed, vgas,  qgas,  vmach, qmach, voth, qoth, vz, qz, pz )) %>%
  full_join(subset(QLAND,    select=c("FIPS", "HLAND2002_ltavg", "HLAND2017_ltavg", "qc"))) %>%
  full_join(subset(PLAND, select=c("FIPS", "RENT_TOT02", "RENT_TOT17", "RENT02", "RENT17", "pc"))) %>%
  full_join(subset(WATER, select=c("FIPS", "WAT02", "WAT15", "qw"))) %>%
  full_join(subset(PHIC, select=c(FIPS, CROPLAND_2002_ltavg, shr_past, phic))) %>%
  full_join(subset(VZ_wide_ltavg, select=c("FIPS", "VZ_2002_ltavg", "VZ_2017_ltavg"))) %>%
  rename(VZ_02 = VZ_2002_ltavg,
         VZ_17 = VZ_2017_ltavg) %>%
  # removes FIPS not in SIMPLE
  filter(FIPS %in% FIPSinSIMPLE$FIPS) %>%
  # replace all other qXXX with 0 if NA
  mutate(across(everything(), ~ ifelse(is.infinite(.), 90, .))) %>%
  mutate(across(everything(), ~ ifelse(is.na(.), 0, .))) %>%
  left_join(FIPS2PR %>% dplyr::select(FIPS, PR9)) %>%
  left_join(FIPS2CZ)

#         b. data_cz ----
data_cz <- data.frame(CZ = 1:606)  %>%
  full_join(subset(QC_CZ, select=c(CZ, VCROP2002_ltavg, VCROP2017_ltavg, vc, q_agg, p_agg, qc_old, q_gr, q_fn, q_vg, q_ot, tgr, tfn, tvg, tot))) %>% # only q_agg and qaug before 111124
  rename(qcrop = qc_old,q = q_agg, vcrop = vc) %>%
  mutate(vcrop = ifelse(vcrop>250, 250, vcrop)) %>%
  full_join(QZ_CZ %>% dplyr::select(CZ, vchem, qchem, vseed, qseed, vgas,  qgas,  vmach, qmach, voth, qoth, vz, qz, pz )) %>%
  full_join(subset(QLAND_CZ,    select=c("CZ", "HLAND2002_ltavg", "HLAND2017_ltavg", "qc"))) %>%
  full_join(subset(PLAND_CZ, select=c(CZ, RENT_TOT02, RENT_TOT17, RENT02, RENT17, pc))) %>%
  full_join(subset(WATER_CZ, select=c(CZ, WAT02, WAT15, qw))) %>%
  full_join(subset(PHIC_CZ, select=c(CZ, CROPLAND_2002_ltavg, shr_past, phic))) %>%
  full_join(subset(VZ_CZ_wide_ltavg, select=c("CZ", "VZ_2002_ltavg", "VZ_2017_ltavg"))) %>%
  left_join(subset(shares2010, select=c(CZ, vshr_nit))) %>%
  mutate(q_adj = ifelse( q/(1-vshr_nit)     < -85, -85, q/(1-vshr_nit)),
         p_adj = ifelse( p_agg/(1-vshr_nit) < -85, -85, p_agg/(1-vshr_nit))) %>%
  mutate(q_adj = case_when(q_adj >250 ~ 250, TRUE ~ q_adj)) %>%
  rename(VZ_02 = VZ_2002_ltavg,
         VZ_17 = VZ_2017_ltavg) %>%
  full_join(subset(VLAB_CZ, select=c(CZ, VLAB2002_ltavg, VLAB2017_ltavg))) %>%
  full_join(subset(immigrant_CZ, select=c(CZ, s_immi))) %>%
  filter(CZ %in% c(1:605)) %>%
  mutate(across(everything(), ~ ifelse(is.infinite(.), 90, .))) %>%
  mutate(across(everything(), ~ ifelse(. < -95, 95, .))) %>% # added 10/30/25
  mutate(across(everything(), ~ ifelse(is.na(.), 0, .))) 

#         c. data_pr ----
# similar to df.levels.target.pr in NM.R file
data_pr <- data.frame(PR9 = 1:9)  %>%
  full_join(subset(QC_PR, select=c(PR9, VCROP2002_ltavg, VCROP2017_ltavg, vc, q_agg, p_agg, qc_old, q_gr, q_fn, q_vg, q_ot))) %>% # only q_agg and qaug before 111124
  rename(qcrop = qc_old,q = q_agg, vcrop = vc) %>%
  mutate(vcrop = ifelse(vcrop>250, 250, vcrop)) %>%
  full_join(QZ_PR %>% dplyr::select(PR9, vchem, qchem, vseed, qseed, vgas,  qgas,  vmach, qmach, voth, qoth, vz, qz, pz )) %>%
  full_join(QLAND_PR %>% dplyr::select(PR9, HLAND2002_ltavg, HLAND2017_ltavg, qc)) %>%
  full_join(PLAND_PR %>% dplyr::select(PR9, RENT_TOT02, RENT_TOT17, RENT02, RENT17, pc)) %>%
  full_join(subset(WATER_PR, select=c(PR9, WAT02, WAT15, qw))) %>%
  full_join(subset(PHIC_PR, select=c(PR9, CROPLAND_2002_ltavg, phic))) %>% #, phic1, phic2, ipf_lev, c2uland, cp2uland))) %>% # shr_past, 
  full_join(subset(AOCROP_PR, select=c(PR9, ao_pr))) %>%
  full_join(VZ_PR_wide_ltavg %>% dplyr::select(PR9, VZ_2002_ltavg, VZ_2017_ltavg)) %>%
  rename(VZ_02 = VZ_2002_ltavg,
         VZ_17 = VZ_2017_ltavg) %>%
  full_join(subset(VLAB_PR, select=c(PR9, VLAB2002_ltavg, VLAB2017_ltavg))) %>%
  mutate(across(everything(), ~ ifelse(is.infinite(.), 90, .))) %>%
  mutate(across(everything(), ~ ifelse(is.na(.), 0, .))) %>%
  filter(PR9 %in% c(1:9))

#    ii.  LABOR: Loop to add QLAB and PHIL in df.for.combo = c(FLS_XX NAWS_XX) ----
#         a. QLAB dfs ----
# example
QLABz_NAWS_CL <- QLAB_CZ %>%
  dplyr::select(CZ, starts_with("VLAB"), vl, matches("NAWS.*low")) %>%
  rename(QLAB_Hours2002 = QLAB02_NAWS_low,
         QLAB_Hours2017 = QLAB17_NAWS_low,
         WAGE02         = WAGE02_NAWS_low,
         WAGE17         = WAGE02_NAWS_low,
         ql             = ql_NAWS_low,
         pl             = pl_NAWS_low)

# function to replicate example above
make_QLAB_subset <- function(data, source, level) {
  
  # detect name of the dataset passed in
  data_name <- deparse(substitute(data))
  # decide prefix and key variable based on dataset name
  if (grepl("QLAB_CZ", data_name)) {
    agg <- "z"
    keyvar <- "CZ"
  } else {
    agg <- "a"
    keyvar <- "PR9"
  }
  
  # Construct suffix for naming
  suffix <- paste0("_", source, "_C", toupper(substr(level, 1, 1)))
  
  # construct final object name
  obj_name <- paste0("QLAB", agg, suffix)
  
  df <- data %>%
    dplyr::select(all_of(keyvar), starts_with("VLAB"), vl, matches(paste0(source, ".*", level)) ) %>%
    dplyr::rename(QLAB_Hours2002 = paste0("QLAB02_", source, "_", level),
                  QLAB_Hours2017 = paste0("QLAB17_", source, "_", level),
                  WAGE02         = paste0("WAGE02_", source, "_", level),
                  WAGE17         = paste0("WAGE17_", source, "_", level),
                  ql             = paste0("ql_", source, "_", level),
                  pl             = paste0("pl_", source, "_", level))
  
  # assign to global environment with dynamic name
  assign(obj_name, df, envir = .GlobalEnv)
  
  message("✅ Created object: ", obj_name)}

make_QLAB_subset(QLAB_CZ, "NAWS", "low") # QLABz_NAWS_CL
make_QLAB_subset(QLAB_CZ, "NAWS", "med")
make_QLAB_subset(QLAB_CZ, "NAWS", "high")
make_QLAB_subset(QLAB_CZ, "FLS",  "low") # QLABz_FLS_CL
make_QLAB_subset(QLAB_CZ, "FLS",  "med")
make_QLAB_subset(QLAB_CZ, "FLS",  "high")

make_QLAB_subset(QLAB_PR, "NAWS", "low") # QLABz_NAWS_CL
make_QLAB_subset(QLAB_PR, "NAWS", "med")
make_QLAB_subset(QLAB_PR, "NAWS", "high")
make_QLAB_subset(QLAB_PR, "FLS",  "low") # QLABz_FLS_CL
make_QLAB_subset(QLAB_PR, "FLS",  "med")
make_QLAB_subset(QLAB_PR, "FLS",  "high")

#         b. Define lists df.ql.name and df.phil.name to get dfz.out and dfa.out ----
dfz.ql.name   <- c("QLABz_NAWS_CL", "QLABz_NAWS_CM", "QLABz_NAWS_CH",
                   "QLABz_FLS_CL",  "QLABz_FLS_CM",  "QLABz_FLS_CH")
dfz.phil.name <- c("PHILz_NL", "PHILz_NM", "PHILz_NH",
                   "PHILz_CL", "PHILz_CM", "PHILz_CH")

# output df data names
dfz.out.name <- c("NAWS_LNL", "NAWS_MNL", "NAWS_HNL",
                  "FLS_LNL",  "FLS_MNL",  "FLS_HNL",
                  
                  "NAWS_LNM", "NAWS_MNM", "NAWS_HNM",
                  "FLS_LNM",  "FLS_MNM",  "FLS_HNM",
                  
                  "NAWS_LNH", "NAWS_MNH", "NAWS_HNH",
                  "FLS_LNH",  "FLS_MNH",  "FLS_HNH",
                  
                  "NAWS_LCL", "NAWS_MCL", "NAWS_HCL",
                  "FLS_LCL",  "FLS_MCL",  "FLS_HCL",
                  
                  "NAWS_LCM", "NAWS_MCM", "NAWS_HCM",
                  "FLS_LCM",  "FLS_MCM",  "FLS_HCM",
                  
                  "NAWS_LCH", "NAWS_MCH", "NAWS_HCH",
                  "FLS_LCH",  "FLS_MCH",  "FLS_HCH")

# QLAB and PHIL data - PR level
dfa.ql.name   <- c("QLABa_NAWS_CL", "QLABa_NAWS_CM", "QLABa_NAWS_CH",
                   "QLABa_FLS_CL",  "QLABa_FLS_CM",  "QLABa_FLS_CH")
dfa.phil.name <- c("PHILa_NL", "PHILa_NM", "PHILa_NH",
                   "PHILa_CL", "PHILa_CM", "PHILa_CH")

# output data names
dfa.out.name <- c("NAWS_LNL", "NAWS_MNL", "NAWS_HNL",
                  "FLS_LNL",  "FLS_MNL",  "FLS_HNL",
                  
                  "NAWS_LNM", "NAWS_MNM", "NAWS_HNM",
                  "FLS_LNM",  "FLS_MNM",  "FLS_HNM",
                  
                  "NAWS_LNH", "NAWS_MNH", "NAWS_HNH",
                  "FLS_LNH",  "FLS_MNH",  "FLS_HNH",
                  
                  "NAWS_LCL", "NAWS_MCL", "NAWS_HCL",
                  "FLS_LCL",  "FLS_MCL",  "FLS_HCL",
                  
                  "NAWS_LCM", "NAWS_MCM", "NAWS_HCM",
                   "FLS_LCM",  "FLS_MCM",  "FLS_HCM",
                   
                   "NAWS_LCH", "NAWS_MCH", "NAWS_HCH",
                   "FLS_LCH",  "FLS_MCH",  "FLS_HCH")

#            Functions: Scenarios (QLAB + PHIL) function ----
scenarios <- function(df.ql.name, df.phil.name, unit) {
  df.lab <- get(df.ql.name) %>% 
    dplyr::select(all_of(unit), VLAB2002_ltavg, VLAB2017_ltavg, QLAB_Hours2002, QLAB_Hours2017, WAGE02, WAGE17, vl, ql, pl)
  
  df.phil <- get(df.phil.name)
  
  df <- left_join(df.lab, df.phil, by = unit) %>%
    mutate(phil_lev = phil/100 * QLAB_Hours2002) %>%
    #slice(1:605) %>%
    filter(!is.na(.data[[unit]]))
  
  return(df) }

# General runner function to create output list
run_scenarios <- function(df.ql.names, df.phil.names, df.out.names, unit) {
  # Generate the combination grid internally
  combos <- expand.grid(df.ql = df.ql.names, df.phil = df.phil.names, stringsAsFactors = FALSE)
  
  df.out <- list()
  for (i in seq_len(nrow(combos))) {
    df.out[[df.out.names[i]]] <- scenarios(df.ql.name   = combos$df.ql[i],
                                           df.phil.name = combos$df.phil[i],
                                           unit         = unit )}
  return(df.out)
}

#         c. Define dfz.out, dfa.out ----
# CZ level
dfz.out <- run_scenarios(df.ql.names  = dfz.ql.name,
                         df.phil.names = dfz.phil.name,
                         df.out.names  = dfz.out.name, unit         = "CZ" )

# PR level
dfa.out <- run_scenarios(df.ql.names  = dfa.ql.name,
                         df.phil.names = dfa.phil.name,
                         df.out.names  = dfa.out.name, unit         = "PR9" )
saveRDS(dfa.out, file.path("temp/0_laba_scenarios_071326.RDS"))
cat("  Saved (interim): temp/0_laba_scenarios_071326.RDS\n")


# 4:    SAVE files in /temp/ ----
cat("[4] Saving output files to temp/...\n")

write_xlsx(shares2010, path="temp/0_shares2010_071326.xlsx")
cat("  Saved: temp/0_shares2010_071326.xlsx\n")
write_xlsx(data_fips,  path="temp/0_data_fips_071326.xlsx")
cat("  Saved: temp/0_data_fips_071326.xlsx\n")
write_xlsx(data_cz,    path="temp/0_data_cz_071326.xlsx")
cat("  Saved: temp/0_data_cz_071326.xlsx\n")
write_xlsx(data_pr,    path="temp/0_data_pr_071326.xlsx")
cat("  Saved: temp/0_data_pr_071326.xlsx\n")

saveRDS(dfz.out,       file.path("temp/0_labz_scenarios_071326.RDS"))
cat("  Saved: temp/0_labz_scenarios_071326.RDS\n")
saveRDS(dfa.out,       file.path("temp/0_laba_scenarios_071326.RDS"))
cat("  Saved: temp/0_laba_scenarios_071326.RDS\n")
saveRDS(PHILz_list,    file.path("temp/0_PHILz_071326.RDS"))
cat("  Saved: temp/0_PHILz_071326.RDS\n")
saveRDS(PHILa_list,    file.path("temp/0_PHILa_071326.RDS"))
cat("  Saved: temp/0_PHILa_071326.RDS\n")

write_xlsx(QLAB_F,     path="temp/0_QLABOR_FIPS_071326.xlsx")
cat("  Saved: temp/0_QLABOR_FIPS_071326.xlsx\n")
write_xlsx(QLAB_CZ,    path="temp/0_QLABOR_CZ_071326.xlsx")
cat("  Saved: temp/0_QLABOR_CZ_071326.xlsx\n")
write_xlsx(QLAB_PR,    path="temp/0_QLABOR_PR_071326.xlsx")
cat("  Saved: temp/0_QLABOR_PR_071326.xlsx\n")

cat("0a_PRESTAGE.R complete. All prestage outputs saved to temp/.\n")