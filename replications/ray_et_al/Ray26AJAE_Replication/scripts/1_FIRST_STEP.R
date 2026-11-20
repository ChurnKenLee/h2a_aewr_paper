# ===========================================================================
#   GOAL: First Stage Estimation 
#   1: Reads in the files generated in 1_PRESTAGE.R
#   2: First Stage Estimation
#   Note: The first stage estimation is repeated for 30 runs 
#         requiring long processing times
#         The Results of the FSE are reported in the list: temp/1_list_est.rds
# ===========================================================================

# 1. Set Path ---- 
rm(list = ls())
project_root <- getwd()
setwd(project_root)

# Libraries 
library(readxl)
library(tidyr)
library(writexl)      # write_excel
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

cat("Libraries loaded.\n")
options(warn = -1)
options(dplyr.summarise.inform = FALSE)

# 2. Read in files ----
cat("Reading crosswalks and first-stage input data...\n")
CZ2PR <- read_excel(file.path("in/Crosswalk/CZ2PR.xlsx")) %>%
  rename(PR9 = PR) %>%
  filter(CZ %in% c(1:605))

CZ_names <- read_excel(file.path("in/Crosswalk/CZ_names.xlsx"))
CZ_names <- CZ_names[1:605, ]
cz = c(CZ_names$CZ)

# Creating a data frame with numeric and character columns directly
PR <- data.frame(
  PR9 = as.numeric(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
  pr = c("Heartland", "Northern Crescent", "Northern Great Plain", "Prairie Gateway", 
         "Eastern Uplands", "Southern Seaboard", "Fruitful Rim", 
         "Basin & Range", "Mississippi Portal", "US"))

pr_names = c("Heartland", "NorthCrescen", "NorthGreatPl", "PrairieGatew", "EastUplands", "SouthSeabr", "FruitfulRim", "BasinaRange", "MississippiP",
             "E_Euro", "N_Afr", "SSA", "S_Amer", "AUS_NZ", " EU", "S_Asia", "CC_Amer", "S_Afr", "SE_Asia", "CAN", "CHN_MNG", "M_East", "JPN_KR", "C_Asia")

# input for FSE
data_cz    <- read_xlsx(file.path("temp/0_data_cz_071326.xlsx"))
data_pr    <- read_xlsx(file.path("temp/0_data_pr_071326.xlsx"))
data_fips  <- read_xlsx(file.path("temp/0_data_fips_071326.xlsx"))
shares2010 <- read_xlsx(file.path("temp/0_shares2010_071326.xlsx"))
dfz.out    <- readRDS(file.path("temp/0_labz_scenarios_071326.RDS"))
dfa.out    <- readRDS(file.path("temp/0_laba_scenarios_071326.RDS"))

cat("Input data loaded.\n")

# 3. FIRST STAGE ESTIMATION ----
cat("Setting up first-stage estimation: objective function and scenario grid...\n")

#    i. Define objective function ----
source("scripts/NM_obj_function.R")

#   ii. Define Scenarios ----
CMARKUP <- c("L", "M", "H")
IMM     <- c("L", "M", "H")
SECTOR  <- c("nonag", "cs")  
LABDATA <- c("NAWS", "FLS")

# use combo short to design df.for.combo in correct sequence
combo_short <- expand.grid(CMARKUP = CMARKUP, SECTOR = SECTOR, IMM = IMM, stringsAsFactors = FALSE)
# run again by replacing NAWS with FLS 
df.for.combo <- c(
  "NAWS_LNL",  "NAWS_MNL", "NAWS_HNL",
  "NAWS_LNM",  "NAWS_MNM", "NAWS_HNM",
  "NAWS_LNH",  "NAWS_MNH", "NAWS_HNH",
  "NAWS_LCL",  "NAWS_MCL", "NAWS_HCL",
  "NAWS_LCM",  "NAWS_MCM", "NAWS_HCM",
  "NAWS_LCH",  "NAWS_MCH", "NAWS_HCH",
  "FLS_LNL",   "FLS_MNL",  "FLS_HNL",
  "FLS_LNM",   "FLS_MNM",  "FLS_HNM",
  "FLS_LNH",   "FLS_MNH",  "FLS_HNH",
  "FLS_LCL",   "FLS_MCL",  "FLS_HCL",
  "FLS_LCM",   "FLS_MCM",  "FLS_HCM",
  "FLS_LCH",   "FLS_MCH",  "FLS_HCH") # pasted in 'create_cmf_pr_combo_files.R' "FLS_MCH",


#  iii. Define parameters ----
mean_sig = 0.26
max.ELAB = 1.01
lower    = c(5,   5,   5,   0.5 * mean_sig, 0.1)        # Lower limits for each parameter
upper    = c(200, 200, 200, 1.5 * mean_sig, max.ELAB  )  # Upper limits for each parameter

st.pr   = 1             # starting PR number
end.pr  = 9             # starting PR number
sreg      = st.pr:end.pr
num.pr  = length(sreg)

do.pr    = c(1:9)
saveCores= 5
run      = 30  # number of runs of the algorithm
k        = length(lower)   # number of unknowns to calibrate (aL,aZ,aR, aF, es_lz) 
iter.max = 45 
iter.mid = 25
dev.tolerance = 0.01
dev.diff.tol  = (dev.tolerance)^2

#   iv. Estimation with Nelder-Mead  ----
# source("scripts/FSE_PP.R") # based on dopar_110525.R
# saveRDS(list.est, file = paste0("temp/1_list_est.rds"))
#   v.  Plot estimates ----
cat("Generating first-stage estimation plots (PLOT_FSE.R)...\n")
source("scripts/PLOT_FSE.R")

cat("1_FIRST_STEP.R complete.\n")
