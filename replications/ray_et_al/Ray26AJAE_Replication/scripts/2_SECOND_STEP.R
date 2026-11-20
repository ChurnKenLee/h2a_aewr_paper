# ===========================================================================
#   GOAL: Second Stage Validation & Identification
#   1: Validation - Use first stage estimates to replicate historical 
#                   labor market outcomes 2002-2017
#   2: Decomposition - Attribute historical change to drivers of change
#   3: Identification - Estimate coefficients
# ===========================================================================

# 1. Set Path ---- 
cat("[1] Setting up path and libraries...\n")
rm(list = ls())
project_root <- getwd()
setwd(project_root)
dir.create("plots/Figure7")
dir.create("plots/Figure8")
options(warn = -1)
options(dplyr.summarise.inform = FALSE)

#    Libraries 
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

# 2. Read in Files & define ----
cat("[2] Reading input files and defining crosswalks...\n")
sel_pr <- c(1,2,3,7)
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
PHILz_list <- readRDS(file.path("temp/0_PHILz_071326.RDS"))
PHILa_list <- readRDS(file.path("temp/0_PHILa_071326.RDS"))
QLAB_F     <- read_xlsx(file.path("temp/0_QLABOR_FIPS_071326.xlsx"))
QLAB_CZ    <- read_xlsx(file.path("temp/0_QLABOR_CZ_071326.xlsx"))
QLAB_PR    <- read_xlsx(file.path("temp/0_QLABOR_PR_071326.xlsx"))

fse.est    <- read_excel(paste0("temp/1_pfp_est_pr.xlsx"))
fse.nonus <- data.frame(CZ         = 606:620, 
                        PR9        = 10:24, # change to 10:24 after all regions have FSE
                        AFL_median = 0,
                        AFZ_median = 0,
                        AFR_median = 0,
                        ESLZ_median = 0.26,
                        ELAB_median = 0.8)

FIPS2CZ   <- read_excel(file.path("in/Crosswalk/FIPS2CZ_251024.xlsx")) %>%
  as.data.frame()

CZ2PR <- read_excel(file.path("in/Crosswalk/CZ2PR.xlsx")) %>%
  rename(PR9 = PR) %>%
  filter(CZ %in% c(1:605))

FIPS2PR   <- read_excel(file.path("in/Crosswalk/FIPS2PR_071024.xlsx")) %>% # use this
  dplyr:: select(FIPS, PR) %>%
  rename(PR9 = PR) %>%
  rbind(c(FIPS = 8014, PR9 = 4)) %>%
  dplyr::select(FIPS) %>%
  full_join(FIPS2CZ) %>%
  full_join(CZ2PR) %>%
  filter(FIPS!=12025) %>% # 12086, updated FIPS for 12025 included
  unique() 

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

CZ_names <- read_excel(file.path("in/Crosswalk/CZ_names.xlsx"))
CZ_names <- CZ_names[1:605, ]
cz = c(CZ_names$CZ)

outlier.cz <- read_excel(file.path("in/List_CZ.xlsx"))
cz.drop <- outlier.cz$CZ
  
cpi0217 <- 137.12 # 2002: 100, 2017: 137.12

do.pr <- c(1:9)

# define groups of combos
combo_groups <- list(
  NAWS_C = c("NAWS_LCL", "NAWS_MCL", "NAWS_HCL",
             "NAWS_LCM", "NAWS_MCM", "NAWS_HCM",
             "NAWS_LCH", "NAWS_MCH", "NAWS_HCH"),
  NAWS_N = c("NAWS_LNL", "NAWS_MNL", "NAWS_HNL",
             "NAWS_LNM", "NAWS_MNM", "NAWS_HNM",
             "NAWS_LNH", "NAWS_MNH", "NAWS_HNH"),
  FLS_C  = c("FLS_LCL",  "FLS_MCL",  "FLS_HCL",
             "FLS_LCM",  "FLS_MCM",  "FLS_HCM",
             "FLS_LCH",  "FLS_MCH",  "FLS_HCH"),
  FLS_N  = c("FLS_LNL",  "FLS_MNL",  "FLS_HNL",
             "FLS_LNM",  "FLS_MNM",  "FLS_HNM",
             "FLS_LNH",  "FLS_MNH",  "FLS_HNH"))

# 3. Historical Validation, write exogenous controls in .har files ----
cat("[3] Writing exogenous control files (.har) for historical validation...\n")
#    i. Write non-labor files ----
df.exo.controls <- data_cz %>%
  left_join(CZ2PR) %>%
  dplyr::select(PR9, CZ, q, p_adj, p_agg, q_adj, qw, pc, pz, phic) %>%
  mutate(across(everything(), ~ ifelse(. < -95, 95, .))) %>%
  mutate(across(-c(CZ,PR9), ~ replace(., CZ %in% cz.drop, 0)))

exo.controls   <- list()
exo.controls[["QWAT"]] <- array(as.vector(as.matrix(df.exo.controls$qw)),  # Data as a vector
                              dim = c(605, 1),  # Set dimensions (adjust if needed)
                              dimnames = list(dim1name = cz,
                                              dim2name = "QWAT"))
exo.controls[["PLND"]] <- array(as.vector(as.matrix(df.exo.controls$pc)),  # Data as a vector
                              dim = c(605, 1),  # Set dimensions (adjust if needed)
                              dimnames = list(dim1name = cz,
                                              dim2name = "PLND"))
exo.controls[["QAUG"]] <- array(as.vector(as.matrix(df.exo.controls$q_adj)),  # Data as a vector
                              dim = c(605, 1),  # Set dimensions (adjust if needed)
                              dimnames = list(dim1name = cz,
                                              dim2name = "QAUG"))
exo.controls[["PAUG"]] <- array(as.vector(as.matrix(df.exo.controls$p_adj)),  # Data as a vector
                              dim = c(605, 1),  # Set dimensions (adjust if needed)
                              dimnames = list(dim1name = cz,
                                              dim2name = "PAUG"))
exo.controls[["PAGG"]] <- array(as.vector(as.matrix(df.exo.controls$p_agg)),  # Data as a vector
                              dim = c(605, 1),  # Set dimensions (adjust if needed)
                              dimnames = list(dim1name = cz,
                                              dim2name = "PAGG"))
exo.controls[["PNLB"]] <- array(as.vector(as.matrix(df.exo.controls$pz)),  # Data as a vector
                              dim = c(605, 1),  # Set dimensions (adjust if needed)
                              dimnames = list(dim1name = cz,
                                              dim2name = "PNLB"))
exo.controls[["PHIC"]] <- array(as.vector(as.matrix(df.exo.controls$phic)),  # Data as a vector
                              dim = c(605, 1),  # Set dimensions (adjust if needed)
                              dimnames = list(dim1name = cz,
                                              dim2name = "PHIC"))
suppressMessages(write_har(exo.controls, file.path("shf/OBS_DATA.har")))  # Save the file with the indexed name

#   ii. Write Labor files ----
#     Labor supply shock (low, med, high = L,M,H) x (ConSe, Non-Ag = C,N)
PHIL_Nz <- data.frame(CZ = 1:605) %>%
  mutate(
    phil_low  = PHILz_list[["PHILz_NL"]]$phil,
    phil      = PHILz_list[["PHILz_NM"]]$phil,
    phil_high = PHILz_list[["PHILz_NH"]]$phil) %>%
  mutate(across(-CZ, ~ replace(., CZ %in% cz.drop, 0)))

PHIL_Cz <- data.frame(CZ = 1:605) %>%
  mutate(phil_low  = PHILz_list[["PHILz_CL"]]$phil,
         phil      = PHILz_list[["PHILz_CM"]]$phil,
         phil_high = PHILz_list[["PHILz_CH"]]$phil) %>%
  mutate(across(-CZ, ~ replace(., CZ %in% cz.drop, 0)))

PHIL_Na <- data.frame(PR9 = 1:9) %>%
  mutate(phil_low  = PHILa_list[["PHILa_NL"]]$phil,
         phil      = PHILa_list[["PHILa_NM"]]$phil,
         phil_high = PHILa_list[["PHILa_NH"]]$phil)

PHIL_Ca <- data.frame(PR9 = 1:9) %>%
  mutate(phil_low  = PHILa_list[["PHILa_CL"]]$phil,
         phil      = PHILa_list[["PHILa_CM"]]$phil,
         phil_high = PHILa_list[["PHILa_CH"]]$phil)

# write .har files
philz.n <- list()
philz.n[["PHIL"]] <- array(as.vector(PHIL_Nz$phil),  # Data as a vector
                          dim = c(605, 1),  # Set dimensions (adjust if needed)
                          dimnames = list(dim1name = cz,
                                          dim2name = "PHIL"))
# 
philz.c <- list()
philz.c[["PHIL"]] <- array(as.vector(PHIL_Cz$phil),  # Data as a vector
                          dim = c(605, 1),  # Set dimensions (adjust if needed)
                          dimnames = list(dim1name = cz,
                                          dim2name = "PHIL"))
suppressMessages(write_har(philz.n, file.path("shf/NonAg_M.har")))
suppressMessages(write_har(philz.c, file.path("shf/ConSe_M.har")))

philz.n <- list()
philz.n[["PHIL"]] <- array(as.vector(PHIL_Nz$phil_low),  # Data as a vector
                           dim = c(605, 1),  # Set dimensions (adjust if needed)
                           dimnames = list(dim1name = cz,
                                           dim2name = "PHIL"))

philz.c <- list()
philz.c[["PHIL"]] <- array(as.vector(PHIL_Cz$phil_low),  # Data as a vector
                           dim = c(605, 1),  # Set dimensions (adjust if needed)
                           dimnames = list(dim1name = cz,
                                           dim2name = "PHIL"))
suppressMessages(write_har(philz.n, file.path("shf/NonAg_L.har")))
suppressMessages(write_har(philz.c, file.path("shf/ConSe_L.har")))

philz.n <- list()
philz.n[["PHIL"]] <- array(as.vector(PHIL_Nz$phil_high),  # Data as a vector
                           dim = c(605, 1),  # Set dimensions (adjust if needed)
                           dimnames = list(dim1name = cz,
                                           dim2name = "PHIL"))

philz.c <- list()
philz.c[["PHIL"]] <- array(as.vector(PHIL_Cz$phil_high),  # Data as a vector
                           dim = c(605, 1),  # Set dimensions (adjust if needed)
                           dimnames = list(dim1name = cz,
                                           dim2name = "PHIL"))
suppressMessages(write_har(philz.n, file.path("shf/NonAg_H.har")))
suppressMessages(write_har(philz.c, file.path("shf/ConSe_H.har")))

#  iii. Write parameter files ----
for (grp in unique(fse.est$group)) {
  group_name <- grp
  
  fse.cz <- CZ2PR %>%
    left_join(fse.est %>% filter(group == group_name) %>% dplyr::select(PR9, AFL_median, AFZ_median, AFR_median, ESLZ_median, ELAB_median)) %>%
    mutate(across(c(starts_with("AF")), ~ ifelse(CZ %in% cz.drop, 0, .))) %>%
    rbind(fse.nonus) %>%
    arrange(CZ)
  
  fse.pr <- fse.cz %>%
    dplyr::select(-CZ) %>%
    group_by(PR9) %>%
    summarize(across(everything(), mean)) %>%
    arrange(PR9)
  
  parm.est <- list()
  parm.est[["AFL"]] <- array(as.vector(fse.cz$AFL_median[1:605]),  # Data as a vector
                             dim = c(605, 1),  # Set dimensions (adjust if needed)
                             dimnames = list(dim1name = cz[1:605],
                                             dim2name = "AFL"))
  parm.est[["AFZ"]] <- array(as.vector(fse.cz$AFZ_median[1:605]),  # Data as a vector
                             dim = c(605, 1),  # Set dimensions (adjust if needed)
                             dimnames = list(dim1name = cz[1:605],
                                             dim2name = "AFZ"))
  parm.est[["AFR"]] <- array(as.vector(fse.cz$AFR_median),  # Data as a vector
                             dim = c(605, 1),  # Set dimensions (adjust if needed)
                             dimnames = list(dim1name = cz[1:605],
                                             dim2name = "AFR"))
  parm.est[["ESLZ"]] <- array(as.vector(fse.pr$ESLZ_median),  # Data as a vector
                              dim = c(24, 1),  # Set dimensions (adjust if needed)
                              dimnames = list(dim1name = pr_names,
                                              dim2name = "ESLZ"))
  parm.est[["ELAB"]] <- array(as.vector(fse.pr$ELAB_median),  # Data as a vector
                              dim = c(24, 1),  # Set dimensions (adjust if needed)
                              dimnames = list(dim1name = pr_names,
                                              dim2name = "ELAB"))
  suppressMessages(write_har(parm.est, file.path("shf", paste0("FSE-", group_name, "M.har"))))
}

#   iv. LAB data: Low med high from NAWS & FLS ----

# helper function for QL, PL for NAWS and FLS
metrics  <- c("QLAB_Hours2002","QLAB_Hours2017","WAGE02","WAGE17")
prefixes <- c("QLAB02",        "QLAB17",        "WAGE02","WAGE17")

build_labor_df <- function(group_name, combos) {
  
  message("Processing group: ", group_name)
  
  # start df
  df <- data.frame(CZ = 1:605)
  
  # dynamically add combo columns
  for (combo in combos) {
    for (i in seq_along(metrics)) {
      metric <- metrics[i]
      pre <- prefixes[i]
      newcol <- paste0(pre, "_", combo)
      df[[newcol]] <- dfz.out[[combo]][[metric]]
    }
  }
  
  # compute mean & quantiles
  df <- df %>%
    rowwise() %>%
    mutate(
      QLAB02      = mean(c_across(starts_with("QLAB02_"))),
      QLAB02_low  = quantile(c_across(starts_with("QLAB02_")), 0.125),
      QLAB02_high = quantile(c_across(starts_with("QLAB02_")), 0.875),
      
      QLAB17      = mean(c_across(starts_with("QLAB17_"))),
      QLAB17_low  = quantile(c_across(starts_with("QLAB17_")), 0.125),
      QLAB17_high = quantile(c_across(starts_with("QLAB17_")), 0.875),
      
      WAGE02      = mean(c_across(starts_with("WAGE02_"))),
      WAGE02_low  = quantile(c_across(starts_with("WAGE02_")), 0.125),
      WAGE02_high = quantile(c_across(starts_with("WAGE02_")), 0.875),
      
      WAGE17      = mean(c_across(starts_with("WAGE17_"))),
      WAGE17_low  = quantile(c_across(starts_with("WAGE17_")), 0.125),
      WAGE17_high = quantile(c_across(starts_with("WAGE17_")), 0.875),
      
      ql      = (QLAB17      - QLAB02)      / QLAB02 * 100,
      ql_low  = (QLAB17_low  - QLAB02_low)  / QLAB02_low  * 100,
      ql_high = (QLAB17_high - QLAB02_high) / QLAB02_high * 100,
      
      # edited 05-Mar-26
      pl      = (WAGE17      - (WAGE02     *cpi0217/100))  / (WAGE02     *cpi0217/100) * 100,
      pl_low  = (WAGE17_low  - (WAGE02_low *cpi0217/100))  / (WAGE02_low *cpi0217/100) * 100,
      pl_high = (WAGE17_high - (WAGE02_high*cpi0217/100))  / (WAGE02_high*cpi0217/100) * 100
      
    ) %>%
    ungroup() %>%
    select(CZ, QLAB02, QLAB02_low, QLAB02_high,
           QLAB17, QLAB17_low, QLAB17_high,
           WAGE02, WAGE02_low, WAGE02_high,
           WAGE17, WAGE17_low, WAGE17_high,
           ql, ql_low, ql_high,
           pl, pl_low, pl_high) %>%
    left_join(data_cz %>% select(CZ, VLAB2002_ltavg, VLAB2017_ltavg), by = "CZ")
  
  df
}

# combo_groups defined in Plot_FSE
for (grp in names(combo_groups)) {
  combos <- combo_groups[[grp]]
  df_name <- paste0(grp, "_data_laborz")
  
  assign(df_name, build_labor_df(grp, combos))
}
# generates NAWS_N_data_laborz, which summarizes over the scenarios in (NAWS_N, NAWS_C, FLS_N, FLS_C)


aggregate_to_PR <- function(df_cz, CZ2PR) {
  df_cz %>% 
    left_join(CZ2PR, by = "CZ") %>%
    group_by(PR9) %>%
    summarise(
      QLAB02      = sum(QLAB02, na.rm = TRUE),
      QLAB02_low  = sum(QLAB02_low, na.rm = TRUE),
      QLAB02_high = sum(QLAB02_high, na.rm = TRUE),
      
      QLAB17      = sum(QLAB17, na.rm = TRUE),
      QLAB17_low  = sum(QLAB17_low, na.rm = TRUE),
      QLAB17_high = sum(QLAB17_high, na.rm = TRUE),
      
      VLAB02      = sum(VLAB2002_ltavg, na.rm = TRUE),
      VLAB17      = sum(VLAB2017_ltavg, na.rm = TRUE)
    ) %>%
    ungroup() %>%
    mutate(
      ql      = (QLAB17 - QLAB02) / QLAB02 * 100,
      ql_low  = (QLAB17_low - QLAB02_low) / QLAB02_low * 100,
      ql_high = (QLAB17_high - QLAB02_high) / QLAB02_high * 100,
      
      WAGE02       = VLAB02 / QLAB02,
      WAGE02_high  = VLAB02 / QLAB02_low,
      WAGE02_low   = VLAB02 / QLAB02_high,
      
      WAGE17       = VLAB17 / QLAB17,
      WAGE17_high  = VLAB17 / QLAB17_low,
      WAGE17_low   = VLAB17 / QLAB17_high,
      
      # edited 05-Mar-26, only pl, pl_low & pl_high were correct
      pl      = (WAGE17       - (WAGE02     *cpi0217/100)) / (WAGE02     *cpi0217/100) * 100,
      pl_low  = (WAGE17_low   - (WAGE02_low *cpi0217/100)) / (WAGE02_low *cpi0217/100) * 100,
      pl_high = (WAGE17_high  - (WAGE02_high*cpi0217/100)) / (WAGE02_high*cpi0217/100) * 100
    ) %>%
    select(
      PR9, QLAB02, QLAB02_low, QLAB02_high,
      QLAB17, QLAB17_low, QLAB17_high,
      WAGE02, WAGE02_low, WAGE02_high,
      WAGE17, WAGE17_low, WAGE17_high,
      ql, ql_low, ql_high,
      pl, pl_low, pl_high
    )
}
NAWS_N_data_labora <- aggregate_to_PR(NAWS_N_data_laborz, CZ2PR)
NAWS_C_data_labora <- aggregate_to_PR(NAWS_C_data_laborz, CZ2PR)
FLS_N_data_labora  <- aggregate_to_PR(FLS_N_data_laborz, CZ2PR)
FLS_C_data_labora  <- aggregate_to_PR(FLS_C_data_laborz, CZ2PR)

#    v. Plot Figures 4 & 5 ----
source("scripts/Data_plots.R")
#   vi. Define required functions ----
run.simple <- function(cmd) {
  
  # Run the command and capture the output
  output <- system(cmd, intern = TRUE)
  
  # Check if the success message is in the output
  if (any(grepl("The program has completed without error.", output))) {
    message("Program completed successfully.")  }
  else {
    stop("Error: Program did not complete successfully.")
  } 
  }

#      Functions to read results
extractz.sl4 <- function(data, headers, n = 605, prefix) {
  # Initialize an empty matrix to hold the data
  combined_data <- matrix(NA, nrow = n, ncol = length(headers) + 1)  # +1 for CZ column
  colnames(combined_data) <- c("CZ", headers)  # Set the column names
  combined_data[, 1] <- 1:n # Fill in the CZ column
  
  # Loop through each header and extract data into the matrix
  for (i in seq_along(headers)) {
    header <- headers[i]
    combined_data[, i + 1] <- data[[header]][1:n]}  # Extract data and fill the matrix
  merged_data <- as.data.frame(combined_data) # Convert the matrix to a data frame
  # Rename columns except "CZ" by adding the prefix
  colnames(merged_data) <- ifelse(colnames(merged_data) == "CZ",
                                  "CZ",
                                  paste0(prefix, colnames(merged_data)))
  
  return(merged_data) }

extracta.sl4 <- function(data, headers, n = 24) {
  combined_data <- matrix(NA, nrow = n, ncol = length(headers) + 1)  # +1 for PR9 column
  colnames(combined_data) <- c("PR9", headers)  # Set the column names
  combined_data[, 1] <- 1:n # Fill in the CZ column
  
  # Loop through each header and extract data into the matrix
  for (i in seq_along(headers)) {
    header <- headers[i]
    combined_data[, i + 1] <- data[[header]][1:n]}  # Extract data and fill the matrix
  merged_data <- as.data.frame(combined_data) # Convert the matrix to a data frame
  return(merged_data) }

# List of headers to extract 
headerz <- c("p_QLANDz",   "p_QWATERz",    "p_QLABORz",  "p_QNLABORz",  
             "p_PLANDz",                   "p_PLABORz",  "p_PNLABORz",  
             "p_QCROPz",   "p_QAUGLANDz",  "p_PAUGLANDz", 
             "s_QLABORz",  "s_QAUGLANDz",  "s_QLANDz")

headera <- c("p_PAUGLANDa", "p_QAUGLANDa", "p_QLABORa",   "p_PLABORa", "p_QLANDa",   "p_PLANDa", "p_QNLABORa", "p_PNLABORa")

summarize_bars_lab_ssa <- function(df, var_prefix, source, which.pr, scale_factor = 1) {
  
  # SSA variable names follow e.g.: sim.WAGE17, SSA.low.WAGE17, SSA.high.WAGE17  
  sim_var  <- paste0("sim.",      var_prefix, "17")
  low_var  <- paste0("SSA.low.",  var_prefix, "17")
  high_var <- paste0("SSA.high.", var_prefix, "17")
  
  df %>%
    rowwise() %>%
    mutate(mean02 = mean(c_across(all_of(paste0(var_prefix, c("02_NAWS_med", "02_FLS_med")))), na.rm = TRUE) / scale_factor,
           mean17 = mean(c_across(all_of(paste0(var_prefix, c("17_NAWS_med", "17_FLS_med")))), na.rm = TRUE) / scale_factor,
           low02  = min(c_across(all_of(paste0(var_prefix, c("02_NAWS_low", "02_FLS_low")))), na.rm = TRUE) / scale_factor,
           high02 = max(c_across(all_of(paste0(var_prefix, c("02_NAWS_high", "02_FLS_high")))), na.rm = TRUE) / scale_factor,
           low17  = min(c_across(all_of(paste0(var_prefix, c("17_NAWS_low", "17_FLS_low")))), na.rm = TRUE) / scale_factor,
           high17 = max(c_across(all_of(paste0(var_prefix, c("17_NAWS_high", "17_FLS_high")))), na.rm = TRUE) / scale_factor,
           # ---- Correct SSA values 
           meanSSA = .data[[sim_var]]  / scale_factor,
           lowSSA  = .data[[low_var]]  / scale_factor,
           highSSA = .data[[high_var]] / scale_factor ) %>%
    ungroup() %>%
    select(PR9, mean02, mean17, low02, high02, low17, high17, meanSSA, lowSSA, highSSA) %>%
    pivot_longer(cols = c(mean02, mean17, meanSSA), names_to = "year", values_to = "mean") %>%
    mutate(year = case_when(year == "mean02"  ~ "2002",
                            year == "mean17"  ~ "2017",
                            year == "meanSSA" ~ "Simulated-2017"),
           low = case_when(year == "2002"    ~ low02,
                           year == "2017"    ~ low17,
                           year == "Simulated-2017"~ lowSSA),
           high = case_when(year == "2002"    ~ high02,
                            year == "2017"    ~ high17,
                            year == "Simulated-2017"~ highSSA)) %>%
    select(PR9, year, mean, low, high) %>%
    filter(PR9 %in% which.pr)
}

# --- Themes ---
theme_top <- theme_minimal(base_size = 12) +
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        legend.position = "top",
        legend.title = element_blank(),
        plot.title = element_text(face = "bold", hjust = 0.5))

theme_bottom <- theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(size = 12, vjust = 0.5, hjust = 0.5, face = "bold", color = "black"),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 12, face = "bold", color = "black"),
        legend.position = "top",
        legend.title = element_blank(),
        plot.title = element_text(face = "bold", hjust = 0.5))

#  vii. Run Median Run ----
cat("Running GEMPACK SIMPLE-G validation scenarios (FLS/NAWS, NM3/CM3)...\n")
run.simple('simplez.exe -cmf 2-Validation-FLS-NM3.cmf')
run.simple('simplez.exe -cmf 2-Validation-NAWS-NM3.cmf')
run.simple('simplez.exe -cmf 2-Validation-FLS-CM3.cmf')
run.simple('simplez.exe -cmf 2-Validation-NAWS-CM3.cmf')
cat("GEMPACK scenario runs complete.\n")


# 4. Validation Plots ----
cat("[4] Generating validation plots...\n")
#    i.  Generate CIs for parms and shks ----
# Dynamic Scenario Names: (NAWS,FLS) (N,C) ---
rm(source, phil)

sources <- c("NAWS", "FLS")
phils   <- c("N", "C")

for (source in sources) {
  for (phil in phils) {
    # Dynamic scenario def ----
    if (phil == "N") {
      phila         <- PHIL_Na
    } else {
      phila         <- PHIL_Ca
    }
    
    #     ii. FSE estimates 
    fse.cz <- CZ2PR %>%
      # not changing fse estimates for group_name
      left_join(fse.est %>% filter(group == paste0(source, "_", phil)) %>% dplyr::select(PR9, AFL_median, AFZ_median, AFR_median, ESLZ_median, ELAB_median)) %>%
      arrange(CZ)
    
    fse.pr <- fse.cz %>%
      dplyr::select(-CZ) %>%
      group_by(PR9) %>%
      summarize(across(everything(), mean)) %>%
      arrange(PR9)
    
    #  iii. Calculate the CIs: AXL, PHIL 
    pfp.est.wCI <- fse.est %>%
      # not scenario specific
      filter(group == paste0(source, "_", phil)) %>%
      dplyr::select(PR9, starts_with("AF"), "ESLZ_median", "ESLZ_mad", "ELAB_median", "ELAB_mad") %>%
      
      mutate(across(contains("_mad"), ~ if_else(.x == 0, 1, .x))) %>%
      
      # Compute raw CI bounds
      mutate(AFL_low_raw  = AFL_median  - (1.96 * AFL_mad),
             AFL_high_raw = AFL_median + (1.96 * AFL_mad),
             AFR_low_raw  = AFR_median  - (1.96 * AFR_mad),
             AFR_high_raw = AFR_median + (1.96 * AFR_mad),
             AFZ_low_raw  = AFZ_median  - (1.96 * AFZ_mad),
             AFZ_high_raw = AFZ_median + (1.96 * AFZ_mad),
             
             ESLZ_low  = ESLZ_median  - (1.96 * ESLZ_mad),
             ESLZ_high = ESLZ_median  + (1.96 * ESLZ_mad),
             ELAB_low  = ELAB_median  - (1.96 * ELAB_mad),
             ELAB_high = ELAB_median  + (1.96 * ELAB_mad),
             
             # Apply rule: if low < 0 -> use mad =2
             AFL_low  = if_else(AFL_low_raw <= 0,  AFL_median  - 2, AFL_low_raw),
             AFR_low  = if_else(AFR_low_raw <= 0,  AFR_median  - 2, AFR_low_raw),
             AFZ_low  = if_else(AFZ_low_raw <= 0,  AFZ_median  - 2, AFZ_low_raw),
             
             AFL_high = AFL_high_raw,
             AFR_high = AFR_high_raw,
             AFZ_high = AFZ_high_raw) %>%
      dplyr::select(-contains("_raw")) %>%
      
      # Final CI width = min absolute side (avoid extreme asymmetry)
      rowwise() %>%
      mutate(AFL_CI = round( min(AFL_median - AFL_low, AFL_high - AFL_median), 2),
             AFR_CI = round( min(AFR_median - AFR_low, AFR_high - AFR_median), 2),
             AFZ_CI = round( min(AFZ_median - AFZ_low, AFZ_high - AFZ_median), 2),
             
             ESLZ_CI = round( min(ESLZ_median - ESLZ_low, ESLZ_high - ESLZ_median), 2),
             ELAB_CI = round( min(ELAB_median - ELAB_low, ELAB_high - ELAB_median), 2)) %>%
      ungroup() %>%
      
      cbind(pr = pr_names[1:9])
    pfp.est.wCI
    
    # using phila for SSA
    phila_wCI <- phila %>%
      mutate(phil_low_pct  = abs(phil_low  - phil) / phil * 100,
             phil_high_pct =    (phil_high - phil) / phil * 100) %>%
      rowwise() %>%
      mutate(philL_CI      = round( min(phil_low_pct, phil_high_pct), 2)) %>%
      cbind(pr = pr_names[1:9])
    
    
    # .txt file for SSA with shocks ----
    # Create lines
    lines_AFL  <- sprintf('Vary p_AFLABORa("%s")   = %.1f;CI TD', pfp.est.wCI$pr, pfp.est.wCI$AFL_CI)
    lines_AFR  <- sprintf('Vary p_AFNLABORa("%s")  = %.1f;CI TD', pfp.est.wCI$pr, pfp.est.wCI$AFR_CI)
    lines_AFZ  <- sprintf('Vary p_AFLANDWTRa("%s") = %.1f;CI TD', pfp.est.wCI$pr, pfp.est.wCI$AFZ_CI)
    lines_PHIL <- sprintf('Vary s_QLABORa("%s")    = %.1f;CI TD', phila_wCI$pr  , phila_wCI$philL_CI)
    
    # Combine with blank lines between groups
    output_lines <- c(lines_AFL, "", lines_AFR, "", lines_AFZ, "", lines_PHIL)
    output_lines
    
    # Write to file
    writeLines(output_lines, file.path(paste0("temp/3_SSA_SHK_", source, "_", phil, ".txt")))
    
    cat("✅ Text file created: vFor shhks \n")
    
    # .txt file for SSA with parms
    # Create lines
    lines_ESLZ  <- sprintf('Vary ESUB_LZa("%s") = %.2f;CI TD', pfp.est.wCI$pr, pfp.est.wCI$ESLZ_CI)
    lines_ELAB  <- sprintf('Vary ELABORa("%s")  = %.2f;CI TD', pfp.est.wCI$pr, pfp.est.wCI$ELAB_CI)
    
    # Combine with blank lines between groups
    output_lines <- c(lines_ESLZ, "", lines_ELAB)
    
    # Write to file
    writeLines(output_lines, file.path(paste0("temp/3_SSA_PARM_", source, "_", phil, ".txt")))
    
    cat("✅ Text file created: vFor PARMSxt\n")
    
  }
  
}

# Use the .TXT files to manually generate the CI for pl.sim and ql.sim
# save them using patter in iv.
# then run the next loop only for NAWS, (N,C)

#   ii.  Read in the median run ----
wpfpz.naws.n    = extractz.sl4( read_SL4("out/2-Validation-NAWS-NM3.sl4", toLowerCase = F) , headerz, prefix = "pfp_") %>%
  as.data.frame() %>%
  mutate(CZ = 1:605) %>%
  left_join(CZ2PR) 

wpfpz.naws.c    = extractz.sl4( read_SL4("out/2-Validation-NAWS-CM3.sl4", toLowerCase = F) , headerz, prefix = "pfp_") %>%
  as.data.frame() %>%
  mutate(CZ = 1:605) %>%
  left_join(CZ2PR) 

wpfpa.naws    = extracta.sl4( read_SL4("out/2-Validation-NAWS-NM3.sl4", toLowerCase = F) , headera) %>%
  as.data.frame() %>%
  slice(1:9, )

wpfpz.fls.n    = extractz.sl4( read_SL4("out/2-Validation-FLS-NM3.sl4", toLowerCase = F) , headerz, prefix = "pfp_") %>%
  as.data.frame() %>%
  mutate(CZ = 1:605) %>%
  left_join(CZ2PR)

wpfpz.fls.c    = extractz.sl4( read_SL4("out/2-Validation-FLS-CM3.sl4", toLowerCase = F) , headerz, prefix = "pfp_") %>%
  as.data.frame() %>%
  mutate(CZ = 1:605) %>%
  left_join(CZ2PR)

wpfpa.fls    = extracta.sl4( read_SL4("out/2-Validation-FLS-NM3.sl4", toLowerCase = F) , headera) %>%
  as.data.frame() %>%
  slice(1:9, )

#  iii.  Historical Validation plots (pl, ql, pc, qc) & Figure 7 from paper ----
ci = 95 # 99
t_val = ifelse(ci == 95, 1.96, 2.576)
rm(source, phil)

for (source in sources) {
  #if(source == "FLS") next
  for (phil in phils) {
    
    # Dynamic scenario def, ssa.m1 and ssa.sd files ----
    if (source == "NAWS" & phil == "N") {
      
      data_labor_pr    <- NAWS_N_data_labora
      hist.post.z      <- get(paste0("wpfpz.", tolower(source), ".", tolower(phil)))
      phila            <- PHIL_Na
      philz            <- PHIL_Nz
      ssa.shk.m.file   <- paste0("3_SSA_wTRIM_SHK_", source, "_", phil, "_m1.sl4")
      ssa.shk.sd.file  <- paste0("3_SSA_wTRIM_SHK_", source, "_", phil, "_sd.sl4")
      
    } else if (source == "NAWS" & phil == "C") {
      
      data_labor_pr    <- NAWS_C_data_labora
      #hist.post.a      <- get(paste0("wpfpa.", tolower(source)))
      hist.post.z      <- get(paste0("wpfpz.", tolower(source), ".", tolower(phil)))
      phila            <- PHIL_Ca
      philz            <- PHIL_Cz
      ssa.shk.m.file   <- paste0("3_SSA_wTRIM_SHK_", source, "_", phil, "_m1.sl4")
      ssa.shk.sd.file  <- paste0("3_SSA_wTRIM_SHK_", source, "_", phil, "_sd.sl4")
      
    } else if (source == "FLS" & phil == "N") {
      
      data_labor_pr    <- FLS_N_data_labora
      #hist.post.a      <- get(paste0("wpfpa.", tolower(source)))
      hist.post.z      <- get(paste0("wpfpz.", tolower(source), ".", tolower(phil)))
      phila            <- PHIL_Na
      philz            <- PHIL_Nz
      ssa.shk.m.file   <- paste0("3_SSA_wTRIM_SHK_", source, "_", phil, "_m1.sl4")
      ssa.shk.sd.file  <- paste0("3_SSA_wTRIM_SHK_", source, "_", phil, "_sd.sl4")
      
    } else if (source == "FLS" & phil == "C") {
      
      data_labor_pr    <- FLS_C_data_labora
      #hist.post.a      <- get(paste0("wpfpa.", tolower(source)))
      hist.post.z      <- get(paste0("wpfpz.", tolower(source), ".", tolower(phil)))
      phila            <- PHIL_Ca
      philz            <- PHIL_Cz
      ssa.shk.m.file   <- paste0("3_SSA_wTRIM_SHK_", source, "_", phil, "_m1.sl4")
      ssa.shk.sd.file  <- paste0("3_SSA_wTRIM_SHK_", source, "_", phil, "_sd.sl4")
      
    } 
    
    message("Read the SSA files")
    SSA_mean_a    = extracta.sl4( read_SL4(paste0("ssa/", ssa.shk.m.file), toLowerCase = F) , headera) %>%
      as.data.frame() %>%
      slice(1:9, )
    
    SSA_mean_z    = extractz.sl4( read_SL4(paste0("ssa/", ssa.shk.m.file), toLowerCase = F) , headerz, prefix = "ssa_m_") %>%
      as.data.frame() %>%
      mutate(across(-CZ, ~ ifelse(CZ %in% cz.drop, 0, .))) %>%
      slice(1:605, ) %>%
      left_join(CZ2PR)
    
    SSA_sd_z    = extractz.sl4( read_SL4(paste0("ssa/", ssa.shk.sd.file), toLowerCase = F) , headerz, prefix = "ssa_sd_") %>%
      as.data.frame() %>%
      mutate(across(-CZ, ~ ifelse(CZ %in% cz.drop, 0, .))) %>%
      slice(1:605, ) %>%
      left_join(CZ2PR)
    # same few lines of SIM.HIST, uses simulated m directly
    pattern <- paste0("_", source, ".*_med")
    
    message("Calculate .sim levels at CZ")
    SSA.HIST    <- data_fips %>%
      dplyr::select(FIPS, CZ, HLAND2002_ltavg, HLAND2017_ltavg, RENT02, RENT17, VLAB2002_ltavg, VLAB2017_ltavg, VZ_02, VZ_17) %>%
      full_join(QLAB_F %>% dplyr::select(FIPS, matches(pattern))) %>%
      rename_with(~ "WAGE02", contains("WAGE02")) %>%
      rename_with(~ "QLAB02", contains("QLAB02")) %>%
      rename_with(~ "WAGE17", contains("WAGE17")) %>%
      rename_with(~ "QLAB17", contains("QLAB17")) %>%
      mutate(VLAND17 = RENT17 * HLAND2017_ltavg) %>%
      #left_join(subset(SSA_mean_z, select=c(CZ, ssa_m_p_QLANDz,  ssa_m_p_PLANDz,  ssa_m_p_QLABORz,  ssa_m_p_PLABORz,  ssa_m_p_QNLABORz,  ssa_m_p_PNLABORz))) %>%
      left_join(hist.post.z %>% dplyr::select(CZ, pfp_p_QLANDz,  pfp_p_PLANDz,  pfp_p_QLABORz,  pfp_p_PLABORz,  pfp_p_QNLABORz,  pfp_p_PNLABORz)) %>%
      # some very large simulated results leading to large PR/LS results
      # mutate(pfp_p_PLABORz = ifelse(PR9==3 & pfp_p_PLABORz > 100, 100, pfp_p_PLABORz),
      #        pfp_p_PLABORz = ifelse(PR9==8 & pfp_p_PLABORz > 100, 100, pfp_p_PLABORz),
      #        pfp_p_PLANDz  = ifelse(PR9==3 & pfp_p_PLANDz  > 100, 100, pfp_p_PLANDz),
      #        pfp_p_PLANDz  = ifelse(PR9==8 & pfp_p_PLANDz  > 100, 100, pfp_p_PLANDz)) %>%
      mutate(sim.QLAND17 = round( HLAND2002_ltavg*(1+pfp_p_QLANDz/100),0),
             sim.RENT17    = round( RENT02 * 1.37 * (1 + pfp_p_PLANDz/100),0),
             sim.VLAND17  = round( sim.RENT17   * sim.QLAND17,0),
             sim.QLAB17  = round( QLAB02 *(1+pfp_p_QLABORz/100),0),
             sim.WAGE17    = round( WAGE02 * 1.37 * (1 + pfp_p_PLABORz/100), 1),
             sim.VLAB17  = round( sim.WAGE17   * sim.QLAB17,0),
             sim.VZ17    = round( VZ_02*cpi0217/100*(1+(pfp_p_QNLABORz + pfp_p_PNLABORz)/100),0)) %>%
      
      left_join(subset(SSA_sd_z,   select=c(CZ, ssa_sd_p_QLANDz, ssa_sd_p_PLANDz, ssa_sd_p_QLABORz, ssa_sd_p_PLABORz, ssa_sd_p_QNLABORz, ssa_sd_p_PNLABORz))) %>%
      mutate(ssa_low_p_QLANDz    = pfp_p_QLANDz   - (t_val * ssa_sd_p_QLANDz),
             ssa_low_p_PLANDz    = pfp_p_PLANDz   - (t_val * ssa_sd_p_PLANDz),
             ssa_low_p_QLABORz   = pfp_p_QLABORz  - (t_val * ssa_sd_p_QLABORz),
             ssa_low_p_PLABORz   = pfp_p_PLABORz  - (t_val * ssa_sd_p_PLABORz),
             ssa_low_p_QNLABORz  = pfp_p_QNLABORz - (t_val * ssa_sd_p_QNLABORz),
             ssa_low_p_PNLABORz  = pfp_p_PNLABORz - (t_val * ssa_sd_p_PNLABORz),
             
             ssa_high_p_QLANDz    = pfp_p_QLANDz   + (t_val * ssa_sd_p_QLANDz),
             ssa_high_p_PLANDz    = pfp_p_PLANDz   + (t_val * ssa_sd_p_PLANDz),
             ssa_high_p_QLABORz   = pfp_p_QLABORz  + (t_val * ssa_sd_p_QLABORz),
             ssa_high_p_PLABORz   = pfp_p_PLABORz  + (t_val * ssa_sd_p_PLABORz),
             ssa_high_p_QNLABORz  = pfp_p_QNLABORz + (t_val * ssa_sd_p_QNLABORz),
             ssa_high_p_PNLABORz  = pfp_p_PNLABORz + (t_val * ssa_sd_p_PNLABORz)) %>%
      
      mutate(SSA.low.QLAND17 = HLAND2002_ltavg         *( 1 + ssa_low_p_QLANDz/100),
             SSA.low.RENT17    = RENT02   * cpi0217/100         * (1 + ssa_low_p_PLANDz/100),
             SSA.low.VLAND17 = SSA.low.QLAND17       * SSA.low.RENT17,
             
             SSA.low.QLAB17  = QLAB02    * (1 + ssa_low_p_QLABORz/100),
             SSA.low.WAGE17    = WAGE02 * cpi0217/100           * (1 + ssa_low_p_PLABORz/100),
             SSA.low.VLAB17  = SSA.low.QLAB17  *  SSA.low.WAGE17,
             
             SSA.low.VZ17    = VZ_02*cpi0217/100*(1 +(ssa_low_p_QNLABORz + ssa_low_p_PNLABORz)/100),
             
             SSA.high.QLAND17 = HLAND2002_ltavg       *( 1 + ssa_high_p_QLANDz/100),
             SSA.high.RENT17    = RENT02   * cpi0217/100 * (1 + ssa_high_p_PLANDz/100),
             SSA.high.VLAND17 = SSA.high.QLAND17 * SSA.high.RENT17,
             
             SSA.high.QLAB17  = QLAB02  * (1 + ssa_high_p_QLABORz/100),
             SSA.high.WAGE17    = WAGE02 * cpi0217/100   * (1 + ssa_high_p_PLABORz/100),
             SSA.high.VLAB17  = SSA.high.QLAB17  *  SSA.high.WAGE17,
             
             SSA.high.VZ17    = VZ_02*cpi0217/100*(1 +(ssa_high_p_QNLABORz + ssa_high_p_PNLABORz)/100)) %>%
      
      left_join(STATE) %>%
      left_join(FIPS2PR)
    
    message("Calculate .sim levels at PR")
    SSA.HIST.PR <- SSA.HIST %>%
      group_by(PR9) %>%
      summarize(across(c(sim.QLAB17, sim.VLAB17, sim.VZ17, sim.QLAND17, sim.VLAND17,
                         SSA.low.QLAB17,  SSA.high.QLAB17,  SSA.low.VLAB17,  SSA.high.VLAB17, SSA.low.VZ17,  SSA.high.VZ17,
                         SSA.low.QLAND17, SSA.high.QLAND17, SSA.low.VLAND17, SSA.high.VLAND17), sum, na.rm=T)) %>%
      mutate(sim.WAGE17        = round( sim.VLAB17       / sim.QLAB17,        1),
             SSA.low.WAGE17    = round( SSA.low.VLAB17   / SSA.low.QLAB17,    1),
             SSA.high.WAGE17   = round( SSA.high.VLAB17  / SSA.high.QLAB17 ,  1),
             sim.RENT17        = round( sim.VLAND17      / sim.QLAND17,        1),
             SSA.low.RENT17    = round( SSA.low.VLAND17  / SSA.low.QLAND17,   1),
             SSA.high.RENT17   = round( SSA.high.VLAND17 / SSA.high.QLAND17 , 1)) %>%
      #left_join(SIM.HIST.PR %>% dplyr::select(PR9, sim.QLAB17, sim.WAGE17, sim.VLAB17, sim.VZ17, sim.QLAND17)) %>%
      dplyr::select(PR9, ends_with(".QLAB17"), ends_with(".WAGE17"), ends_with(".QLAND17"), ends_with(".RENT17"), ends_with(".VZ17"), ends_with(".VLAND17"), ends_with(".VLAB17")) %>%
      filter(!is.na(PR9))
    SSA.HIST.PR
    
    QLAB_PR_EST <- QLAB_PR %>%
      left_join(SSA.HIST.PR %>% dplyr::select(PR9, contains("WAGE17"), contains("QLAB17")))
    names(QLAB_PR_EST)
    
    #       c. Plot historical validation ----
    message("Prep to plot the bar graphs")
    #          SEL.PR ----
    # --- Prepare data ---
    wage_levels <- summarize_bars_lab_ssa(QLAB_PR_EST, "WAGE", source, which.pr = sel_pr) %>%
      left_join(PR, by = "PR9") %>%
      mutate(pr = factor(pr, levels = PR$pr))
    
    qlab_levels <- summarize_bars_lab_ssa(QLAB_PR_EST, "QLAB", source, which.pr = sel_pr, 1000000) %>% # Million Units
      left_join(PR, by = "PR9") %>%
      mutate(pr = factor(pr, levels = PR$pr))
    
    # --- Plots ---
    p1 <- ggplot(qlab_levels, aes(x = pr, y = mean, fill = factor(year))) +
      geom_col(position = position_dodge(width = 0.8), width = 0.7) +
      geom_errorbar(aes(ymin = low, ymax = high),
                    position = position_dodge(width = 0.8), width = 0.2) +
      labs(y = "Labor hours (million)", title = "A. Employment (Labor Hours)") +
      scale_fill_manual(values = c("2002" = "tan3", "2017" = "steelblue", "Simulated-2017" = "seagreen")) +
      theme_bottom
    
    p2 <- ggplot(wage_levels, aes(x = pr, y = mean, fill = factor(year))) +
      geom_col(position = position_dodge(width = 0.8), width = 0.7) +
      geom_errorbar(aes(ymin = low, ymax = high),
                    position = position_dodge(width = 0.8), width = 0.2) +
      labs(y = "Wage (Current USD)", title = "B. Wages ($/hr)") +
      scale_fill_manual(values = c("2002" = "tan3", "2017" = "steelblue", "Simulated-2017" = "seagreen")) +
      theme_bottom +
      scale_x_discrete(labels = function(x) sub(" ", "\n", x))
    
    # --- Combine (QLAB/WAGE top, ql/pl bottom) ---
    which <- if (1 %in% unique(wage_levels$PR9)) "sel" else "oth"
    message("Prep to plot the bar graphs")
    historical.validation.plot <- (p1 + p2) 
    ggsave(filename = file.path(paste0("plots/Figure7/hist_vali_lab_", which ,"_", source, "_", phil, ".png")), historical.validation.plot, width = 10, height = 5, dpi = 300, bg = "white")
    cat("  Saved: plots/Figure7/hist_vali_lab_", which, "_", source, "_", phil, ".png\n", sep = "")
    
    #          OTH.PR ----
    oth.pr = setdiff(do.pr, sel_pr)
    # --- Prepare data ---
    wage_levels <- summarize_bars_lab_ssa(QLAB_PR_EST, "WAGE", source, which.pr = oth.pr) %>%
      left_join(PR, by = "PR9") %>%
      mutate(pr = factor(pr, levels = PR$pr))
    
    qlab_levels <- summarize_bars_lab_ssa(QLAB_PR_EST, "QLAB", source, which.pr = oth.pr, 1000000) %>% # Million Units
      left_join(PR, by = "PR9") %>%
      mutate(pr = factor(pr, levels = PR$pr))
    
    # --- Plots ---
    p1 <- ggplot(qlab_levels, aes(x = pr, y = mean, fill = factor(year))) +
      geom_col(position = position_dodge(width = 0.8), width = 0.7) +
      geom_errorbar(aes(ymin = low, ymax = high),
                    position = position_dodge(width = 0.8), width = 0.2) +
      labs(y = "Labor hours (million)", title = "A. Employment (Labor Hours)") +
      scale_fill_manual(values = c("2002" = "tan3", "2017" = "steelblue", "Simulated-2017" = "seagreen")) +
      theme_bottom +
      scale_x_discrete(labels = function(x) sub(" ", "\n", x))
    
    p2 <- ggplot(wage_levels, aes(x = pr, y = mean, fill = factor(year))) +
      geom_col(position = position_dodge(width = 0.8), width = 0.7) +
      geom_errorbar(aes(ymin = low, ymax = high),
                    position = position_dodge(width = 0.8), width = 0.2) +
      labs(y = "Wage (Current USD)", title = "B. Wages ($/hr)") +
      scale_fill_manual(values = c("2002" = "tan3", "2017" = "steelblue", "Simulated-2017" = "seagreen")) +
      theme_bottom +
      scale_x_discrete(labels = function(x) sub(" ", "\n", x))
    
    which <- if (1 %in% unique(wage_levels$PR9)) "sel" else "oth"
    
    message("Prep to plot the bar graphs")
    historical.validation.plot <- (p1 + p2) 
    #ggsave(filename = paste0(fig.path, "/hist_vali_lab_", which ,"_", source, "_", phil, ".png"), historical.validation.plot, width = 10, height = 5, dpi = 300, bg = "white")
    
  }
}


# 5. Attribute historical change to Drivers ----
cat("[5] Attributing historical change to drivers (decomposition)...\n")
#  Define function
decomp.driver <- function(sim_var, obs_var) {
  # same function as process_variable, with parm.ssa
  decomp <- data.frame(decomp.in[[sim_var]][1:9, ]) %>%
    mutate(PR9 = 1:9) %>%
    left_join(PR) %>%
    rename(sim = TOTAL) %>% # sim = simulated value for sim_var
    # !!sym(obs_var): This converts the string obs_var into a symbol, allowing dplyr to use it dynamically.
    left_join(data_pr       %>% dplyr::select(PR9, q, phic, pc, qc, qz), by = "PR9") %>%
    left_join(phila %>% dplyr::select(PR9, phil)) %>%
    left_join(data_labor_pr %>% dplyr::select(PR9, pl, ql)) %>%
    rename(obs = !!sym(obs_var))  %>% # obs = observed value for sim_var
    mutate(Q.con    = round( ifelse(sim / obs > 0, Q / sim * obs, 0), 2),
           AFL.con  = round( ifelse(sim / obs > 0, AFL / sim * obs, 0), 2),
           AFZ.con  = round( ifelse(sim / obs > 0, AFZ / sim * obs, 0), 2),
           AFR.con  = round( ifelse(sim / obs > 0, AFR / sim * obs, 0), 2),
           PHIC.con = round( ifelse(sim / obs > 0, PHIC / sim * obs, 0), 2),
           PHIL.con = round( ifelse(sim / obs > 0, PHIL / sim * obs, 0), 2),
           QW.con   = round( ifelse(sim / obs > 0, QW / sim * obs, 0), 2),
           PZ.con   = round( ifelse(sim / obs > 0, PZ / sim * obs, 0), 2)) %>%
    mutate(test = rowSums(across(ends_with(".con")))) %>%
    dplyr::select(PR9, sim, Q.con, AFL.con, AFZ.con, AFR.con, PHIC.con, PHIL.con, QW.con, PZ.con, obs, test)
  # save as a list or df wihth appropriate name
  return(decomp)}

#   i.   Decomp median run ----
for (source in sources) {
  #if (source == "FLS") next
  for (phil in phils) {
    
    # Dynamic scenario def, ssa.m1 and ssa.sd files ----
    if (source == "NAWS" & phil == "N") {
      
      data_labor_pr    <- NAWS_N_data_labora
      hist.post.a      <- get(paste0("wpfpz.", tolower(source), ".", tolower(phil)))
      phila            <- PHIL_Na
      
    } else if (source == "NAWS" & phil == "C") {
      
      data_labor_pr    <- NAWS_C_data_labora
      hist.post.a      <- get(paste0("wpfpz.", tolower(source), ".", tolower(phil)))
      phila            <- PHIL_Ca
      
    } else if (source == "FLS" & phil == "N") {
      
      data_labor_pr    <- FLS_N_data_labora
      hist.post.a      <- get(paste0("wpfpz.", tolower(source), ".", tolower(phil)))
      phila            <- PHIL_Na
      
    } else if (source == "FLS" & phil == "C") {
      
      data_labor_pr    <- FLS_C_data_labora
      hist.post.a      <- get(paste0("wpfpz.", tolower(source), ".", tolower(phil)))
      phila            <- PHIL_Ca
      
    } 
    
    decomp.in <- read_SL4(paste0("out/2-Validation-", source, "-", phil, "M3.sl4"), toLowerCase = F)
    
    decomp_pl_report <- decomp.driver("p_PLABORa", "pl")
    cat("  Decomposition (", source, "-", phil, ") pl report:", nrow(decomp_pl_report), "rows\n")
    
    decomp_ql_report <- decomp.driver("p_QLABORa", "ql")
    cat("  Decomposition (", source, "-", phil, ") ql report:", nrow(decomp_ql_report), "rows\n")
    
    # edited 19-03-26, did manual for NAWS_N, saved with R2
    write_xlsx(decomp_pl_report, path=paste0("temp/3_pl_decomp_", source, "-", phil, ".xlsx" ))
    cat("  Saved: temp/3_pl_decomp_", source, "-", phil, ".xlsx\n", sep = "")
    write_xlsx(decomp_ql_report, path=paste0("temp/3_ql_decomp_", source, "-", phil, ".xlsx" ))
    cat("  Saved: temp/3_ql_decomp_", source, "-", phil, ".xlsx\n", sep = "")
  }
}

#  ii.   Populate Table 4 ----
tab_pl           <- as.data.frame(matrix(NA, nrow = 12, ncol = 4))
colnames(tab_pl) <- c(PR$pr[1], PR$pr[3], PR$pr[2], PR$pr[7])
rownames(tab_pl) <- c("WAGE02", "WAGE17", "pl", "exp", "PFP_L", "PFP_R", "PFP_I", "PHIL_C", "PHIL_L", "QW", "PZ", "check")
tab_pl[1, ]  <- round( c(NAWS_N_data_labora$WAGE02[1], NAWS_N_data_labora$WAGE02[3], NAWS_N_data_labora$WAGE02[2], NAWS_N_data_labora$WAGE02[7]),1)
tab_pl[2, ]  <- round( c(NAWS_N_data_labora$WAGE17[1], NAWS_N_data_labora$WAGE17[3], NAWS_N_data_labora$WAGE17[2], NAWS_N_data_labora$WAGE17[7]),1)
tab_pl[3, ]  <- round( c(NAWS_N_data_labora$pl[1],     NAWS_N_data_labora$pl[3],     NAWS_N_data_labora$pl[2],     NAWS_N_data_labora$pl[7]),1)
tab_pl[4, ]  <- round( c(decomp_pl_report$Q.con[1],    decomp_pl_report$Q.con[3],    decomp_pl_report$Q.con[2],    decomp_pl_report$Q.con[7]),1)
tab_pl[5, ]  <- round( c(decomp_pl_report$AFL.con[1],  decomp_pl_report$AFL.con[3],  decomp_pl_report$AFL.con[2],  decomp_pl_report$AFL.con[7]),1)
tab_pl[6, ]  <- round( c(decomp_pl_report$AFR.con[1],  decomp_pl_report$AFR.con[3],  decomp_pl_report$AFR.con[2],  decomp_pl_report$AFR.con[7]),1)
tab_pl[7, ]  <- round( c(decomp_pl_report$AFZ.con[1],  decomp_pl_report$AFZ.con[3],  decomp_pl_report$AFZ.con[2],  decomp_pl_report$AFZ.con[7]),1)
tab_pl[8, ]  <- round( c(decomp_pl_report$PHIC.con[1], decomp_pl_report$PHIC.con[3], decomp_pl_report$PHIC.con[2], decomp_pl_report$PHIC.con[7]),1)
tab_pl[9, ]  <- round( c(decomp_pl_report$PHIL.con[1], decomp_pl_report$PHIL.con[3], decomp_pl_report$PHIL.con[2], decomp_pl_report$PHIL.con[7]),1)
tab_pl[10, ]  <- round( c(decomp_pl_report$QW.con[1], decomp_pl_report$QW.con[3], decomp_pl_report$QW.con[2], decomp_pl_report$QW.con[7]),1)
tab_pl[11, ]  <- round( c(decomp_pl_report$PZ.con[1], decomp_pl_report$PZ.con[3], decomp_pl_report$PZ.con[2], decomp_pl_report$PZ.con[7]),1)
tab_pl[12, ]  <- tab_pl[4, ] + tab_pl[5, ] + tab_pl[6, ] + tab_pl[7, ] + tab_pl[8, ] + tab_pl[9, ] + tab_pl[10, ] + tab_pl[11, ]

tab_ql           <- as.data.frame(matrix(NA, nrow = 12, ncol = 4))
colnames(tab_ql) <- c(PR$pr[1], PR$pr[3], PR$pr[2], PR$pr[7])
rownames(tab_ql) <- c("QLAB02", "QLAB17", "ql", "exp", "PFP_L", "PFP_R", "PFP_I", "PHIL_C", "PHIL_L", "QW.con", "PZ.con", "check")
tab_ql[1, ]  <- round( c(NAWS_N_data_labora$QLAB02 [1], NAWS_N_data_labora$QLAB02[3], NAWS_N_data_labora$QLAB02[2], NAWS_N_data_labora$QLAB02[7]) / 1000000,1)
tab_ql[2, ]  <- round( c(NAWS_N_data_labora$QLAB17[1],  NAWS_N_data_labora$QLAB17[3], NAWS_N_data_labora$QLAB17[2], NAWS_N_data_labora$QLAB17[7]) / 1000000,1)
tab_ql[3, ]  <- round( c(NAWS_N_data_labora$ql[1],      NAWS_N_data_labora$ql[3],     NAWS_N_data_labora$ql[2],     NAWS_N_data_labora$ql[7]),1)
tab_ql[4, ]  <- round( c(decomp_ql_report$Q.con[1],     decomp_ql_report$Q.con[3],    decomp_ql_report$Q.con[2],    decomp_ql_report$Q.con[7]),1)
tab_ql[5, ]  <- round( c(decomp_ql_report$AFL.con[1],   decomp_ql_report$AFL.con[3],  decomp_ql_report$AFL.con[2],  decomp_ql_report$AFL.con[7]),1)
tab_ql[6, ]  <- round( c(decomp_ql_report$AFR.con[1],   decomp_ql_report$AFR.con[3],  decomp_ql_report$AFR.con[2],  decomp_ql_report$AFR.con[7]),1)
tab_ql[7, ]  <- round( c(decomp_ql_report$AFZ.con[1],   decomp_ql_report$AFZ.con[3],  decomp_ql_report$AFZ.con[2],  decomp_ql_report$AFZ.con[7]),1)
tab_ql[8, ]  <- round( c(decomp_ql_report$PHIC.con[1],  decomp_ql_report$PHIC.con[3], decomp_ql_report$PHIC.con[2], decomp_ql_report$PHIC.con[7]),1)
tab_ql[9, ]  <- round( c(decomp_ql_report$PHIL.con[1],  decomp_ql_report$PHIL.con[3], decomp_ql_report$PHIL.con[2], decomp_ql_report$PHIL.con[7]),1)
tab_ql[10, ]  <- round( c(decomp_ql_report$QW.con[1], decomp_ql_report$QW.con[3], decomp_ql_report$QW.con[2], decomp_ql_report$QW.con[7]),1)
tab_ql[11, ]  <- round( c(decomp_ql_report$PZ.con[1], decomp_ql_report$PZ.con[3], decomp_ql_report$PZ.con[2], decomp_ql_report$PZ.con[7]),1)
tab_ql[12, ] <- tab_ql[4, ] + tab_ql[5, ] + tab_ql[6, ] + tab_ql[7, ] + tab_ql[8, ] + tab_ql[9, ] + tab_ql[10, ] + tab_ql[11, ]

write_xlsx(tab_pl, path=paste0("table/3_TABLE4_pl.xlsx" ))
write_xlsx(tab_ql, path=paste0("table/3_TABLE4_ql.xlsx" ))

# 6. Coefficient Estimates: Figure 8 ----
cat("[6] Plotting coefficient estimates (Figure 8)...\n")
#   i: Define Combo of wage estimates (NAWS, FLS) and phil (N, C) ----
# define Manually
source <- "NAWS"
phil   <- "N"

#  ii: Read in SSA results for beta ----
beta_pl <- read_xlsx(path=paste0("in/ssa/3_ssa_", source, "-", phil, "_beta_pl", "_729.xlsx" )) 
beta_ql <- read_xlsx(path=paste0("in/ssa/3_ssa_", source, "-", phil, "_beta_ql", "_729.xlsx" )) 

# iii: Plot ----
#       A ----
custom_order = c(1,3,2,7)
sel_var = c("beta.PHIL", "beta.AFL", "beta.AFZ", "beta.AFR", "beta.PHIC")
beta_order = rev(sel_var)

# Convert data to long format and join with the PR data
df_long <- beta_pl %>%
  filter(if_all(starts_with("beta."), ~ .x != 0)) %>%
  filter(PR9 %in% sel_pr) %>%
  dplyr::select(PR9, sel_var) %>%
  pivot_longer(cols = sel_var,
               names_to = "Variable",
               values_to = "Value") %>%
  group_by(PR9, Variable) %>%
  mutate(stats = list(boxplot.stats(Value)$stats), 
         lower = stats[[1]][1],
         upper = stats[[1]][5],
         outlier = Value < lower | Value > upper) %>%
  ungroup() %>%
  filter(!outlier) %>%
  select(-stats, -lower, -upper) %>%
  left_join(PR, by = "PR9")

df_long$Variable <- factor(df_long$Variable, levels = beta_order)

# Reorder the 'pr' column to ensure the desired order
ordered_pr_levels <- PR$pr[match(custom_order, PR$PR9)]  # your custom order (not reversed)
df_long$pr <- factor(df_long$pr, levels = rev(ordered_pr_levels), ordered = TRUE)  # for plotting (top-to-bottom)
df_long$pr_legend <- factor(df_long$pr, levels = ordered_pr_levels)  # for legend (left-to-right)

label_positions <- tibble(pr = rev(PR$pr[PR$PR9 %in% sel_pr]))

nudge_x = c(-0.35,-0.1, 0.1, 0.35, # for up or down for PHIC(FR, NC, NGP, H)
            -0.35,-0.1, 0.1, 0.35, # AR
            -0.35,-0.1, 0.1, 0.35, #AI
            -0.35,-0.1, 0.1, 0.35, #AL
            -0.35,-0.1, 0.2, 0.35) #PHIL (FR,NC,NGP,H)   
nudge_y = c(0.2,   0.2,   0.2,   0.1, # left or right for PHIC, (FR,NC,NGP,H)
            -0.25, -0.25, -0.25,-0.25, #AR
            -0.25, -0.35, -0.35,-0.25, #AI
            -0.4,  -0.5,  -0.45, -0.15,  #AL
            0.15,  -0.25,  0.15,  0.01) # PHIL

label_data <- df_long %>%
  group_by(Variable, pr) %>%
  summarise(y = median(Value), .groups = "drop") %>%
  left_join(label_positions, by = "pr") %>%
  mutate(nudge_x = nudge_x,
         nudge_y = nudge_y)

plot.pl <- ggplot(df_long, aes(x = Variable, y = Value, color = pr)) +
  # dots for the median
  stat_summary(fun = median, geom = "point", position = position_dodge(0.6), size = 5) +
  # error bars
  stat_summary(aes(group = pr), 
               fun.data = function(x) {data.frame(ymin = quantile(x, 0.025),
                                                  ymax = quantile(x, 0.975),
                                                  middle = median(x)) }, 
               geom = "errorbar", width = 0.2, position = position_dodge(0.6)) +
  # custom-positioned labels
  geom_text(data = label_data,
            aes(x = as.numeric(Variable) + nudge_x, 
                y = y + nudge_y, 
                label = sprintf("%.2f", y), color = pr),
            position = position_dodge(width = 0.6),
            size = 7, show.legend = F) +
  theme_light() +
  labs(title = " ", x = " ", y = " ", fill = "Production Regions: ") + 
  scale_color_manual(values = setNames(brewer.pal(length(ordered_pr_levels), "Set2"), ordered_pr_levels), breaks = ordered_pr_levels) +
  coord_flip() +  # Flip coordinates for horizontal bars
  
  # Vertical lines at 0.5 intervals (dashed grey)
  geom_hline(yintercept = 0, color = "black", size = 0.8, linetype = "dashed") +  
  geom_vline(xintercept = c(1.5, 2.5, 3.5), linetype = "dashed", color = "gray50", size = 0.5) +
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  
        legend.position = "bottom",  
        legend.direction = "horizontal",  
        legend.title = element_text(size = 25),
        legend.text  = element_text(size = 15), 
        axis.text.x  = element_text(size = 15),  
        axis.text.y  = element_text(size = 20, face = "bold", vjust = 1, color = "black"),
        panel.border = element_rect(color = "black", size = 1)) +  
  guides(color = guide_legend(title = NULL)) +
  scale_x_discrete(labels = c(
    "beta.AFL"  = expression(beta[aL]),
    "beta.AFZ"  = expression(beta[aI]),
    "beta.AFR"  = expression(beta[aR]),
    "beta.PHIL" = expression(beta[phi~L]),
    "beta.PHIC" = expression(beta[phi~C]))) +
  scale_y_continuous(breaks = seq(round(min(df_long$Value), 0), round(max(df_long$Value), 0), by = 1)) 
plot.pl  

#       B ----
# Convert data to long format and join with the PR data
df_long <- beta_ql %>%
  filter(if_all(starts_with("beta."), ~ .x != 0)) %>%
  filter((PR9) %in% sel_pr) %>%
  dplyr::select(PR9, sel_var) %>%
  pivot_longer(cols = sel_var,
               names_to = "Variable",
               values_to = "Value") %>%
  group_by(PR9, Variable) %>%
  mutate(stats = list(boxplot.stats(Value)$stats), 
         lower = stats[[1]][1],
         upper = stats[[1]][5],
         outlier = Value < lower | Value > upper) %>%
  ungroup() %>%
  filter(!outlier) %>%
  select(-stats, -lower, -upper) %>%
  left_join(PR, by = "PR9")

df_long$Variable <- factor(df_long$Variable, levels = beta_order)
# Reorder the 'pr' column to ensure the desired order
ordered_pr_levels <- PR$pr[match(custom_order, PR$PR9)]  # your custom order (not reversed)
df_long$pr <- factor(df_long$pr, levels = rev(ordered_pr_levels), ordered = TRUE)  # for plotting (top-to-bottom)
df_long$pr_legend <- factor(df_long$pr, levels = ordered_pr_levels)  # for legend (left-to-right)

label_positions <- tibble(pr = rev(PR$pr[PR$PR9 %in% sel_pr]))

nudge_x = c(-0.35,-0.1, 0.1, 0.4, # for up or down for PHIL, FR, NC, NGP, H
            -0.35,-0.1, 0.1, 0.4,
            -0.35,-0.1, 0.1, 0.4,
            -0.35,-0.1, 0.1, 0.4,
            -0.35,-0.1, 0.1, 0.4)  
nudge_y = c(0.1,  0.2, 0.3, 0.25, # left or right for PHIC, (FR,NC,NGP,H)
            -0.25,  -0.25, -0.3, -0.2,  #AR   (FR, NC, NGP, H)
            -0.25,  -0.4, -0.4,-0.5,  #AI   (FR, NC, NGP, H)
            -0.2,  -0.6, -0.8,-0.5,  #AL   (FR, NC, NGP, H)
            -0.2,  -0.4, -0.75,-0.25)  #PHIL (FR, NC, NGP, H)

label_data <- df_long %>%
  group_by(Variable, pr) %>%
  summarise(y = median(Value), .groups = "drop") %>%
  left_join(label_positions, by = "pr") %>%
  mutate(nudge_x = nudge_x,
         nudge_y = nudge_y) %>%
  # Fig7_ed
  mutate(label_with_star = ifelse(pr == "Heartland" & Variable == "beta.AFL",
                                  paste0(sprintf("%.2f", y), "*"),
                                  sprintf("%.2f", y)))

plot.ql <- ggplot(df_long, aes(x = Variable, y = Value, color = pr)) +
  # dots for the median
  stat_summary(fun = median, geom = "point", position = position_dodge(0.6), size = 5) +
  stat_summary(aes(group = pr),
               fun.data = function(x) {
                 low  <- quantile(x, 0.025)
                 high <- quantile(x, 0.975)
                 med  <- median(x)
                 
                 # cap the lower whisker at -3.5
                 low_capped <- pmax(low, -3.5) # using -6.5 for NAWS+C and -4.5 for FLS+C
                 
                 data.frame(ymin   = low_capped,
                            ymax   = high,
                            middle = med) }, geom = "errorbar", width = 0.2, position = position_dodge(0.6) ) +
  
  # custom-positioned labels
  geom_text(data = label_data,
            aes(x = as.numeric(Variable) + nudge_x, 
                y = y + nudge_y, 
                # Fig7_ed, 
                label = label_with_star, color = pr),
            #label = sprintf("%.2f", y), color = pr),
            position = position_dodge(width = 0.6),
            size = 7, show.legend = F) +
  theme_light() +
  labs(title = " ", x = " ", y = " ", fill = "Production Regions: ") + 
  scale_color_manual(values = setNames(brewer.pal(length(ordered_pr_levels), "Set2"), ordered_pr_levels), breaks = ordered_pr_levels) +
  coord_flip() +  # Flip coordinates for horizontal bars
  
  # Vertical lines at 0.5 intervals (dashed grey)
  geom_hline(yintercept = 0, color = "black", size = 0.8, linetype = "dashed") +  
  geom_vline(xintercept = c(1.5, 2.5, 3.5), linetype = "dashed", color = "gray50", size = 0.5) +
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  
        legend.position = "bottom",  
        legend.direction = "vertical",  
        legend.title = element_text(size = 25),
        legend.text  = element_text(size = 15), 
        axis.text.x  = element_text(size = 15),  
        axis.text.y  = element_text(size = 20, face = "bold", vjust = 1, color = "black"),
        panel.border = element_rect(color = "black", size = 1)) +  
  guides(color = guide_legend(title = NULL)) +
  scale_x_discrete(labels = c(
    "beta.AFL"  = expression(lambda[aL]),
    "beta.AFZ"  = expression(lambda[aI]),
    "beta.AFR"  = expression(lambda[aR]),
    "beta.PHIL" = expression(lambda[phi~L]),
    "beta.PHIC" = expression(lambda[phi~C]))) +
  scale_y_continuous(breaks = seq(round(min(df_long$Value), 0), round(max(df_long$Value), 0), by = 1)) 
plot.ql

#       C ----
sel_var = "beta.Q"
beta_order = rev(sel_var)

# Convert data to long format and join with the PR data
df_long <- beta_pl %>%
  select(-ends_with(".PHIC")) %>% # direction of PHIC not clear, check analytics ".PHIC"
  filter(if_all(starts_with("beta."), ~ .x != 0)) %>%
  filter((PR9) %in% sel_pr) %>%
  dplyr::select(PR9, sel_var) %>%
  pivot_longer(cols = sel_var,
               names_to = "Variable",
               values_to = "Value") %>%
  group_by(PR9, Variable) %>%
  mutate(stats = list(boxplot.stats(Value)$stats), 
         lower = stats[[1]][1],
         upper = stats[[1]][5],
         outlier = Value < lower | Value > upper) %>%
  ungroup() %>%
  filter(!outlier) %>%
  select(-stats, -lower, -upper) %>%
  left_join(PR, by = "PR9")

df_long$Variable <- factor(df_long$Variable, levels = beta_order)

# Reorder the 'pr' column to ensure the desired order
ordered_pr_levels <- PR$pr[match(sel_pr, PR$PR9)]  # your custom order (not reversed)
df_long$pr <- factor(df_long$pr, levels = rev(ordered_pr_levels), ordered = TRUE)  # for plotting (top-to-bottom)
df_long$pr_legend <- factor(df_long$pr, levels = ordered_pr_levels)  # for legend (left-to-right)

label_positions <- tibble(pr = rev(PR$pr[PR$PR9 %in% sel_pr]))

nudge_x = c(-0.35,-0.1, 0.1, 0.4)  
nudge_y = c(0.0,  -0.33, 0.4,0.01) #left to right (FR, NGP, NC, H)

label_data <- df_long %>%
  group_by(Variable, pr) %>%
  summarise(y = median(Value), .groups = "drop") %>%
  left_join(label_positions, by = "pr") %>%
  mutate(nudge_x = nudge_x,
         nudge_y = nudge_y)

plot.pl.Q <- ggplot(df_long, aes(x = Variable, y = Value, color = pr)) +
  # dots for the median
  stat_summary(fun = median, geom = "point", position = position_dodge(0.6), size = 5) +
  # error bars
  stat_summary(aes(group = pr), 
               fun.data = function(x) {data.frame(ymin = quantile(x, 0.025),
                                                  ymax = quantile(x, 0.975),
                                                  middle = median(x)) }, 
               geom = "errorbar", width = 0.2, position = position_dodge(0.6)) +
  # custom-positioned labels
  geom_text(data = label_data,
            aes(x = as.numeric(Variable) + nudge_x, 
                y = y + nudge_y, 
                label = sprintf("%.2f", y), color = pr),
            position = position_dodge(width = 0.6),
            size = 7, show.legend = F) +
  theme_light() +
  labs(title = " ", x = " ", y = "Coefficient estimate", fill = "Production Regions: ") + 
  scale_color_manual(values = setNames(brewer.pal(length(ordered_pr_levels), "Set2"), ordered_pr_levels), breaks = ordered_pr_levels) +
  coord_flip() +  # Flip coordinates for horizontal bars
  
  # Vertical lines at 0.5 intervals (dashed grey)
  geom_hline(yintercept = 0, color = "black", size = 0.8, linetype = "dashed") +  
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  
        legend.position = "bottom",  
        legend.direction = "horizontal",  
        legend.title = element_text(size = 25),
        legend.text  = element_text(size = 15), 
        axis.text.x  = element_text(size = 20),  
        axis.text.y  = element_text(size = 20, face = "bold", vjust = 1, color = "black"),
        panel.border = element_rect(color = "black", size = 1)) +  
  guides(color = guide_legend(title = NULL)) +
  scale_x_discrete(labels = c("beta.Q"  = expression(beta[q]))) +
  scale_y_continuous(breaks = seq(round(min(df_long$Value), 0), round(max(df_long$Value), 0), by = 1)) 
plot.pl.Q  

#       D ----
# Convert data to long format and join with the PR data
df_long <- beta_ql %>%
  select(-ends_with(".PHIC")) %>% # direction of PHIC not clear, check analytics ".PHIC"
  filter(if_all(starts_with("beta."), ~ .x != 0)) %>%
  filter((PR9) %in% sel_pr) %>%
  dplyr::select(PR9, sel_var) %>%
  pivot_longer(cols = sel_var,
               names_to = "Variable",
               values_to = "Value") %>%
  group_by(PR9, Variable) %>%
  mutate(stats = list(boxplot.stats(Value)$stats), 
         lower = stats[[1]][1],
         upper = stats[[1]][5],
         outlier = Value < lower | Value > upper) %>%
  ungroup() %>%
  filter(!outlier) %>%
  select(-stats, -lower, -upper) %>%
  left_join(PR, by = "PR9")

df_long$Variable <- factor(df_long$Variable, levels = beta_order)
# Reorder the 'pr' column to ensure the desired order
ordered_pr_levels <- PR$pr[match(custom_order, PR$PR9)]  # your custom order (not reversed)
df_long$pr <- factor(df_long$pr, levels = rev(ordered_pr_levels), ordered = TRUE)  # for plotting (top-to-bottom)
df_long$pr_legend <- factor(df_long$pr, levels = ordered_pr_levels)  # for legend (left-to-right)

label_positions <- tibble(pr = rev(PR$pr[PR$PR9 %in% sel_pr]))

nudge_x = c(-0.35,-0.1, 0.25, 0.4)  
nudge_y = c(0.2,  -0.55, 0.35, 0.01)

label_data <- df_long %>%
  group_by(Variable, pr) %>%
  summarise(y = median(Value), .groups = "drop") %>%
  left_join(label_positions, by = "pr") %>%
  mutate(nudge_x = nudge_x,
         nudge_y = nudge_y)  %>%
  # Fig7_ed
  mutate(label_with_star = ifelse(pr == "Heartland" & Variable == "beta.Q",
                                  paste0(sprintf("%.2f", y), "*"),
                                  sprintf("%.2f", y)))

plot.ql.Q <- ggplot(df_long, aes(x = Variable, y = Value, color = pr)) +
  # dots for the median
  stat_summary(fun = median, geom = "point", position = position_dodge(0.6), size = 5) +
  stat_summary(aes(group = pr),
               fun.data = function(x) {
                 low  <- quantile(x, 0.025)
                 high <- quantile(x, 0.975)
                 med  <- median(x)
                 
                 # cap the lower whisker at 5
                 high_capped <- pmin(high, 5)
                 
                 data.frame(ymin   = low,
                            ymax   = high_capped,
                            middle = med) }, geom = "errorbar", width = 0.2, position = position_dodge(0.6) ) +
  geom_text(data = label_data,
            aes(x = as.numeric(Variable) + nudge_x, 
                y = y + nudge_y, 
                # Fig7_ed, 
                label = label_with_star, color = pr),
            #label = sprintf("%.2f", y), color = pr),
            position = position_dodge(width = 0.6),
            size = 7, show.legend = F) +
  theme_light() +
  labs(title = " ", x = " ", y = "Coefficient estimate", fill = "Production Regions: ") + 
  scale_color_manual(values = setNames(brewer.pal(length(ordered_pr_levels), "Set2"), ordered_pr_levels), breaks = ordered_pr_levels) +
  coord_flip() +  # Flip coordinates for horizontal bars
  
  # Vertical lines at 0.5 intervals (dashed grey)
  geom_hline(yintercept = 0, color = "black", size = 0.8, linetype = "dashed") +  
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  
        legend.position = "bottom",  
        legend.direction = "horizontal",  
        legend.title = element_text(size = 25),
        legend.text  = element_text(size = 15), 
        axis.text.x  = element_text(size = 20),  
        axis.text.y  = element_text(size = 20, face = "bold", vjust = 1, color = "black"),
        panel.border = element_rect(color = "black", size = 1)) +  
  guides(color = guide_legend(title = NULL)) +
  scale_x_discrete(labels = c("beta.Q"  = expression(lambda[q]))) +
  scale_y_continuous(breaks = seq(round(min(df_long$Value), 0), round(max(df_long$Value), 0), by = 3)) 
plot.ql.Q

#       legend ----
plot.for.legend <- ggplot(df_long, aes(x = Variable, y = Value, color = pr)) +
  stat_summary(fun = median, geom = "point", position = position_dodge(0.6), size = 5) +
  # custom-positioned labels
  theme_light() +
  labs(title = " ", x = " ", y = " ", fill = "Production Regions: ") + 
  scale_color_manual(values = setNames(brewer.pal(length(ordered_pr_levels), "Set2"), ordered_pr_levels), breaks = ordered_pr_levels) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_text(size = 18, face = "bold"),
    legend.text = element_text(size = 18)
  ) +
  labs(color = "Production Regions:")

# Force the plot build to get grobs, without this the get_egend command can not see the plot fully when we add bottom and horizontal to the plot_for_legend comm
g <- ggplotGrob(plot.for.legend)

# Find the legend grob by name ("guide-box")
legend_index <- which(sapply(g$grobs, function(x) x$name) == "guide-box")

# Extract the legend grob
legend_shared <- g$grobs[[legend_index]]
legend_shared

#       combined ----
png(paste0("plots/Figure8/", source, "_", phil, "_ed.png"), units = "in", width = 12, height = 12, res = 300)
# Create the plot grid directly
ggdraw() + 
  draw_plot(plot.pl + theme(legend.position = "none"),    
            x=0,   y= 0.26, width = 0.5, height = 0.73) +
  draw_label("Coefficient Estimates for Change in Wages", 
             x = 0.25, y = 0.98, fontface = "bold", size = 18, hjust = 0.5) +
  draw_label("A: Supply Shock & Productivity Changes", 
             x = 0.25, y = 0.95, fontface = "bold", size = 15, hjust = 0.5) +
  
  draw_plot(plot.ql + theme(legend.position = "none"),
            x=0.5,    y= 0.26, width = 0.49, height = 0.73) +
  draw_label("Coefficient Estimates for Change in Employment", 
             x = 0.75, y = 0.98, fontface = "bold", size = 18, hjust = 0.5) +
  draw_label("B: Supply Shock & Productivity Changes", 
             x = 0.75, y = 0.95, fontface = "bold", size = 15, hjust = 0.5) +
  
  draw_plot(plot.pl.Q + theme(legend.position = "none"),    
            x=0,   y= 0.06, width = 0.5, height = 0.22) +
  draw_label("C: Expansion Effect", 
             x = 0.25, y = 0.24, fontface = "bold", size = 15, hjust = 0.5) +
  
  draw_plot(plot.ql.Q + theme(legend.position = "none"),    
            x=0.5, y= 0.06, width = 0.49, height = 0.22) +
  draw_label("D: Expansion Effect", 
             x = 0.75, y = 0.24, fontface = "bold", size = 15, hjust = 0.5)  +
  # Shared legend at bottom
  draw_plot(legend_shared, x = 0.25, y = 0, width = 0.5, height = 0.1) 
dev.off()

cat("2_SECOND_STEP.R complete.\n")
cat("Find Plots in Paper in plots folder, tables in tables folder \n")
cat("END \n")


