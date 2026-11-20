# ============================================================
#  SETUP: Paste the path to your replication folder here
#  Use forward slashes, e.g. "C:/Users/Jane/Ray26AJAE_Replication"
# ============================================================

rm(list = ls())
project_root <- "D:/Models/Labor/PFP/Ray26AJAE_Replication" # <-- UPDATE WITH PATH WHERE YOU UNZIP THE REPLICATION FOLDER
setwd(paste0(project_root))

packages <- c(
  "rlang",
  "readxl",
  "tidyr",
  "writexl",
  "sf",
  "ggplot2",
  "RColorBrewer",
  "viridis",
  "scales",
  "cowplot",
  "ggrepel",
  "dplyr",
  "ipumsr",
  "readstata13",
  "HARr",
  "doParallel", 
  "doRNG",
  "patchwork"
)

installed_packages <- rownames(installed.packages())

options(repos = c(CRAN = "https://cloud.r-project.org"))

for (pkg in packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
}