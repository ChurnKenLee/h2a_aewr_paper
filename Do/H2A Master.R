## H2A Master Files ##
## Phil Hoxie ##
## January 11, 2024 ##

gc()
rm(list = ls())

## Master Paths ## -------------------------------------------------------------
if (!exists("path_code", mode = "function")) {
  source(file.path("code", "paths.R"))
}
ensure_project_dirs()

folder_dir <- as_dir(path_root())
folder_do <- as_dir(path_do())
folder_data <- as_dir(path_processed())
folder_raw <- as_dir(path_raw())
folder_output <- as_dir(path_outputs())

## Packages ## -----------------------------------------------------------------
library(tidyverse)
library(arrow)
library(tidylog, warn.conflicts = FALSE)

## Run Sub Scripts -------------------------------------------------------------

# state min wages

source(paste0(folder_do, "H2A Pull Min Wages.R"), echo = TRUE)

## Clean

source(paste0(folder_do, "H2A Clean and Load.R"), echo = TRUE)

## Build Dataset

source(paste0(folder_do, "H2A Build Dataset.R"), echo = TRUE)

## Analysis Figs and Tables

## DDD + Figs ##
source(paste0(folder_do, "H2A Analysis Figs and Tables.R"), echo = TRUE)
