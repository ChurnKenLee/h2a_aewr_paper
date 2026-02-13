## H2A Master Files ##
## Phil Hoxie ##
## January 11, 2024 ##

gc()
rm(list = ls())

## Master Paths ## -------------------------------------------------------------

# folder_dir <- paste0("C:/Users/",Sys.info()["user"],"/Dropbox/H-2A Paper/")
# folder_do <- paste0(folder_dir, "Do/")
# folder_data <- paste0(folder_dir, "Data Int/")
# folder_output <- paste0(folder_dir, "Output/")

# folder_dir <- "R:/Hoxie/H-2A Paper/"
# folder_do <- "R:/Hoxie/H-2A Paper/Do/"
# folder_data <- "R:/Hoxie/H-2A Paper/Data Int/"
# folder_output <- "R:/Hoxie/H-2A Paper/Output/"

library(here)
folder_dir <- paste0(here(), "/")
folder_do <- paste0(folder_dir, "Do/")
folder_data <- paste0(folder_dir, "Data Int/")
folder_output <- paste0(folder_dir, "Output/")

## Packages ## -----------------------------------------------------------------

library(tidyverse)
library(arrow)
library(tidylog, warn.conflicts = FALSE)

## Run Sub Scripts -------------------------------------------------------------

# state min wages

source(paste0(folder_do, "H2A Pull Min Wages.R"), echo = TRUE)

## Load files from Ken 

# source(paste0(folder_do, "Read Parquet.R"), echo = TRUE)

## Clean 

source(paste0(folder_do, "H2A Clean and Load.R"), echo = TRUE)

## Build Dataset 

source(paste0(folder_do, "H2A Build Dataset.R"), echo = TRUE)

## Analysis Figs and Tables 

## DDD + Figs ##

source(paste0(folder_do, "H2A Analysis Figs and Tables.R"), echo = TRUE)




