
# check to make sure you have the necessary packages installed
packages <- c("data.table", "purrr", "tidyverse", "here", "readxl")

# Check if packages are installed and install them if they are not
for (package in packages) {
  if (!package %in% installed.packages()) {
    install.packages(package)
  }
}

library(data.table)
library(purrr)
library(tidyverse)
library(here)
library(readxl)

# Set the working directory to the root of the Repo file
your_path_before_Repo = "/Users/alexandrahill/Library/CloudStorage/Box-Box/01_research/ali_kaleb/H-2A Project/Repo"

warning(paste0("Does this file path look correct? ",here(your_path_before_Repo,'Code','00_Setup.r \n'),
"If not, please update the 'your_path_before_Repo' variable in the 00_Setup.r file."))


source(here(your_path_before_Repo,'Code','01_combine_h2a.r'))