rm(list = ls())
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)
library(janitor)

fls <- read_parquet(path_int("fls_region.parquet"))
fls <- fls %>% arrange(region_name, estimate_year, )
