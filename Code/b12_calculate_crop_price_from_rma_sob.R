library(here)
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)

rm(list = ls())

rma_sob <- read_parquet(here("binaries", "rma_sob.parquet"))


