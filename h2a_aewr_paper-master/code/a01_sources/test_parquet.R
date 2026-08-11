library("arrow")
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
test <- read_parquet(path_int(
  "h2a_prediction_elastic_net_model_cutoff_2008.parquet"
))
test <- test %>% filter(!is.na(scale))
