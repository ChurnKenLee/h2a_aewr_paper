## H2A: Read Parquet Files from Binaries 
## Phil Hoxie
## 11/21/2023
# library(tidyverse) # in master file 
library(arrow)
library(haven)
#library(sparklyr)

## requires running master file first ##

folder_binary <- paste0(folder_dir, "files_for_phil/")  
folder_dataint <-  paste0(folder_dir, "Data Int/")  

files <- list.files(path = folder_binary, all.files = T)

keep <- str_detect(files, ".parquet")

files <- as.data.frame(files)

files$keep <- keep

files <- files %>% 
  filter(keep == TRUE)

# subset 

files <- files %>% 
  filter(
    files == "state_year_min_wage.parquet" | 
      files == "oews_county_aggregated.parquet" | 
      files == "h2a_aggregated.parquet" | 
      files == "croplandcros_county_crop_acres.parquet" | 
      files == "aewr.parquet" |
      files == "nass_census_selected_obs.parquet")

## Loop over files to read parquet files ##

for (i in 1:length(files$files)){
  df <- read_parquet(file = paste0(folder_binary, files$files[i]))
  # write_dta(df, path = paste0(folder_dataint, gsub(".parquet", "",files$files[i]),".dta" ))
  # write.csv(df, file = paste0(folder_dataint, gsub(".parquet", "",files$files[i]),".csv" ))
  print(i/length(files$files))
  rm(df)
}
