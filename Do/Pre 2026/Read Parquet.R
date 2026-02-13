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

## Loop over files to read parquet files ##

for (i in 1:length(files$files)){
  df <- read_parquet(file = paste0(folder_binary, files$files[i]))
 # write_dta(df, path = paste0(folder_dataint, gsub(".parquet", "",files$files[i]),".dta" ))
  write.csv(df, file = paste0(folder_dataint, gsub(".parquet", "",files$files[i]),".csv" ))
  print(i/length(files$files))
  rm(df)
}

# gsub(".parquet", "",files$files[1]) 
# 
# length(files$files) ## this many iterations to handle 
# 
# # 1
# 
# df <- read_parquet(file = paste0(folder_binary, files$files[1]))
# names(df)[39] <- "CV_pct"
# 
# write_dta(df, path = paste0(folder_dataint, gsub(".parquet", "",files$files[1]),".dta" ))
# print(1/length(files$files))
# rm(df)
# 
# # 2
# 
# df <- read_parquet(file = paste0(folder_binary, files$files[2]))
# write_dta(df, path = paste0(folder_dataint, gsub(".parquet", "",files$files[2]),".dta" ))
# print(2/length(files$files))
# rm(df)
# 
# # 3
# 
# df <- read_parquet(file = paste0(folder_binary, files$files[3]))
# write_dta(df, path = paste0(folder_dataint, gsub(".parquet", "",files$files[3]),".dta" ))
# print(3/length(files$files))
# 
# # 4
# 
# df <- read_parquet(file = paste0(folder_binary, files$files[4]))
# write_dta(df, path = paste0(folder_dataint, gsub(".parquet", "",files$files[4]),".dta" ))
# print(4/length(files$files))
# 
# # 5
# 
# df <- read_parquet(file = paste0(folder_binary, files$files[5]))
# write_dta(df, path = paste0(folder_dataint, gsub(".parquet", "",files$files[5]),".dta" ))
# print(5/length(files$files))
# 
# # 6
# 
# df <- read_parquet(file = paste0(folder_binary, files$files[6]))
# write_dta(df, path = paste0(folder_dataint, gsub(".parquet", "",files$files[6]),".dta" ))
# print(6/length(files$files))
# 
# # 7
# 
# df <- read_parquet(file = paste0(folder_binary, files$files[7]))
# write_dta(df, path = paste0(folder_dataint, gsub(".parquet", "",files$files[7]),".dta" ))
# print(7/length(files$files))
# 
# # 8
# 
# df <- read_parquet(file = paste0(folder_binary, files$files[8]))
# names(df)
# names(df)[37] <- "CV_pct"
# write_dta(df, path = paste0(folder_dataint, gsub(".parquet", "",files$files[8]),".dta" ))
# print(8/length(files$files))