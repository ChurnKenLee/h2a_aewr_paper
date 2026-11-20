# Extract county-quarter private-sector crop-production (NAICS 111) pay and
# employment from raw QCEW quarterly single files. Reads only the years asked for.
# Args: 1 = scratch output dir (outside the repo); 2 = first year; 3 = last year;
#       4 = optional directory holding <year>_qtrly_singlefile.zip (default data/raw/qcew).
# Writes qcew_crop_county_quarter_<first>_<last>.parquet to the output dir.
source("explorations/episode_2025/ep_helpers.R")
out_dir <- id_output_dir()
args <- commandArgs(trailingOnly = TRUE)
first_year <- as.integer(args[2]); last_year <- as.integer(args[3])
raw_dir <- if (length(args) >= 4) args[4] else path_raw("qcew")
stopifnot(is.finite(first_year), is.finite(last_year), first_year <= last_year, dir.exists(raw_dir))

keep <- c("area_fips", "own_code", "industry_code", "agglvl_code", "year", "qtr", "disclosure_code", "qtrly_estabs",
  "month1_emplvl", "month2_emplvl", "month3_emplvl", "total_qtrly_wages", "avg_wkly_wage")
one_year <- function(y) {
  zip <- file.path(raw_dir, paste0(y, "_qtrly_singlefile.zip"))
  stopifnot(file.exists(zip))
  member <- utils::unzip(zip, list = TRUE)$Name
  member <- member[grepl("singlefile\\.csv$", member)]
  stopifnot(length(member) == 1)
  pick <- function(x, pos) x[x$own_code == "5" & x$industry_code == "111" & x$agglvl_code == "75", keep]
  d <- readr::read_csv_chunked(unz(zip, member), readr::DataFrameCallback$new(pick), chunk_size = 500000,
    col_types = readr::cols(.default = readr::col_character()), progress = FALSE)
  message(y, ": ", nrow(d), " county-quarter rows from ", member)
  d
}
d <- bind_rows(lapply(first_year:last_year, one_year)) %>%
  filter(grepl("^[0-9]{5}$", area_fips), !grepl("(000|999)$", area_fips)) %>%
  transmute(county_fips = harmonize_county_fips_2010(area_fips), year = as.integer(year), qtr = as.integer(qtr),
    disclosed = !(disclosure_code %in% "N"), establishments = as.numeric(qtrly_estabs),
    emp = (as.numeric(month1_emplvl) + as.numeric(month2_emplvl) + as.numeric(month3_emplvl)) / 3,
    total_qtrly_wages = as.numeric(total_qtrly_wages), avg_wkly_wage = as.numeric(avg_wkly_wage)) %>%
  filter(!is.na(county_fips)) %>%
  group_by(county_fips, year, qtr) %>%
  summarise(disclosed = all(disclosed), establishments = sum(establishments), emp = sum(emp),
    total_qtrly_wages = sum(total_qtrly_wages), avg_wkly_wage = sum(avg_wkly_wage * emp) / sum(emp), .groups = "drop")
stopifnot(!anyDuplicated(d[c("county_fips", "year", "qtr")]), is.character(d$county_fips), min(d$year) >= first_year, max(d$year) <= last_year)
write_parquet(d, file.path(out_dir, sprintf("qcew_crop_county_quarter_%d_%d.parquet", first_year, last_year)))
print(d %>% group_by(year) %>% summarise(rows = n(), mean_quarterly_emp_k = sum(emp[disclosed], na.rm = TRUE) / 4e3, disclosed_rows = sum(disclosed)), n = 30)
