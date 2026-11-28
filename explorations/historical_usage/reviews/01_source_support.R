# Coverage checks for the two public data extensions suggested by the reviewers.
source("code/paths.R")
source(path_code("c00_shared", "geography.R"))
library(data.table)
setDTthreads(2L)

# One observation per WA case with unambiguous disclosed piece-rate fields.
fields <- c("fiscal_year", "case_number", "worksite_state", "piece_rate_offer", "piece_rate_unit")
offers <- as.data.table(arrow::read_parquet(path_int("h2a_disclosure_records.parquet"),
                                           col_select = tidyselect::all_of(fields)))
offers <- unique(offers[worksite_state == "WA" & fiscal_year %between% c(2017L, 2025L) &
                        !is.na(case_number) & nzchar(trimws(case_number))])
conflicting <- offers[, .N, by = .(fiscal_year, case_number)][N > 1L]
offers <- offers[!conflicting, on = .(fiscal_year, case_number)]
offers[, positive := suppressWarnings(as.numeric(piece_rate_offer)) > 0]
offers[, attachment := grepl("addend|attachment|additional crop|additional wage", piece_rate_unit,
                             ignore.case = TRUE)]
offers[, fruit := grepl("apple|cherr|pear", piece_rate_unit, ignore.case = TRUE)]
print(offers[, .(cases = .N, positive_rate = sum(positive, na.rm = TRUE),
                 attachment_pointer = sum(attachment),
                 positive_and_names_fruit = sum(positive & fruit, na.rm = TRUE)), by = fiscal_year
             ][order(fiscal_year)])
cat("Conflicting WA case-year keys excluded:", nrow(conflicting), "\n")

# Read the original county-level 4-digit industry. The existing 3-digit
# extract uses a different aggregation-level code. Keep only observed rows.
archive <- path_raw("qcew", "2024_qtrly_singlefile.zip")
member <- unzip(archive, list = TRUE)$Name
member <- member[endsWith(member, ".csv")]
stopifnot(length(member) == 1L)
input <- unz(archive, member, open = "r")
header <- readLines(input, n = 1L)
data <- list()
repeat {
  lines <- readLines(input, n = 50000L)
  if (!length(lines)) break
  selected <- lines[grepl(',"1151",', lines, fixed = TRUE)]
  if (!length(selected)) next
  chunk <- fread(text = paste(c(header, selected), collapse = "\n"),
                 select = c("area_fips", "own_code", "industry_code", "agglvl_code",
                            "qtr", "disclosure_code"), colClasses = "character", na.strings = NULL)
  county <- chunk[industry_code == "1151" & agglvl_code == "76" &
                  own_code %chin% c("1", "2", "3", "5") & grepl("^[0-9]{5}$", area_fips)]
  county[, county_fips := harmonize_county_fips_2010(area_fips)]
  data[[length(data) + 1L]] <- county[state_from_county_fips(county_fips) <= "56" &
                                      county_code_from_county_fips(county_fips) != "999"]
}
close(input)
county_quarters <- rbindlist(data)[, .(disclosed = all(disclosure_code == "")), by = .(county_fips, qtr)]
print(county_quarters[, .(reported_counties = uniqueN(county_fips), reported_county_quarters = .N,
                          disclosed_county_quarters = sum(disclosed))])
