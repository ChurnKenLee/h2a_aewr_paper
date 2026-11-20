# Identification search, strategy A2 (Phase 2 acquisition): tidy FY2006 and
# FY2007 OFLC H-2A disclosure files (pre-trend years for A2).
#
# Usage (repository root):
#   Rscript --vanilla explorations/identification_search/A2/02_disclosures_fy2006_2007.R
#
# Inputs (read only):
#   data/raw/h2a_disclosures_fy2006_2007/H2a_FY2006_text.zip  (FLC Data Center,
#       Wayback capture 2010-09-20 of download/H2a_%20FY2006_text.zip)
#   data/raw/h2a_disclosures_fy2006_2007/H2a_FY2007_text.zip  (capture
#       2010-09-20 of download/H2a_FY2007_text.zip)
#   data/intermediate/h2a_disclosure_records.parquet  (FY2008 case numbers,
#       overlap check only)
# Output:
#   explorations/identification_search/sources/h2a_disclosures_fy2006_2007.csv
#
# The receipt date is not a field in either file. It is decoded from the case
# number (A|C-YYDDD-NNNNN: NPC letter, two-digit year, Julian day), which is
# the OFLC convention for these case numbers; validated below against FY2007
# Decision_Date (receipt must precede decision).

suppressMessages({
  library(data.table)
  library(arrow)
})
raw_dir <- "data/raw/h2a_disclosures_fy2006_2007"
out_dir <- "explorations/identification_search/sources"
retrieved <- "2026-11-05"

read_member <- function(zipfile, member) {
  tmp <- tempfile()
  dir.create(tmp)
  utils::unzip(file.path(raw_dir, zipfile), files = member, exdir = tmp)
  fread(file.path(tmp, member), colClasses = "character", encoding = "Latin-1", na.strings = "")
}

money <- function(x) as.numeric(gsub("[$,]", "", x))
mdy_date <- function(x) as.IDate(as.Date(sub(" .*$", "", x), format = "%m/%d/%Y"))
julian_from_case <- function(cn) {
  m <- regmatches(cn, regexec("^[A-Z]-(\\d{2})(\\d{3})-", cn))
  vapply(m, function(z) {
    if (length(z) < 3) return(NA_character_)
    d <- as.Date(sprintf("20%s-01-01", z[2])) + as.integer(z[3]) - 1
    format(d)
  }, character(1))
}

f6 <- read_member("H2a_FY2006_text.zip", "H2a_ FY2006.txt")
d6 <- f6[, .(
  fiscal_year = 2006L, case_number = Case_Number, application_type = NA_character_,
  decision_date = as.IDate(NA), case_status = Case_Status,
  certification_begin_date = mdy_date(Certification_Begin_Date),
  certification_end_date = mdy_date(Certification_End_Date),
  employer_name = Employer_Name, employer_address_1 = Employer_Address_1,
  employer_address_2 = Employer_Address_2, employer_city = Employer_City,
  employer_state = Employer_State, employer_postal_code = Employer_Postal_Code,
  agent_attorney_name = NA_character_, job_title = Job_Title,
  nbr_workers_requested = as.numeric(Nbr_Workers_Requsted),
  nbr_workers_certified = as.numeric(Nbr_Workers_Certified),
  basic_hours_per_week = as.numeric(Basic_Nbr_Hours),
  basic_rate_of_pay = money(Basic_Rate_of_Pay), basic_unit_of_pay = Wage_Rate_Per,
  alien_work_state = NA_character_,
  source_file = "H2a_FY2006_text.zip:H2a_ FY2006.txt",
  source_url = "https://web.archive.org/web/20100920023721/http://flcdatacenter.com/download/H2a_%20FY2006_text.zip"
)]

f7 <- read_member("H2a_FY2007_text.zip", "H-2A_FY2007.txt")
# The FY2007 text file has no header; field order from
# H-2A_Disclosure_File_Format.doc (same zip).
stopifnot(ncol(f7) == 20)
setnames(f7, c("Case_Number", "Application_Type", "Decision_Date", "Case_Status",
               "Certification_Begin_Date", "Certification_End_Date", "Employer_Name",
               "Employer_Address_1", "Employer_Address_2", "Employer_City", "Employer_State",
               "Employer_Postal_Code", "Agent_Attorney_Name", "Job_Title",
               "Nbr_Workers_Requested", "Nbr_Workers_Certified", "Basic_Hours_Per_Week",
               "Basic_Rate_of_Pay", "Basic_Unit_of_Pay", "Alien_Work_State"))
d7 <- f7[, .(
  fiscal_year = 2007L, case_number = Case_Number, application_type = Application_Type,
  decision_date = mdy_date(Decision_Date), case_status = Case_Status,
  certification_begin_date = mdy_date(Certification_Begin_Date),
  certification_end_date = mdy_date(Certification_End_Date),
  employer_name = Employer_Name, employer_address_1 = Employer_Address_1,
  employer_address_2 = Employer_Address_2, employer_city = Employer_City,
  employer_state = Employer_State, employer_postal_code = Employer_Postal_Code,
  agent_attorney_name = Agent_Attorney_Name, job_title = Job_Title,
  nbr_workers_requested = as.numeric(Nbr_Workers_Requested),
  nbr_workers_certified = as.numeric(Nbr_Workers_Certified),
  basic_hours_per_week = as.numeric(Basic_Hours_Per_Week),
  basic_rate_of_pay = money(Basic_Rate_of_Pay), basic_unit_of_pay = Basic_Unit_of_Pay,
  alien_work_state = Alien_Work_State,
  source_file = "H2a_FY2007_text.zip:H-2A_FY2007.txt",
  source_url = "https://web.archive.org/web/20100920025123/http://flcdatacenter.com/download/H2a_FY2007_text.zip"
)]

d <- rbind(d6, d7)
d[, received_date_from_case_number := as.IDate(julian_from_case(case_number))]
d[, retrieval_date := retrieved]
fwrite(d, file.path(out_dir, "h2a_disclosures_fy2006_2007.csv"))

cat("rows:", nrow(d), "\n")
print(d[, .(rows = .N, cases = uniqueN(case_number),
            certified_workers = sum(nbr_workers_certified, na.rm = TRUE),
            requested_workers = sum(nbr_workers_requested, na.rm = TRUE),
            hours_known = mean(!is.na(basic_hours_per_week)),
            rate_known = mean(!is.na(basic_rate_of_pay)),
            recv_decoded = mean(!is.na(received_date_from_case_number)),
            recv_min = min(received_date_from_case_number, na.rm = TRUE),
            recv_max = max(received_date_from_case_number, na.rm = TRUE)), by = fiscal_year])
print(d[, .N, by = .(fiscal_year, case_status)][order(fiscal_year, -N)])
print(d[, .N, by = .(fiscal_year, basic_unit_of_pay)][order(fiscal_year, -N)])
cat("FY2007 receipt <= decision share:",
    d[fiscal_year == 2007 & !is.na(decision_date), mean(received_date_from_case_number <= decision_date, na.rm = TRUE)], "\n")
cat("FY2007 decision date range:", format(range(d[fiscal_year == 2007]$decision_date, na.rm = TRUE)), "\n")
cat("FY2007 work state = employer state share:",
    d[fiscal_year == 2007, mean(alien_work_state == employer_state, na.rm = TRUE)], "\n")
# Multi-row cases (association members) and overlap with the local FY2008 file.
print(d[, .(rows_per_case_gt1 = mean(.N > 1)), by = .(fiscal_year, case_number)][, .(share_multi = mean(rows_per_case_gt1)), by = fiscal_year])
r8 <- as.data.table(read_parquet("data/intermediate/h2a_disclosure_records.parquet",
                                 col_select = c("case_number", "fiscal_year")))
cat("case numbers also in local FY2008+ records:",
    d[, uniqueN(case_number[case_number %in% r8$case_number]), by = fiscal_year][, paste(fiscal_year, V1)], "\n")
cat("FY2006 & FY2007 shared case numbers:", length(intersect(d6$case_number, d7$case_number)), "\n")
