# Identification search, strategy A2 (Phase 3 design, step 1a): employer-
# location certified rows and the county x receipt-FY panel, FY2006-FY2011.
#
# Usage (repository root):
#   Rscript --vanilla explorations/identification_search/A2/03_rows_panel.R "$S"
# Writes $S/A2/a2_rows.parquet (row level) and $S/A2/a2_county_rfy_panel.parquet.
#
# Why employer location: the FY2006-07 disclosures (the only pre-period beyond
# FY2008) report the employer's address and no worksite. To keep one geography
# in every year, all years FY2006-FY2011 are located by employer ZIP -> county
# (2010 ZCTA-county relationship file, the county holding the largest share of
# the ZCTA population). The shared worksite panel is used only for a FY2008
# geography-agreement check (pre-period).
#
# Uniform row rules for every disclosure file FY2006-FY2012:
# - positive certified workers and a 'Certified' status;
# - association masters dropped: within a multi-row case, a row is a master if
#   it carries an organization flag (FY2008+ files, the b01 pre-FY2016 rule) or
#   its certified count equals the sum of the other rows' (any year), provided
#   another row has positive certified workers;
# - case numbers in several files: the record in the latest file is kept;
# - receipt date: case_received_date (FY2008+), decoded from the case number
#   (FY2006-07); receipt FY = federal FY of the receipt date;
# - weeks = (end - begin) / 7 from period dates (FY2008+, b01 basis) or
#   certification dates (FY2006-07); rows without valid dates dropped;
# - outcome hours = certified positions x weeks x 40 (constant weekly hours:
#   weekly hours are missing for every FY2008-FY2010 case, so a constant is
#   used in every year; PPML with unit FE is invariant to the constant);
# - monthly-paid rows (range herders and other special procedures, outside the
#   OWL schedule) are flagged and excluded from the primary outcome.
#
# Integrity: no outcome is related to any dose here. Printed checks are counts,
# geocoding coverage and FY2008 (pre-period) comparisons only.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(readr); library(tidyr)})
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "A2"); dir.create(out, showWarnings = FALSE)
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
CONST_WEEKLY_HOURS <- 40
fed_fy <- function(d) as.integer(format(d, "%Y")) + as.integer(as.integer(format(d, "%m")) >= 10)
num <- function(x) suppressWarnings(as.numeric(x))

# ---- ZIP -> county ----------------------------------------------------------
z <- read_csv(file.path(path_root(), "data/raw/geographic_crosswalks/census/zcta_county_rel_10.txt"),
  col_types = cols(.default = "c")) %>%
  transmute(zip5 = ZCTA5, county_fips = harmonize_county_fips_2010(GEOID), zpoppct = num(ZPOPPCT), zareapct = num(ZAREAPCT)) %>%
  arrange(zip5, desc(zpoppct), desc(zareapct), county_fips) %>% group_by(zip5) %>% slice(1) %>% ungroup() %>%
  select(zip5, county_fips)
# Fallbacks for ZIPs that are not ZCTAs (PO boxes) or missing: (1) employer
# city + state matched to a 2010 Census place or county subdivision name (first
# listed county); (2) the numerically nearest ZCTA in the employer's state.
cn <- read_csv(file.path(path_root(), "data/raw/geographic_crosswalks/census/national_county.txt"), col_names = FALSE,
  col_types = cols(.default = "c")) %>%
  transmute(st = X1, cname = toupper(X4), county_fips = harmonize_county_fips_2010(paste0(X2, X3)))
strip_type <- function(x) toupper(trimws(sub(" (city|town|village|borough|CDP|township|municipality|city and borough|plantation|gore|grant|location|purchase|comunidad|zona urbana|unorganized territory|UT|CCD)$", "", x, ignore.case = TRUE)))
pl <- read_delim(file.path(path_root(), "data/raw/geographic_crosswalks/census/national_places.txt"), delim = "|",
  col_types = cols(.default = "c"), locale = locale(encoding = "latin1")) %>% transmute(st = STATE, city = strip_type(PLACENAME), cname = toupper(trimws(sub(",.*$", "", COUNTY))))
cs <- read_csv(file.path(path_root(), "data/raw/geographic_crosswalks/census/national_cousub.txt"), col_types = cols(.default = "c"), locale = locale(encoding = "latin1")) %>%
  transmute(st = STATE, city = strip_type(COUSUBNAME), cname = toupper(COUNTYNAME))
city_cty <- bind_rows(pl, cs) %>% inner_join(cn, by = c("st", "cname")) %>% distinct(st, city, .keep_all = TRUE) %>% select(st, city, county_fips_city = county_fips)
zst <- z %>% mutate(st_fips = substr(county_fips, 1, 2), zn = as.numeric(zip5))
st_abb <- c(setNames(state.abb, toupper(state.name)), "DISTRICT OF COLUMBIA" = "DC", "PUERTO RICO" = "PR")
abb_fips <- distinct(cn %>% transmute(st, st_fips = substr(county_fips, 1, 2)))
nearest_zcta <- function(zip, stf) {
  out <- rep(NA_character_, length(zip))
  for (i in which(!is.na(zip) & !is.na(stf))) { c <- zst[zst$st_fips == stf[i], ]
    if (nrow(c)) out[i] <- c$county_fips[which.min(abs(c$zn - as.numeric(zip[i])))] }
  out
}

# ---- FY2006-07 rows ---------------------------------------------------------
o <- read_csv("explorations/identification_search/sources/h2a_disclosures_fy2006_2007.csv", col_types = cols(.default = "c")) %>%
  transmute(file_fy = as.integer(fiscal_year), case_number, case_status, organization_flag = NA_character_,
    employer_name, employer_state, employer_city, zip_raw = employer_postal_code, cert = num(nbr_workers_certified),
    begin = as.Date(certification_begin_date), end = as.Date(certification_end_date),
    received = as.Date(received_date_from_case_number), wage = num(basic_rate_of_pay), unit = basic_unit_of_pay,
    job_title)
# ---- FY2008-12 rows ---------------------------------------------------------
n <- open_dataset(path_int("h2a_disclosure_records.parquet")) %>% filter(fiscal_year >= 2008, fiscal_year <= 2012) %>%
  select(fiscal_year, case_number, case_status, organization_flag, employer_name, employer_state, employer_city, employer_postal_code,
    nbr_workers_certified, period_begin_date, period_end_date, case_received_date, wage_rate, wage_unit, job_title) %>% collect() %>%
  transmute(file_fy = as.integer(fiscal_year), case_number, case_status, organization_flag, employer_name, employer_state, employer_city,
    zip_raw = employer_postal_code, cert = num(nbr_workers_certified), begin = as.Date(period_begin_date),
    end = as.Date(period_end_date), received = as.Date(case_received_date), wage = num(wage_rate), unit = wage_unit, job_title)
r <- bind_rows(o, n) %>% mutate(row_id = row_number())

# Latest file per case number
latest <- r %>% group_by(case_number) %>% summarise(keep_fy = max(file_fy), .groups = "drop")
cat("case numbers in more than one file:", sum(table(distinct(r, case_number, file_fy)$case_number) > 1), "\n")
r <- r %>% inner_join(latest, by = "case_number") %>% filter(file_fy == keep_fy) %>% select(-keep_fy)

# Masters
r <- r %>% group_by(file_fy, case_number) %>%
  mutate(n_rows = n(), c0 = coalesce(cert, 0), tot = sum(c0), others_pos = (tot - c0) > 0,
    master = n_rows > 1 & others_pos & ((!is.na(organization_flag) & organization_flag != "") | abs(c0 - (tot - c0)) < 0.5)) %>%
  ungroup()
cat("master rows dropped by file FY:\n"); print(table(r$file_fy[r$master]))
cat("FY2008+ agreement of flag and sum rules among multi-row cases:\n")
print(with(r %>% filter(file_fy >= 2008, n_rows > 1, others_pos), table(flag = !is.na(organization_flag) & organization_flag != "",
  sum_rule = abs(c0 - (tot - c0)) < 0.5)))

rows <- r %>% filter(!master, grepl("^Certified", case_status), cert > 0) %>%
  mutate(zip5 = gsub("[^0-9]", "", coalesce(zip_raw, "")), zip5 = ifelse(nchar(zip5) == 4, paste0("0", zip5), zip5),
    zip5 = ifelse(nchar(zip5) %in% c(5, 9), substr(zip5, 1, 5), NA_character_),
    st = toupper(trimws(employer_state)), st = ifelse(nchar(st) == 2, st, unname(st_abb[st])),
    city = toupper(trimws(gsub("\\s+", " ", employer_city)))) %>%
  left_join(z, by = "zip5") %>% mutate(geo_method = ifelse(!is.na(county_fips), "zcta", NA_character_)) %>%
  left_join(city_cty, by = c("st", "city")) %>%
  mutate(geo_method = ifelse(is.na(county_fips) & !is.na(county_fips_city), "city_place", geo_method),
    county_fips = coalesce(county_fips, county_fips_city)) %>% select(-county_fips_city) %>%
  left_join(abb_fips, by = "st")
nz <- is.na(rows$county_fips)
rows$county_fips[nz] <- nearest_zcta(rows$zip5[nz], rows$st_fips[nz])
rows$geo_method[nz & !is.na(rows$county_fips)] <- "nearest_zcta_in_state"
cat("geocoding method by file FY (rows):\n"); print(table(rows$file_fy, rows$geo_method, useNA = "ifany"))
rows <- rows %>%
  mutate(state_fips = state_from_county_fips(county_fips), unit_l = tolower(trimws(unit)),
    monthly = unit_l %in% c("mth", "monthly", "month"), hourly = unit_l %in% c("hr", "hourly", "hour"),
    days = as.numeric(end - begin), dates_ok = !is.na(begin) & !is.na(end) & days > 0,
    weeks = ifelse(dates_ok, days / 7, NA_real_), posweeks = cert * weeks, hours_c = posweeks * CONST_WEEKLY_HOURS,
    fy_receipt = fed_fy(received))
cat("\nrows kept by file FY; geocoded share; dates ok; monthly share:\n")
print(rows %>% group_by(file_fy) %>% summarise(rows = n(), geocoded = mean(!is.na(county_fips)),
  geocoded_cert = sum(cert[!is.na(county_fips)]) / sum(cert), dates_ok = mean(dates_ok), monthly_cert = sum(cert[monthly]) / sum(cert),
  hourly_cert = sum(cert[hourly]) / sum(cert), recv_na = mean(is.na(received)), cert = sum(cert), .groups = "drop"), width = 200)
cat("state mismatch (ZIP county state vs employer state), geocoded rows:", round(mean(rows$st_fips != rows$state_fips, na.rm = TRUE), 4), "\n")
write_parquet(rows %>% select(row_id, file_fy, case_number, employer_name, employer_state, zip5, geo_method, county_fips, state_fips,
  received, fy_receipt, begin, end, cert, weeks, posweeks, hours_c, wage, unit, hourly, monthly, dates_ok, job_title),
  file.path(out, "a2_rows.parquet"))

# ---- County x receipt-FY panel, FY2006-FY2011 -------------------------------
pr <- rows %>% filter(!is.na(county_fips), dates_ok, !monthly, fy_receipt %in% 2006:2011)
panel <- pr %>% group_by(county_fips, year = fy_receipt) %>%
  summarise(cert_hours_c = sum(hours_c), cert_pos = sum(cert), cert_posweeks = sum(posweeks), n_rows = n(),
    hourly_hours_c = sum(hours_c[hourly & !is.na(wage) & wage > 0 & wage <= 100]),
    wage_x_hours = sum((wage * hours_c)[hourly & !is.na(wage) & wage > 0 & wage <= 100]), .groups = "drop")
assert_geo_columns(panel, "county_fips")
stopifnot(!anyDuplicated(panel[c("county_fips", "year")]))
write_parquet(panel, file.path(out, "a2_county_rfy_panel.parquet"))
cat("\npanel rows by receipt FY (counties with any hours):\n"); print(table(panel$year))

# ---- Pre-period checks (FY2008 only) ----------------------------------------
cr <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("case_number", "fiscal_year", "county_fips", "fy_receipt", "cert_pos", "cert_posweeks"))
c8 <- cr %>% filter(fy_receipt == 2008)
cat("shared rows FY2008 receipts with missing posweeks:", sum(is.na(c8$cert_posweeks)), "\n"); c8$cert_posweeks <- coalesce(c8$cert_posweeks, 0)
e8 <- pr %>% filter(fy_receipt == 2008)
cat("\nFY2008 receipts: shared worksite rows cert_pos", round(sum(c8$cert_pos)), "posweeks", round(sum(c8$cert_posweeks)),
  "| employer-location rows cert_pos", round(sum(e8$cert)), "posweeks", round(sum(e8$posweeks)), "\n")
# Case-level geography agreement: employer county vs worksite county (FY2008 receipts)
ec <- e8 %>% group_by(case_number) %>% summarise(emp_cty = list(unique(county_fips)), w = sum(posweeks), .groups = "drop")
wc <- c8 %>% group_by(case_number) %>% summarise(ws_cty = list(unique(county_fips)), .groups = "drop")
g <- inner_join(ec, wc, by = "case_number") %>% rowwise() %>%
  mutate(same_county = any(emp_cty %in% ws_cty), same_state = any(substr(emp_cty, 1, 2) %in% substr(ws_cty, 1, 2))) %>% ungroup()
cat("FY2008 receipts, posweeks-weighted share of cases whose employer county is among worksite counties:",
  round(weighted.mean(g$same_county, g$w), 3), "| same state:", round(weighted.mean(g$same_state, g$w), 3), "| cases:", nrow(g), "\n")
saveRDS(g %>% select(case_number, w, same_county, same_state), file.path(out, "a2_fy2008_geo_agreement.rds"))
