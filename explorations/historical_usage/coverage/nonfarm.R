# Do national changes in three nonfarm industries predict local employment and H-2A filings?
# Uses frozen 2007 shares, outside-state changes, and only cached project inputs.
source("code/paths.R")
source(path_code("c00_shared", "geography.R"))
library(data.table)
library(fixest)
setDTthreads(2L)
setFixest_nthreads(2L)

industries <- c("1012", "1013", "1026") # Construction, manufacturing, leisure and hospitality.
q <- arrow::open_dataset(path_int("qcew.parquet"))
q <- dplyr::filter(q, own_code == "5", year >= 2005, year <= 2024,
                   industry_code %in% c("10", "1012", "1013", "1026"),
                   agglvl_code %in% c("13", "53", "71", "73"))
q <- as.data.table(dplyr::collect(dplyr::select(q, area_fips, agglvl_code, industry_code,
                                               year, disclosure_code, annual_avg_emplvl)))
panel <- as.data.table(arrow::read_parquet(path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "state_fips", "year", "nbr_workers_certified_start_year",
                 "nbr_workers_certified_fiscal_year", "nbr_applications_start_year")))
assert_geo_columns(panel, c("county_fips", "state_fips"))
setnames(panel, c("nbr_workers_certified_start_year", "nbr_workers_certified_fiscal_year",
                  "nbr_applications_start_year"), c("positions", "fiscal_positions", "filings"))
stopifnot(!anyDuplicated(panel, by = c("county_fips", "year")))
counties <- unique(panel[, .(county_fips, state_fips)])

local <- q[agglvl_code %chin% c("71", "73") & grepl("^[0-9]{5}$", area_fips)]
local[, county_fips := harmonize_county_fips_2010(area_fips)]
local <- merge(local, counties, by = "county_fips")
local[, geography_priority := as.integer(area_fips != county_fips)]
local <- local[local[, .I[geography_priority == min(geography_priority)],
                     by = .(county_fips, industry_code, year)]$V1]
local <- local[disclosure_code == "" & annual_avg_emplvl >= 0]
stopifnot(!anyDuplicated(local, by = c("county_fips", "industry_code", "year")))
shares <- dcast(local[year == 2007L], county_fips + state_fips ~ industry_code,
                value.var = "annual_avg_emplvl")
shares <- shares[complete.cases(shares) & get("10") > 0]
shares <- melt(shares, id.vars = c("county_fips", "state_fips", "10"), measure.vars = industries,
               variable.name = "industry_code", value.name = "employment", variable.factor = FALSE)
shares[, share := employment / get("10")]
stopifnot(shares[, all(sum(share) <= 1.00001), by = county_fips][, all(V1)])

national <- q[agglvl_code == "13" & area_fips == "US000" & disclosure_code == "",
              .(year, industry_code, us_employment = annual_avg_emplvl)]
state <- q[agglvl_code == "53" & grepl("^[0-9]{2}000$", area_fips) & disclosure_code == "",
           .(state_fips = state_fips(substr(area_fips, 1, 2)), year, industry_code,
             state_employment = annual_avg_emplvl)]
shocks <- merge(state, national, by = c("year", "industry_code"))
setorder(shocks, state_fips, industry_code, year)
shocks[, outside := us_employment - state_employment]
shocks[, growth := fifelse(year == shift(year) + 1L, outside / shift(outside) - 1, NA_real_),
       by = .(state_fips, industry_code)]
shocks <- shocks[is.finite(growth), .(year = year + 1L, state_fips, industry_code, growth)]
dose <- merge(shares, shocks, by = c("state_fips", "industry_code"), allow.cartesian = TRUE)
dose <- dose[, .(industries = .N, dose_pp = 100 * sum(share * growth)), by = .(county_fips, year)
             ][industries == 3L & year %between% c(2008L, 2025L)]
x <- merge(dose, panel, by = c("county_fips", "year"))
setorder(x, county_fips, year)
stopifnot(!anyDuplicated(x, by = c("county_fips", "year")), uniqueN(x$year) == 18L)

# Intermediate check: did employment actually move across these industries per baseline private job?
actual <- local[industry_code %chin% industries,
                .(industries = .N, jobs = sum(annual_avg_emplvl)), by = .(county_fips, year)][industries == 3L]
setorder(actual, county_fips, year)
actual[, change := fifelse(year == shift(year) + 1L, jobs - shift(jobs), NA_real_), by = county_fips]
actual <- merge(actual, unique(shares[, .(county_fips, baseline_employment = get("10"))]), by = "county_fips")
actual <- actual[is.finite(change), .(county_fips, year = year + 1L, local_pp = 100 * change / baseline_employment)]
x <- merge(x, actual, by = c("county_fips", "year"), all.x = TRUE)
before <- panel[, .(county_fips, year = year + 2L, prior_positions = positions)]
x <- merge(x, before, by = c("county_fips", "year"), all.x = TRUE)

print(x[, .(county_years = .N, counties = uniqueN(county_fips), jurisdictions = uniqueN(state_fips),
            years = uniqueN(year), filing_equivalents = sum(filings), position_starts = sum(positions),
            fiscal_positions = sum(fiscal_positions), share_filing = sum(filings)/sum(panel$filings),
            share_starts = sum(positions)/sum(panel$positions),
            share_fiscal_panel = sum(fiscal_positions)/sum(panel$fiscal_positions))])
audit <- as.data.table(arrow::read_parquet(path_int("h2a_case_allocation_audit.parquet"),
                        col_select = c("fiscal_year", "case_allocation_reconciled", "case_certified_target")))
national_fy <- audit[case_allocation_reconciled == TRUE & fiscal_year %between% c(2008L,2025L),
                     .(all_fy = sum(case_certified_target)), by = .(year = fiscal_year)]
by_year <- merge(x[, .(supported_fy = sum(fiscal_positions)), by = year], national_fy, by = "year")
by_year[, share := supported_fy / all_fy]
print(by_year[, .(national_fy_share = sum(supported_fy)/sum(all_fy),
                 minimum_annual_share = min(share), maximum_annual_share = max(share))])

within <- feols(dose_pp ~ 1 | county_fips + state_fips^year, x, fixef.rm = "none")
x[, dose_resid := resid(within)]
info <- x[, .(squares = sum(dose_resid^2)), by = year][, fraction := squares/sum(squares)]
print(info[order(-fraction)][1:4])
print(info[, .(effective_years = 1/sum(fraction^2))])
for (term in c("local_pp", "log1p(positions)", "log1p(filings)", "log1p(prior_positions)")) {
  model <- feols(as.formula(paste(term, "~ dose_pp | county_fips + state_fips^year")), x,
                 vcov = ~state_fips + year, notes = FALSE)
  cat(term, "N", nobs(model), "estimate, two-way cluster 95% interval\n")
  print(c(coef(model), confint(model, parm = "dose_pp")))
}
for (remove in list(2021:2022, c(2009:2011, 2021:2022))) {
  model <- feols(log1p(positions) ~ dose_pp | county_fips + state_fips^year, x[!year %in% remove],
                 vcov = ~state_fips + year, notes = FALSE)
  cat("Positions, excluding outcome years", remove, "\n")
  print(c(coef(model), confint(model, parm = "dose_pp")))
}
