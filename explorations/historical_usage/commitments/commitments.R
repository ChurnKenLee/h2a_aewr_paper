# Historical October commitments and deviations from trailing regional AEWR growth.
source("code/paths.R")
source(path_code("c00_shared", "geography.R"))
library(data.table)
library(fixest)
setDTthreads(2L)
setFixest_nthreads(2L)
out <- commandArgs(trailingOnly = TRUE)
stopifnot(length(out) == 1L)
dir.create(out, showWarnings = FALSE, recursive = TRUE)
emit <- function(x, filename) fwrite(x, file.path(out, paste0(filename, ".csv")))
read_cols <- function(path, cols) as.data.table(arrow::read_parquet(path, col_select = tidyselect::all_of(cols)))

# One unambiguous FY-valid decision date per case. Do not resolve conflicting dates.
d <- read_cols(path_int("h2a_disclosure_records.parquet"), c("fiscal_year", "case_number", "decision_date"))
d <- d[fiscal_year %between% c(2010, 2016)]
d[, decision := as.IDate(decision_date, format = "%Y-%m-%d")]
d[year(decision) + as.integer(month(decision) >= 10L) != fiscal_year, decision := as.IDate(NA)]
d <- d[, .(decision = if (uniqueN(decision, na.rm = TRUE) == 1L) min(decision, na.rm = TRUE) else as.IDate(NA)),
       by = .(fiscal_year, case_number)]
by_fy <- function(partition) {
  x <- read_cols(path_int("h2a_worksite_months", sprintf("fy_%d.parquet", partition)),
                 c("fiscal_year", "case_number", "county_fips", "month", "allocated_certification_position_days"))
  stopifnot(all(x$fiscal_year == partition))
  x <- x[month >= as.Date("2011-10-01") & month <= as.Date("2015-09-01") &
           !is.na(county_fips) & allocated_certification_position_days > 0]
  x <- merge(x, d, by = c("fiscal_year", "case_number"), all.x = TRUE)
  x[, service_fy := year(month) + as.integer(month(month) >= 10L)]
  x[, target_month := month(month) <= 9L]
  x[, early := !is.na(decision) & decision <= as.IDate(sprintf("%d-10-31", service_fy - 1L))]
  x[, .(pd = sum(allocated_certification_position_days),
        jan_sep = sum(allocated_certification_position_days[target_month]),
        jan_mar = sum(allocated_certification_position_days[month(month) %in% 1:3]),
        unknown = sum(allocated_certification_position_days[target_month & is.na(decision)]),
        early_pd = sum(allocated_certification_position_days[target_month & early])),
    by = .(county_fips, service_fy)]
}
base <- rbindlist(lapply(2010:2016, by_fy))
base <- base[, lapply(.SD, sum), by = .(county_fips, service_fy)]
stopifnot(identical(sort(unique(base$service_fy)), 2012:2015),
          all(base$early_pd <= base$jan_sep + 1e-6))
emit(base[, lapply(.SD, sum), by = service_fy, .SDcols = c("pd", "jan_sep", "unknown", "early_pd")],
     "baseline_timing")
exposure <- base[, .(early_pd = sum(early_pd), pd = sum(pd), js = sum(jan_sep), jan_mar = sum(jan_mar),
                     unknown = sum(unknown), active_years = .N), by = county_fips]
# Positive baseline capacity required. Unknown target-month decision dates are
# excluded from the primary sample instead of treated as uncommitted.
exposure <- exposure[pd > 0 & unknown == 0]
exposure[, B := early_pd/pd]
assert_geo_columns(exposure, "county_fips")

panel <- read_cols(path_processed("county_year_panel.parquet"),
                   c("county_fips", "year", "state_fips", "aewr_region_id", "aewr", "man_hours_certified_fiscal_year"))
assert_geo_columns(panel, c("county_fips", "state_fips", "aewr_region_id"))
stopifnot(!anyDuplicated(panel, by = c("county_fips", "year")))
rates <- unique(panel[year %between% c(2012L, 2024L), .(aewr_region_id, year, aewr)])
stopifnot(!anyDuplicated(rates, by = c("aewr_region_id", "year")), all(is.finite(rates$aewr) & rates$aewr > 0))
setorder(rates, aewr_region_id, year)
rates[, growth := log(aewr) - shift(log(aewr)), by = aewr_region_id]
rates[, moving_mean := (shift(growth) + shift(growth, 2L) + shift(growth, 3L))/3, by = aewr_region_id]
rates[, shock := growth - moving_mean]
rates <- rates[year %between% c(2017L, 2024L)]
notices <- fread(path_code("a01_sources", "aewr_rate_schedule_manifest.csv"))
rates <- merge(rates, notices[, .(year, publication_date, effective_start)], by = "year")
stopifnot(nrow(rates) == 17L * 8L, all(is.finite(rates$shock)),
          all(rates$publication_date > as.IDate(sprintf("%d-10-31", rates$year - 1L))))
emit(rates, "shock_panel")
basepanel <- merge(panel[, .(county_fips, year, state_fips, aewr_region_id,
                             hours = man_hours_certified_fiscal_year)], exposure, by = "county_fips")
setorder(basepanel, county_fips, year)
basepanel[, `:=`(later_hours = shift(hours, type = "lead"), prior_hours = shift(hours)), by = county_fips]
p <- merge(basepanel, rates[, .(aewr_region_id, year, shock)], by = c("aewr_region_id", "year"))
stopifnot(all(is.finite(p$later_hours)), all(is.finite(p$prior_hours)), all(p$later_hours >= 0),
          nrow(p) == uniqueN(p$county_fips) * 8L)
p[, dose := B * shock]
p[, within_js_dose := fifelse(js > 0, early_pd/js, NA_real_) * shock]
p[, q1_dose := jan_mar / pd * shock]
p[, js_dose := js / pd * shock]
# A common dose unit permits interpretation of every fit as an expected-hours contrast.
fit <- function(x, outcome, dose = "dose", national_slope = FALSE, seasonal = FALSE) {
  fe <- if (national_slope) "county_fips + state_fips^year + year[B]" else "county_fips + state_fips^year"
  rhs <- if (seasonal) paste(dose, "+ q1_dose + js_dose") else dose
  fepois(as.formula(sprintf("%s ~ %s | %s", outcome, rhs, fe)), data = x, notes = FALSE,
         vcov = ~aewr_region_id, glm.iter = 100L)
}
record <- function(model, x, spec, dose = "dose") {
  keep <- obs(model)
  g <- uniqueN(x$aewr_region_id[keep])
  beta <- unname(coef(model)[dose])
  serr <- unname(se(model)[dose])
  crit <- qt(0.975, g - 1)
  contrast <- 0.25 * 0.05
  stopifnot(isTRUE(model$convStatus), is.finite(beta), is.finite(serr), serr > 0)
  data.table(spec, counties = uniqueN(x$county_fips[keep]), regions = g, n = length(keep),
             beta, se = serr, lower = beta - crit * serr, upper = beta + crit * serr,
             contrast_percent = 100 * expm1(contrast * beta),
             lower_percent = 100 * expm1(contrast * (beta - crit * serr)),
             upper_percent = 100 * expm1(contrast * (beta + crit * serr)),
             approx_mde_positive_pct = 100 * expm1(contrast * serr * (crit + qnorm(0.8))))
}
fits <- list()
main <- fit(p, "later_hours")
fits[[1]] <- record(main, p, "FYt_plus_1")
fits[[2]] <- record(fit(p, "prior_hours"), p, "placebo_FYt_minus_1")
fits[[3]] <- record(fit(p, "hours"), p, "contemporaneous_FYt")
fits[[4]] <- record(fit(p, "later_hours", national_slope = TRUE), p, "FYt_plus_1_national_year_by_commitment")
fits[[5]] <- record(fit(p, "later_hours", "within_js_dose"), p, "FYt_plus_1_share_among_Jan_Sep",
                    "within_js_dose")
balanced <- p[active_years == 4L]
fits[[6]] <- record(fit(balanced, "later_hours"), balanced, "FYt_plus_1_all_four_baseline_years")
fits[[7]] <- record(fit(p, "later_hours", seasonal = TRUE), p, "FYt_plus_1_control_calendar_mix")
emit(rbindlist(fits), "fits")
sample <- p[obs(main)]
sample[, fit_weight := fitted(main)]
fw <- feols(dose ~ 1 | county_fips + state_fips^year, sample, weights = ~fit_weight, notes = FALSE)
sample[, info := fit_weight * resid(fw)^2]
support <- sample[, .(information = sum(info), n = .N), by = aewr_region_id]
support[, share := information / sum(information)]
deletions <- rbindlist(lapply(sort(unique(sample$aewr_region_id)), function(region) {
  subset <- p[aewr_region_id != region]
  r <- record(fit(subset, "later_hours"), subset, "delete_region")
  r[, omitted_region := region]
  r
}))
G <- nrow(deletions)
jackknife_se <- sqrt((G - 1L) / G * sum((deletions$beta - mean(deletions$beta))^2))
emit(data.table(jackknife_se, lower_percent = 100*expm1(0.0125*(coef(main)[1L] - qt(0.975,G-1L)*jackknife_se)),
                upper_percent = 100*expm1(0.0125*(coef(main)[1L] + qt(0.975,G-1L)*jackknife_se))), "jackknife")
emit(deletions, "region_deletions")
emit(support[order(-share)], "region_information")
counties <- unique(p[, .(county_fips, B, pd, js, active_years)])
summary <- counties[, .(counties = .N, zero_B = sum(B == 0), active_four = sum(active_years == 4L),
                        B_p10 = quantile(B, .1), B_p50 = median(B), B_p90 = quantile(B, .9),
                        B_mean = mean(B), B_weighted = weighted.mean(B, pd),
                        shock_min = min(rates$shock), shock_max = max(rates$shock), shock_sd = sd(rates$shock),
                        effective_regions = 1/sum(support$share^2), maximum_region_share = max(support$share))]
emit(summary, "support")
