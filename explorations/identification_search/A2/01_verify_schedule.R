# Identification search, strategy A2 (Phase 2 acquisition): verification of
# the OES-based AEWR schedule built by 00_aewr_2009_oes_schedule.R.
#
# Usage (repository root; $S is the identification-search scratch root):
#   Rscript --vanilla explorations/identification_search/A2/01_verify_schedule.R "$S"
#
# Checks (no outcome analysis; offered wages only):
#   1. State level: FY2009 certified-weighted mean offered wage of hourly cases
#      received after 2009-01-19 (local case rows) and the schedule-implied
#      Level I floor max(OWL 45-2092 Level I, $7.25), against the 2010 rule's
#      Table 1 column 4 (data/intermediate/aewr_2009_nonstatutory_wages.parquet).
#   2. Case level: share of hourly cases whose offered wage equals an OWL rate
#      (any agricultural SOC, any level; 45-2092 Level I) for the worksite
#      county's OES area, under each wage year, by receipt month. This dates
#      the wage-year switch empirically.
# Outputs: $S/A2/verify_state_table1.csv, $S/A2/verify_case_match_by_month.csv,
#          explorations/identification_search/sources/aewr_2009_oes_verification.csv

suppressMessages({
  library(data.table)
  library(arrow)
})
args <- commandArgs(TRUE)
S <- if (length(args)) args[1] else stop("pass the scratch root")
out_scratch <- file.path(S, "A2")
dir.create(out_scratch, showWarnings = FALSE, recursive = TRUE)
src_dir <- "explorations/identification_search/sources"

sched <- fread(file.path(src_dir, "aewr_2009_oes_schedule.csv"),
               colClasses = c(state_fips = "character", oes_area_code = "character",
                              oes_area_code_owl = "character"))
xw <- fread(file.path(src_dir, "oes_area_county_crosswalk_2009.csv"),
            colClasses = c(county_fips = "character", oes_area_code = "character",
                           state_fips = "character"))
xw <- unique(xw[!is.na(county_fips) & county_fips != "", .(wage_year, county_fips, oes_area_code)])

rows <- as.data.table(read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet")))
rows <- rows[!is.na(received) & received >= as.IDate("2008-10-01") & received <= as.IDate("2010-09-30")]
cat("case rows in window:", nrow(rows), " cases:", uniqueN(rows$case_number), "\n")
print(rows[, .N, by = .(row_source, status)])
cat("hourly wage present share:", mean(!is.na(rows$hourly_wage)), "\n")

# Rates per (wage_year, area): all ag SOC x level values (AEWR incl. $7.25
# floor) and the 45-2092 Level I.
rates <- sched[!is.na(aewr_hourly), .(wage_year, oes_area_code, soc_code, skill_level, aewr_hourly)]
rates <- unique(rates)

cases <- rows[!is.na(hourly_wage) & !is.na(county_fips),
              .(case_number, county_fips, state_fips, received, hourly_wage, cert_pos, fiscal_year, status)]
match_one <- function(wy) {
  m <- merge(cases, xw[wage_year == wy], by = "county_fips", allow.cartesian = TRUE)
  m <- merge(m, rates[wage_year == wy], by = c("wage_year", "oes_area_code"), allow.cartesian = TRUE)
  m[, hit := abs(hourly_wage - aewr_hourly) < 0.005]
  m[, .(any_level = any(hit),
        l1_452092 = any(hit & soc_code == "45-2092" & skill_level == "I"),
        any_452092 = any(hit & soc_code == "45-2092"),
        l1_any_soc = any(hit & skill_level == "I")),
    by = case_number][, wage_year := wy]
}
mm <- rbind(match_one("2008-07-01/2009-06-30"), match_one("2009-07-01/2010-06-30"))
mw <- dcast(mm, case_number ~ wage_year, value.var = c("any_level", "l1_452092", "any_452092"))
setnames(mw, gsub("2008-07-01/2009-06-30", "owl0809", gsub("2009-07-01/2010-06-30", "owl0910", names(mw))))
cl <- unique(cases[, .(case_number, received, hourly_wage)], by = "case_number")
cl <- merge(cl, mw, by = "case_number", all.x = TRUE)
for (v in setdiff(names(mw), "case_number")) set(cl, which(is.na(cl[[v]])), v, FALSE)
cl[, at_725 := abs(hourly_wage - 7.25) < 0.005]
cl[, month := format(received, "%Y-%m")]
by_month <- cl[, .(cases = .N,
                   owl0809_any = mean(any_level_owl0809), owl0809_l1_452092 = mean(l1_452092_owl0809),
                   owl0910_any = mean(any_level_owl0910), owl0910_l1_452092 = mean(l1_452092_owl0910),
                   owl0809_l1_452092_above725 = mean(l1_452092_owl0809 & !at_725),
                   owl0910_l1_452092_above725 = mean(l1_452092_owl0910 & !at_725),
                   at_725 = mean(at_725)), by = month][order(month)]
print(by_month, digits = 3)
fwrite(by_month, file.path(out_scratch, "verify_case_match_by_month.csv"))

# Regime windows by receipt date.
cl[, regime := fcase(received < as.IDate("2009-01-17"), "a_pre_2009-01-17",
                     received < as.IDate("2009-07-01"), "b_2009-01-17_to_06-30",
                     received < as.IDate("2010-03-15"), "c_2009-07-01_to_2010-03-14",
                     default = "d_2010-03-15_on")]
by_regime <- cl[, .(cases = .N,
                    owl0809_any = mean(any_level_owl0809), owl0809_any_452092 = mean(any_452092_owl0809),
                    owl0809_l1_452092 = mean(l1_452092_owl0809),
                    owl0910_any = mean(any_level_owl0910), owl0910_any_452092 = mean(any_452092_owl0910),
                    owl0910_l1_452092 = mean(l1_452092_owl0910),
                    either_any = mean(any_level_owl0809 | any_level_owl0910),
                    owl0809_l1_452092_above725 = mean(l1_452092_owl0809 & !at_725),
                    owl0910_l1_452092_above725 = mean(l1_452092_owl0910 & !at_725),
                    owl0809_any_above725 = mean(any_level_owl0809 & !at_725),
                    owl0910_any_above725 = mean(any_level_owl0910 & !at_725),
                    at_725 = mean(at_725)), by = regime][order(regime)]
print(by_regime, digits = 3)

# ---- State level against Table 1 (column 4: post-Jan-19 FY2009 certified wage)
t1 <- as.data.table(read_parquet("data/intermediate/aewr_2009_nonstatutory_wages.parquet"))
fy9 <- rows[fiscal_year == 2009 & received > as.IDate("2009-01-19") & !is.na(hourly_wage) &
              status %in% c("certified", "partial") & cert_pos > 0]
fy9 <- unique(fy9, by = c("case_number", "county_fips"))
fy9[, wy := fifelse(received < as.IDate("2009-07-01"), "2008-07-01/2009-06-30", "2009-07-01/2010-06-30")]
l1 <- sched[soc_code == "45-2092" & skill_level == "I", .(wy = wage_year, oes_area_code, l1 = aewr_hourly)]
l1 <- merge(xw[, .(wy = wage_year, county_fips, oes_area_code)], unique(l1), by = c("wy", "oes_area_code"))
l1 <- l1[, .(l1 = mean(l1, na.rm = TRUE)), by = .(wy, county_fips)]
fy9 <- merge(fy9, l1, by = c("wy", "county_fips"), all.x = TRUE)
# Allocation rows split a case's positions across counties; weight each row
# by its certified positions (cert_pos is already allocated).
st <- fy9[, .(cases = uniqueN(case_number), cert_pos = sum(cert_pos),
              offered_mean = weighted.mean(hourly_wage, cert_pos),
              l1_floor_mean = weighted.mean(l1, cert_pos, na.rm = TRUE),
              l1_cover = sum(cert_pos[!is.na(l1)]) / sum(cert_pos)), by = state_fips]
st <- merge(st, t1[, .(state_fips, state_name, table1_post = aewr_nonstatutory_application_wage_2009)],
            by = "state_fips", all = TRUE)
st[, `:=`(gap_offered = offered_mean - table1_post, gap_l1 = l1_floor_mean - table1_post)]
print(st[order(-cert_pos)], digits = 3)
both <- st[!is.na(table1_post) & !is.na(l1_floor_mean)]
summ <- data.table(
  check = c("states_compared", "corr_offered_vs_table1", "wmean_abs_gap_offered_vs_table1",
            "corr_L1floor_vs_table1", "wmean_gap_L1floor_minus_table1",
            "share_states_L1floor_le_table1_plus_0.05"),
  value = c(nrow(both), both[, cor(offered_mean, table1_post)],
            both[, weighted.mean(abs(gap_offered), cert_pos)],
            both[, cor(l1_floor_mean, table1_post)],
            both[, weighted.mean(gap_l1, cert_pos)],
            both[, mean(l1_floor_mean <= table1_post + 0.05)]))
print(summ)
fwrite(st, file.path(out_scratch, "verify_state_table1.csv"))

ver <- rbind(
  summ[, .(level = "state", group = "FY2009 received after 2009-01-19", metric = check, value)],
  melt(by_regime[, cases := as.numeric(cases)], id.vars = "regime", variable.name = "metric")[
    , .(level = "case", group = regime, metric = as.character(metric), value)]
)
fwrite(ver, file.path(src_dir, "aewr_2009_oes_verification.csv"))
