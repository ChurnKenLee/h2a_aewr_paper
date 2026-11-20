# Identification search, strategy A3 (Phase 2 acquisition): verification of
# the acquired 2023-2025 OEWS occupation AEWRs.
#
# Usage (repository root, after 00_aewr_2023_oews_schedule.R):
#   Rscript --vanilla explorations/identification_search/A3/01_verify_schedule.R
#
# Checks:
#   1. OFLC's May 2024 workbook vs the BLS May 2024 state file (Wayback copy).
#   2. Recompute the OEWS-based field-and-livestock AEWRs for AK, DC, GU, PR,
#      VI from the May 2024 workbook (employment-weighted H_MEAN) against the
#      values transcribed from OFLC's archived table.
#   3. The FY2026 655.120(b)(1)(ii) skill-level rates in
#      data/intermediate/aewr_occupation_skill_rates.parquet (2025-10-02 vintage)
#      against the acquired May 2024 statewide means for overlapping state-SOCs.
#   4. Offered hourly wages in FY2022-FY2026 disclosures (non-field SOCs) against
#      the acquired rate in force at case receipt, the neighbouring vintages and
#      the FLS AEWR. Offered wages only; no quantity outcomes.
# Outputs:
#   explorations/identification_search/sources/aewr_2023_2025_verification.csv
#   explorations/identification_search/sources/aewr_2023_2025_offer_match_by_soc.csv

suppressMessages({
  library(data.table)
  library(arrow)
  library(readxl)
})
source("code/c00_shared/geography.R")

src <- "explorations/identification_search/sources"
raw <- "data/raw/aewr_2023_oews_occupation"
fls_socs <- c("45-2041", "45-2091", "45-2092", "45-2093", "45-2099", "53-7064")
out <- list()
add <- function(check, group, n, metric, value) {
  out[[length(out) + 1]] <<- data.table(check = check, group = as.character(group), n = as.integer(n),
                                        metric = metric, value = as.numeric(value))
}

rates <- fread(file.path(src, "aewr_2023_2025_occupation_rates.csv"),
               colClasses = list(character = c("state_fips", "soc_code")))
vint <- fread(file.path(src, "aewr_2023_2025_oews_vintages.csv"))
fg <- fread(file.path(src, "aewr_2023_2025_oews_field_group_rates.csv"), colClasses = list(character = "state_fips"))

# ---- 1. OFLC workbook vs BLS state file (May 2024) ---------------------------
tmp <- tempfile(); dir.create(tmp)
utils::unzip(file.path(raw, "bls/oesm24st_wayback_20250601024221.zip"), exdir = tmp)
bls <- as.data.table(read_excel(file.path(tmp, "oesm24st/state_M2024_dl.xlsx"), col_types = "text"))
wb24 <- as.data.table(read_excel(file.path(raw, "flag/BLS-OEWS_National_State_Data_May_2024_Release.xlsx"),
                                 sheet = "State Data", col_types = "text"))
m <- merge(bls[, .(AREA_TITLE, OCC_CODE, h_bls = H_MEAN)], wb24[, .(AREA_TITLE, OCC_CODE, h_wb = H_MEAN)],
           by = c("AREA_TITLE", "OCC_CODE"), all = TRUE)
add("1_oflc_vs_bls_may2024", "all state rows", nrow(m), "share_identical_h_mean", mean(m$h_bls == m$h_wb, na.rm = TRUE))
add("1_oflc_vs_bls_may2024", "all state rows", nrow(m), "rows_only_in_one_source", sum(is.na(m$h_bls) | is.na(m$h_wb)))
add("1_oflc_vs_bls_may2024", "Colorado", sum(bls$AREA_TITLE == "Colorado"), "bls_colorado_state_rows", sum(bls$AREA_TITLE == "Colorado"))

# ---- 2. OEWS field-group AEWR recomputation (May 2024) ------------------------
fgc <- wb24[OCC_CODE %in% fls_socs & AREA_TITLE %in% c("Alaska", "District of Columbia", "Guam", "Puerto Rico", "Virgin Islands")]
fgc[, `:=`(emp = suppressWarnings(as.numeric(TOT_EMP)), h = suppressWarnings(as.numeric(H_MEAN)))]
fgc <- fgc[!is.na(emp) & !is.na(h), .(recomputed = round(sum(emp * h) / sum(emp), 2), n_soc = .N), by = .(state_name = AREA_TITLE)]
fgc <- merge(fgc, fg[oews_reference == "May 2024", .(state_name, published = aewr_hourly)], by = "state_name", all = TRUE)
for (r in seq_len(nrow(fgc))) {
  add("2_field_group_recompute_may2024", fgc$state_name[r], fgc$n_soc[r], "recomputed_minus_published",
      fgc$recomputed[r] - fgc$published[r])
}

# ---- 3. FY2026 (b)(1)(ii) skill rates vs acquired May 2024 means --------------
sk <- as.data.table(read_parquet("data/intermediate/aewr_occupation_skill_rates.parquet"))
sk <- sk[trimws(source_sheet) == "Statewide 655.120(b)(1)(ii)" & as.character(effective_start) == "2025-10-02"]
sk <- dcast(sk, state_fips + soc_code ~ paste0(worker_category, "_", skill_level), value.var = "hourly_aewr")
a24 <- rates[oews_reference == "May 2024", .(state_fips, soc_code, rate_basis, aewr_hourly)]
cmp <- merge(sk, a24, by = c("state_fips", "soc_code"))
add("3_ifr2026_vs_may2024", "overlap state-SOCs", nrow(cmp), "n_overlap", nrow(cmp))
add("3_ifr2026_vs_may2024", "skill-II source SOCs not in acquired", nrow(sk) - nrow(cmp), "n_ifr_only", nrow(sk) - nrow(cmp))
for (b in c("statewide", "national_fallback")) {
  x <- cmp[rate_basis == b & !is.na(aewr_hourly)]
  add("3_ifr2026_vs_may2024", b, nrow(x), "share_US_skill2_equals_may2024_mean", mean(abs(x$US_2 - x$aewr_hourly) < 0.005))
  add("3_ifr2026_vs_may2024", b, nrow(x), "median_US_skill2_minus_mean", median(x$US_2 - x$aewr_hourly))
  add("3_ifr2026_vs_may2024", b, nrow(x), "median_US_skill1_over_mean", median(x$US_1 / x$aewr_hourly))
  add("3_ifr2026_vs_may2024", b, nrow(x), "share_US_skill1_below_mean", mean(x$US_1 < x$aewr_hourly))
}
fwrite(cmp[abs(US_2 - aewr_hourly) >= 0.005 & !is.na(aewr_hourly)][order(state_fips, soc_code)],
       file.path(tempdir(), "skill2_mismatch.csv"))
mm <- cmp[abs(US_2 - aewr_hourly) >= 0.005 & !is.na(aewr_hourly)]
cat("skill-II mismatches by state (top):\n"); print(head(mm[, .N, by = state_fips][order(-N)], 10))

# ---- 4. Offered wages in disclosures ----------------------------------------
d <- as.data.table(read_parquet("data/intermediate/h2a_disclosure_records.parquet",
                                col_select = c("case_number", "fiscal_year", "case_received_date", "case_status",
                                               "soc_code", "wage_rate", "wage_unit", "worksite_state")))
d <- d[fiscal_year >= 2022 & wage_unit == "Hour"]
d[, `:=`(soc6 = substr(soc_code, 1, 7), wage = suppressWarnings(as.numeric(wage_rate)),
         recv = as.IDate(case_received_date))]
d <- d[!is.na(wage) & !is.na(recv) & grepl("^\\d\\d-\\d{4}$", soc6) & !soc6 %in% fls_socs]
st <- unique(rates[, .(state_abbr, state_fips, kansas_order_state)])
d <- merge(d, st, by.x = "worksite_state", by.y = "state_abbr")

# Vintage in force at receipt (state-level Kansas dates; entity membership
# outside the 17 states is not observable).
vint[, k := .I]
vint[, `:=`(s0 = as.IDate(effective_start), e0 = as.IDate(effective_end_exclusive),
            s1 = as.IDate(kansas_group_effective_start), e1 = as.IDate(kansas_group_effective_end_exclusive))]
d[, k := NA_integer_]
for (i in seq_len(nrow(vint))) {
  d[!kansas_order_state & recv >= vint$s0[i] & recv < vint$e0[i], k := i]
  d[kansas_order_state & recv >= vint$s1[i] & recv < vint$e1[i], k := i]
}
d[, period := fcase(recv < as.IDate("2023-03-30"), "0_pre_rule",
                    !is.na(k), paste0(k, "_", vint$oews_reference[pmax(k, 1)]),
                    recv >= as.IDate("2025-08-29") & recv < as.IDate("2025-10-02"), "5_vacatur_2010_method",
                    recv >= as.IDate("2025-10-02"), "6_ifr_2025",
                    default = "other")]
rk <- rates[, .(state_fips, soc_code, oews_reference, aewr_hourly, rate_basis)]
rk[, k := match(oews_reference, vint$oews_reference)]
for (lag in c(0L, -1L, 1L)) {
  nm <- c("0" = "cur", "-1" = "prev", "1" = "next")[as.character(lag)]
  d[, kk := fifelse(period == "0_pre_rule", 1L, k + lag)]
  d[period %in% c("5_vacatur_2010_method", "6_ifr_2025"), kk := 4L + lag]
  d <- merge(d, rk[, .(state_fips, soc6 = soc_code, kk = k, r = aewr_hourly, b = rate_basis)],
             by = c("state_fips", "soc6", "kk"), all.x = TRUE)
  setnames(d, c("r", "b"), paste0(c("rate_", "basis_"), nm))
}
# FLS AEWR in force at receipt
fls <- as.data.table(read_parquet("data/intermediate/aewr_rate_schedule.parquet"))
fls <- fls[reference_year >= 2021, .(state_fips = state_fips(state_fips), aewr, s = as.IDate(effective_start),
                                     e = as.IDate(effective_end_exclusive),
                                     alt = as.IDate(alternative_effective_start))]
d[, fls_rate := NA_real_]
for (i in seq_len(nrow(fls))) {
  f <- fls[i]
  s_k <- if (is.na(f$alt)) f$s else f$alt
  d[state_fips == f$state_fips & !kansas_order_state & recv >= f$s & recv < f$e, fls_rate := f$aewr]
  d[state_fips == f$state_fips & kansas_order_state & recv >= s_k & recv < f$e, fls_rate := f$aewr]
}
# Kansas states keep the prior FLS rate until the alternative start.
fls_k <- fls[!is.na(alt)]
for (i in seq_len(nrow(fls_k))) {
  prev <- fls[state_fips == fls_k$state_fips[i] & e == fls_k$s[i]]
  if (nrow(prev)) d[state_fips == fls_k$state_fips[i] & kansas_order_state & recv >= fls_k$s[i] & recv < fls_k$alt[i],
                    fls_rate := prev$aewr[1]]
}
eq <- function(a, b) !is.na(b) & abs(a - b) < 0.005
d[, `:=`(at_cur = eq(wage, rate_cur), at_prev = eq(wage, rate_prev), at_next = eq(wage, rate_next),
         at_fls = eq(wage, fls_rate), ge_cur = !is.na(rate_cur) & wage >= rate_cur - 0.005)]
summ <- function(x, by) {
  x[, .(n = .N, share_at_cur = mean(at_cur), share_at_prev = mean(at_prev), share_at_next = mean(at_next),
        share_at_fls = mean(at_fls), share_ge_cur = mean(ge_cur),
        share_at_any_oews_or_fls = mean(at_cur | at_prev | at_next | at_fls),
        share_rate_missing = mean(is.na(rate_cur)), share_national_basis = mean(basis_cur %in% "national_fallback")),
    by = by]
}
sp <- summ(d, "period")[order(period)]
cat("\nOffered hourly wage vs acquired AEWR, non-field SOCs, by receipt period:\n"); print(sp)
for (r in seq_len(nrow(sp))) for (mcol in setdiff(names(sp), c("period", "n"))) {
  add("4_offers_vs_rates_by_period", sp$period[r], sp$n[r], mcol, sp[[mcol]][r])
}
sn <- summ(d[period %in% c("1_May 2021", "2_May 2022", "3_May 2023", "4_May 2024")], "period")
add("4_offers_vs_rates_by_period", "pooled_2023_rule_window", sum(sn$n), "share_at_cur",
    d[period %in% sn$period, mean(at_cur)])
add("4_offers_vs_rates_by_period", "pooled_2023_rule_window", sum(sn$n), "share_at_fls",
    d[period %in% sn$period, mean(at_fls)])
# national-fallback rows: matches confirm the fallback convention
nf <- d[period %in% sn$period & basis_cur == "national_fallback"]
add("4_offers_national_fallback", "pooled_2023_rule_window", nrow(nf), "share_at_cur", if (nrow(nf)) mean(nf$at_cur) else NA)
# Colorado, May 2024 vintage (statewide not published; national fallback)
co <- d[period == "4_May 2024" & state_fips == "08"]
add("4_offers_colorado_may2024", "CO", nrow(co), "share_at_cur_national", if (nrow(co)) mean(co$at_cur) else NA)
# Teche Vermilion: Louisiana 53-3032 after 2024-09-19
la <- d[state_fips == "22" & soc6 == "53-3032" & period != "0_pre_rule"]
la[, teche := fifelse(recv > as.IDate("2024-09-19"), "LA_53-3032_after_2024-09-19", "LA_53-3032_2023-03-30_to_2024-09-19")]
lt <- summ(la, "teche")
cat("\nLouisiana heavy-truck drivers around the Teche PI:\n"); print(lt)
for (r in seq_len(nrow(lt))) for (mcol in c("share_at_cur", "share_at_fls", "share_ge_cur")) {
  add("4_offers_teche_louisiana", lt$teche[r], lt$n[r], mcol, lt[[mcol]][r])
}
bysoc <- summ(d[period %in% sn$period], c("soc6"))[order(-n)]
fwrite(bysoc[n >= 20], file.path(src, "aewr_2023_2025_offer_match_by_soc.csv"))
cat("\nTop SOCs, 2023-rule window:\n"); print(head(bysoc, 15))

res <- rbindlist(out)
fwrite(res, file.path(src, "aewr_2023_2025_verification.csv"))
print(res[!grepl("^4_offers_vs_rates_by_period", check)])
