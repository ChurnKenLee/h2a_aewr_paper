# Identification search, strategy A3 (Phase 2 acquisition): the occupation-
# specific (OEWS) AEWRs for non-field-and-livestock SOC codes under the 2023
# AEWR final rule (88 FR 12760, effective 2023-03-30; vacated 2025-08-25 in
# Teche Vermilion v. Chavez-DeRemer, implemented by OFLC from 2025-08-29;
# superseded by the 2025 IFR on 2025-10-02).
#
# Usage (repository root):
#   Rscript --vanilla explorations/identification_search/A3/00_aewr_2023_oews_schedule.R
#
# Inputs (read only):
#   data/raw/aewr_2023_oews_occupation/flag/BLS-OEWS_National_State_Data_May_202[1-4]*.xlsx
#       OFLC's own AEWR lookup workbooks (FLAG AEWR page and its archive):
#       statewide and national OEWS mean hourly wage (H_MEAN) for every SOC.
#   data/intermediate/aewr_occupation_skill_rates.parquet  state name -> FIPS only
# Outputs:
#   explorations/identification_search/sources/aewr_2023_2025_occupation_rates.csv
#   explorations/identification_search/sources/aewr_2023_2025_oews_vintages.csv
#   explorations/identification_search/sources/aewr_2023_2025_oews_field_group_rates.csv
#   explorations/identification_search/sources/aewr_2023_2025_regime_periods.csv
#
# Rule (20 CFR 655.120(b)(1)(ii) as amended in 2023; OFLC FAQ 2023-03-09):
# the AEWR for an SOC outside the six field-and-livestock (FLS) SOCs is the
# statewide annual average hourly gross wage for the SOC reported by OEWS; if
# OEWS reports no statewide wage, the national one. A job spanning several SOCs
# takes the highest applicable AEWR (655.120(b)(5)). The FLAG workbooks hold
# exactly that statewide/national H_MEAN; OFLC's page says the H_MEAN column
# "is the statewide AEWR for the SOC code in your state".

suppressMessages({
  library(data.table)
  library(arrow)
  library(readxl)
})
source("code/c00_shared/geography.R")

raw_dir <- "data/raw/aewr_2023_oews_occupation/flag"
out_dir <- "explorations/identification_search/sources"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
retrieved <- "2026-11-05"

fls_socs <- c("45-2041", "45-2091", "45-2092", "45-2093", "45-2099", "53-7064")
kansas_states <- c("AR", "FL", "GA", "ID", "IN", "IA", "KS", "LA", "MO", "MT",
                   "NE", "ND", "OK", "SC", "TN", "TX", "VA")

# ---- OEWS vintages and the dates OFLC applied them --------------------------
# Default group dates; the Kansas group (17 states plus Miles Berry Farm and
# GFVGA members as of 2024-08-26) differs only for the 2025 update
# (FR 2025-12910: effective 2025-07-11, Kansas group 2025-07-25).
flag <- "https://flag.dol.gov/sites/default/files/"
vint <- data.table(
  oews_reference = c("May 2021", "May 2022", "May 2023", "May 2024"),
  file = c("BLS-OEWS_National_State_Data_May_2021_Release_Effective_3.30.23_to_6.30.23.xlsx",
           "BLS-OEWS_National_State_Data_May_2022_Release.xlsx",
           "BLS-OEWS_National_State_Data_May_2023_Release.xlsx",
           "BLS-OEWS_National_State_Data_May_2024_Release.xlsx"),
  source_url = paste0(flag, c(
    "2023-06/BLS-OEWS_National_State_Data_May%202021%20Release%20Effective%203.30.23%20to%206.30.23.xlsx",
    "2023-06/BLS-OEWS_National_State_Data_May%202022_Release.xlsx",
    "2024-07/BLS-OEWS_National_State_Data_May%202023_Release.xlsx",
    "2025-07/BLS-OEWS_National_State_Data_May%202024_Release.xlsx")),
  effective_start = c("2023-03-30", "2023-07-01", "2024-07-08", "2025-07-11"),
  effective_end_exclusive = c("2023-07-01", "2024-07-08", "2025-07-11", "2025-08-29"),
  kansas_group_effective_start = c("2023-03-30", "2023-07-01", "2024-07-08", "2025-07-25"),
  kansas_group_effective_end_exclusive = c("2023-07-01", "2024-07-08", "2025-07-25", "2025-08-29"),
  authority = c(
    "88 FR 12760 (FR Doc 2023-03756), final rule effective 2023-03-30; FLAG label 'Effective March 30 to June 30, 2023'",
    "FR Doc 2023-12896 (2023-06-16): rates effective 2023-07-01; FLAG label 'Effective July 1, 2023 to July 7, 2024'",
    "FR Doc 2024-13809 (2024-06-24): rates applicable 2024-07-08; FLAG label 'Effective July 8, 2024 to July 10, 2025'",
    "FR Doc 2025-12910 (2025-07-11): effective 2025-07-11, Kansas group 2025-07-25; end = OFLC vacatur implementation (FLAG archive 'Effective July 11, 2025 through August 28, 2025')"),
  link_source = c("FLAG AEWR page archive (live 2026-11-05)", "FLAG AEWR page archive (live 2026-11-05)",
                  "FLAG AEWR page archive (live 2026-11-05)",
                  "FLAG archive PDF 'AEWR Page Archive 8.28.25' (link not on live page)")
)

# ---- state lookup (FIPS from the owning producer's artifact) -----------------
st <- unique(as.data.table(read_parquet("data/intermediate/aewr_occupation_skill_rates.parquet",
                                        col_select = c("state_name", "state_fips"))))
st[, state_fips := state_fips(state_fips)]
abb <- data.table(state_name = c(state.name, "District of Columbia", "Puerto Rico", "Guam", "Virgin Islands"),
                  state_abbr = c(state.abb, "DC", "PR", "GU", "VI"))
st <- merge(st, abb, by = "state_name", all.x = TRUE)
stopifnot(nrow(st) == 54, !anyNA(st$state_abbr), !anyNA(st$state_fips))

num <- function(x) suppressWarnings(as.numeric(x))

rates <- list()
for (i in seq_len(nrow(vint))) {
  v <- vint[i]
  f <- file.path(raw_dir, v$file)
  sd <- as.data.table(read_excel(f, sheet = "State Data", col_types = "text"))
  nd <- as.data.table(read_excel(f, sheet = "National Data", col_types = "text"))
  sd <- sd[, .(state_name = AREA_TITLE, soc_code = OCC_CODE, soc_title = OCC_TITLE, state_h_mean_text = H_MEAN)]
  # The national sheet lists the seven OEWS-only aggregate codes (e.g. 47-4090)
  # twice, as broad and as detailed rows with identical values.
  nd <- unique(nd[, .(soc_code = OCC_CODE, national_title = OCC_TITLE, national_h_mean_text = H_MEAN)])
  stopifnot(!anyDuplicated(sd, by = c("state_name", "soc_code")), !anyDuplicated(nd, by = "soc_code"))
  bad_area <- setdiff(unique(sd$state_name), st$state_name)
  if (length(bad_area)) stop("unknown OEWS area: ", paste(bad_area, collapse = ", "))

  # SOC universe: every code published at state level (detailed codes plus the
  # seven OEWS-only aggregate codes such as 47-4090) and every detailed
  # (non-zero last digit) national code; drop major groups and the total.
  is_major <- function(s) grepl("-0000$", s)
  socs <- union(sd[!is_major(soc_code), soc_code], nd[grepl("[1-9]$", soc_code), soc_code])
  socs <- setdiff(socs, fls_socs)

  g <- CJ(state_name = st$state_name, soc_code = socs)
  g <- merge(g, sd, by = c("state_name", "soc_code"), all.x = TRUE)
  g <- merge(g, nd, by = "soc_code", all.x = TRUE)
  g[is.na(soc_title), soc_title := national_title]
  g[, national_title := NULL]
  # readxl returns numeric cells as full-precision text; OEWS publishes cents.
  g[, `:=`(state_num = round(num(state_h_mean_text), 2), nat_num = round(num(national_h_mean_text), 2))]
  g[, rate_basis := fcase(
    !is.na(state_num), "statewide",
    state_h_mean_text %in% "#", "statewide_topcoded_unavailable",
    !is.na(nat_num), "national_fallback",
    default = "unavailable")]
  g[, aewr_hourly := fcase(rate_basis == "statewide", state_num,
                           rate_basis == "national_fallback", nat_num,
                           default = NA_real_)]
  g[, state_row_published := !is.na(state_h_mean_text)]
  g[, `:=`(state_h_mean = state_num, state_h_mean_code = fifelse(is.na(state_num), state_h_mean_text, NA_character_),
           national_h_mean = nat_num, national_h_mean_code = fifelse(is.na(nat_num), national_h_mean_text, NA_character_))]
  g[, `:=`(state_num = NULL, nat_num = NULL, state_h_mean_text = NULL, national_h_mean_text = NULL)]
  g <- merge(g, st, by = "state_name")
  g[, `:=`(
    oews_reference = v$oews_reference,
    effective_start = v$effective_start,
    effective_end_exclusive = v$effective_end_exclusive,
    kansas_group_effective_start = v$kansas_group_effective_start,
    kansas_group_effective_end_exclusive = v$kansas_group_effective_end_exclusive,
    kansas_order_state = state_abbr %in% kansas_states,
    source_url = v$source_url,
    retrieval_date = retrieved,
    # Long-form authority, cohort rule and confidence wording: see
    # aewr_2023_2025_oews_vintages.csv and the notes file.
    confidence = fcase(
      rate_basis == "statewide", "high",
      rate_basis == "national_fallback" & state_row_published, "high",
      rate_basis == "national_fallback", "medium_high",
      default = "not_determinable")
  )]
  # Colorado has no statewide rows in the May 2024 workbook (BLS did not
  # publish Colorado statewide May 2024 estimates), so the rule gives the
  # national mean; but OFLC's FY2026 IFR file carries Colorado-specific
  # values, so what OFLC applied in Colorado on 2025-07-11..2025-08-28 is
  # uncertain.
  if (v$oews_reference == "May 2024") g[state_abbr == "CO", confidence := "medium"]
  rates[[i]] <- g
}
rates <- rbindlist(rates)
setcolorder(rates, c("state_fips", "state_abbr", "state_name", "soc_code", "soc_title",
                     "oews_reference", "aewr_hourly", "rate_basis", "state_h_mean", "state_h_mean_code",
                     "national_h_mean", "national_h_mean_code", "state_row_published", "effective_start",
                     "effective_end_exclusive", "kansas_order_state",
                     "kansas_group_effective_start", "kansas_group_effective_end_exclusive"))
setorder(rates, effective_start, state_fips, soc_code)
stopifnot(!anyDuplicated(rates, by = c("state_fips", "soc_code", "oews_reference")),
          all(nchar(rates$state_fips) == 2), !any(rates$soc_code %in% fls_socs))
fwrite(rates, file.path(out_dir, "aewr_2023_2025_occupation_rates.csv"))
fwrite(vint[, .(oews_reference, effective_start, effective_end_exclusive, kansas_group_effective_start,
                kansas_group_effective_end_exclusive, authority, link_source, source_url,
                source_file = file.path("data/raw/aewr_2023_oews_occupation/flag", file),
                applies_to = "SOCs outside the six FLS SOCs, job orders (ETA-790/790A) submitted on or after 2023-03-30; highest applicable AEWR across all SOCs of the job (20 CFR 655.120(b)(5)); a higher updated AEWR binds mid-contract (655.120(b)(3))",
                retrieval_date = retrieved)],
       file.path(out_dir, "aewr_2023_2025_oews_vintages.csv"))

# ---- OEWS-computed field-and-livestock AEWRs (AK, DC, GU, PR, VI) -----------
# Transcribed from OFLC's archived AEWR tables (FLAG PDFs; Wayback captures
# agree). The other 49 states use the FLS rate in
# data/intermediate/aewr_rate_schedule.parquet (fls_common_soc_reference).
fg_src <- c(
  "May 2021" = paste0(flag, "2023-12/Historical%20AEWR%20Effective%20March%2030%20to%20June%2030%202023.pdf"),
  "May 2022" = paste0(flag, "2023-12/Historical%20AEWR%20Effective%20July%201%20to%20December%2031%202023.pdf"),
  "May 2023" = paste0(flag, "2024-12/Range%20and%20Non-Range%20Occupations%20for%20Field%20Workers%20and%20Livestock%20Workers%20combined%20Effective%20January%201%20to%20December%2015%2C%202024.pdf"),
  "May 2024" = paste0(flag, "2025-08/AEWR%20Page%20Archive%208.28.25.pdf"))
fg <- data.table(
  state_abbr = rep(c("AK", "DC", "GU", "PR", "VI"), times = 4),
  oews_reference = rep(names(fg_src), each = 5),
  aewr_hourly = c(17.21, 20.33, 10.40, 9.17, 13.24,
                  18.02, 20.72, 10.12, 9.70, 13.95,
                  19.52, 22.23, 10.34, 9.94, 13.96,
                  20.04, 20.79, 10.67, 10.36, 13.69),
  table_label = rep(c("OEWS, last updated July 2022", "OEWS, July 1, 2023",
                      "OEWS, July 8, 2024", "OEWS **, July 11th, 2025"), each = 5)
)
fg <- merge(fg, vint[, .(oews_reference, effective_start, effective_end_exclusive)], by = "oews_reference")
fg <- merge(fg, st[, .(state_abbr, state_fips, state_name)], by = "state_abbr")
fg[, `:=`(soc_codes = paste(fls_socs, collapse = ";"),
          method = "employment-weighted mean of statewide OEWS H_MEAN over the reported FLS SOCs (BLS method; 655.120(b)(1)(i)(B))",
          source_url = fg_src[oews_reference], retrieval_date = retrieved,
          confidence = "high: transcribed from OFLC archived AEWR table")]
setcolorder(fg, c("state_fips", "state_abbr", "state_name", "soc_codes", "oews_reference", "aewr_hourly",
                  "effective_start", "effective_end_exclusive"))
setorder(fg, effective_start, state_fips)
fwrite(fg, file.path(out_dir, "aewr_2023_2025_oews_field_group_rates.csv"))

# ---- regime periods for non-field SOCs ---------------------------------------
per <- data.table(
  period_id = c("pre_2023_rule", "oews_may2021", "oews_may2022", "oews_may2023", "oews_may2024",
                "vacatur_2010_method", "ifr_2025_skill"),
  start = c(NA, "2023-03-30", "2023-07-01", "2024-07-08", "2025-07-11", "2025-08-29", "2025-10-02"),
  end_exclusive = c("2023-03-30", "2023-07-01", "2024-07-08", "2025-07-11", "2025-08-29", "2025-10-02", "2026-08-03"),
  kansas_group_start = c(NA, "2023-03-30", "2023-07-01", "2024-07-08", "2025-07-25", "2025-08-29", "2025-10-02"),
  kansas_group_end_exclusive = c("2023-03-30", "2023-07-01", "2024-07-08", "2025-07-25", "2025-08-29", "2025-10-02", "2026-08-03"),
  nonfield_aewr = c("state/region FLS AEWR (2010 method; single rate for all non-range jobs)",
                    "statewide OEWS H_MEAN by SOC, May 2021 (national if no statewide)",
                    "statewide OEWS H_MEAN by SOC, May 2022", "statewide OEWS H_MEAN by SOC, May 2023",
                    "statewide OEWS H_MEAN by SOC, May 2024",
                    "single statewide FLS AEWR for all non-range jobs (2010 method); AK, DC, GU, PR, VI listed N/A",
                    "OEWS skill-level AEWRs (655.120(b)(1)(ii) under 2025 IFR); see aewr_occupation_skill_rates.parquet"),
  cohort_rule = c("job order submitted before 2023-03-30 (and certified cases) keep 2010 method until next AEWR adjustment for the SOC (FAQ Q3)",
                  "job order (ETA-790/790A) submitted on/after 2023-03-30",
                  "updated AEWR applies to work performed from effective date; higher rate binds mid-contract (655.120(b)(3))",
                  "as above", "as above; Kansas-order states/entities from 2025-07-25",
                  "OFLC announced 2025-08-28 that AEWRs follow the 2010 rule; FLAG archive dates the new table from 2025-08-29; no FR notice for this switch",
                  "IFR, FR Doc 2025-19365 (published 2025-10-02)"),
  entity_exceptions = c("",
                        "", "",
                        "Teche Vermilion PI (W.D. La.): from job orders filed after 2024-09-19 (and pending then), DOL may not enforce the 2023 rule for H-2A workers in Louisiana sugar cane farming/processing hired by named plaintiffs and members (as of 2024-11-07) of Teche Vermilion SCGA, Cora Texas G&H, American Sugar Cane League (amended PI 2024-12-05, Exhibit 1 member list)",
                        "Teche Vermilion PI as above",
                        "rule vacated 2025-08-25 (judgment filed 2025-08-26); PI made permanent", ""),
  source = c("FLAG Wayback 20230322091505; FAQ 2023-03-09",
             "FLAG archive; FR 2023-03756", "FR 2023-12896; FLAG archive", "FR 2024-13809; FLAG archive",
             "FR 2025-12910; FLAG 'AEWR Page Archive 8.28.25'",
             "OFLC announcement 2025-08-28 (MN DEED copy); FLAG Wayback 20250830184252; FLAG archive label; IFR 2025-19365 preamble",
             "FR 2025-19365")
)
fwrite(per, file.path(out_dir, "aewr_2023_2025_regime_periods.csv"))

cat("rows:", nrow(rates), "\n")
print(rates[, .N, by = .(oews_reference, rate_basis)][order(oews_reference, rate_basis)])
print(rates[, .(n_soc = uniqueN(soc_code), n_state = uniqueN(state_fips)), by = oews_reference])
