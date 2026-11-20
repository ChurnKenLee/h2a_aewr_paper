# Identification search, strategy A1 (Phase 3 design): sourced table of the
# state minimum wage that is an H-2A wage floor under 20 CFR 655.120(a)(1)(v),
# FY2025-FY2026, by state and (NY, OR) minimum-wage area.
#
# Usage (repository root):
#   Rscript --vanilla explorations/identification_search/A1/00_minwage_table.R
# Output: explorations/identification_search/A1/sources/state_ag_minwage_fy2025_2026.csv
#
# The table is a snapshot table: one row per state x area x snapshot date with
# the general state rate in force on that date and the floor applicable to
# H-2A field workers. Agricultural coverage comes from
# code/a01_sources/state_ag_minwage_coverage.csv (read here, not modified).
# Rates come from the U.S. DOL WHD "State Minimum Wage Laws" page at three
# dates (two Wayback captures and the live page), state sources for FL, OR, NY,
# NJ and ME. Values are typed in below from those pages; every row carries its
# citation, URL, retrieval date and confidence.
#
# Snapshots:
#   2025-09-29  value in force on 2025-09-29 (DOL page updated 2025-07-31).
#               Every 2024-25 change was an increase, so this is an upper bound
#               for the whole FY2025 Q1-Q3 window (used only to show that the
#               minimum wage never exceeded the FY2025 AEWR).
#   2025-10-01  as 2025-09-29 except FL ($14.00 from 2025-09-30).
#   2026-01-01  in force 2026-01-01 through 2026-06-30 (DOL page updated
#               2026-01-01; the page updated 2026-07-01 shows changes only in
#               AK, DC and OR on 2026-07-01).
#   2026-07-01  DOL page updated 2026-07-01 (AK, DC, OR steps).
#   2026-09-30  FL $15.00 (scheduled; DOL page of 2025-07-31). All other
#               states carry 2026-07-01 values; MI's scheduled $15.00 on
#               2027-01-01 is recorded in notes only.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr)})

out_dir <- file.path("explorations", "identification_search", "A1", "sources")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
retrieved <- "2026-11-05"

src <- list(
  dol_20250731 = c("U.S. DOL WHD, State Minimum Wage Laws (page 'Updated July 31, 2025'), Wayback capture 2025-09-30",
    "https://web.archive.org/web/20250930023625/https://www.dol.gov/agencies/whd/minimum-wage/state"),
  dol_20260101 = c("U.S. DOL WHD, State Minimum Wage Laws (page 'Updated January 1, 2026'), Wayback capture 2026-03-15",
    "https://web.archive.org/web/20260315090750/https://www.dol.gov/agencies/whd/minimum-wage/state"),
  dol_20260701 = c("U.S. DOL WHD, State Minimum Wage Laws (page 'Last updated July 1, 2026'), live page",
    "https://www.dol.gov/agencies/whd/minimum-wage/state"),
  fl = c("DOL WHD page of 2025-07-31: 'The Florida minimum wage is scheduled to increase by $1.00 every September 30th until reaching $15.00 on September 30, 2026'; Fla. Const. art. X sec. 24(c) (Amendment 2, 2020); Lowndes, 'Florida's Minimum Wage Increases to $14.00/Hour at Month-End' (2025)",
    "https://web.archive.org/web/20250930023625/https://www.dol.gov/agencies/whd/minimum-wage/state ; https://www.lowndes-law.com/newsroom/insights/floridas-minimum-wage-increases-to-14-00-hour-at-month-end"),
  or = c("Oregon BOLI minimum wage schedule (July 1, 2024 / 2025 / 2026: standard, Portland Metro, nonurban counties); ORS 653.025 (Portland Metro = inside the urban growth boundary of a metropolitan service district); ORS 653.026 (list of nonurban counties)",
    "https://www.oregon.gov/boli/workers/pages/minimum-wage-schedule.aspx ; https://oregon.public.law/statutes/ors_653.025 ; https://oregon.public.law/statutes/ors_653.026"),
  ny = c("DOL WHD pages (regional rates: NYC, Nassau, Suffolk, Westchester vs remainder of state); NYSDOL Farm Industry Minimum Wage poster LS110 (2026 version: $17.00 NYC/Long Island/Westchester, $16.00 rest of state; via search snippet of the Chinese-language LS110 PDF)",
    "https://web.archive.org/web/20260315090750/https://www.dol.gov/agencies/whd/minimum-wage/state ; https://dol.ny.gov/farm-industry-minimum-wage-poster-ls110 ; https://forms.labor.ny.gov/WP/LS110zho.pdf"),
  nj = c("NJ P.L.2019, c.32 farm minimum-wage schedule ($13.40 in 2025, $14.20 in 2026) as recorded in code/a01_sources/state_ag_minwage_coverage.csv (NJDOL MW-220 (1/26) poster)",
    "https://www.nj.gov/labor/wageandhour/assets/PDFs/MW-220(1-26)WageAndHourAbstractPoster.pdf"),
  me = c("26 M.R.S. 651-655 (agricultural minimum wage from 2026-01-01; base $14.65 CPI-indexed, $15.10 for 2026) as recorded in code/a01_sources/state_ag_minwage_coverage.csv; DOL WHD general rate $15.10 (page updated 2026-01-01)",
    "https://legislature.maine.gov/statutes/26/title26sec652.html ; https://web.archive.org/web/20260315090750/https://www.dol.gov/agencies/whd/minimum-wage/state"),
  ma = c("M.G.L. c.151 sec. 2A agricultural minimum $8.00 as recorded in code/a01_sources/state_ag_minwage_coverage.csv",
    "https://malegislature.gov/Laws/GeneralLaws/PartI/TitleXXI/Chapter151/Section2A")
)

# General (non-farm basic) state rate by snapshot. NA = no state law or below
# federal (the federal $7.25 applies to FLSA-covered employers).
g <- tribble(
  ~state_abbrev, ~area_id, ~s20250929, ~s20251001, ~s20260101, ~s20260701, ~s20260930,
  "AL", "statewide", NA, NA, NA, NA, NA,
  "AK", "statewide", 13.00, 13.00, 13.00, 14.00, 14.00,
  "AZ", "statewide", 14.70, 14.70, 15.15, 15.15, 15.15,
  "AR", "statewide", 11.00, 11.00, 11.00, 11.00, 11.00,
  "CA", "statewide", 16.50, 16.50, 16.90, 16.90, 16.90,
  "CO", "statewide", 14.81, 14.81, 15.16, 15.16, 15.16,
  "CT", "statewide", 16.35, 16.35, 16.94, 16.94, 16.94,
  "DE", "statewide", 15.00, 15.00, 15.00, 15.00, 15.00,
  "DC", "statewide", 17.95, 17.95, 17.95, 18.40, 18.40,
  "FL", "statewide", 13.00, 14.00, 14.00, 14.00, 15.00,
  "GA", "statewide", 5.15, 5.15, 5.15, 5.15, 5.15,
  "HI", "statewide", 14.00, 14.00, 16.00, 16.00, 16.00,
  "ID", "statewide", 7.25, 7.25, 7.25, 7.25, 7.25,
  "IL", "statewide", 15.00, 15.00, 15.00, 15.00, 15.00,
  "IN", "statewide", 7.25, 7.25, 7.25, 7.25, 7.25,
  "IA", "statewide", 7.25, 7.25, 7.25, 7.25, 7.25,
  "KS", "statewide", 7.25, 7.25, 7.25, 7.25, 7.25,
  "KY", "statewide", 7.25, 7.25, 7.25, 7.25, 7.25,
  "LA", "statewide", NA, NA, NA, NA, NA,
  "ME", "statewide", 14.65, 14.65, 15.10, 15.10, 15.10,
  "MD", "statewide", 15.00, 15.00, 15.00, 15.00, 15.00,
  "MA", "statewide", 15.00, 15.00, 15.00, 15.00, 15.00,
  "MI", "statewide", 12.48, 12.48, 13.73, 13.73, 13.73,
  "MN", "statewide", 11.13, 11.13, 11.41, 11.41, 11.41,
  "MS", "statewide", NA, NA, NA, NA, NA,
  "MO", "statewide", 13.75, 13.75, 15.00, 15.00, 15.00,
  "MT", "statewide", 10.55, 10.55, 10.85, 10.85, 10.85,
  "NE", "statewide", 13.50, 13.50, 15.00, 15.00, 15.00,
  "NV", "statewide", 12.00, 12.00, 12.00, 12.00, 12.00,
  "NH", "statewide", 7.25, 7.25, 7.25, 7.25, 7.25,
  "NJ", "statewide", 15.49, 15.49, 15.92, 15.92, 15.92,
  "NM", "statewide", 12.00, 12.00, 12.00, 12.00, 12.00,
  "NY", "downstate", 16.50, 16.50, 17.00, 17.00, 17.00,
  "NY", "upstate", 15.50, 15.50, 16.00, 16.00, 16.00,
  "NC", "statewide", 7.25, 7.25, 7.25, 7.25, 7.25,
  "ND", "statewide", 7.25, 7.25, 7.25, 7.25, 7.25,
  "OH", "statewide", 10.70, 10.70, 11.00, 11.00, 11.00,
  "OK", "statewide", 7.25, 7.25, 7.25, 7.25, 7.25,
  "OR", "standard", 15.05, 15.05, 15.05, 15.55, 15.55,
  "OR", "portland_metro", 16.30, 16.30, 16.30, 16.80, 16.80,
  "OR", "nonurban", 14.05, 14.05, 14.05, 14.55, 14.55,
  "PA", "statewide", 7.25, 7.25, 7.25, 7.25, 7.25,
  "RI", "statewide", 15.00, 15.00, 16.00, 16.00, 16.00,
  "SC", "statewide", NA, NA, NA, NA, NA,
  "SD", "statewide", 11.50, 11.50, 11.85, 11.85, 11.85,
  "TN", "statewide", NA, NA, NA, NA, NA,
  "TX", "statewide", 7.25, 7.25, 7.25, 7.25, 7.25,
  "UT", "statewide", 7.25, 7.25, 7.25, 7.25, 7.25,
  "VT", "statewide", 14.01, 14.01, 14.42, 14.42, 14.42,
  "VA", "statewide", 12.41, 12.41, 12.77, 12.77, 12.77,
  "WA", "statewide", 16.66, 16.66, 17.13, 17.13, 17.13,
  "WV", "statewide", 8.75, 8.75, 8.75, 8.75, 8.75,
  "WI", "statewide", 7.25, 7.25, 7.25, 7.25, 7.25,
  "WY", "statewide", 5.15, 5.15, 5.15, 5.15, 5.15
)

# County rules for sub-state areas (county FIPS strings).
ny_downstate <- c("36005", "36047", "36061", "36081", "36085", "36059", "36103", "36119")
or_nonurban <- c("41001", "41011", "41013", "41015", "41019", "41021", "41023", "41025", "41031", "41035",
  "41037", "41045", "41049", "41055", "41059", "41061", "41063", "41069")
area_rule <- c(
  statewide = "all counties",
  downstate = paste("New York City, Nassau, Suffolk, Westchester:", paste(ny_downstate, collapse = " ")),
  upstate = "all other New York counties",
  standard = "all Oregon counties not listed as nonurban (ORS 653.026); includes Multnomah, Washington and Clackamas because farms lie outside the Metro urban growth boundary",
  portland_metro = "inside the Metro urban growth boundary only; no county is assigned (H-2A worksites are outside the UGB); recorded for completeness",
  nonurban = paste("ORS 653.026 nonurban counties:", paste(or_nonurban, collapse = " ")))

cov <- read_csv(file.path("code", "a01_sources", "state_ag_minwage_coverage.csv"), col_types = cols(.default = "c"))
cov_on <- function(st, date) {
  r <- cov %>% filter(state_abbrev == st, as.Date(effective_start) <= as.Date(date),
    is.na(effective_end) | effective_end == "" | as.Date(effective_end) >= as.Date(date))
  stopifnot(nrow(r) == 1)
  r
}

snap <- c(s20250929 = "2025-09-29", s20251001 = "2025-10-01", s20260101 = "2026-01-01", s20260701 = "2026-07-01", s20260930 = "2026-09-30")
valid_through <- c(s20250929 = "2025-09-29", s20251001 = "2025-12-31", s20260101 = "2026-06-30", s20260701 = "2026-09-29", s20260930 = "")
dol_src <- c(s20250929 = "dol_20250731", s20251001 = "dol_20250731", s20260101 = "dol_20260101", s20260701 = "dol_20260701", s20260930 = "dol_20260701")

rows <- list()
for (i in seq_len(nrow(g))) for (k in names(snap)) {
  st <- g$state_abbrev[i]; area <- g$area_id[i]; d <- snap[[k]]; gen <- g[[k]][i]
  cv <- cov_on(st, d)
  covers <- cv$state_min_covers_hired_farmworkers
  basis <- NA_character_; floor <- NA_real_; s <- src[[dol_src[[k]]]]; conf <- "high"; note <- ""
  if (st == "MA") { floor <- 8.00; basis <- "ag_specific_rate"; s <- src$ma; note <- "General minimum excludes farm work; separate $8.00 farm rate" }
  else if (st == "NJ") { floor <- if (d < "2026-01-01") 13.40 else 14.20; basis <- "ag_specific_rate"; s <- src$nj
    note <- "Separate farm schedule under P.L.2019, c.32 (general rate and seasonal/small-employer rate are higher)" }
  else if (st == "ME" && d >= "2026-01-01") { floor <- 15.10; basis <- "ag_specific_rate"; s <- src$me; conf <- "medium"
    note <- "Agricultural minimum wage from 2026-01-01 (26 M.R.S. 652); equals the general rate in 2026; dollar figure from secondary reports of Maine DOL" }
  else if (covers == "FALSE") { floor <- 7.25; basis <- "federal_minimum"
    note <- paste0("State law excludes hired farmworkers (", cv$statute_citation, "); FLSA $7.25 applies") }
  else if (is.na(gen) || gen < 7.25) { floor <- 7.25; basis <- "federal_minimum"; note <- "No state rate above the federal minimum" }
  else { floor <- gen; basis <- "state_minimum"
    if (covers == "partial") { note <- paste0("Coverage partial in the coverage table; H-2A-size crop employers treated as covered: ", substr(cv$exemption_text_summary, 1, 200)); conf <- "medium" } }
  if (st == "FL") { s <- src$fl; note <- paste(note, "FL steps on September 30: $13.00 (2024-09-30), $14.00 (2025-09-30), $15.00 (2026-09-30)") }
  if (st == "OR") { s <- src$or; note <- paste(note, "Oregon steps on July 1; three tiers") }
  if (st == "NY") { s <- src$ny; conf <- "medium"; note <- paste(note, "Farm workers covered at the general regional rate (Minimum Wage Order for Farm Workers); 2026 farm poster value from a search snippet") }
  if (st == "MI") note <- paste(note, "$12.48 from 2025-02-21; $13.73 from 2026-01-01; $15.00 scheduled 2027-01-01 (DOL)")
  if (st == "OH") note <- paste(note, "Large-employer rate (gross receipts >= $394,000 in 2025, $405,000 in 2026); H-2A employers assumed above the threshold")
  if (st == "MT") note <- paste(note, "Rate for businesses with sales over $110,000")
  if (cv$confidence %in% c("low", "medium") && basis == "state_minimum") conf <- "medium"
  rows[[length(rows) + 1]] <- tibble(state_abbrev = st, area_id = area, area_county_rule = area_rule[[area]],
    snapshot_date = d, valid_through = valid_through[[k]], general_min_wage = gen,
    h2a_state_floor = round(floor, 2), floor_basis = basis, ag_coverage_in_coverage_table = covers,
    source_citation = s[1], source_url = s[2], source_retrieved = retrieved, confidence = conf, notes = trimws(note))
}
tab <- bind_rows(rows)
fips <- cov %>% distinct(state_abbrev, state_fips)
tab <- tab %>% left_join(fips, by = "state_abbrev") %>% relocate(state_fips, .after = state_abbrev)
stopifnot(!anyNA(tab$state_fips), !anyDuplicated(tab[c("state_abbrev", "area_id", "snapshot_date")]))
write_csv(tab, file.path(out_dir, "state_ag_minwage_fy2025_2026.csv"), na = "")
write_csv(tibble(area_id = c(rep("downstate", length(ny_downstate)), rep("nonurban", length(or_nonurban))),
  county_fips = c(ny_downstate, or_nonurban)), file.path(out_dir, "minwage_substate_counties.csv"))
message("wrote ", nrow(tab), " rows")
print(tab %>% select(state_abbrev, area_id, snapshot_date, h2a_state_floor) %>%
  pivot_wider(names_from = snapshot_date, values_from = h2a_state_floor) %>% as.data.frame())
