# Identification search, strategy A2 (Phase 2 acquisition): the OES-based
# AEWR schedules in force under the December 2008 H-2A rule (73 FR 77110,
# effective 2009-01-17; displaced by the 2010 rule for applications filed on or
# after 2010-03-15).
#
# Usage (repository root):
#   Rscript --vanilla explorations/identification_search/A2/00_aewr_2009_oes_schedule.R
#
# Inputs (read only):
#   data/raw/aewr_2009_oes/OWL_2009_TEXT.zip  FLC Data Center OWL, wage year
#       7/2008-6/2009 (BLS OES May 2007), Wayback capture 2010-01-03
#   data/raw/aewr_2009_oes/OWL_2010_TEXT.zip  wage year 7/2009-6/2010
#       (BLS OES May 2008), Wayback capture 2010-01-03
#   data/intermediate/oews_area_definitions.parquet  county FIPS lookup only
# Outputs:
#   explorations/identification_search/sources/aewr_2009_oes_schedule.csv
#   explorations/identification_search/sources/oes_area_county_crosswalk_2009.csv
#
# 20 CFR 655.108(e) (2008): "The AEWR will be based on published wage data for
# the occupation, skill level, and geographical area from the [BLS] OES survey.
# The NPC will obtain wage information on the AEWR using the On-line Wage
# Library (OWL) ... This wage shall not be less than the July 24, 2009 Federal
# minimum wage of $7.25." The OWL "All Industries" table (ALC_WAGE) is the
# source; the ACWIA education table does not apply to H-2A.
# 655.108(a): the employer must offer the highest of the AEWR "in effect at the
# time recruitment for a position is begun", the prevailing wage, or the
# Federal or State minimum wage. The wage year therefore attaches to the date
# recruitment began (NPC wage determination), not strictly to the filing date.

suppressMessages({
  library(data.table)
  library(arrow)
})

raw_dir <- "data/raw/aewr_2009_oes"
out_dir <- "explorations/identification_search/sources"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
retrieved <- "2026-11-05"

wage_years <- data.table(
  owl_file = c("OWL_2009_TEXT.zip", "OWL_2010_TEXT.zip"),
  wayback_ts = c("20100103170946", "20100103171029"),
  wage_year = c("2008-07-01/2009-06-30", "2009-07-01/2010-06-30"),
  oes_reference = c("BLS OES May 2007", "BLS OES May 2008"),
  rule_2008_from = c("2009-01-17", "2009-07-01"),
  rule_2008_to = c("2009-06-30", "2010-03-14"),
  owl_posted = c("FLC wage data updated 2008-07-01 (file dated 2008-07-07; Job Zones updated 2008-10-13)",
                 "FLC wage data updated 2009-07-01 (file dated 2009-06-30)")
)
wage_years[, source_url := sprintf(
  "https://web.archive.org/web/%s/http://flcdatacenter.com/download/%s",
  wayback_ts, owl_file)]

# Agricultural and H-2A-relevant SOC codes (2000 SOC, as used by OWL).
keep_soc <- function(s) {
  grepl("^45-", s) | s %in% c("53-7064", "53-7062", "53-3032", "53-3033",
                              "11-9011", "11-9012", "49-3041", "37-3011",
                              "51-3022", "47-2061")
}

abb_fips <- data.table(
  state_abbr = c(state.abb, "DC", "PR", "GU", "VI"),
  state_name = toupper(c(state.name, "District of Columbia", "Puerto Rico", "Guam", "Virgin Islands"))
)
fips_ref <- as.data.table(read_parquet("data/intermediate/oews_area_definitions.parquet"))
st_map <- unique(fips_ref[!is.na(state_fips), .(state_name = toupper(oews_state_name), state_fips)])
abb_fips <- merge(abb_fips, st_map, by = "state_name", all.x = TRUE)
abb_fips[state_abbr == "GU" & is.na(state_fips), state_fips := "66"]
abb_fips[state_abbr == "VI" & is.na(state_fips), state_fips := "78"]
abb_fips[state_abbr == "PR" & is.na(state_fips), state_fips := "72"]

# OWL drops the leading zero of 7-digit nonmetropolitan (BOS) area codes
# (e.g. "100001" = Alabama BOS 0100001). MSA, division and NECTA codes are
# 5 digits. Keep the code as a string.
pad_area <- function(a) ifelse(nchar(a) == 6, paste0("0", a), a)

norm_name <- function(x) {
  x <- toupper(x)
  x <- sub(",\\s*[A-Z]{2}$", "", x)
  x <- gsub("\\(.*\\)", "", x)
  x <- trimws(x)
  x <- gsub("\\bST\\.? ", "SAINT ", x)
  x <- gsub("\\bSTE\\.? ", "SAINTE ", x)
  x <- gsub(" (COUNTY|PARISH|BOROUGH|CENSUS AREA|CITY AND BOROUGH|MUNICIPALITY|MUNICIPIO|TOWN|PLANTATION|GORE|GRANT|LOCATION|PURCHASE|TOWNSHIP)$", "", x)
  x <- gsub("[^A-Z0-9]", "", x)
  x
}

read_owl <- function(zipfile, member) {
  tmp <- tempfile()
  dir.create(tmp)
  utils::unzip(file.path(raw_dir, zipfile), files = member, exdir = tmp)
  fread(file.path(tmp, member), colClasses = "character", encoding = "Latin-1")
}

sched <- list()
xw <- list()
for (i in seq_len(nrow(wage_years))) {
  wy <- wage_years[i]
  w <- read_owl(wy$owl_file, "ALC_WAGE.txt")
  g <- read_owl(wy$owl_file, "Geography.txt")
  soc <- read_owl(wy$owl_file, "Soc.txt")[, .(SocCode, SocTitle)]

  # ---- crosswalk: OWL area x county/town -> county FIPS
  g[, `:=`(oes_area_code_owl = Area, oes_area_code = pad_area(Area))]
  g[, is_town := grepl("\\(", CountyTownName)]
  g[, county_part := trimws(sub("\\s*\\(.*$", "", CountyTownName))]
  g[, town_part := ifelse(is_town, trimws(sub("^.*\\((.*)\\).*$", "\\1", CountyTownName)), NA_character_)]
  g <- merge(g, abb_fips[, .(state_abbr, state_fips)], by.x = "StateAb", by.y = "state_abbr", all.x = TRUE)

  ref <- fips_ref[year %in% c(2008, 2009) & !is.na(county_fips),
                  .(state_fips, oews_township_code, nm = norm_name(oews_county_name), county_fips)]
  ref_cty <- unique(ref[oews_township_code %in% c("000", NA), .(state_fips, nm, county_fips)])
  ref_town <- unique(ref[!oews_township_code %in% c("000", NA),
                         .(state_fips, nm = sub("CITY$", "", nm), county_fips)])
  ref_cty <- ref_cty[, .(county_fips = county_fips[1], n_cand = uniqueN(county_fips)), by = .(state_fips, nm)]
  ref_town <- ref_town[, .(county_fips = county_fips[1], n_cand = uniqueN(county_fips)), by = .(state_fips, nm)]

  g[, nm := ifelse(is_town, sub("CITY$", "", norm_name(town_part)), norm_name(county_part))]
  g[, nm_cty := norm_name(county_part)]
  g <- merge(g, ref_town[, .(state_fips, nm, fips_town = county_fips, n_town = n_cand)],
             by = c("state_fips", "nm"), all.x = TRUE)
  g <- merge(g, ref_cty[, .(state_fips, nm_cty = nm, fips_cty = county_fips, n_cty = n_cand)],
             by = c("state_fips", "nm_cty"), all.x = TRUE)
  g[, `:=`(county_fips = NA_character_, match_method = "unmatched")]
  g[is_town & !is.na(fips_town) & n_town == 1, `:=`(county_fips = fips_town, match_method = "town_name")]
  g[is.na(county_fips) & !is.na(fips_cty) & n_cty == 1,
    `:=`(county_fips = fips_cty, match_method = ifelse(is_town, "county_of_town", "county_name"))]
  # Fallback 1 (New England towns): the county part of "COUNTY (TOWN)" takes
  # the county FIPS that the state's matched towns with the same county part
  # have (unique only).
  town_cty <- g[match_method == "town_name", .(f = uniqueN(county_fips), cf = county_fips[1]),
                by = .(state_fips, nm_cty)][f == 1]
  g <- merge(g, town_cty[, .(state_fips, nm_cty, cf)], by = c("state_fips", "nm_cty"), all.x = TRUE)
  g[is.na(county_fips) & is_town & !is.na(cf), `:=`(county_fips = cf, match_method = "county_of_town")]
  # Fallback 2: OWL truncates names at 22 characters; unique prefix match
  # within state against county names.
  for (r in which(is.na(g$county_fips) & !g$is_town)) {
    cand <- ref_cty[state_fips == g$state_fips[r] & startsWith(nm, g$nm_cty[r]) & nchar(g$nm_cty[r]) >= 6]
    if (uniqueN(cand$county_fips) == 1) {
      set(g, r, "county_fips", cand$county_fips[1])
      set(g, r, "match_method", "county_name_prefix")
    }
  }
  # Fallback 3: hand-assigned names (2010-vintage county FIPS, per
  # content/contracts/geographic-codes.md). Clifton Forge city (51560) merged
  # into Alleghany County (51005) in 2001. Yellowstone National Park (MT) and
  # the Oak Ridge Reservation (TN) are federal land and stay unmatched, as do
  # PR, GU and VI (not in the project county frame).
  manual <- data.table(
    StateAb = c("AK", "AK", "AR", "FL", "NY", "SD", "VA", rep("ME", 5)),
    CountyTownName = c("ALEUTIAN ISLANDS EAST", "ALEUTIAN ISLANDS WEST", "HOT SPRINGS",
                       "MIAMI-DADE-OLD-DEF", "ORANGE", "SHANNON", "CLIFTON FORGE CIT",
                       "WASHINGTON (EAST CENTRAL WASHINGTON - U TS)",
                       "WASHINGTON (EAST CENTRAL WASHINGTON)",
                       "WASHINGTON (INDIAN TWP RESERVATION)",
                       "WASHINGTON (NORTH WASHINGTON)",
                       "WASHINGTON (PASSAMOQUODDY PLEASANT POINT INDIAN)"),
    mfips = c("02013", "02016", "05059", "12086", "36071", "46113", "51005", rep("23029", 5))
  )
  g <- merge(g, manual, by = c("StateAb", "CountyTownName"), all.x = TRUE)
  g[is.na(county_fips) & !is.na(mfips), `:=`(county_fips = mfips, match_method = "manual")]
  xw[[i]] <- g[, .(wage_year = wy$wage_year, owl_file = wy$owl_file,
                   oes_area_code, oes_area_code_owl, oes_area_name = AreaName,
                   state_abbr = StateAb, state_fips, county_town_name = CountyTownName,
                   is_new_england_town = is_town, county_fips, match_method)]

  # ---- schedule
  w <- w[keep_soc(SocCode)]
  w <- merge(w, soc, by = "SocCode", all.x = TRUE)
  w <- melt(w, id.vars = c("Area", "SocCode", "SocTitle", "GeoLvl"),
            measure.vars = paste0("Level", 1:4), variable.name = "lvl", value.name = "wage_raw")
  w[, owl_hourly_wage := as.numeric(trimws(wage_raw))]
  w[, skill_level := c("I", "II", "III", "IV")[as.integer(sub("Level", "", lvl))]]
  areas <- unique(g[, .(Area = oes_area_code_owl, oes_area_code, oes_area_name = AreaName,
                        state_abbr = StateAb, state_fips)])
  w <- merge(w, areas, by = "Area", allow.cartesian = TRUE, all.x = TRUE)
  w[, aewr_hourly := pmax(owl_hourly_wage, 7.25)]
  sched[[i]] <- w[, .(
    wage_year = wy$wage_year, oes_reference = wy$oes_reference,
    rule_2008_filing_from = wy$rule_2008_from, rule_2008_filing_to = wy$rule_2008_to,
    state_abbr, state_fips, oes_area_code, oes_area_code_owl = Area, oes_area_name,
    soc_code = SocCode, soc_title = SocTitle, owl_geo_level = as.integer(GeoLvl),
    skill_level, owl_hourly_wage, aewr_hourly,
    floor_725_binds = !is.na(owl_hourly_wage) & owl_hourly_wage < 7.25,
    owl_posted = wy$owl_posted, source_url = wy$source_url,
    retrieval_date = retrieved,
    confidence = ifelse(is.na(owl_hourly_wage), "no_wage_published",
                        "high: official OWL file; wage-year-to-date mapping medium (see notes)")
  )]
}

sched <- rbindlist(sched)
xw <- rbindlist(xw)
setorder(sched, wage_year, state_abbr, oes_area_code, soc_code, skill_level)
setorder(xw, wage_year, state_abbr, oes_area_code, county_town_name)

stopifnot(sched[, .N, by = .(wage_year, state_abbr, oes_area_code, soc_code, skill_level)][, all(N == 1)])
fwrite(sched, file.path(out_dir, "aewr_2009_oes_schedule.csv"))
fwrite(xw, file.path(out_dir, "oes_area_county_crosswalk_2009.csv"))

cat("schedule rows:", nrow(sched), "\n")
print(sched[, .(rows = .N, areas = uniqueN(oes_area_code), socs = uniqueN(soc_code),
                states = uniqueN(state_abbr), missing_wage = sum(is.na(owl_hourly_wage)),
                floor_binds = sum(floor_725_binds)), by = wage_year])
print(sched[soc_code == "45-2092", .(rows = .N, geo1 = mean(owl_geo_level == 1),
                                     l1_mean = mean(owl_hourly_wage[skill_level == "I"], na.rm = TRUE),
                                     l1_sd = sd(owl_hourly_wage[skill_level == "I"], na.rm = TRUE)),
            by = wage_year])
cat("crosswalk rows:", nrow(xw), "\n")
print(xw[, .N, by = .(wage_year, match_method)])
print(xw[match_method == "unmatched"][, .(wage_year, state_abbr, oes_area_code, county_town_name)][1:40])
