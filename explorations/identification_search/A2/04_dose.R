# Identification search, strategy A2 (Phase 3 design, step 1b): the effective-
# floor dose by county x receipt FY, FY2006-FY2011.
#
# Usage (repository root, after 03_rows_panel.R):
#   Rscript --vanilla explorations/identification_search/A2/04_dose.R "$S"
# Writes $S/A2/a2_county_dose.parquet (county x FY dose and components),
# $S/A2/a2_receipt_weights.csv (omega), $S/A2/a2_area_floors.csv.
#
# Floor in force on receipt date d for a county c in state s:
#   FLS periods (to 2009-01-16; from 2010-03-15):
#     max(FLS AEWR_s in force on d, state agricultural minimum wage_s(year of d))
#   OES periods (2009-01-17 to 2009-06-30: OWL 7/2008-6/2009 = OES May 2007;
#   2009-07-01 to 2010-03-14: OWL 7/2009-6/2010 = OES May 2008):
#     max(OWL 45-2092 Level I of c's OES area, 7.25, state ag minimum wage_s(year))
# The FY dose is x_ct = sum_d omega_d x floor_c(d), with omega the national
# FY2006-FY2008 share of certified hours (employer-location rows, hourly and
# other non-monthly pay) by receipt month, spread uniformly over the days of
# the month. omega is predetermined, so the realized 2009-10 filing spikes
# do not enter the dose.
# Blank 45-2092 Level I cells: primary = dose missing (county excluded);
# sensitivity = fill chain (same area, other wage year 45-2092 L1; then same
# area-year 45-2099 L1; then 45-2091 L1).
# Counties spanning several OES areas (New England): assigned to the area with
# the most towns in the OWL geography file (ties: lowest code).
#
# Integrity: uses pre-period (FY2006-08) national receipt-month hours only.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(readr); library(tidyr)})
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "A2")
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
FYS <- 2006:2011
WY09 <- "2008-07-01/2009-06-30"; WY10 <- "2009-07-01/2010-06-30"
D_OES_START <- as.Date("2009-01-17"); D_WY10 <- as.Date("2009-07-01"); D_FLS_BACK <- as.Date("2010-03-15")
AEWR_FLOOR_2008_RULE <- 7.25
num <- function(x) suppressWarnings(as.numeric(x))

# ---- OWL schedule: Level I cells -------------------------------------------
sch <- read_csv("explorations/identification_search/sources/aewr_2009_oes_schedule.csv", col_types = cols(.default = "c")) %>%
  filter(skill_level == "I", soc_code %in% c("45-2092", "45-2099", "45-2091")) %>%
  transmute(wage_year, state_fips = state_fips(state_fips), oes_area_code, soc_code, w = num(owl_hourly_wage)) %>%
  pivot_wider(names_from = soc_code, values_from = w)
sch <- sch %>% group_by(state_fips, oes_area_code) %>%
  mutate(l1_other_wy = ifelse(wage_year == WY09, `45-2092`[wage_year == WY10][1], `45-2092`[wage_year == WY09][1])) %>% ungroup() %>%
  mutate(l1 = `45-2092`, blank = is.na(l1),
    l1_fill = coalesce(l1, l1_other_wy, `45-2099`, `45-2091`),
    fill_source = case_when(!blank ~ "published", !is.na(l1_other_wy) ~ "45-2092 other wage year",
      !is.na(`45-2099`) ~ "45-2099 L1", !is.na(`45-2091`) ~ "45-2091 L1", TRUE ~ "none"))
write_csv(sch, file.path(out, "a2_area_floors.csv"))
cat("45-2092 L1 blank cells by wage year:\n"); print(sch %>% filter(blank) %>% count(wage_year, fill_source))

# ---- County -> area (per wage year) ------------------------------------------
xw <- read_csv("explorations/identification_search/sources/oes_area_county_crosswalk_2009.csv", col_types = cols(.default = "c")) %>%
  filter(!is.na(county_fips)) %>% mutate(county_fips = harmonize_county_fips_2010(county_fips), state_fips = state_fips(state_fips))
ca <- xw %>% count(wage_year, county_fips, state_fips, oes_area_code, name = "towns") %>%
  group_by(wage_year, county_fips) %>% mutate(n_areas = n()) %>% arrange(desc(towns), oes_area_code, .by_group = TRUE) %>% slice(1) %>% ungroup()
stopifnot(all(substr(ca$county_fips, 1, 2) == ca$state_fips))
cat("counties spanning >1 area (assigned to modal area):", sum(ca$n_areas > 1 & ca$wage_year == WY09), "\n")

# ---- FLS AEWR schedule and minimum wages -------------------------------------
fls <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>% filter(occupation_scope == "nonrange_reference", reference_year %in% 2005:2012) %>%
  transmute(state_fips = state_fips(state_fips), fls = aewr, start = as.Date(effective_start), end_x = as.Date(effective_end_exclusive))
fls_states <- sort(unique(fls$state_fips))
mw <- read_parquet(path_int("state_real_minwages.parquet")) %>% filter(year %in% 2005:2011) %>%
  transmute(state_fips = state_fips(state_fips), cal_year = as.integer(year), mw = prevailing_ag_min_wage)

# ---- Receipt-month weights omega (FY2006-FY2008 national) --------------------
rows <- read_parquet(file.path(out, "a2_rows.parquet")) %>% filter(fy_receipt %in% 2006:2008, dates_ok, !monthly)
om <- rows %>% mutate(m = as.integer(format(received, "%m"))) %>% group_by(m) %>% summarise(h = sum(hours_c), .groups = "drop") %>%
  mutate(omega = h / sum(h)) %>% select(m, omega)
write_csv(om, file.path(out, "a2_receipt_weights.csv"))
cat("omega by receipt month:\n"); print(om %>% mutate(omega = round(omega, 4)) %>% as.data.frame())

days <- tibble(d = seq(as.Date("2005-10-01"), as.Date("2011-09-30"), by = "day")) %>%
  mutate(fy = as.integer(format(d, "%Y")) + as.integer(as.integer(format(d, "%m")) >= 10), m = as.integer(format(d, "%m")),
    cal_year = as.integer(format(d, "%Y"))) %>%
  group_by(fy, m) %>% mutate(ndays = n()) %>% ungroup() %>% left_join(om, by = "m") %>% mutate(wd = omega / ndays,
    regime = case_when(d < D_OES_START ~ "FLS", d < D_WY10 ~ "OES09", d < D_FLS_BACK ~ "OES10", TRUE ~ "FLS"))
# FLS segments: weight by state through the FLS rate in force on each day
fls_day <- days %>% filter(regime == "FLS") %>% select(d, fy, cal_year, wd) %>%
  cross_join(tibble(state_fips = fls_states)) %>%
  left_join(fls, by = "state_fips", relationship = "many-to-many") %>% filter(d >= start, d < end_x) %>%
  group_by(state_fips, d) %>% filter(n() == 1) %>% ungroup()
stopifnot(nrow(fls_day) == sum(days$regime == "FLS") * length(fls_states))
fls_state_fy <- fls_day %>% left_join(mw, by = c("state_fips", "cal_year")) %>%
  group_by(state_fips, fy) %>% summarise(w_fls = sum(wd), x_fls = sum(wd * pmax(fls, mw, na.rm = TRUE)), fls_mean = sum(wd * fls) / sum(wd), .groups = "drop")
oes_w <- days %>% filter(regime != "FLS") %>% group_by(fy, regime, cal_year) %>% summarise(w = sum(wd), .groups = "drop")
cat("OES-regime weight by FY:\n"); print(oes_w %>% group_by(fy, regime) %>% summarise(w = sum(w), .groups = "drop"))

# ---- County dose -------------------------------------------------------------
cty <- ca %>% filter(state_fips %in% fls_states) %>%
  select(wage_year, county_fips, state_fips, oes_area_code, n_areas) %>%
  left_join(sch %>% select(wage_year, state_fips, oes_area_code, l1, l1_fill, blank, fill_source), by = c("wage_year", "state_fips", "oes_area_code"))
stopifnot(!anyNA(cty$blank))
cw <- cty %>% pivot_wider(id_cols = c(county_fips, state_fips), names_from = wage_year,
  values_from = c(oes_area_code, l1, l1_fill, blank, n_areas)) %>%
  rename_with(~ sub(WY09, "09", sub(WY10, "10", .x, fixed = TRUE), fixed = TRUE))
cw <- cw %>% filter(!is.na(oes_area_code_09), !is.na(oes_area_code_10))
cat("counties with both wage-year areas:", nrow(cw), "in", n_distinct(cw$state_fips), "states\n")
oes_c <- tidyr::crossing(cw, oes_w) %>% left_join(mw, by = c("state_fips", "cal_year")) %>%
  mutate(fl_state = pmax(AEWR_FLOOR_2008_RULE, mw, na.rm = TRUE),
    l1r = ifelse(regime == "OES09", l1_09, l1_10), l1f = ifelse(regime == "OES09", l1_fill_09, l1_fill_10),
    floor = pmax(l1r, fl_state), floor_fill = pmax(l1f, fl_state), trunc = fl_state > l1f) %>%
  group_by(county_fips, fy) %>%
  summarise(w_oes = sum(w), x_oes = sum(w * floor), x_oes_fill = sum(w * floor_fill), x_oes_raw = sum(w * l1f),
    x_oes_state = sum(w * fl_state), trunc_share = sum(w * trunc) / sum(w), .groups = "drop")
dose <- tidyr::crossing(cw %>% select(county_fips, state_fips, starts_with("oes_area_code"), starts_with("l1"), starts_with("blank"), n_areas_09), fy = FYS) %>%
  left_join(fls_state_fy, by = c("state_fips", "fy")) %>% left_join(oes_c, by = c("county_fips", "fy")) %>%
  mutate(no_oes = is.na(w_oes), across(c(w_oes, x_oes, x_oes_fill, x_oes_raw, x_oes_state, trunc_share), ~ ifelse(no_oes, 0, .x)),
    x = x_fls + x_oes, x_fill = x_fls + x_oes_fill,
    # untruncated (raw OWL L1, blanks filled) and the state-level floor component, for the truncation statistics
    x_raw = x_fls + x_oes_raw, x_statefloor = x_fls + x_oes_state,
    area_09 = paste(state_fips, oes_area_code_09), area_10 = paste(state_fips, oes_area_code_10),
    blank_any = blank_09 | blank_10) %>%
  rename(year = fy)
stopifnot(all(abs(dose$w_fls + dose$w_oes - 1) < 1e-9))
# Exposure (time-invariant): within-state contrast carried by the OES regimes
dose <- dose %>% group_by(county_fips) %>%
  mutate(E09 = x[year == 2009] - x[year == 2008], E10 = x[year == 2010] - x[year == 2008]) %>% ungroup()
assert_geo_columns(dose, c("county_fips", "state_fips"))
stopifnot(!anyDuplicated(dose[c("county_fips", "year")]))
write_parquet(dose, file.path(out, "a2_county_dose.parquet"))
cat("\ndose summary (all counties, unweighted) by FY:\n")
print(dose %>% group_by(year) %>% summarise(n = n(), mean_x = mean(x, na.rm = TRUE), sd_x = sd(x, na.rm = TRUE),
  na_x = sum(is.na(x)), mean_w_oes = mean(w_oes), .groups = "drop"))
cat("blank-cell counties (primary dose NA):", sum(dose$blank_any & dose$year == 2009), "\n")
