# Identification search, strategy A2 (Phase 3 design, step 1d): support
# statistics, truncation statistics, pre-period residual variance and the
# node-period share (gate DGP rule). Builds the design frames used by 07_power.R.
#
# Usage (repository root, after 03-05):
#   Rscript --vanilla explorations/identification_search/A2/06_support.R "$S"
# Writes $S/A2/a2_support.csv, a2_truncation.csv, a2_node_share.csv,
# a2_design_frames.rds.
#
# Integrity: outcomes enter only for FY2006-FY2008 (calibration weights mu_i
# and FE-only residual variance). FY2009-FY2011 outcomes are never read here;
# the policy frame carries the dose only.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr); library(fixest)})
source(path_code("designs", "border_discontinuity", "power.R"))
source("explorations/identification_search/shared/power_harness.R")
setFixest_nthreads(1)
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "A2")
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
PRE <- 2006:2008; FYS <- 2006:2011

dose <- read_parquet(file.path(out, "a2_county_dose.parquet"))
ctl <- read_parquet(file.path(out, "a2_county_controls.parquet")) %>% select(county_fips, constr_share, manuf_share)
pan <- read_parquet(file.path(out, "a2_county_rfy_panel.parquet")) %>% filter(year %in% PRE) %>% select(county_fips, year, cert_hours_c)
stopifnot(all(pan$year %in% PRE))

base <- dose %>% left_join(ctl, by = "county_fips") %>%
  mutate(state_year = paste(state_fips, year), area_id = area_09, assign_id = paste(area_09, area_10, sep = "|"),
    nonmetro = nchar(oes_area_code_09) == 7)
cat("counties without QCEW controls:", n_distinct(base$county_fips[is.na(base$constr_share)]), "(dropped)\n")
base <- base %>% filter(!is.na(constr_share))

# Pre-period calibration frame (FY2006-08) and PPML weights mu_i
pre_all <- base %>% filter(year %in% PRE) %>% left_join(pan, by = c("county_fips", "year")) %>%
  mutate(cert_hours_c = coalesce(cert_hours_c, 0))
pol_all <- base

frame <- function(xcol, keep_blank = FALSE) {
  ok <- pol_all %>% group_by(county_fips) %>% summarise(ok = all(!is.na(.data[[xcol]])), .groups = "drop") %>% filter(ok)
  pol <- pol_all %>% semi_join(ok, by = "county_fips") %>% mutate(x = .data[[xcol]])
  pre <- pre_all %>% semi_join(ok, by = "county_fips")
  lin <- ish_ppml_linearize(pre, pol, "cert_hours_c", "county_fips")
  list(pre = lin$pre, pol = lin$policy, y = lin$y, w = lin$weights)
}
# FWL: residualize the dose on FE + (construction, manufacturing share) x FY
resid_controls <- function(pol, w, fe_fml = "county_fips + state_year") {
  f <- as.formula(paste("x ~ i(year, constr_share, ref = 2008) + i(year, manuf_share, ref = 2008) |", fe_fml))
  m <- feols(f, data = pol, weights = pol[[w]])
  r <- resid(m, na.rm = FALSE); r[is.na(r)] <- 0  # singleton FE cells are fit exactly
  pol$x_unres <- pol$x; pol$x <- r; pol
}

sup_rows <- list()
support <- function(tag, pol, w, fe, jk = "state_fips", other = "area_id", unit = "county_fips", note = "") {
  pp <- ish_prep(pol, unit, "year", fe, jk, NULL, w, x = "x", clusters = c(jk, other), jk_cluster = jk)
  lev <- rowsum(pp$xw^2, match(pp$data[[jk]], pp$jk))[, 1]
  la <- rowsum(pp$xw^2, pp$data[[other]])[, 1]
  wm <- function(v) sum(pp$w * v) / sum(pp$w)
  r <- tibble(run = tag, units = length(pp$units), periods = length(pp$periods), rows = nrow(pp$data), states = length(pp$jk),
    states_with_dose = sum(lev > 1e-12 * pp$sxx), areas = length(la), areas_with_dose = sum(la > 1e-12 * pp$sxx),
    dose_sd_after_fe = sqrt(pp$sxx / sum(pp$w)), effective_states = pp$sxx^2 / sum(lev^2), max_state_leverage = max(lev) / pp$sxx,
    max_state = pp$jk[which.max(lev)], effective_areas = pp$sxx^2 / sum(la^2), max_area_leverage = max(la) / pp$sxx,
    max_area = names(la)[which.max(la)], mean_dose_w = wm(pp$data$x_unres %||% pp$data$x), note = note)
  sup_rows[[tag]] <<- r; print(as.data.frame(r)); invisible(r)
}
`%||%` <- function(a, b) if (is.null(a)) b else a

FE <- c("county_fips", "state_year")
# P: primary (blank cells excluded, controls partialled out)
fp <- frame("x"); fp$pol <- resid_controls(fp$pol, fp$w)
support("P_primary", fp$pol, fp$w, FE, note = "county (employer location) FE + state x FY FE; recession controls x FY partialled; PPML weights = FY2006-08 mean hours")
# Residual SD of the PPML-linearized pre outcome after FE (weighted)
pre_pp <- ish_prep(fp$pre, "county_fips", "year", FE, "area_id", NULL, fp$w)
rsd <- sqrt(sum(ish_within(cbind(pre_pp$data[[fp$y]]), pre_pp)^2) / sum(pre_pp$w))
cat("pre-period (FY2006-08) PPML-linearized residual SD after county + state x FY FE:", round(rsd, 4), "\n")
ns <- bind_rows(
  ish_node_period_share(fp$pre, fp$y, "county_fips", "year", FE, "area_id", weights = fp$w) %>% mutate(node = "area_id (OES area x state)"),
  ish_node_period_share(fp$pre, fp$y, "county_fips", "year", c("county_fips", "year"), "state_fips", weights = fp$w) %>% mutate(node = "state_fips, county + FY FE (reference only)"))
ns$ratio <- ns$observed_node_period_share / ns$pure_unit_noise_share
print(ns); write_csv(ns, file.path(out, "a2_node_share.csv"))

f0 <- frame("x"); f0$pol$x_unres <- f0$pol$x
support("S_no_controls", f0$pol, f0$w, FE, note = "primary without recession controls")
ff <- frame("x_fill"); ff$pol <- resid_controls(ff$pol, ff$w)
support("S_blank_filled", ff$pol, ff$w, FE, note = "blank 45-2092 L1 cells filled (other wage year, then 45-2099, 45-2091)")
fr <- frame("x_raw"); fr$pol <- resid_controls(fr$pol, fr$w)
support("S_untruncated_counterfactual", fr$pol, fr$w, FE, note = "counterfactual dose: raw OWL L1 (filled), no 7.25 / state-minimum truncation")
fpm <- frame("x"); fpm$pol <- resid_controls(fpm$pol %>% mutate(nm_year = paste(nonmetro, year), state_nm_year = paste(state_fips, nonmetro, year)), fpm$w,
  "county_fips + state_nm_year")
support("S_state_x_nonmetro_x_FY", fpm$pol %>% mutate(state_nm_year = paste(state_fips, nonmetro, year)), fpm$w, c("county_fips", "state_nm_year"),
  note = "FE state x nonmetro-area x FY (absorbs metro/nonmetro trends within state)")
fu <- frame("x"); fu$pol <- resid_controls(fu$pol %>% mutate(one = 1), "one")
support("S_unweighted_counties_with_pre_use", fu$pol %>% mutate(one = 1), "one", FE, note = "equal weights, counties with FY2006-08 use")
# Area-level unit (OES area x state): aggregate pre outcome; dose is constant within area in the primary frame
ar_pol <- fp$pol %>% group_by(area_id, state_fips, year) %>%
  summarise(x_unres = weighted.mean(x_unres, ppml_weight), constr_share = weighted.mean(constr_share, ppml_weight),
    manuf_share = weighted.mean(manuf_share, ppml_weight), xsd = sd(x_unres), .groups = "drop")
cat("max within-area SD of the county dose (primary frame):", signif(max(ar_pol$xsd, na.rm = TRUE), 3), "\n")
ar_pre <- fp$pre %>% group_by(area_id, state_fips, year) %>% summarise(cert_hours_c = sum(cert_hours_c), .groups = "drop") %>%
  mutate(state_year = paste(state_fips, year))
ar_pol <- ar_pol %>% mutate(x = x_unres, state_year = paste(state_fips, year))
la <- ish_ppml_linearize(ar_pre, ar_pol, "cert_hours_c", "area_id")
la$policy <- resid_controls(la$policy, la$weights, "area_id + state_year")
support("S_area_unit", la$policy %>% mutate(area2 = area_id), la$weights, c("area_id", "state_year"), other = "area2", unit = "area_id",
  note = "unit = OES area x state (employer location), area FE + state x FY FE")

# Worksite-county frame (shared panel, receipt FY2008 weights): dose support only (one pre-year: no calibration possible)
ws <- read_parquet(file.path(S, "shared", "outcomes_county_fy.parquet"), col_select = c("county_fips", "year", "cert_hours_rfy")) %>%
  filter(year == 2008, cert_hours_rfy > 0) %>% transmute(county_fips, ppml_weight = cert_hours_rfy)
wpol <- base %>% filter(year %in% 2008:2011) %>% inner_join(ws, by = "county_fips") %>% group_by(county_fips) %>% filter(all(!is.na(x))) %>% ungroup() %>%
  mutate(x = x)
wpol <- resid_controls(wpol, "ppml_weight")
support("S_worksite_county_FY2008_11", wpol, "ppml_weight", FE, note = "shared worksite panel geography (rfy), FY2008 hours weights, FY2008-11; support only")

sup <- bind_rows(sup_rows) %>% mutate(preperiod_residual_sd_primary = rsd)
write_csv(sup, file.path(out, "a2_support.csv"))

# ---- Truncation statistics ----------------------------------------------------
fl <- read_csv(file.path(out, "a2_area_floors.csv"), show_col_types = FALSE)
mu <- fp$pre %>% distinct(county_fips, ppml_weight)
d09 <- base %>% filter(year == 2009) %>% select(county_fips, state_fips, l1_09, l1_10, l1_fill_09, l1_fill_10, blank_any) %>%
  inner_join(mu, by = "county_fips")
mw <- read_parquet(path_int("state_real_minwages.parquet")) %>% filter(year %in% 2009:2010) %>%
  transmute(state_fips = state_fips(state_fips), year, mw = prevailing_ag_min_wage) %>% pivot_wider(names_from = year, values_from = mw, names_prefix = "mw")
d09 <- d09 %>% left_join(mw, by = "state_fips") %>%
  mutate(fl09 = pmax(7.25, mw2009, na.rm = TRUE), fl10 = pmax(7.25, mw2009, mw2010, na.rm = TRUE),
    f09 = pmax(l1_fill_09, fl09), f10 = pmax(l1_fill_10, fl10), t09 = fl09 >= l1_fill_09, t10 = fl10 >= l1_fill_10)
wsd <- function(v, g, w) { m <- ave(v * w, g, FUN = sum) / ave(w, g, FUN = sum); sqrt(sum(w * (v - m)^2) / sum(w)) }
usd <- function(v, g) { m <- ave(v, g, FUN = mean); sqrt(mean((v - m)^2)) }
tr <- bind_rows(lapply(c("09", "10"), function(k) {
  raw <- d09[[paste0("l1_fill_", k)]]; f <- d09[[paste0("f", k)]]; t <- d09[[paste0("t", k)]]; w <- d09$ppml_weight
  tibble(wage_year = k, counties = nrow(d09), share_counties_truncated = mean(t), share_preperiod_hours_truncated = sum(w[t]) / sum(w),
    within_state_sd_raw_unweighted = usd(raw, d09$state_fips), within_state_sd_floor_unweighted = usd(f, d09$state_fips),
    within_state_sd_raw_hours_w = wsd(raw, d09$state_fips, w), within_state_sd_floor_hours_w = wsd(f, d09$state_fips, w),
    share_within_state_var_surviving_hours_w = (wsd(f, d09$state_fips, w) / wsd(raw, d09$state_fips, w))^2)
}))
print(as.data.frame(tr)); write_csv(tr, file.path(out, "a2_truncation.csv"))
saveRDS(list(primary = fp, no_controls = f0, blank_filled = ff, area = la, nonmetro = fpm), file.path(out, "a2_design_frames.rds"))
