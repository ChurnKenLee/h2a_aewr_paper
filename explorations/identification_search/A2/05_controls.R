# Identification search, strategy A2 (Phase 3 design, step 1c): predetermined
# 2009-recession exposure controls (declared before any outcome contrast).
#
# Usage (repository root):
#   Rscript --vanilla explorations/identification_search/A2/05_controls.R "$S"
# Writes $S/A2/a2_county_controls.parquet: county_fips, constr_share,
# manuf_share (QCEW 2006-2007 mean private annual-average employment in NAICS
# 23 and NAICS 31-33 over total private employment), suppression flags.
#
# Source: BLS QCEW annual singlefiles 2006 and 2007 (data/raw/qcew, read only).
# Rows: own_code 5 (private); agglvl 71 (county total, industry 10) and 74
# (county NAICS sector: 23, 31-33). Suppressed sector cells (disclosure 'N',
# employment 0) are imputed with the state's employment-weighted share among
# disclosed counties, and flagged.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "A2"); tmp <- file.path(out, "qcew_tmp"); dir.create(tmp, showWarnings = FALSE)
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
q <- rbindlist(lapply(2006:2007, function(y) {
  zf <- file.path(path_root(), "data", "raw", "qcew", paste0(y, "_annual_singlefile.zip"))
  nm <- unzip(zf, list = TRUE)$Name[1]
  f <- unzip(zf, files = nm, exdir = tmp, overwrite = TRUE)
  d <- fread(f, select = c("area_fips", "own_code", "industry_code", "agglvl_code", "year", "disclosure_code", "annual_avg_emplvl"),
    colClasses = list(character = c("area_fips", "own_code", "industry_code", "agglvl_code", "disclosure_code")))
  unlink(f)  # scratch copy of the extracted CSV only
  d[own_code == "5" & ((agglvl_code == "71" & industry_code == "10") | (agglvl_code == "74" & industry_code %in% c("23", "31-33")))]
}))
q <- as_tibble(q) %>% filter(!grepl("999$", area_fips), !substr(area_fips, 1, 2) %in% c("72", "78")) %>%
  mutate(county_fips = harmonize_county_fips_2010(area_fips), sup = disclosure_code == "N",
    ind = recode(industry_code, "10" = "tot", "23" = "constr", "31-33" = "manuf"))
w <- q %>% group_by(county_fips, ind) %>% summarise(emp = mean(annual_avg_emplvl), sup = any(sup), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = ind, values_from = c(emp, sup))
w <- w %>% mutate(state_fips = state_from_county_fips(county_fips),
  constr_share = ifelse(!coalesce(sup_constr, TRUE) & emp_tot > 0, emp_constr / emp_tot, NA_real_),
  manuf_share = ifelse(!coalesce(sup_manuf, TRUE) & emp_tot > 0, emp_manuf / emp_tot, NA_real_)) %>%
  group_by(state_fips) %>%
  mutate(constr_imputed = is.na(constr_share), manuf_imputed = is.na(manuf_share),
    constr_share = coalesce(constr_share, sum(emp_constr[!constr_imputed], na.rm = TRUE) / sum(emp_tot[!constr_imputed], na.rm = TRUE)),
    manuf_share = coalesce(manuf_share, sum(emp_manuf[!manuf_imputed], na.rm = TRUE) / sum(emp_tot[!manuf_imputed], na.rm = TRUE))) %>%
  ungroup() %>% select(county_fips, state_fips, emp_tot, constr_share, manuf_share, constr_imputed, manuf_imputed)
assert_geo_columns(w, c("county_fips", "state_fips"))
stopifnot(!anyDuplicated(w$county_fips))
write_parquet(w, file.path(out, "a2_county_controls.parquet"))
cat("counties:", nrow(w), "| imputed construction:", sum(w$constr_imputed), "| imputed manufacturing:", sum(w$manuf_imputed), "\n")
print(summary(w[c("constr_share", "manuf_share")]))
