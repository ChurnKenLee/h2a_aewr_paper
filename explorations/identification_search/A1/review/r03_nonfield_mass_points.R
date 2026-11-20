# POST-OUTCOME REVIEW CHECK (Phase 6 blind review of A1; not declared in the
# A1 registry). Identification search, strategy A1.
#
# Are FY2025 non-field H-2A hourly offers bunched at a state x SOC mass point
# (the signature of a binding occupation-specific floor such as the 2023-rule
# OEWS AEWR, which is absent from aewr_rate_schedule.parquet), and how did
# their offers change in FY2026 (receipts >= 2025-10-02) relative to field
# offers in the same state? Predetermined FY2025 data plus FY2026 offered wages.
# Usage (repository root):
#   R_LIBS_USER=... Rscript --vanilla explorations/identification_search/A1/review/r03_nonfield_mass_points.R "$S"
# Writes $S/A1/review/r03_nonfield_mass_points.csv, r03_wage_change_field_nonfield.csv
source("explorations/identification_search/A1/03_helpers.R")
a1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(a1_out_dir(S), "review")
cr <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("case_number", "fiscal_year", "fy_receipt", "in_rq13d", "received", "state_fips", "cert_hours", "hourly_wage", "soc7", "at_aewr")) %>%
  filter(in_rq13d, fy_receipt %in% 2025:2026, is.finite(cert_hours), cert_hours > 0, is.finite(hourly_wage), hourly_wage > 0, hourly_wage <= 100,
    fy_receipt == 2025 | received >= as.Date("2025-10-02"))
disc <- arrow::open_dataset(path_int("h2a_disclosure_records.parquet")) %>% filter(fiscal_year %in% 2025:2026) %>%
  select(case_number, fiscal_year, soc_code) %>% collect() %>%
  mutate(soc7_d = substr(gsub("[^0-9-]", "", coalesce(soc_code, "")), 1, 7)) %>% distinct(case_number, fiscal_year, .keep_all = TRUE)
cr <- cr %>% left_join(disc %>% select(case_number, fiscal_year, soc7_d), by = c("case_number", "fiscal_year")) %>%
  mutate(soc = ifelse(!is.na(soc7) & soc7 != "", soc7, soc7_d), grp = ifelse(is.na(soc) | soc == "", "unknown", ifelse(soc %in% FIELD_SOC, "field", "nonfield")))
# Mass point: share of state x SOC hours within $0.05 of the hours-modal wage (cells with >= 5 rows)
mp <- cr %>% filter(fy_receipt == 2025, grp != "unknown") %>% mutate(wr = round(hourly_wage, 2)) %>%
  group_by(grp, state_fips, soc) %>% filter(n() >= 5) %>%
  mutate(mode = { t <- tapply(cert_hours, wr, sum); as.numeric(names(t)[which.max(t)]) }) %>%
  summarise(h = sum(cert_hours), at_mode = sum(cert_hours[abs(hourly_wage - mode) <= 0.05]), flagged_at_aewr = sum(cert_hours[at_aewr %in% TRUE]), .groups = "drop") %>%
  group_by(grp) %>% summarise(cells = n(), hours = sum(h), share_at_cell_mode = sum(at_mode) / hours, share_flagged_at_aewr = sum(flagged_at_aewr) / hours, .groups = "drop") %>%
  mutate(label = "POST-OUTCOME review check")
print(mp); write_csv(mp, file.path(out, "r03_nonfield_mass_points.csv"))
wc <- cr %>% filter(grp != "unknown") %>% group_by(grp, year = fy_receipt) %>%
  summarise(hours = sum(cert_hours), mean_wage = sum(hourly_wage * cert_hours) / hours, .groups = "drop") %>%
  group_by(grp) %>% mutate(change = mean_wage - lag(mean_wage), pct = change / lag(mean_wage)) %>% ungroup() %>% mutate(label = "POST-OUTCOME review check")
print(wc); write_csv(wc, file.path(out, "r03_wage_change_field_nonfield.csv"))
