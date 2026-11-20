# POST-OUTCOME REVIEW CHECK (Phase 6 blind review of A1; not declared in the
# A1 registry). Identification search, strategy A1.
#
# Robustness of the one significant A1 secondary design, SD5 (employer x state,
# unit FE + state x FY FE, PPML certified rq13d hours, eta = -0.0221, state
# CRV1 p = 0.0056):
#   Q1 reproduce SD5;
#   Q2 drop units whose b is imputed (no known-AEWR hourly hours in FY2025);
#   Q3 intensive margin (units positive in both years) and exit LPM;
#   Q4 control for the unit's FY2025 non-field SOC hours share x 1(FY2026);
#   Q5 field-only units (FY2025 SOC-known hours all field);
#   Q6 employer-level placebo FY2024 -> FY2025 with b from FY2024 and the same
#      FY2026 cut (the F1 analogue that the registry did not declare for SD5);
#   Q7 employer-ID linkage: share of rq13d hours without employer_id_balanced by
#      FY, and the FY2025 incumbents' FY2026 exit rate by b bin;
#   Q8 leave-one-state-out range.
# Usage (repository root):
#   R_LIBS_USER=... Rscript --vanilla explorations/identification_search/A1/review/r02_sd5_robustness.R "$S"
# Writes $S/A1/review/r02_*.csv
source("explorations/identification_search/A1/03_helpers.R")
a1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(a1_out_dir(S), "review"); dir.create(out, showWarnings = FALSE)

cr <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("case_number", "fiscal_year", "fy_receipt", "in_rq13d", "state_fips", "employer_id_balanced", "cert_hours",
    "hourly_wage", "aewr_known", "at_aewr", "soc7")) %>% filter(in_rq13d, fy_receipt %in% 2024:2026)
link <- cr %>% filter(is.finite(cert_hours)) %>% group_by(fy_receipt) %>%
  summarise(hours = sum(cert_hours), share_no_employer_id = sum(cert_hours[is.na(employer_id_balanced)]) / hours,
    na_cert_hours_rows = NA_integer_, .groups = "drop") %>%
  mutate(na_cert_hours_rows = sapply(fy_receipt, function(y) sum(cr$fy_receipt == y & !is.finite(cr$cert_hours))))
print(link); write_csv(link, file.path(out, "r02_q7_employer_linkage.csv"))

disc <- arrow::open_dataset(path_int("h2a_disclosure_records.parquet")) %>% filter(fiscal_year %in% 2024:2026) %>%
  select(case_number, fiscal_year, soc_code) %>% collect() %>%
  mutate(soc7_d = substr(gsub("[^0-9-]", "", coalesce(soc_code, "")), 1, 7)) %>% distinct(case_number, fiscal_year, .keep_all = TRUE)
cr <- cr %>% filter(is.finite(cert_hours), !is.na(employer_id_balanced)) %>%
  left_join(disc %>% select(case_number, fiscal_year, soc7_d), by = c("case_number", "fiscal_year")) %>%
  mutate(soc = ifelse(!is.na(soc7) & soc7 != "", soc7, soc7_d), soc_known = !is.na(soc) & soc != "", field = soc %in% FIELD_SOC,
    hourly = is.finite(hourly_wage) & hourly_wage > 0 & hourly_wage <= 100, unit = paste(employer_id_balanced, state_fips))
emp <- cr %>% group_by(unit, state_fips, year = fy_receipt) %>%
  summarise(h = sum(cert_hours), kn = sum(cert_hours[hourly & aewr_known %in% TRUE]),
    at = sum(cert_hours[hourly & aewr_known %in% TRUE & at_aewr %in% TRUE]),
    knf = sum(cert_hours[hourly & aewr_known %in% TRUE & field]), atf = sum(cert_hours[hourly & aewr_known %in% TRUE & at_aewr %in% TRUE & field]),
    nonfield_sh = sum(cert_hours[soc_known & !field]) / sum(cert_hours[soc_known]), all_field = sum(cert_hours[soc_known & !field]) == 0 & any(soc_known),
    .groups = "drop")

eb <- read_parquet(file.path(S, "A1", "a1_employer_doses.parquet"))
pe <- tidyr::crossing(eb %>% select(unit, state_fips, aewr_region_id, e25, eff_cut_h, b, lvl_d1), year = 2025:2026) %>%
  left_join(emp %>% select(unit, year, h), by = c("unit", "year")) %>% mutate(h = coalesce(h, 0)) %>%
  left_join(emp %>% filter(year == 2025) %>% select(unit, kn, nonfield_sh, all_field, knf, atf), by = "unit") %>%
  mutate(post = as.numeric(year == 2026), x = ifelse(year == 2025, e25, lvl_d1), state_year = paste(state_fips, year),
    b_known = kn > 0, nonfield_post = coalesce(nonfield_sh, 0) * post)
both <- pe %>% group_by(unit) %>% filter(all(h > 0)) %>% ungroup()
ex <- pe %>% filter(year == 2026) %>% mutate(exit = as.numeric(h == 0), dose = b * eff_cut_h, state_year = paste(state_fips, 2026))

q1 <- a1_est("Q1_SD5_reproduce", pe, "h", "x", c("unit", "state_year"), role = "review_post_outcome", declared = FALSE)
loo <- tibble(state_fips = rownames(q1$B), eta_loo = q1$B[, 1]) %>% arrange(eta_loo)
print(head(loo, 3)); print(tail(loo, 3)); write_csv(loo, file.path(out, "r02_q8_sd5_leave_one_state_out.csv"))
res <- list(q1$rows,
  a1_est("Q2_SD5_b_known_only", pe %>% filter(b_known), "h", "x", c("unit", "state_year"), role = "review_post_outcome", declared = FALSE)$rows,
  a1_est("Q3a_SD5_intensive_both_years", both, "h", "x", c("unit", "state_year"), role = "review_post_outcome", declared = FALSE)$rows,
  a1_est("Q3b_exit_LPM_unweighted", ex, "exit", "dose", "state_fips", method = "ols", role = "review_post_outcome", declared = FALSE)$rows,
  a1_est("Q4_SD5_nonfield_control", pe, "h", "x", c("unit", "state_year"), ctrl = "nonfield_post", role = "review_post_outcome", declared = FALSE)$rows,
  a1_est("Q5_SD5_field_only_units", pe %>% filter(all_field %in% TRUE), "h", "x", c("unit", "state_year"), role = "review_post_outcome", declared = FALSE)$rows)

# Q6 placebo FY2024 -> FY2025 (b from FY2024, FY2024 incumbents)
inc24 <- emp %>% filter(year == 2024, h > 0)
st24 <- inc24 %>% group_by(state_fips) %>% summarise(st = sum(at) / sum(kn), .groups = "drop")
stv <- eb %>% distinct(state_fips, e25, eff_cut_h, aewr_region_id)
pl <- inc24 %>% left_join(st24, by = "state_fips") %>% transmute(unit, state_fips, b24 = ifelse(kn > 0, at / kn, st)) %>%
  inner_join(stv, by = "state_fips")
pp <- tidyr::crossing(pl, year = 2024:2025) %>% left_join(emp %>% select(unit, year, h), by = c("unit", "year")) %>%
  mutate(h = coalesce(h, 0), x = e25 + (year == 2025) * b24 * eff_cut_h, state_year = paste(state_fips, year))
res[[length(res) + 1]] <- a1_est("Q6_SD5_placebo_fy2024_fy2025", pp, "h", "x", c("unit", "state_year"), role = "review_post_outcome", declared = FALSE)$rows
# Same placebo FY2023 -> FY2024 is not possible with this case-row file (FY2024-26 read); report FY2024->25 only.
r <- bind_rows(res) %>% mutate(label = "POST-OUTCOME review check")
print(r %>% select(spec_id, term, estimate, se, p, se_jackknife, p_jackknife, p_region_crv1, n, units, clusters), width = 220)
write_csv(r, file.path(out, "r02_sd5_robustness.csv"))

# Exit rate by b bin (FY2025 incumbents) and hours-weighted
eb_bin <- ex %>% left_join(emp %>% filter(year == 2025) %>% select(unit, h25 = h), by = "unit") %>%
  mutate(bin = cut(b, c(-0.01, 0.25, 0.75, 0.99, 1))) %>% group_by(bin) %>%
  summarise(units = n(), hours25 = sum(h25), exit_rate = mean(exit), hours_weighted_exit = sum(h25 * exit) / sum(h25),
    mean_nonfield_sh = mean(nonfield_sh, na.rm = TRUE), .groups = "drop")
print(eb_bin); write_csv(eb_bin, file.path(out, "r02_q7_exit_by_b.csv"))
message("done ", format(Sys.time()))
