# POST-OUTCOME REVIEW CHECK (Phase 6 blind review of A1; not declared in the
# A1 registry). Identification search, strategy A1.
#
# Purpose: test whether the near-zero A1 first stage (pi_U = 0.033, F = 0.10)
# is a feature of employer behaviour or an artefact of how the FY2025 binding
# share b_c is measured. The shared at-AEWR flag (shared/01_outcomes.R) compares
# FY2025 hourly offers only with the field/livestock reference AEWR
# (aewr_rate_schedule: nonrange / fls_common_soc reference). Under the 2023 AEWR
# rule (effective 2023-03-30), non-field SOCs had occupation-specific OEWS AEWRs
# that are not in the repository schedule, so a non-field job paid exactly its
# own AEWR is counted as "above the AEWR" (b < 1).
#
# Checks (all post-outcome, descriptive or re-estimation):
#   R1  reproduce pi_U and F from the case rows and the A1 dose file;
#   R2  FY2025 rq13d known-AEWR hourly hours: at-AEWR share by field vs
#       non-field SOC (SOC from case rows, else h2a_disclosure_records);
#       share of "not at AEWR" hours that are non-field;
#   R3  field-only binding share b_f and field-only offered wage omega_f;
#       first stages (county, state x FY FE, registry weights) using b_f;
#   R4  employer x state first stage (incumbents with hourly offers in both
#       years): change in offered wage on b_e x (F^U - e25), state FE;
#   R5  primary reduced form with the field-only binding share (post-outcome).
#
# Usage (repository root):
#   R_LIBS_USER=... Rscript --vanilla explorations/identification_search/A1/review/r01_binding_share_and_first_stage.R "$S"
# Writes $S/A1/review/r01_*.csv
source("explorations/identification_search/A1/03_helpers.R")
a1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(a1_out_dir(S), "review"); dir.create(out, showWarnings = FALSE)
FE_W <- c("county_fips", "state_year")

doses <- a1_doses(S) %>% left_join(a1_ppml_weights(S), by = "county_fips") %>% mutate(ppml_w = coalesce(ppml_w, 0))

# ---- Case rows with SOC -------------------------------------------------------------
cr <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("case_number", "fiscal_year", "fy_receipt", "in_rq13d", "received", "county_fips", "state_fips",
    "employer_id_balanced", "cert_hours", "hourly_wage", "aewr_known", "at_aewr", "soc7")) %>%
  filter(in_rq13d, fy_receipt %in% 2025:2026, is.finite(cert_hours), cert_hours > 0, county_fips != "00000")
disc <- arrow::open_dataset(path_int("h2a_disclosure_records.parquet")) %>% filter(fiscal_year %in% 2024:2026) %>%
  select(case_number, fiscal_year, soc_code) %>% collect() %>%
  mutate(soc7_d = substr(gsub("[^0-9-]", "", coalesce(soc_code, "")), 1, 7)) %>% distinct(case_number, fiscal_year, .keep_all = TRUE)
cr <- cr %>% left_join(disc %>% select(case_number, fiscal_year, soc7_d), by = c("case_number", "fiscal_year")) %>%
  mutate(soc = ifelse(!is.na(soc7) & soc7 != "", soc7, soc7_d), soc_known = !is.na(soc) & soc != "",
    field = soc %in% FIELD_SOC, hourly = is.finite(hourly_wage) & hourly_wage > 0 & hourly_wage <= 100)

# ---- R2: who is "not at AEWR" in FY2025 ------------------------------------------------
k25 <- cr %>% filter(fy_receipt == 2025, hourly, aewr_known %in% TRUE)
a25 <- doses %>% distinct(state_fips, e25)
r2 <- k25 %>% left_join(a25, by = "state_fips") %>% mutate(grp = ifelse(!soc_known, "soc_unknown", ifelse(field, "field", "nonfield"))) %>%
  group_by(grp) %>% summarise(known_hours = sum(cert_hours), at_aewr_share = sum(cert_hours[at_aewr %in% TRUE]) / known_hours,
    not_at_hours = sum(cert_hours[!(at_aewr %in% TRUE)]),
    median_wage_over_field_aewr = { o <- order(hourly_wage / e25); x <- (hourly_wage / e25)[o]; w <- cert_hours[o]; x[which(cumsum(w) >= sum(w) / 2)[1]] },
    .groups = "drop") %>%
  mutate(share_of_all_known_hours = known_hours / sum(known_hours), share_of_not_at_hours = not_at_hours / sum(not_at_hours))
print(r2, width = 200); write_csv(r2, file.path(out, "r01_r2_fy2025_not_at_aewr_by_soc.csv"))
# Top non-field SOCs among not-at hours
r2b <- k25 %>% filter(soc_known, !field, !(at_aewr %in% TRUE)) %>% group_by(soc) %>% summarise(hours = sum(cert_hours), .groups = "drop") %>%
  arrange(desc(hours)) %>% mutate(share = hours / sum(hours)) %>% head(10)
print(r2b); write_csv(r2b, file.path(out, "r01_r2b_top_nonfield_soc.csv"))
# Not-at field hours: how far above the field AEWR?
r2c <- k25 %>% filter(field, !(at_aewr %in% TRUE)) %>% left_join(a25, by = "state_fips") %>%
  mutate(gap = hourly_wage - e25, bin = cut(gap, c(-Inf, -0.05, 0.5, 1, 2, 5, Inf))) %>% group_by(bin) %>%
  summarise(hours = sum(cert_hours), .groups = "drop") %>% mutate(share = hours / sum(hours))
print(r2c); write_csv(r2c, file.path(out, "r01_r2c_field_not_at_gap.csv"))
# Field rows below the FY2025 AEWR: by receipt month, and equality with the prior (FY2024) AEWR
a24 <- doses %>% distinct(state_fips, a24)
r2e <- k25 %>% filter(field, !(at_aewr %in% TRUE)) %>% left_join(a25, by = "state_fips") %>% left_join(a24, by = "state_fips") %>%
  filter(hourly_wage - e25 <= -0.05) %>% mutate(eq_a24 = abs(hourly_wage - a24) <= 0.05, rmon = format(received, "%Y-%m")) %>%
  group_by(state_fips) %>% summarise(hours = sum(cert_hours), share_eq_a24 = sum(cert_hours[eq_a24]) / hours,
    med_gap = median(hourly_wage - e25), .groups = "drop") %>% arrange(desc(hours)) %>% mutate(share = hours / sum(hours))
print(head(r2e, 12)); write_csv(r2e, file.path(out, "r01_r2e_field_below_aewr_by_state.csv"))

# ---- County binding share among field hours; field-only offered wage ----------------
bf <- k25 %>% filter(field) %>% group_by(county_fips) %>% summarise(at = sum(cert_hours[at_aewr %in% TRUE]), kn = sum(cert_hours), .groups = "drop")
nf <- k25 %>% group_by(county_fips) %>% summarise(nonfield_known_share = sum(cert_hours[soc_known & !field]) / sum(cert_hours), .groups = "drop")
doses <- doses %>% left_join(bf, by = "county_fips") %>% left_join(nf, by = "county_fips") %>% group_by(state_fips) %>%
  mutate(st_f = sum(at, na.rm = TRUE) / sum(kn, na.rm = TRUE)) %>% ungroup() %>%
  mutate(b_f = pmin(pmax(ifelse(is.finite(kn) & kn > 0, at / kn, st_f), 0), 1), b_f = ifelse(is.finite(b_f), b_f, b),
    d1_u_f = b_f * eff_cut_u, d1_h_f = b_f * eff_cut_h)
bsum <- doses %>% filter(is.finite(kn), kn > 0) %>% summarise(counties = n(),
  hw_mean_b = weighted.mean(b, b_kn), hw_mean_b_f = weighted.mean(b_f, kn),
  share_counties_b_f_eq1 = mean(b_f >= 0.9999), cor_b_nonfield_share = cor(b, nonfield_known_share, use = "complete.obs"))
print(bsum); write_csv(bsum, file.path(out, "r01_r3_binding_share_field_only.csv"))

om_all <- cr %>% filter(hourly, fy_receipt == 2025 | received >= as.Date("2025-10-02"))
mk_om <- function(d) d %>% group_by(county_fips, year = fy_receipt) %>%
  summarise(omega = sum(hourly_wage * cert_hours) / sum(cert_hours), .groups = "drop")
fs_frame <- function(om) {
  tidyr::crossing(doses, year = 2025:2026) %>% inner_join(om, by = c("county_fips", "year")) %>%
    group_by(county_fips) %>% filter(n() == 2, all(ppml_w > 0)) %>% ungroup() %>%
    mutate(post = as.numeric(year == 2026), state_year = paste(state_fips, year), region_year = paste(aewr_region_id, year),
      xU = e25 + post * d1_u, xUf = e25 + post * d1_u_f, xUraw = e25 + post * eff_cut_u, bpost = b * post, bfpost = b_f * post,
      nfpost = coalesce(nonfield_known_share, 0) * post)
}
fsA <- fs_frame(mk_om(om_all)); fsF <- fs_frame(mk_om(om_all %>% filter(field)))
r3 <- bind_rows(
  a1_est("R1_reproduce_FS_U", fsA, "omega", "xU", FE_W, method = "ols", wvar = "ppml_w", role = "review_post_outcome", declared = FALSE)$rows,
  a1_est("R3a_FS_all_rows_b_field", fsA, "omega", "xUf", FE_W, method = "ols", wvar = "ppml_w", role = "review_post_outcome", declared = FALSE)$rows,
  a1_est("R3b_FS_field_rows_b_field", fsF, "omega", "xUf", FE_W, method = "ols", wvar = "ppml_w", role = "review_post_outcome", declared = FALSE)$rows,
  a1_est("R3c_FS_field_rows_b_all", fsF, "omega", "xU", FE_W, method = "ols", wvar = "ppml_w", role = "review_post_outcome", declared = FALSE)$rows,
  a1_est("R3d_FS_all_rows_b_and_nonfield", fsA, "omega", c("bpost", "nfpost"), FE_W, method = "ols", wvar = "ppml_w", role = "review_post_outcome", declared = FALSE)$rows,
  a1_est("R3e_FS_all_rows_unweighted", fsA, "omega", "xU", FE_W, method = "ols", role = "review_post_outcome", declared = FALSE)$rows,
  a1_est("R3f_FS_raw_cut_region_FE", fsA, "omega", "xUraw", c("county_fips", "region_year"), method = "ols", wvar = "ppml_w", role = "review_post_outcome", declared = FALSE)$rows
) %>% mutate(F = (estimate / se)^2, label = "POST-OUTCOME review check")
print(r3 %>% select(spec_id, term, estimate, se, p, F, n, clusters), width = 200)
write_csv(r3, file.path(out, "r01_r3_first_stages.csv"))

# ---- R4: employer x state first stage --------------------------------------------------
ew <- cr %>% filter(hourly, !is.na(employer_id_balanced), fy_receipt == 2025 | received >= as.Date("2025-10-02")) %>%
  group_by(unit = paste(employer_id_balanced, state_fips), state_fips, year = fy_receipt) %>%
  summarise(omega = sum(hourly_wage * cert_hours) / sum(cert_hours), h = sum(cert_hours),
    kn = sum(cert_hours[aewr_known %in% TRUE]), at = sum(cert_hours[aewr_known %in% TRUE & at_aewr %in% TRUE]),
    fieldsh = sum(cert_hours[field]) / sum(cert_hours[soc_known]), .groups = "drop")
e25w <- ew %>% filter(year == 2025, kn > 0) %>% transmute(unit, b_e = at / kn, w25 = h, fieldsh25 = fieldsh)
st_cut <- doses %>% filter(!(state_fips == "36" & area_id == "downstate"), !(state_fips == "41" & area_id == "nonurban")) %>%
  distinct(state_fips, eff_cut_u, e25)
ef <- ew %>% inner_join(e25w, by = "unit") %>% group_by(unit) %>% filter(n() == 2) %>% ungroup() %>%
  left_join(st_cut, by = "state_fips") %>% left_join(doses %>% distinct(state_fips, aewr_region_id), by = "state_fips") %>%
  filter(!is.na(aewr_region_id), is.finite(eff_cut_u)) %>%
  mutate(post = as.numeric(year == 2026), xe = e25 + post * b_e * eff_cut_u, state_year = paste(state_fips, year),
    county_fips = unit, fieldpost = coalesce(fieldsh25, 0) * post, bpost = b_e * post)
r4 <- bind_rows(
  a1_est("R4a_FS_employer_state", ef, "omega", "xe", c("unit", "state_year"), method = "ols", wvar = "w25", role = "review_post_outcome", declared = FALSE)$rows,
  a1_est("R4b_FS_employer_state_field_only", ef %>% filter(fieldsh25 >= 0.999), "omega", "xe", c("unit", "state_year"), method = "ols", wvar = "w25", role = "review_post_outcome", declared = FALSE)$rows,
  a1_est("R4c_FS_employer_state_b_and_field", ef, "omega", c("bpost", "fieldpost"), c("unit", "state_year"), method = "ols", wvar = "w25", role = "review_post_outcome", declared = FALSE)$rows
) %>% mutate(F = (estimate / se)^2, label = "POST-OUTCOME review check")
print(r4 %>% select(spec_id, term, estimate, se, p, F, n, units, clusters), width = 200)
write_csv(r4, file.path(out, "r01_r4_employer_first_stage.csv"))
# Descriptive: change in employer offered wage by FY2025 b_e bin (field-only employers)
r4d <- ef %>% filter(fieldsh25 >= 0.999) %>% select(unit, year, omega, b_e, w25, eff_cut_u) %>%
  tidyr::pivot_wider(names_from = year, values_from = omega, names_prefix = "om") %>%
  mutate(bin = cut(b_e, c(-0.01, 0.25, 0.75, 0.99, 1)), d = om2026 - om2025) %>% group_by(bin) %>%
  summarise(units = n(), w = sum(w25), mean_d_omega = weighted.mean(d, w25), mean_cut_u = weighted.mean(eff_cut_u, w25), .groups = "drop")
print(r4d); write_csv(r4d, file.path(out, "r01_r4d_employer_wage_change_by_b.csv"))

# ---- R5: primary reduced form with the field-only binding share --------------------------
p <- a1_panel(S, doses, 2025:2026, "cert_hours_rq13d") %>% mutate(post = as.numeric(year == 2026),
  x = e25 + post * d1_h, xf = e25 + post * d1_h_f)
r5 <- bind_rows(
  a1_est("R5a_reproduce_primary", p, "cert_hours_rq13d", "x", FE_W, role = "review_post_outcome", declared = FALSE)$rows,
  a1_est("R5b_primary_b_field", p, "cert_hours_rq13d", "xf", FE_W, role = "review_post_outcome", declared = FALSE)$rows) %>%
  mutate(label = "POST-OUTCOME review check")
print(r5 %>% select(spec_id, estimate, se, p, se_jackknife, n), width = 200)
write_csv(r5, file.path(out, "r01_r5_reduced_form_b_field.csv"))
message("done ", format(Sys.time()))
