# Identification search, strategy A3 (Phase 3 design, step 1a): pre-rule
# employer x worksite-state units, pre-rule job cells and the statutory floor
# dose for the 2023 occupation-method AEWR rule and its 2025 reversal.
#
# Usage (repository root, after 00-01):
#   Rscript --vanilla explorations/identification_search/A3/02_units_dose.R "$S"
# Writes $S/A3/a3_units.parquet (one row per pre-rule unit: shares, type, PPML
# weights), a3_cells.parquet (employer x state x pre-rule SOC cells),
# a3_floors_daily.parquet, a3_unit_dose.parquet (unit x receipt FY dose, two
# receipt windows: rfy = full receipt FY, FY2020-FY2025, the primary; rq13d =
# receipts 1 Oct-30 Jun decided by 30 Jun, FY2020-FY2026, the FY2026 secondary),
# a3_cell_dose.parquet (employer x state x {field, nonfield} cells, design
# secondary), a3_pre_outcomes.parquet (FY2020-FY2022 only), a3_build.csv.
#
# Integrity: only FY2020-FY2022 (pre-rule) receipt-FY outcomes are read
# (cell definitions, shares, PPML weights, offered wages). No post-rule outcome
# is read or written here. Treatment comes from the pre-rule SOC on the
# disclosure record (h2a_disclosure_records by case_number x fiscal_year), never
# from post-rule SOC.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "A3")
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
dir.create(out, showWarnings = FALSE)
src <- "explorations/identification_search/sources"

PRE <- 2020:2022; FYS <- 2020:2026
FIELD_SOC <- c("45-2041", "45-2091", "45-2092", "45-2093", "45-2099", "53-7064")
NO_FLS <- c("02", "11", "66", "72", "78")  # AK, DC, GU, PR, VI: no FLS AEWR before 2023
KANSAS <- c("AR", "FL", "GA", "ID", "IN", "IA", "KS", "LA", "MO", "MT", "NE", "ND", "OK", "SC", "TN", "TX", "VA")
CUTOFF <- as.IDate("2023-03-30"); RAMP_END <- as.IDate("2023-04-29")  # job order -> application lag (<= 30 days)
TECHE <- as.IDate("2024-09-20"); VACATUR <- as.IDate("2025-08-29"); IFR <- as.IDate("2025-10-02")
log_rows <- list(); note <- function(k, v) log_rows[[length(log_rows) + 1]] <<- data.table(item = k, value = as.character(v))

# SOC groups for the share x FY controls (national occupation shocks)
soc_group <- function(soc) fcase(soc %in% FIELD_SOC, "field", soc == "53-3032", "g_53_3032", soc == "53-3033", "g_53_3033",
  soc %in% c("53-3053", "53-3052", "53-3058"), "g_53_305x", soc == "47-2061", "g_47_2061", soc == "49-3041", "g_49_3041",
  soc == "45-1011", "g_45_1011", substr(soc, 1, 2) == "47", "g_47_other", default = "g_other_nf")

# ---- Pre-rule case rows (FY2020-FY2022 receipt, all decision dates) -----------------
cr <- as.data.table(open_dataset(file.path(S, "shared", "outcomes_case_rows.parquet")) |>
  dplyr::filter(fy_receipt %in% PRE) |>
  dplyr::select(case_number, fiscal_year, fy_receipt, received, state_fips, county_fips, employer_id_balanced,
                cert_hours, cert_pos, req_pos, hourly_wage, in_rq13d) |> dplyr::collect())
stopifnot(all(cr$fy_receipt %in% PRE))
cr[, cert_hours := fcoalesce(cert_hours, 0)]
disc <- as.data.table(open_dataset(path_int("h2a_disclosure_records.parquet")) |>
  dplyr::filter(fiscal_year >= 2019, fiscal_year <= 2023) |>
  dplyr::select(case_number, fiscal_year, soc_code, job_title, employer_name) |> dplyr::collect())
disc <- disc[, .(soc = substr(gsub("[^0-9-]", "", fcoalesce(soc_code[soc_code != ""][1], "")), 1, 7),
                 job_title = job_title[1], employer_name = employer_name[1]), by = .(case_number, fiscal_year)]
cr <- merge(cr, disc, by = c("case_number", "fiscal_year"), all.x = TRUE)
note("pre_rows", nrow(cr)); note("pre_hours", sum(cr$cert_hours))
note("pre_hours_soc_missing_share", cr[, sum(cert_hours[is.na(soc) | !grepl("^\\d\\d-\\d{4}$", soc)]) / sum(cert_hours)])
cr <- cr[!is.na(soc) & grepl("^\\d\\d-\\d{4}$", soc) & !is.na(state_fips) & state_fips != "00" & cert_hours > 0]
cr[, state_fips := state_fips(state_fips)]
note("pre_hours_no_fls_state_share", cr[, sum(cert_hours[state_fips %in% NO_FLS]) / sum(cert_hours)])
cr <- cr[!state_fips %in% NO_FLS]
cr[, `:=`(field = soc %in% FIELD_SOC, grp = soc_group(soc))]
note("pre_hours_nonfield_share", cr[, sum(cert_hours[!field]) / sum(cert_hours)])

# ---- Rates ------------------------------------------------------------------------
rates <- fread(file.path(src, "aewr_2023_2025_occupation_rates.csv"), colClasses = list(character = c("state_fips", "soc_code")))
rates[, `:=`(s0 = as.IDate(effective_start), e0 = as.IDate(effective_end_exclusive),
             s1 = as.IDate(kansas_group_effective_start), e1 = as.IDate(kansas_group_effective_end_exclusive))]
st <- unique(rates[, .(state_fips, state_abbr, kansas_order_state)])
fls <- as.data.table(read_parquet(path_int("aewr_rate_schedule.parquet")))
fls <- fls[occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference") & worker_category == "covered_workers" &
           effective_end_exclusive > as.IDate("2019-01-01"),
           .(state_fips = state_fips(state_fips), aewr, s = as.IDate(effective_start), e = as.IDate(effective_end_exclusive),
             alt = as.IDate(alternative_effective_start))]
mw <- as.data.table(read_parquet(path_int("state_real_minwages.parquet")))[year %in% 2019:2025,
  .(state_fips = state_fips(state_fips), cyear = as.integer(year), mw = prevailing_ag_min_wage)]
# GA and WY end in 2023 (federal $7.25 binds there): carry the last year forward
mw <- CJ(state_fips = unique(mw$state_fips), cyear = 2019:2025)[mw, mw := i.mw, on = c("state_fips", "cyear")]
setorder(mw, state_fips, cyear); mw[, mw := nafill(mw, "locf"), by = state_fips]
mw26 <- fread("explorations/identification_search/A1/sources/state_ag_minwage_fy2025_2026.csv", colClasses = list(character = "state_fips"))
mw26 <- mw26[area_id %in% c("statewide", "upstate", "standard") & snapshot_date %in% c("2025-10-01", "2026-01-01"),
             .(state_fips, snap = as.IDate(snapshot_date), mw = h2a_state_floor)]
sk <- as.data.table(read_parquet(path_int("aewr_occupation_skill_rates.parquet")))
sk <- sk[worker_category == "H2A" & as.character(effective_start) == "2025-10-02",
         .(state_fips = state_fips(state_fips), soc_code, skill_level, rate = hourly_aewr, sheet = trimws(source_sheet))]
sk_field <- dcast(sk[soc_code == "FIELD_LIVESTOCK_COMBINED"], state_fips ~ paste0("f_sk", skill_level), value.var = "rate")
sk_nf <- dcast(sk[sheet == "Statewide 655.120(b)(1)(ii)"], state_fips + soc_code ~ paste0("n_sk", skill_level), value.var = "rate")

states <- sort(unique(cr$state_fips))
stopifnot(all(states %in% fls$state_fips), all(states %in% sk_field$state_fips))
days <- data.table(d = seq(as.IDate("2019-10-01"), as.IDate("2026-06-30"), by = "day"))
days[, `:=`(m = month(d), fy = year(d) + (month(d) >= 10L))]
days[, rq := m <= 6L | m >= 10L]  # rq13d receipt window: 1 Oct - 30 Jun

# Field floor F_s(d): max(FLS AEWR in force [Kansas-order states: alternative start], state ag minimum)
# before 2025-10-02; max(H-2A skill-I (b)(1)(i) rate, state ag minimum) from 2025-10-02.
fd <- CJ(state_fips = states, d = days$d)
fd <- merge(fd, st[, .(state_fips, kansas = kansas_order_state)], by = "state_fips")
fd[, fls := NA_real_]
for (i in seq_len(nrow(fls))) {
  f <- fls[i]; sk_start <- if (is.na(f$alt)) f$s else f$alt
  fd[state_fips == f$state_fips & !kansas & d >= f$s & d < f$e, fls := f$aewr]
  fd[state_fips == f$state_fips & kansas & d >= sk_start & d < f$e, fls := f$aewr]
  if (!is.na(f$alt)) {  # Kansas group keeps the prior rate until the alternative start
    prev <- fls[state_fips == f$state_fips & e == f$s]
    if (nrow(prev)) fd[state_fips == f$state_fips & kansas & d >= f$s & d < f$alt, fls := prev$aewr[1]]
  }
}
fd[, cyear := year(d)]
fd <- merge(fd, mw, by = c("state_fips", "cyear"), all.x = TRUE)
fd[d >= as.IDate("2025-10-01"), mw := NA_real_]
for (sn in sort(unique(mw26$snap))) {
  fd <- merge(fd, mw26[snap == sn, .(state_fips, mw_new = mw)], by = "state_fips", all.x = TRUE)
  fd[d >= sn, mw := mw_new]; fd[, mw_new := NULL]
}
fd <- merge(fd, sk_field, by = "state_fips", all.x = TRUE)
stopifnot(!anyNA(fd[d < IFR]$fls), !anyNA(fd$mw), !anyNA(fd$f_sk1))
fd[, F := fifelse(d < IFR, pmax(fls, mw), pmax(f_sk1, mw))]
fd[, F_sk2 := fifelse(d < IFR, F, pmax(f_sk2, mw))]

# Non-field floor N_sg(d) for each (state, pre-rule SOC) in use.
#   before 2023-03-30: F (2010 method; one rate for all non-range jobs)
#   2023-03-30 to 2025-08-28: OEWS vintage in force at receipt (Kansas-group dates by worksite state).
#     Primary: max(OEWS_sg, F) (highest applicable AEWR; a non-field H-2A job is treated as also
#     carrying field duties, so a below-FLS OEWS rate does not lower its floor). Statutory variant
#     (sensitivity): max(OEWS_sg, state minimum).
#     2023-03-30 to 2023-04-29: linear ramp in logs (job orders submitted before the cutoff keep
#     the 2010 method; applications follow job orders by up to 30 days).
#     Teche Vermilion proxy units (Louisiana sugar cane): F from 2024-09-20.
#   2025-08-29 to 2025-10-01 (vacatur): F (inside FY2025 rfy; outside the rq13d window).
#   from 2025-10-02: max(H-2A skill-I (b)(1)(ii) rate_sg, F); variant skill II.
#   Unavailable OEWS hourly means (annual-only SOCs): N = F (no determinable occupation rate).
socs <- sort(unique(cr[field == FALSE, soc]))
nf_keys <- unique(cr[field == FALSE, .(state_fips, soc)])
rk <- rates[soc_code %in% socs, .(state_fips, soc = soc_code, oews_reference, r = aewr_hourly, rate_basis, s0, e0, s1, e1)]
nd <- merge(fd[, .(state_fips, d, kansas, F, F_sk2, mw)], nf_keys, by = "state_fips", allow.cartesian = TRUE)
nd[, oews := NA_real_]
for (v in unique(rk$oews_reference)) {
  x <- rk[oews_reference == v]
  nd <- merge(nd, x[, .(state_fips, soc, r, s0, e0, s1, e1)], by = c("state_fips", "soc"), all.x = TRUE)
  nd[!kansas & !is.na(s0) & d >= s0 & d < e0, oews := r]
  nd[kansas & !is.na(s1) & d >= s1 & d < e1, oews := r]
  nd[, c("r", "s0", "e0", "s1", "e1") := NULL]
}
nd <- merge(nd, sk_nf[, .(state_fips, soc = soc_code, n_sk1, n_sk2)], by = c("state_fips", "soc"), all.x = TRUE)
in_rule <- function(d) d >= CUTOFF & d < VACATUR
lam <- function(d) pmin(1, pmax(0, as.numeric(d - CUTOFF + 1L) / as.numeric(RAMP_END - CUTOFF + 1L)))
# rule_rate / ifr_rate are NA where no occupation rate is determinable (then N = base).
mk <- function(rule_rate, ifr_rate, base, teche = FALSE, ramp = TRUE) {
  lb <- log(base)
  tgt <- fifelse(is.na(rule_rate), lb, log(rule_rate))
  if (teche) tgt <- fifelse(nd$d >= TECHE, lb, tgt)
  l <- if (ramp) lam(nd$d) else as.numeric(nd$d >= CUTOFF)
  o <- lb + l * (tgt - lb)
  o <- fifelse(nd$d >= VACATUR & nd$d < IFR, lb, o)
  fifelse(nd$d >= IFR, fifelse(is.na(ifr_rate), lb, log(ifr_rate)), o)
}
nd[, lN := mk(pmax(oews, F), pmax(n_sk1, F), F)]
nd[, lN_teche := mk(pmax(oews, F), pmax(n_sk1, F), F, teche = TRUE)]
nd[, lN_stat := mk(pmax(oews, mw), pmax(n_sk1, mw), F)]
nd[, lN_sk2 := mk(pmax(oews, F_sk2), pmax(n_sk2, F_sk2), F_sk2)]
nd[, lN_noramp := mk(pmax(oews, F), pmax(n_sk1, F), F, ramp = FALSE)]
# Teche-proxy versions of the variants (each variant differs from the primary in one dimension)
nd[, lN_stat_t := mk(pmax(oews, mw), pmax(n_sk1, mw), F, teche = TRUE)]
nd[, lN_sk2_t := mk(pmax(oews, F_sk2), pmax(n_sk2, F_sk2), F_sk2, teche = TRUE)]
nd[, lN_noramp_t := mk(pmax(oews, F), pmax(n_sk1, F), F, teche = TRUE, ramp = FALSE)]
stopifnot(nd[d < CUTOFF, all(abs(lN - log(F)) < 1e-12)])
note("nf_state_soc_keys", nrow(nf_keys))
note("nf_keys_oews_unavailable_share", nd[d == as.IDate("2024-01-15"), mean(is.na(oews))])
note("nf_keys_ifr_missing_share", nd[d == as.IDate("2026-01-15"), mean(is.na(n_sk1))])

# Predetermined receipt-month weights: national FY2020-FY2022 certified hours by receipt month,
# per window (rfy: all 12 months; rq13d: Oct-Jun receipts decided by 30 Jun). Uniform within month.
cr[, rm := month(received)]
WIN <- list(rfy = list(fys = 2020:2025, rows = rep(TRUE, nrow(cr)), day = rep(TRUE, nrow(days))),
            rq13d = list(fys = 2020:2026, rows = cr$in_rq13d, day = days$rq))
fy_F <- fy_N <- list()
for (wn in names(WIN)) {
  om <- cr[WIN[[wn]]$rows, .(h = sum(cert_hours)), by = rm][, .(rm, w = h / sum(h))]
  dw <- merge(days[WIN[[wn]]$day & fy %in% WIN[[wn]]$fys], om, by.x = "m", by.y = "rm")
  dw[, wd := w / .N, by = .(fy, m)][, wd := wd / sum(wd), by = fy]
  note(paste0("receipt_month_weights_", wn), paste(sprintf("%d:%.3f", om[order(rm)]$rm, om[order(rm)]$w), collapse = " "))
  fy_F[[wn]] <- merge(fd[, .(state_fips, d, lF = log(F), lF_sk2 = log(F_sk2))], dw[, .(d, fy, wd)], by = "d")[,
    .(lF = sum(wd * lF), lF_sk2 = sum(wd * lF_sk2)), by = .(state_fips, fy)][, window := wn]
  vv <- c("lN", "lN_teche", "lN_stat", "lN_sk2", "lN_noramp", "lN_stat_t", "lN_sk2_t", "lN_noramp_t")
  fy_N[[wn]] <- merge(nd[, c("state_fips", "soc", "d", vv), with = FALSE], dw[, .(d, fy, wd)], by = "d")[,
    lapply(.SD, function(z) sum(wd * z)), by = .(state_fips, soc, fy), .SDcols = vv][, window := wn]
}
fy_F <- rbindlist(fy_F); fy_N <- rbindlist(fy_N)
write_parquet(merge(fd[, .(state_fips, d, F, F_sk2, fls, mw)], nd[, .(state_fips, d, soc, oews, n_sk1, lN)], by = c("state_fips", "d"), allow.cartesian = TRUE),
  file.path(out, "a3_floors_daily.parquet"))

# ---- Units (employer x worksite state) and pre-rule cells ----------------------------
cells <- cr[, .(hours = sum(cert_hours), cases = uniqueN(case_number)), by = .(employer_id_balanced, state_fips, soc, field, grp)]
units <- cr[, .(mu = sum(cert_hours) / length(PRE), mu_rq13d = sum(cert_hours[in_rq13d]) / length(PRE), hours_pre = sum(cert_hours)),
            by = .(employer_id_balanced, state_fips)]
nfh <- cells[field == FALSE, .(nf_hours = sum(hours)), by = .(employer_id_balanced, state_fips)]
units <- merge(units, nfh, by = c("employer_id_balanced", "state_fips"), all.x = TRUE)
units[, `:=`(nf_hours = fcoalesce(nf_hours, 0))][, s_nf := nf_hours / hours_pre]
dom <- cells[field == FALSE][order(-hours)][, .(dom_nf_soc = soc[1], dom_nf_grp = grp[1]), by = .(employer_id_balanced, state_fips)]
units <- merge(units, dom, by = c("employer_id_balanced", "state_fips"), all.x = TRUE)
units[, type := fifelse(s_nf >= 0.5, dom_nf_grp, "field")]
units[, `:=`(node = paste(state_fips, type), unit_id = paste(employer_id_balanced, state_fips, sep = "|"))]
# Teche Vermilion proxy: Louisiana units whose employer name or any pre-rule job title mentions sugar cane
cane <- cr[state_fips == "22" & grepl("cane|sugar", paste(employer_name, job_title), ignore.case = TRUE),
           unique(paste(employer_id_balanced, state_fips, sep = "|"))]
units[, teche := unit_id %in% cane]
note("teche_proxy_units", sum(units$teche)); note("teche_proxy_nf_hours_pre", units[teche == TRUE, sum(nf_hours)])
grp_sh <- dcast(cells[field == FALSE, .(h = sum(hours)), by = .(employer_id_balanced, state_fips, grp)],
                employer_id_balanced + state_fips ~ paste0("sh_", grp), value.var = "h", fill = 0)
units <- merge(units, grp_sh, by = c("employer_id_balanced", "state_fips"), all.x = TRUE)
shc <- grep("^sh_", names(units), value = TRUE)
for (v in shc) set(units, j = v, value = fcoalesce(units[[v]], 0) / units$hours_pre)
stopifnot(!anyDuplicated(units$unit_id), all(units$mu > 0))
write_parquet(units, file.path(out, "a3_units.parquet"))
write_parquet(cells, file.path(out, "a3_cells.parquet"))

# ---- Dose: unit x FY ---------------------------------------------------------------
# x_ut = s_field,u log F_st + sum_g s_ug log N_sgt (pre-rule hours shares by SOC)
cw <- cells[, .(employer_id_balanced, state_fips, soc, field, hours)]
cw <- merge(cw, units[, .(employer_id_balanced, state_fips, hours_pre, teche)], by = c("employer_id_balanced", "state_fips"))
cw[, sh := hours / hours_pre]
cfy <- cw[, .(fy = FYS), by = .(employer_id_balanced, state_fips, soc, field, sh, teche)]
cfy <- merge(cfy, fy_F, by = c("state_fips", "fy"), allow.cartesian = TRUE)
cfy <- merge(cfy, fy_N, by = c("state_fips", "soc", "fy", "window"), all.x = TRUE)
stopifnot(!anyNA(cfy[field == FALSE]$lN))
for (v in c("lN", "lN_teche", "lN_stat", "lN_noramp", "lN_stat_t", "lN_noramp_t")) cfy[field == TRUE, (v) := lF]
cfy[field == TRUE, `:=`(lN_sk2 = lF_sk2, lN_sk2_t = lF_sk2)]
cfy[, lN_primary := fifelse(teche, lN_teche, lN)]
dose <- cfy[, .(x = sum(sh * lN_primary), x_noteche = sum(sh * lN), x_stat = sum(sh * fifelse(teche, lN_stat_t, lN_stat)),
                x_sk2 = sum(sh * fifelse(teche, lN_sk2_t, lN_sk2)), x_noramp = sum(sh * fifelse(teche, lN_noramp_t, lN_noramp)),
                lF = lF[1], lF_sk2 = lF_sk2[1]),
            by = .(employer_id_balanced, state_fips, fy, window)]
dose[, `:=`(D = x - lF, unit_id = paste(employer_id_balanced, state_fips, sep = "|"))]
stopifnot(nrow(dose) == nrow(units) * (length(WIN$rfy$fys) + length(WIN$rq13d$fys)), !anyNA(dose$x))
write_parquet(dose, file.path(out, "a3_unit_dose.parquet"))

# Cell secondary: employer x state x {field, nonfield}; nonfield cell dose = within-cell share-weighted log N
cfy[, kind := fifelse(field, "field", "nonfield")]
cdose <- cfy[window == "rfy", .(x = sum(sh * lN_primary) / sum(sh), lF = lF[1]), by = .(employer_id_balanced, state_fips, kind, fy)]
cmu <- cw[, .(mu = sum(hours) / length(PRE)), by = .(employer_id_balanced, state_fips, kind = fifelse(field, "field", "nonfield"))]
cdose <- merge(cdose, cmu, by = c("employer_id_balanced", "state_fips", "kind"))
cdose[, cell_id := paste(employer_id_balanced, state_fips, kind, sep = "|")]
write_parquet(cdose, file.path(out, "a3_cell_dose.parquet"))

# ---- Pre-rule outcomes (FY2020-FY2022 only) -----------------------------------------
pre <- cr[, .(cert_hours = sum(cert_hours), cert_pos = sum(cert_pos, na.rm = TRUE), req_pos = sum(req_pos, na.rm = TRUE),
              cert_hours_rq13d = sum(cert_hours[in_rq13d]), req_pos_rq13d = sum(req_pos[in_rq13d], na.rm = TRUE)),
          by = .(employer_id_balanced, state_fips, fy = fy_receipt)]
pre_cell <- cr[, .(cert_hours = sum(cert_hours)), by = .(employer_id_balanced, state_fips, kind = fifelse(field, "field", "nonfield"), fy = fy_receipt)]
# pre-rule hourly offers (hours-weighted log), nonfield SOCs: for the predetermined pass-through bound
offers <- cr[field == FALSE & !is.na(hourly_wage) & hourly_wage > 0 & hourly_wage <= 100,
             .(lw = sum(cert_hours * log(hourly_wage)) / sum(cert_hours), h = sum(cert_hours)), by = .(employer_id_balanced, state_fips, soc, fy = fy_receipt)]
write_parquet(pre, file.path(out, "a3_pre_outcomes.parquet"))
write_parquet(pre_cell, file.path(out, "a3_pre_cell_outcomes.parquet"))
write_parquet(offers, file.path(out, "a3_pre_offers_nonfield.parquet"))

note("units", nrow(units)); note("units_with_nf", sum(units$s_nf > 0)); note("units_nf_majority", sum(units$s_nf >= 0.5))
note("employers", uniqueN(units$employer_id_balanced)); note("states", uniqueN(units$state_fips))
note("nf_cells_employer_state_soc", nrow(cells[field == FALSE])); note("nf_cells_hours_pre", cells[field == FALSE, sum(hours)])
note("nf_hours_share_pre", cells[, sum(hours[!field]) / sum(hours)])
fwrite(rbindlist(log_rows), file.path(out, "a3_build.csv"))
print(rbindlist(log_rows))
