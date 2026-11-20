# F1 blind review (Phase 6), POST-OUTCOME check r02: the floor classification.
#  (a) Recomputes F_app and candidate floors by hand (direct reads of the source
#      tables, not f1_classify) for a random sample of 40 FY2026 M26 field orders and
#      compares classes.
#  (b) Skill assignment: FY2026 field non-piece class / matched-floor mix by experience
#      months (does exp >= 3 behave like a legal skill-II floor?).
#  (c) Where below_skill2 / below_floor offers sit relative to US/H-2A skill rates.
#  (d) Tolerance: 'above' offers within $0.05-$0.25 of a candidate floor.
#  (e) 'above' offers that equal a (b)(1)(ii) SOC rate of ANY SOC in the state
#      (combined-duty orders; unseen floor), and FY2026 shared values (S3) vs. round numbers.
#  (f) at_floor offers censored above F_app (c_floor > F_app + 0.05): P2 counts them as unbound at F_app.
#  (g) NY / OR sub-state areas.
# Usage: R_LIBS_USER=... Rscript --vanilla explorations/identification_search/F1/review/r02_classification.R "$S"
source("explorations/identification_search/F1/03_est_helpers.R")
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "F1", "review"); dir.create(out, showWarnings = FALSE)
o <- read_parquet(file.path(S, "F1", "f1_orders.parquet"))
doses <- f1_doses(S); ST48 <- unique(doses$state_fips)
n26 <- o %>% filter(fy == 2026, new_rule, !piece, soc_grp == "field", state_fips %in% ST48)
M26 <- n26 %>% filter(cls %in% c("above", "at_floor", "at_old_aewr"))

# ---- (a) hand check -----------------------------------------------------------------
sk <- read_parquet(path_int("aewr_occupation_skill_rates.parquet"))
st_rates <- sk %>% filter(trimws(source_sheet) == "Statewide 655.120(b)(1)(i)", as.Date(effective_start) == as.Date("2025-10-02"))
mwt <- read.csv("explorations/identification_search/A1/sources/state_ag_minwage_fy2025_2026.csv", colClasses = c(state_fips = "character"))
ref <- read_parquet(path_int("aewr_rate_schedule.parquet")) %>% filter(occupation_scope %in% c("nonrange_reference", "fls_common_soc_reference"), worker_category == "covered_workers")
set.seed(20261105)
smp <- bind_rows(M26 %>% filter(cls == "above") %>% slice_sample(n = 15), M26 %>% filter(cls == "at_floor") %>% slice_sample(n = 15),
  M26 %>% filter(cls == "at_old_aewr") %>% slice_sample(n = 5), n26 %>% filter(cls == "below_skill2") %>% slice_sample(n = 5))
hand <- smp %>% rowwise() %>% mutate(
  us1 = as.numeric(st_rates$hourly_aewr[st_rates$state_fips == state_fips & st_rates$worker_category == "US" & st_rates$skill_level == 1][1]),
  us2 = as.numeric(st_rates$hourly_aewr[st_rates$state_fips == state_fips & st_rates$worker_category == "US" & st_rates$skill_level == 2][1]),
  h1 = as.numeric(st_rates$hourly_aewr[st_rates$state_fips == state_fips & st_rates$worker_category == "H2A" & st_rates$skill_level == 1][1]),
  h2 = as.numeric(st_rates$hourly_aewr[st_rates$state_fips == state_fips & st_rates$worker_category == "H2A" & st_rates$skill_level == 2][1]),
  mw_vals = paste(sort(unique(mwt$h2a_state_floor[mwt$state_fips == state_fips & as.Date(mwt$snapshot_date) <= coalesce(end, received) &
    (is.na(as.Date(mwt$valid_through)) | as.Date(mwt$valid_through) >= received)])), collapse = "/"),
  old_vals = paste(sort(unique(ref$aewr[ref$state_fips == state_fips & as.Date(ref$effective_start) <= as.Date("2025-09-30") &
    (is.na(ref$effective_end_exclusive) | as.Date(ref$effective_end_exclusive) > as.Date("2024-10-01"))])), collapse = "/")) %>% ungroup() %>%
  mutate(hand_F = pmax(ifelse(skill == "II", us2, us1), coalesce(mw_max, 7.25)),
    near = mapply(function(w, a, b, c, d, m) { v <- c(a, b, c, d, as.numeric(strsplit(m, "/")[[1]]), 7.25); any(abs(w - v) <= 0.05, na.rm = TRUE) }, wage, us1, us2, h1, h2, mw_vals),
    near_old = mapply(function(w, m) any(abs(w - as.numeric(strsplit(m, "/")[[1]])) <= 0.05), wage, old_vals),
    hand_cls = case_when(near ~ "at_floor", near_old ~ "at_old_aewr", wage > hand_F + 0.05 ~ "above", skill == "II" & wage > us1 + 0.05 ~ "below_skill2", TRUE ~ "below_floor"),
    agree = hand_cls == cls | (cls == "at_floor" & hand_cls != "at_floor" & grepl("soc_rate", floor_types)))
print(hand %>% select(state_fips, mw_area, received, wage, exp_months, skill, us1, us2, h1, h2, mw_vals, F_app, hand_F, floor_types, cls, hand_cls, agree), n = 50, width = 300)
message("hand-check agreement: ", sum(hand$agree), " / ", nrow(hand), "; F_app equal: ", sum(abs(hand$F_app - hand$hand_F) < 1e-9))
write_csv(hand %>% select(-any_of("old_rates")), file.path(out, "r02a_hand_check.csv"))

# ---- (b) skill / experience ---------------------------------------------------------------
n26 <- n26 %>% mutate(at_us1 = abs(wage - us_s1_ref) <= TOL) %>% left_join(f1_state_skill_rates(), by = "state_fips") %>%
  mutate(at_us2 = abs(wage - US_s2) <= TOL, at_h2a = abs(wage - H2A_s1) <= TOL | abs(wage - H2A_s2) <= TOL,
    btw = wage > US_s1 + TOL & wage < US_s2 - TOL, below_us1 = wage < US_s1 - TOL)
n26 <- n26 %>% mutate(across(c(at_us1, at_us2, at_h2a, btw, below_us1), ~ coalesce(.x, FALSE)))
message("FY2026 field non-piece orders with missing statewide skill rates: ", sum(is.na(n26$US_s2)), "; missing exp_months: ", sum(is.na(n26$exp_months)))
ex <- n26 %>% group_by(exp_class) %>% summarise(orders = n(), sh_at_US_s1 = sum(hours[at_us1]) / sum(hours),
  sh_at_US_s2 = sum(hours[at_us2]) / sum(hours), sh_at_H2A = sum(hours[at_h2a]) / sum(hours), sh_between_s1_s2 = sum(hours[btw]) / sum(hours),
  sh_below_US_s1 = sum(hours[below_us1]) / sum(hours), sh_above = sum(hours[cls == "above"]) / sum(hours),
  sh_below_skill2 = sum(hours[cls == "below_skill2"]) / sum(hours), sh_below_floor = sum(hours[cls == "below_floor"]) / sum(hours), hours = sum(hours), .groups = "drop")
print(ex, width = 300); write_csv(ex, file.path(out, "r02b_skill_by_experience.csv"))

# ---- (c) below classes ----------------------------------------------------------------------
bl <- n26 %>% filter(cls %in% c("below_skill2", "below_floor")) %>% mutate(
  pos = case_when(wage < H2A_s1 - TOL ~ "below H-2A skill I", wage < US_s1 - TOL ~ "between H-2A s1 and US s1", wage <= US_s1 + TOL ~ "at US s1 (?)",
    wage < US_s2 - TOL ~ "between US s1 and US s2", TRUE ~ "above US s2 (MW floor binding)"),
  mwfl = !is.na(mw_max) & mw_max > US_s1 + TOL) %>%
  group_by(cls, skill, pos, mwfl) %>% summarise(orders = n(), hours = sum(hours), .groups = "drop")
print(bl, n = 40); write_csv(bl, file.path(out, "r02c_below_positions.csv"))

# ---- (d) tolerance: near-misses among 'above' ---------------------------------------------
fl <- f1_state_skill_rates(); so <- f1_soc_skill_rates(); mw <- f1_minwage()
cand_state <- bind_rows(fl %>% pivot_longer(-state_fips, values_to = "v") %>% select(state_fips, v),
  mw %>% select(state_fips, v = mw), tibble(state_fips = ST48, v = 7.25)) %>% filter(is.finite(v)) %>% distinct()
ab <- M26 %>% filter(cls == "above")
dist <- ab %>% select(oid, state_fips, wage, hours) %>% inner_join(cand_state, by = "state_fips", relationship = "many-to-many") %>%
  group_by(oid, hours) %>% summarise(dmin = min(abs(wage - v)), .groups = "drop")
nm <- tibble(window = c("(0.05, 0.10]", "(0.10, 0.25]", "(0.25, 0.50]", "> 0.50"),
  hours_share_of_above = c(sum(dist$hours[dist$dmin <= 0.10]), sum(dist$hours[dist$dmin > 0.10 & dist$dmin <= 0.25]),
    sum(dist$hours[dist$dmin > 0.25 & dist$dmin <= 0.5]), sum(dist$hours[dist$dmin > 0.5])) / sum(dist$hours))
print(nm); write_csv(nm, file.path(out, "r02d_near_miss.csv"))

# ---- (e) above offers at ANY (b)(1)(ii) SOC rate of the state; S3 shared values ------------------
soc_all <- sk %>% filter(trimws(source_sheet) == "Statewide 655.120(b)(1)(ii)", as.Date(effective_start) == as.Date("2025-10-02")) %>%
  transmute(state_fips, soc7 = substr(soc_code, 1, 7), v = as.numeric(hourly_aewr), wc = worker_category, sl = skill_level) %>% filter(is.finite(v))
hit_any <- ab %>% select(oid, state_fips, wage, hours) %>% inner_join(soc_all %>% filter(wc == "US"), by = "state_fips", relationship = "many-to-many") %>%
  filter(abs(wage - v) <= 0.005) %>% group_by(oid, hours) %>% summarise(n_soc = n_distinct(soc7), socs = paste(head(unique(soc7), 4), collapse = ","), .groups = "drop")
# chance benchmark: same test with wage + 0.37 (shifted off any exact match)
hit_shift <- ab %>% transmute(oid, state_fips, wage = wage + 0.37, hours) %>% inner_join(soc_all %>% filter(wc == "US"), by = "state_fips", relationship = "many-to-many") %>%
  filter(abs(wage - v) <= 0.005) %>% distinct(oid, hours)
ee <- tibble(stat = c("above hours share exactly (to the cent) at some US (b)(1)(ii) SOC rate of the state", "same, wages shifted by +$0.37 (chance benchmark)"),
  value = c(sum(hit_any$hours) / sum(ab$hours), sum(hit_shift$hours) / sum(ab$hours)))
print(ee)
# S3 shared values: cents digits and roundness
k <- o %>% filter(cls == "above", !piece) %>% mutate(wr = round(wage, 2)) %>% group_by(fy, state_fips, soc_grp, wr) %>% summarise(n_emp = n_distinct(employer_id), .groups = "drop")
sv <- bind_rows(
  M26 %>% filter(cls == "above") %>% mutate(wr = round(wage, 2)) %>% left_join(k, by = c("fy", "state_fips", "soc_grp", "wr")) %>% mutate(set = "FY2026 M26 above"),
  o %>% filter(fy == 2025, !piece, soc_grp == "field", state_fips %in% ST48, cls == "above") %>% mutate(wr = round(wage, 2)) %>%
    left_join(k, by = c("fy", "state_fips", "soc_grp", "wr")) %>% mutate(set = "FY2025 M25 above")) %>%
  filter(is.finite(wr)) %>% mutate(shared = coalesce(n_emp, 0L) >= 3, round_dollar = abs(wr - round(wr)) < 0.005, round_quarter = abs(wr * 4 - round(wr * 4)) < 0.005)
svs <- sv %>% group_by(set, shared) %>% summarise(orders = n(), sh_round_dollar = sum(hours[round_dollar]) / sum(hours),
  sh_round_quarter = sum(hours[round_quarter]) / sum(hours), hours = sum(hours), .groups = "drop")
print(svs); write_csv(bind_rows(ee %>% mutate(set = "e"), svs %>% mutate(stat = "S3 roundness")), file.path(out, "r02e_unseen_floors.csv"))
top <- sv %>% filter(shared) %>% group_by(set, state_fips, wr) %>% summarise(orders = n(), employers = n_distinct(employer_id), hours = sum(hours), .groups = "drop") %>%
  group_by(set) %>% mutate(share_of_shared_hours = hours / sum(hours)) %>% arrange(set, desc(hours)) %>% slice_head(n = 12) %>%
  left_join(doses %>% filter(mw_area == "statewide" | TRUE) %>% group_by(state_fips) %>% summarise(a24 = first(a24), a25 = first(a25), US_s1 = first(US_s1), US_s2 = first(US_s2), .groups = "drop"), by = "state_fips")
top <- top %>% mutate(wr_exact = sprintf("%.2f", wr))
# match shared values to: prior-year AEWRs of the state (any reference AEWR 2019-2025), any field-SOC (b)(1)(ii) rate (US or H-2A, skill I/II)
allref <- ref %>% transmute(state_fips, v = as.numeric(aewr), start = as.Date(effective_start)) %>% filter(start >= as.Date("2019-01-01"))
fsoc <- soc_all %>% filter(soc7 %in% FIELD_SOC)
top <- top %>% rowwise() %>% mutate(
  match_ref_aewr = paste(format(sort(unique(allref$start[allref$state_fips == state_fips & abs(allref$v - wr) <= 0.005]))), collapse = ","),
  match_field_soc_rate = paste(unique(paste0(fsoc$soc7, ":", fsoc$wc, fsoc$sl)[fsoc$state_fips == state_fips & abs(fsoc$v - wr) <= 0.005]), collapse = ","),
  n_states_same_value = sum(abs(unique(sv$wr[sv$set == set & sv$shared]) - wr) < 0.005)) %>% ungroup()
print(top %>% select(set, state_fips, wr_exact, orders, employers, share_of_shared_hours, a24, a25, US_s1, US_s2, match_ref_aewr, match_field_soc_rate), n = 30, width = 250)
xs <- sv %>% filter(shared) %>% group_by(set, wr = round(wr, 2)) %>% summarise(states = n_distinct(state_fips), hours = sum(hours), .groups = "drop") %>%
  group_by(set) %>% mutate(share = hours / sum(hours)) %>% arrange(set, desc(hours)) %>% slice_head(n = 6)
print(xs); write_csv(top, file.path(out, "r02e_top_shared_values.csv"))

# ---- (f) at_floor censored above F_app ------------------------------------------------------
ff <- M26 %>% filter(cls == "at_floor") %>% mutate(above_fapp = c_floor > F_app + TOL) %>% group_by(above_fapp) %>%
  summarise(orders = n(), hours = sum(hours), .groups = "drop") %>% mutate(share_of_M26_hours = hours / sum(M26$hours))
ff2 <- M26 %>% filter(cls == "at_floor", c_floor > F_app + TOL) %>% count(floor_types, skill, wt = hours, name = "hours") %>% arrange(desc(hours))
print(ff); print(ff2, n = 15); write_csv(ff, file.path(out, "r02f_atfloor_above_fapp.csv"))

# ---- (g) NY / OR -----------------------------------------------------------------------------
ny <- n26 %>% filter(state_fips %in% c("36", "41")) %>% group_by(state_fips, mw_area, cls) %>% summarise(orders = n(), hours = sum(hours), mw_max = paste(unique(mw_max), collapse = "/"), .groups = "drop")
print(ny, n = 30)
sub <- f1_county_area(); print(sub %>% filter(substr(county_fips, 1, 2) %in% c("36", "41")) %>% count(substr(county_fips, 1, 2), area_id))
write_csv(ny, file.path(out, "r02g_ny_or.csv"))
