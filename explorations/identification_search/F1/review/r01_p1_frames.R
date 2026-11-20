# F1 blind review (Phase 6), POST-OUTCOME check r01: the P1 incumbent frame.
# Reproduces P1 on the declared 285-unit INC frame and the 352-unit design frame,
# decomposes the 67 units that the declared frame drops (FY2026 classes, experience,
# dose, wage change), and re-estimates theta on alternative frames that do not
# condition on the FY2026 offer class. Also: FY2026 class mix of incumbents (how many
# 'above every floor' FY2025 units sit at a floor in FY2026), leave-one-state-out
# influence, and a Holm adjustment across the declared primary hypotheses.
#
# Usage: R_LIBS_USER=... Rscript --vanilla explorations/identification_search/F1/review/r01_p1_frames.R "$S"
# Writes $S/F1/review/r01_*.csv. Reads only $S/F1/f1_orders.parquet and $S/A1/a1_state_doses.csv.
source("explorations/identification_search/F1/03_est_helpers.R")
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "F1", "review"); dir.create(out, showWarnings = FALSE)
o <- read_parquet(file.path(S, "F1", "f1_orders.parquet"))
doses <- f1_doses(S); ST48 <- unique(doses$state_fips)
q <- o %>% filter(state_fips %in% ST48, !piece, soc_grp == "field", !is.na(employer_id))

base <- q %>% filter(fy == 2025) %>% group_by(employer_id, state_fips) %>%
  summarise(h = sum(hours), sh_above = sum(hours[cls == "above"]) / sum(hours), any_unknown = any(cls == "floor_unknown"),
    mw_area = mw_area[which.max(hours)], w_base = sum(wage * hours) / sum(hours),
    sh_exp3 = sum(hours[!is.na(exp_months) & exp_months >= 3]) / sum(hours), .groups = "drop") %>%
  filter(!any_unknown, sh_above >= 0.9999)
n26 <- q %>% filter(fy == 2026, new_rule) %>% group_by(employer_id, state_fips) %>%
  summarise(w_all = sum(wage * hours) / sum(hours),
    w_m26 = if (any(cls %in% c("above", "at_floor", "at_old_aewr"))) sum((wage * hours)[cls %in% c("above", "at_floor", "at_old_aewr")]) / sum(hours[cls %in% c("above", "at_floor", "at_old_aewr")]) else NA_real_,
    h26 = sum(hours), sh_at_floor26 = sum(hours[cls == "at_floor"]) / sum(hours), sh_above26 = sum(hours[cls == "above"]) / sum(hours),
    sh_old26 = sum(hours[cls == "at_old_aewr"]) / sum(hours), sh_b2_26 = sum(hours[cls == "below_skill2"]) / sum(hours),
    sh_bf26 = sum(hours[cls == "below_floor"]) / sum(hours), any_m26 = any(cls %in% c("above", "at_floor", "at_old_aewr")),
    mixed = any(cls %in% c("above", "at_floor", "at_old_aewr")) & any(cls %in% c("below_skill2", "below_floor")),
    exp26 = sum(hours[!is.na(exp_months) & exp_months >= 3]) / sum(hours),
    fapp26 = sum(F_app * hours) / sum(hours), .groups = "drop")
d <- base %>% inner_join(n26, by = c("employer_id", "state_fips")) %>% left_join(doses, by = c("state_fips", "mw_area")) %>%
  mutate(dF = eff_cut_u, dw_all = w_all - w_base, dw_m26 = w_m26 - w_base)
stopifnot(!anyNA(d$dF))
message("design frame units: ", nrow(d), "; declared INC (any M26 order): ", sum(d$any_m26), "; mixed units: ", sum(d$mixed))

fits <- bind_rows(
  p1_fit(d$dF[d$any_m26], d$dw_m26[d$any_m26], d$h[d$any_m26], d$state_fips[d$any_m26]) %>% mutate(frame = "declared INC (285): presence and wage on M26 orders"),
  p1_fit(d$dF, d$dw_all, d$h, d$state_fips) %>% mutate(frame = "design frame (352): presence and wage on all new-rule orders"),
  p1_fit(d$dF[d$any_m26], d$dw_all[d$any_m26], d$h[d$any_m26], d$state_fips[d$any_m26]) %>% mutate(frame = "285 units, wage on all new-rule orders (order-level selection removed)"),
  p1_fit(d$dF[!d$mixed & d$any_m26], d$dw_m26[!d$mixed & d$any_m26], d$h[!d$mixed & d$any_m26], d$state_fips[!d$mixed & d$any_m26]) %>% mutate(frame = "285 minus mixed units"))
# Skill-floor dose on the 352 frame
d <- d %>% mutate(dF_skill = ifelse(sh_exp3 > 0.5, pmax(US_s2, mw_20260101) - e25, eff_cut_u))
fits <- bind_rows(fits,
  p1_fit(d$dF_skill, d$dw_all, d$h, d$state_fips) %>% mutate(frame = "design frame (352), skill-floor dose"),
  p1_fit(d$dF, d$dw_all, rep(1, nrow(d)), d$state_fips) %>% mutate(frame = "design frame (352), unweighted"))
print(fits %>% select(frame, estimate, intercept, se, ci_low, ci_high, p_theta0, p_theta1, n_units, n_states), width = 250)
write_csv(fits, file.path(out, "r01_p1_frames.csv"))

# Who are the 67?
cmp <- d %>% mutate(grp = ifelse(any_m26, "kept (any M26 order)", "dropped (only below_skill2 / below_floor)")) %>% group_by(grp) %>%
  summarise(units = n(), states = n_distinct(state_fips), hours25 = sum(h), mean_dF = sum(h * dF) / sum(h), sd_dF = sqrt(sum(h * (dF - sum(h * dF) / sum(h))^2) / sum(h)),
    mean_dw = sum(h * dw_all) / sum(h), mean_w25 = sum(h * w_base) / sum(h), mean_w26 = sum(h * w_all) / sum(h), mean_fapp26 = sum(h * fapp26) / sum(h),
    exp3_share25 = sum(h * sh_exp3) / sum(h), exp3_share26 = sum(h * exp26) / sum(h),
    sh_below_skill2 = sum(h * sh_b2_26) / sum(h), sh_below_floor = sum(h * sh_bf26) / sum(h),
    sh_at_floor26 = sum(h * sh_at_floor26) / sum(h), sh_above26 = sum(h * sh_above26) / sum(h), sh_old26 = sum(h * sh_old26) / sum(h), .groups = "drop")
print(cmp, width = 300); write_csv(cmp, file.path(out, "r01_dropped_vs_kept.csv"))
# Correlation of dropping with dose (unit level, hours-weighted)
message(sprintf("hours-weighted corr(dropped, dF) on 352: %.3f", cov.wt(cbind(!d$any_m26, d$dF), wt = d$h, cor = TRUE)$cor[1, 2]))

# FY2026 position of declared incumbents: how many 'above every floor' FY2025 units are at a floor in FY2026
pos <- d %>% filter(any_m26) %>% mutate(pos26 = case_when(sh_at_floor26 >= 0.9999 ~ "all FY2026 hours at a floor",
  sh_above26 >= 0.9999 ~ "all FY2026 hours above every floor", sh_old26 >= 0.9999 ~ "all at old AEWR", TRUE ~ "mixed")) %>%
  group_by(pos26) %>% summarise(units = n(), mean_dw = sum(h * dw_m26) / sum(h), mean_dF = sum(h * dF) / sum(h),
    mean_premium25 = sum(h * (w_base - a25)) / sum(h), hours_share = sum(h), .groups = "drop") %>% mutate(hours_share = hours_share / sum(hours_share))
print(pos); write_csv(pos, file.path(out, "r01_inc_fy2026_position.csv"))

# Leave-one-state-out estimates on both frames
loso <- function(dd, y, lab) {
  g <- sort(unique(dd$state_fips))
  tibble(frame = lab, state_out = g, theta = vapply(g, function(k) { i <- dd$state_fips != k; wls(dd$dF[i], y[i], dd$h[i])[2] }, numeric(1)),
    units_out = as.integer(table(dd$state_fips)[g]), hours_share_out = as.numeric(tapply(dd$h, dd$state_fips, sum)[g] / sum(dd$h)))
}
dk <- d %>% filter(any_m26)
L <- bind_rows(loso(dk, dk$dw_m26, "declared 285"), loso(d, d$dw_all, "design 352"))
print(L %>% group_by(frame) %>% summarise(min = min(theta), max = max(theta), state_min = state_out[which.min(theta)], state_max = state_out[which.max(theta)]))
write_csv(L, file.path(out, "r01_loso.csv"))

# State-level picture: hours-weighted state mean dw vs dF (does the slope come from a few states?)
stl <- d %>% group_by(state_fips) %>% summarise(units = n(), units_kept = sum(any_m26), dF = sum(h * dF) / sum(h),
  dw_all = sum(h * dw_all) / sum(h), dw_kept = sum((h * dw_m26)[any_m26]) / sum(h[any_m26]), h = sum(h), .groups = "drop") %>% arrange(dF)
print(stl, n = 40); write_csv(stl, file.path(out, "r01_state_means.csv"))

# Holm across the declared primary hypotheses that gate the reading (P1 H0a, H0b; P3 V1)
p3 <- read_csv(file.path(S, "F1", "f1_p3.csv"), show_col_types = FALSE) %>% filter(grepl("^primary", scope), grid == "m=1")
p1 <- read_csv(file.path(S, "F1", "f1_p1.csv"), show_col_types = FALSE) %>% filter(item == "P1_theta")
# P3 p-value: bootstrap-normal approximation from the reported SE (percentile draws not saved)
p_p3 <- 2 * pnorm(-abs(p3$estimate / p3$se))
ps <- c(P1_H0a_theta0 = p1$p_theta0, P1_H0b_theta1 = p1$p_theta1, P3_V1 = p_p3)
holm <- tibble(hyp = names(ps), p = ps, p_holm = p.adjust(ps, "holm"), p_holm_P1_only = c(p.adjust(ps[1:2], "holm"), NA))
print(holm); write_csv(holm, file.path(out, "r01_holm.csv"))
