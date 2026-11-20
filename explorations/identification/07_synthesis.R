# Track 5: what each design can rule out, in elasticity units, side by side.
# Reads the CSVs written by 01, 02, 05, 06 (and 04 when present) in the
# scratch dir, plus the bite scratch dir's Anderson-Rubin sets. Estimands
# differ across rows and are never pooled.
source("explorations/identification/id_helpers.R")
out_dir <- id_output_dir()
bite_dir <- commandArgs(trailingOnly = TRUE)[2]
if (is.na(bite_dir)) bite_dir <- "/tmp/h2a-bite-cross-design"
rd <- function(f) read_csv(file.path(out_dir, f), show_col_types = FALSE)

cu <- rd("common_unit_all.csv"); pr <- rd("precision_estimates.csv")
t2 <- rd("track2_estimates.csv"); t2t <- rd("track2_relprice_translation.csv"); t3 <- rd("track3_headline.csv")
row <- function(design, outcome, estimand, assumption, cluster, estimate, se, wild_p, el, lo, hi, ci_type, mde_el, source)
  tibble(design, outcome, estimand, assumption, cluster, estimate, se, wild_p, elasticity = el,
    ci_low = lo, ci_high = hi, ci_type, mde_elasticity = mde_el, source)

out <- list()
# Region-year designs, average effect (plain panel, IV with AR sets).
ar <- read_csv(file.path(bite_dir, "iv_ar_sets.csv"), show_col_types = FALSE) %>%
  filter(control_set == "bea", model == "baseline_iv", cluster == "aewr_region_id", outcome %in% c("cert", "hours"))
for (nm in c("cert", "hours")) {
  b <- pr %>% filter(spec == "1_baseline_ratio", outcome == nm, cluster == "aewr_region_id")
  o <- b %>% filter(estimator == "OLS"); i <- b %>% filter(estimator == "IV"); a <- ar %>% filter(outcome == nm)
  out[[length(out) + 1]] <- row("Plain panel OLS (county+year FE, ratio)", nm, "average effect", "region-year AEWR deviations exogenous given controls",
    "region CRV1 (17)", o$estimate, o$se, o$wild_p, o$elasticity, o$elasticity_low, o$elasticity_high, "CRV1 t(16)", o$mde_elasticity, "02_precision.R")
  out[[length(out) + 1]] <- row("Panel IV column 4", nm, "average effect (IV)", "dissimilar-donor exclusion",
    "region CRV1 (17)", i$estimate, i$se, i$wild_p, i$elasticity, id_elasticity(a$ar_low, i$mean_price, i$mean_y),
    id_elasticity(a$ar_high, i$mean_price, i$mean_y), paste("Anderson-Rubin", a$ar_type), i$mde_elasticity, "02_precision.R + bite iv_ar_sets.csv")
  for (sp in c("2_ppml_counts", "3_ratio_farm_emp_weighted", "4_region_year_aggregate")) for (est in unique(pr$estimator[pr$spec == sp])) {
    x <- pr %>% filter(spec == sp, estimator == est, outcome == nm, cluster %in% c("aewr_region_id", "aewr_region_id pairs bootstrap (199)"))
    if (!nrow(x)) next
    out[[length(out) + 1]] <- row(paste0("Phase 0 ", sp, " ", est), nm,
      switch(sp, `2_ppml_counts` = "average semi-elasticity of total use (size-weighted)",
        `3_ratio_farm_emp_weighted` = "average effect, farm-employment-weighted",
        `4_region_year_aggregate` = "region-year elasticity of total use"),
      if (grepl("IV", est)) "dissimilar-donor exclusion" else "region-year AEWR deviations exogenous",
      x$cluster, x$estimate, x$se, x$wild_p, x$elasticity, x$elasticity_low, x$elasticity_high, "CRV1 / pairs bootstrap, t(16)", x$mde_elasticity, "02_precision.R")
  }
  for (win in c("2011-2019", "2020-2025")) {
    x <- pr %>% filter(spec == paste0("5_employer_panel_", win), outcome == nm, cluster == "aewr_region_id")
    out[[length(out) + 1]] <- row(paste("Phase 0 employer panel", win), paste0("log1p_", nm), "average semi-elasticity, baseline employers",
      "region-year AEWR deviations exogenous", "region CRV1 (17)", x$estimate, x$se, x$wild_p, x$elasticity,
      x$elasticity_low, x$elasticity_high, "CRV1 t(16)", x$mde_elasticity, "02_precision.R")
  }
}
# Bite and border rows (common-unit file).
for (k in seq_len(nrow(cu))) {
  x <- cu[k, ]
  keep <- (grepl("^panel_bite", x$design) & x$cluster == "aewr_region_id" & x$control_set == "bea") |
    (grepl("^border_ring_h1_(screened_)?disjoint_(baseline|heterogeneity)$", x$design)) |
    grepl("^employer_design4", x$design)
  if (!isTRUE(keep)) next
  out[[length(out) + 1]] <- row(x$design, x$outcome, x$estimand,
    if (grepl("average_conversion", x$design)) "beta_c proportional to frozen gap; differential trends valid" else if (grepl("bite", x$design)) "parallel trends across frozen exposure" else if (grepl("border", x$design)) "border continuity; 25 edges" else "bound/unbound comparability; fails cohort replication",
    x$cluster, x$estimate, x$se, x$wild_p, x$elasticity, x$elasticity_low, x$elasticity_high, "CRV1", x$mde_elasticity, "01_common_unit.R")
}
# Track 2.
for (k in which(t2$cluster == "state_fips" & t2$spec %in% c("s1_level__ag_real", "s1_level__gen_real",
  "s3_pairs__statewide_h0_unscreened_within_region__ag_real") & t2$term %in% c("ag_real", "gen_real"))) {
  x <- t2[k, ]
  out[[length(out) + 1]] <- row(paste("Track 2", x$spec), x$outcome, "elasticity w.r.t. domestic floor (real ag/general minimum)",
    "state minimum-wage timing exogenous given region x year FE", "state CRV1", x$estimate, x$se, x$wild_p,
    x$elasticity, x$elasticity_low, x$elasticity_high, "CRV1", x$mde_elasticity, "05_track2_minwage.R")
}
for (k in which(t2t$cluster == "state_fips" & t2t$spec %in% c("s1_level__ag_real", "s1_level__gen_real"))) {
  x <- t2t[k, ]
  out[[length(out) + 1]] <- row(paste("Track 2 translated", x$spec), x$outcome, "implied AEWR elasticity (relative-price elasticity)",
    "only AEWR minus floor matters, and a $1 floor rise raises competing domestic cost by $1", "state CRV1",
    x$implied_aewr_slope, x$se, NA, x$implied_aewr_elasticity, x$implied_low, x$implied_high, "CRV1", x$implied_mde_elasticity, "05_track2_minwage.R")
}
# Track 3.
for (k in which(t3$spec == "base" & t3$cluster == "aewr_region_id" & t3$term == "A")) {
  x <- t3[k, ]
  G <- if ("regions" %in% names(x)) x$regions else 17
  lo <- x$elasticity - qt(.975, G - 1) * x$se * x$elasticity / x$estimate
  hi <- x$elasticity + qt(.975, G - 1) * x$se * x$elasticity / x$estimate
  out[[length(out) + 1]] <- row(paste("Track 3 multi-region", x$calendar, x$window), x$outcome,
    "within-employer regional reallocation elasticity", "regional demand shocks orthogonal to AEWR deviations",
    "region CRV1", x$estimate, x$se, x$wild_p, x$elasticity, pmin(lo, hi), pmax(lo, hi), "CRV1", x$mde_elasticity, "06_track3_multiregion.R")
}
# Track 1 post-policy (08; amendment 1). Jackknife intervals, t(45).
if (file.exists(file.path(out_dir, "track1_post_estimates.csv"))) {
  t1 <- rd("track1_post_estimates.csv") %>%
    filter(spec %in% c("B_primary", "placebo_fy2024_fy2025", "S2_state_trends", "S1_base_static_x_us_skill_1"), outcome != "y_wage_q")
  for (k in seq_len(nrow(t1))) {
    x <- t1[k, ]
    out[[length(out) + 1]] <- row(paste("Track 1", x$spec, x$estimator), x$outcome,
      if (x$spec == "placebo_fy2024_fy2025") "placebo (FY2024 vs FY2025, FY2026 dose)" else "ITT of the statutory statewide rate within former region, FY2026 (size-weighted if PPML)",
      "no dose-correlated FY2026 shocks; drift limited by the one-year contrast", "state jackknife (46)",
      x$estimate, x$se_jackknife, x$wild_p_state, x$elasticity, x$elasticity_low_jk, x$elasticity_high_jk,
      "jackknife t(45)", x$mde_elasticity_jk, "08_track1_estimate.R")
  }
}
# Track 2 under the corrected, dated coverage coding (05 with the fixed c01 output).
if (file.exists(file.path(out_dir, "track2c_estimates.csv"))) {
  t2c <- rd("track2c_estimates.csv"); t2ct <- rd("track2c_relprice_translation.csv")
  for (k in which(t2c$cluster == "state_fips" & t2c$spec == "s1_level__ag_real" & t2c$term == "ag_real")) {
    x <- t2c[k, ]
    out[[length(out) + 1]] <- row("Track 2 corrected coverage s1_level__ag_real", x$outcome, "elasticity w.r.t. domestic floor (real applicable ag minimum)",
      "state minimum-wage timing exogenous given region x year FE", "state CRV1", x$estimate, x$se, x$wild_p,
      x$elasticity, x$elasticity_low, x$elasticity_high, "CRV1", x$mde_elasticity, "05_track2_minwage.R (corrected)")
  }
  for (k in which(t2ct$cluster == "state_fips" & t2ct$spec == "s1_level__ag_real")) {
    x <- t2ct[k, ]
    out[[length(out) + 1]] <- row("Track 2 corrected coverage translated s1_level__ag_real", x$outcome, "implied AEWR elasticity (relative-price elasticity)",
      "only AEWR minus floor matters, and a $1 floor rise raises competing domestic cost by $1", "state CRV1",
      x$implied_aewr_slope, x$se, NA, x$implied_aewr_elasticity, x$implied_low, x$implied_high, "CRV1", x$implied_mde_elasticity, "05_track2_minwage.R (corrected)")
  }
}
# Track 4 (09): elasticity with respect to hourly cost, not the AEWR.
if (file.exists(file.path(out_dir, "track4_estimates.csv"))) {
  t4 <- rd("track4_estimates.csv") %>% filter(spec %in% c("i_region_year_FE", "iii_ppml_region_year_FE"), term == "D")
  for (k in seq_len(nrow(t4))) {
    x <- t4[k, ]
    out[[length(out) + 1]] <- row(paste("Track 4 overtime", x$spec), x$outcome, "elasticity w.r.t. hourly cost (overtime exposure); fails lead check",
      "threshold timing exogenous; 6 treated states", "state CRV1", x$estimate, x$se_state_crv1, x$wild_p_state,
      x$elasticity_cost, x$elasticity_low, x$elasticity_high, "CRV1", x$mde_elasticity, "09_track4_overtime.R")
  }
}
syn <- bind_rows(out) %>% mutate(
  excludes_le_minus_2 = ci_low > -2, excludes_le_minus_1 = ci_low > -1, excludes_le_minus_0.5 = ci_low > -0.5,
  excludes_ge_plus_1 = ci_high < 1)
id_write(syn, out_dir, "synthesis.csv")
print(as.data.frame(syn %>% select(design, outcome, elasticity, ci_low, ci_high, mde_elasticity, excludes_le_minus_1)), digits = 3)
