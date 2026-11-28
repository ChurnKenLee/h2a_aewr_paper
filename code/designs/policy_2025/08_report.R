source("code/designs/policy_2025/helpers.R")
suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(tibble)
})
out <- id_output_dir(); cd_verify()
table <- function(d) {
  d <- as.data.frame(d)
  if (!nrow(d)) return("None.")
  for (name in names(d)) {
    if (is.numeric(d[[name]])) d[[name]] <- ifelse(is.na(d[[name]]),"—",formatC(d[[name]],digits=5,format="fg"))
    d[[name]] <- gsub("|","\\|",replace_na(as.character(d[[name]]),"—"),fixed=TRUE)
  }
  paste(c(paste0("| ",paste(names(d),collapse=" | ")," |"),paste0("| ",paste(rep("---",ncol(d)),collapse=" | ")," |"),
    apply(d,1,function(x)paste0("| ",paste(x,collapse=" | ")," |"))),collapse="\n")
}
read_result <- function(name) read_csv(file.path(out,name),show_col_types=FALSE)
w <- read_result("wage_estimates.csv")
q <- read_result("quantity_estimates.csv")
q_cut <- read_result("quantity_cutoff_audit.csv")
q_conv <- read_result("quantity_cost_conversion.csv")
m <- read_result("mechanism_estimates.csv")
m_desc <- read_result("mechanism_descriptives.csv")
e <- read_result("earnings_interim_estimates.csv")
pre <- read_result("earnings_pre_estimates.csv")
validation <- jsonlite::read_json(file.path(out,"validation.json"))
stopifnot(validation$status=="passed")
lines <- character()
put <- function(...) lines <<- c(lines,paste0(...),"")
put("# The 2025 H-2A policy shift")
put("Generated ",as.character(Sys.time()),". This standalone design uses the complete FY2026 H-2A disclosure (Q4 release) as its baseline since the November 21, 2026 update, with the November 25, 2026 unified minimum-wage vintage update (`history/vintage_update_2026_11_25.json`) restoring federal-floor agricultural minimum wages for GA/WY (2024-2026) and AL/LA/MS/SC/TN (2026) in the 48 reference-schedule envelope rows while leaving all other wage-offer and earnings specifications unchanged. Outcomes had already been inspected; this is not a new preregistration. The 2013 shutdown is a separate historical exploration.")
put("## Offered wages")
main <- filter(w,tier=="R7",tau==2026,specification %in% c("occupation_dw","filer_state_fixed_composition","within_filer_across_state"))
put(table(select(main,specification,weights,n,states,estimate,low,high)))
wp <- filter(main,specification=="occupation_dw",weights=="h0")
put(sprintf("The continuing filer–state–occupation estimate is %.3f dollars per statutory dollar, with a 95%% interval [%.3f, %.3f]. A larger statutory cut is associated with a larger offer reduction.",wp$estimate,wp$low,wp$high))
put("Offers are nominal hourly dollars. The main contrast is FY2025–FY2026, using decisions over the full fiscal year (October 1 through September 30) and baseline certified-hours weights. FY2026 cases received on or after August 3, 2026 fall under the second statewide schedule; the dose and reference rates remain those of October 2, 2025. Former AEWR-region effects absorb regional differences; state deletion jackknife intervals use t(G−1). R7 identifies filers, not worksite businesses or individual farms. Composition checks condition on occupations surviving in both periods.")
put("Effective job-level legal floors remain unresolved. The reference-schedule envelope cannot establish worker category, skill, job-order submission timing, prevailing wage or a market wage. The result does not by itself identify realized pay, employment, monopsony or welfare.")
put("### Linkage and pre-policy comparisons")
put(table(w %>% filter(specification=="occupation_dw",weights=="h0") %>% select(tier,tau,status,n,states,estimate,low,high)))
put("### Composition, selection and applicability")
put(table(read_result("wage_composition_coverage.csv")))
put(table(read_result("wage_schedule_envelope_audit.csv")))
put(table(w %>% filter(tier=="R7",tau==2026,grepl("^selection_|^heterogeneity_",specification)) %>% select(specification,weights,term,status,estimate,low,high,p_holm,low_simultaneous,high_simultaneous)))
put("## County H-2A quantity response (Track 1, FY2025–FY2026)")
put("Added on November 26, 2026 (`history/quantity_amendment_2026_11_26.json`) after prior FY2026 Q1–Q3 and Q4 outcomes had been inspected. Estimates cover all 2,921 positive-2011-farm-employment counties in the 15 multi-state former AEWR regions (46 states) with County and Former-Region × FY fixed effects and exact delete-one-state jackknife (CV3J, t(45)) inference. The national level shift is absorbed; only within-former-region dispersion in the October 2, 2025 statewide rate identifies the slope.")
q_main <- filter(q, window=="full_fy", specification %in% c("B_primary","B_primary_log_rate","B_primary_first_stage"))
put(table(select(q_main,specification,outcome,estimator,states,estimate,se_jackknife,p_jackknife,elasticity,elasticity_low_jk,elasticity_high_jk,mde_elasticity_jk)))
q_hp <- filter(q, window=="full_fy", specification=="B_primary", outcome=="hours", estimator=="PPML")
put(sprintf("On the complete FY2026 Q4 disclosure (full fiscal year October 1–September 30), the size-weighted PPML certified-hours statutory elasticity is %.3f with a 95%% state-jackknife interval [%.3f, %.3f] (MDE %.2f). This provides a credible one-season lower bound on elastic demand rather than a tightly pinned point estimate.",q_hp$elasticity,q_hp$elasticity_low_jk,q_hp$elasticity_high_jk,q_hp$mde_elasticity_jk))
put("### Cutoffs, Q1–Q3 benchmark, and pre-policy placebos (PPML certified hours)")
put(table(q %>% filter(outcome=="hours", estimator=="PPML", specification %in% c("B_primary","placebo_fy2024_fy2025")) %>% select(window,contrast,specification,states,estimate,se_jackknife,elasticity,elasticity_low_jk,elasticity_high_jk,mde_elasticity_jk)))
put("### FY2024–FY2026 window and legal-cutoff coverage audit")
put(table(q_cut))
put("### Post-hoc cost-elasticity conversion (central ratio 1.2401)")
put(table(q_conv %>% filter(cost_ratio_case=="central", outcome %in% c("hours","cert")) %>% select(window,outcome,pass_through_source,pi_pass_through,w_bar_offered,eps_statutory,eps_cost,eps_cost_low_jk,eps_cost_high_jk,mde_eps_cost_jk)))
put("## Mechanism heterogeneity (FY2025–FY2026)")
put("Added on November 26, 2026 (`history/mechanisms_amendment_2026_11_26.json`) as a post-inspection mechanism audit on cached local inputs (not a preregistration). Estimates decompose the Track 1 county quantity response across five mechanism panels on the same 2,921-county, 46-state, 15-former-region frame with County and Former-Region × FY fixed effects, exact delete-one-state jackknife (CV3J) inference, paired 46-state jackknife subgroup contrasts, and Holm step-down multiplicity correction across the 12 declared primary FY2025–FY2026 hypotheses.")
m_prim <- m %>% filter(in_primary_family) %>% select(panel,specification,estimator,states,estimate,se_jackknife,elasticity,elasticity_low_jk,elasticity_high_jk,mde_elasticity_jk,p_jackknife,p_holm_family)
put("### Primary 12-hypothesis mechanism family (Holm-adjusted)")
put(table(m_prim))
put("### Mechanism subgroup shares and CDL thresholds (FY2024–FY2026)")
put(table(m_desc %>% select(year,hours_full_fy,requested_hours,pipeline_approval_ratio,worksite_bus_hours_share,incumbent_1yr_hours_share,h2alc_hours_share,late_start_hours_share,perennial_high_hours_share,perennial_p75_cutoff)))
put("## Covered crop earnings: 2025Q4–2026Q1")
put("The primary interim outcome averages two year-over-year log changes in private crop-production (NAICS 111) average weekly earnings. Every county needs both current/prior pairs, positive disclosed earnings and employment of at least ten in all four observations. The 2022–2024 H-2A intensity interacts with the statutory cut, with intensity and state effects included, equal county weights, and state deletion jackknife inference. States require at least two eligible counties.")
put(table(e %>% filter(specification=="crop_primary",tau>=2025) %>% select(window,n,states,estimate,low,high,percent_at_S03_cut_minus4,percent_low,percent_high)))
ep <- filter(e,window=="interim_2025Q4_2026Q1",specification=="crop_primary")
put(sprintf("At exposure 0.30 and a −$4 cut, the primary estimate implies %.2f%% relative earnings change [%.2f%%, %.2f%%]. This is a model-implied relative difference, not the overall change in earnings.",ep$percent_at_S03_cut_minus4,ep$percent_low,ep$percent_high))
put("### Sensitivities and comparison industries")
put(table(e %>% filter(window=="interim_2025Q4_2026Q1") %>% select(specification,weights,status,n,states,estimate,low,high,percent_at_S03_cut_minus4,percent_low,percent_high)))
put("The employment-weighted, cash-dose and commuting-zone intervals include zero in this vintage. Removing the intensity control brings the point estimate near zero. Quarter-specific estimates suggest concentration in Q4 but do not establish a distinct timing pattern. These findings are suggestive and specification-sensitive; quiet placebos do not establish exogeneity.")
put("Weekly earnings combine pay rates, hours and workforce composition. Incomplete UI classification and sparse covered/excluded groups do not identify a general domestic-worker spillover. Animal production and all-private outcomes are diagnostics, not guaranteed unaffected controls. Differential state shocks and contemporary policy changes remain identification concerns.")
put("### Matched pre-policy windows")
put(table(e %>% filter(window=="matched_two_quarter_placebo") %>% select(tau,n,states,estimate,low,high)))
put("### Coverage and outstanding series")
put(table(read_result("earnings_interim_coverage.csv")))
put(table(read_result("earnings_interim_queue.csv")))
put("The original three-quarter target still requires 2026Q2 and verified publication. No missing quarter is imputed. The authorized interim analysis already inspected outcomes; later windows are updates, not independent confirmations. The historical full-window command and its release guard remain preserved in the exploration archive, outside this runner.")
put("## Reproduction and evidence")
put("```sh\n./scripts/run_policy_2025.sh\n```")
put("The runner reads cached inputs only. Its source-vintage guard stops on drift. [Design and commands](../../code/designs/policy_2025/README.md), [complete coefficient ledger](policy_2025_specifications.md), [provenance](policy_2025/provenance.json), and [artifact validation](policy_2025/validation.json) accompany this report. State deletions, coverage, floor and allocation audits are retained under `outputs/tables/policy_2025/`. Earlier exploration reports remain historical records.")
put("All ",nrow(w)," wage, ",nrow(q)," Track 1 quantity, ",nrow(m)," mechanism-heterogeneity, ",nrow(pre)," pre-policy earnings and ",nrow(e)," interim coefficient rows (",nrow(w)+nrow(q)+nrow(m)+nrow(pre)+nrow(e)," total) were compared with the reference ledgers recorded for this vintage, including sample sizes, statuses and inference, at absolute tolerance 1e−10. Independent wage WLS, quantity PPML, mechanism PPML, earnings OLS and the 12 interim checks also passed. Passing validation establishes computational consistency, not causal identification.")
writeLines(lines,file.path(out,"policy_2025.md"))
appendix <- c("# 2025 policy-shift specification ledger","", "Every declared fit is retained, including unavailable estimates. This ledger is generated from the same CSVs as the report.","",
  "## Wage offers","",table(w),"","## County quantity response (Track 1)","",table(q),"","## Mechanism heterogeneity","",table(m),"","## Pre-policy earnings","",table(pre),"","## Interim earnings","",table(e))
writeLines(appendix,file.path(out,"policy_2025_specifications.md"))

# ------------------------------------------------------------------------------
# Code-Generated Publication Tables & Figure for Manuscript Move 4
# ------------------------------------------------------------------------------
fmt_n <- function(x, d = 3L) if (is.finite(x)) sprintf(paste0("%.", d, "f"), x) else "---"
fmt_i <- function(x) if (is.finite(x)) formatC(as.integer(round(x)), format = "d", big.mark = ",") else "---"
stars <- function(p) if (!is.finite(p)) "" else if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""

# 1. Booktabs Table: table_policy_2025_wages.tex
pick_w <- function(tier_nm, tau_yr, spec_nm, wt_nm, term_nm = "dose") {
  r <- w %>% filter(tier == !!tier_nm, tau == !!tau_yr, specification == !!spec_nm, weights == !!wt_nm, term == !!term_nm)
  stopifnot(nrow(r) == 1L)
  r[1L, ]
}

fmt_wrow <- function(r, lbl, show_holm = FALSE) {
  wt_lbl <- if (r$weights == "h0") "Baseline hrs" else "Equal"
  ci_txt <- if (is.finite(r$low) && is.finite(r$high)) paste0("[", fmt_n(r$low, 3L), ", ", fmt_n(r$high, 3L), "]") else "---"
  lbl_full <- if (show_holm && is.finite(r$p_holm)) paste0(lbl, " (Holm $p=", fmt_n(r$p_holm, 3L), "$)") else lbl
  paste0(
    lbl_full, " & ",
    wt_lbl, " & ",
    fmt_n(r$estimate, 3L), stars(r$p), " & ",
    "[", fmt_n(r$se_jackknife, 3L), "] & ",
    "(", fmt_n(r$se_crv1, 3L), ") & ",
    ci_txt, " & ",
    fmt_i(r$n), " \\\\"
  )
}

tex_wages <- c(
  "\\begin{table}[!htbp]",
  "\\centering",
  "\\caption{2025 Statewide AEWR Rule: Offered-Wage Pass-Through, Linkage Stability, and Selection (FY2024--FY2026)}",
  "\\label{tab:policy_2025_wages}",
  "\\small",
  "\\begin{tabular}{lcccccc}",
  "\\toprule",
  "Specification & Weight & Pass-Through & JK SE & CRV1 SE & 95\\% JK CI & $N$ \\\\",
  "\\midrule",
  "\\multicolumn{7}{l}{\\textit{Panel A: Continuing-Filer Wage Pass-Through (R7 Linkage, FY2025--FY2026)}} \\\\",
  fmt_wrow(pick_w("R7", 2026L, "occupation_dw", "h0"), "Continuing filer--state--occupation (primary)"),
  fmt_wrow(pick_w("R7", 2026L, "occupation_dw", "equal"), "Continuing filer--state--occupation"),
  fmt_wrow(pick_w("R7", 2026L, "filer_state_fixed_composition", "h0"), "Continuing filer--state (fixed occ.\\ composition)"),
  fmt_wrow(pick_w("R7", 2026L, "filer_state_unrestricted", "h0"), "Continuing filer--state (unrestricted)"),
  fmt_wrow(pick_w("R7", 2026L, "within_filer_across_state", "h0"), "Within multi-state filer (filer + region FE)"),
  fmt_wrow(pick_w("R7", 2026L, "within_filer_across_state", "equal"), "Within multi-state filer (filer + region FE)"),
  fmt_wrow(pick_w("R7", 2026L, "receipt_aligned_single_state_common_SOC", "h0"), "Receipt-aligned, single-state, 5 common SOCs"),
  "\\midrule",
  "\\multicolumn{7}{l}{\\textit{Panel B: Linkage-Tier Stability \\& Pre-Policy Placebos (Filer--State--Occupation, Baseline Hours)}} \\\\",
  fmt_wrow(pick_w("R5", 2026L, "occupation_dw", "h0"), "R5 strict linkage tier (FY2025--FY2026)"),
  fmt_wrow(pick_w("R8", 2026L, "occupation_dw", "h0"), "R8 loose linkage tier (FY2025--FY2026)"),
  fmt_wrow(pick_w("FEIN", 2026L, "occupation_dw", "h0"), "FEIN linkage tier (FY2025--FY2026)"),
  fmt_wrow(pick_w("R7", 2025L, "occupation_dw", "h0"), "Pre-policy placebo (R7, FY2024--FY2025)"),
  fmt_wrow(pick_w("R7", 2024L, "occupation_dw", "h0"), "Pre-policy placebo (R7, FY2023--FY2024)"),
  "\\midrule",
  "\\multicolumn{7}{l}{\\textit{Panel C: Continuation Selection \\& Baseline Heterogeneity (R7, FY2025--FY2026)}} \\\\",
  fmt_wrow(pick_w("R7", 2026L, "selection_stay", "h0"), "Selection: stays in hourly-offer panel"),
  fmt_wrow(pick_w("R7", 2026L, "selection_any_cert", "h0"), "Selection: any FY2026 certified positions"),
  fmt_wrow(pick_w("R7", 2026L, "selection_baseline_premium", "equal", "dose:stay"), "Selection: baseline premium ($\\text{dose} \\times \\text{stay}$)"),
  fmt_wrow(pick_w("R7", 2026L, "selection_prechange", "equal", "dose:stay"), "Selection: prior wage change ($\\text{dose} \\times \\text{stay}$)"),
  fmt_wrow(pick_w("R7", 2026L, "heterogeneity_highpremium", "h0", "dose:highpremium"), "Heterogeneity: baseline premium $> \\$0.25$", show_holm = TRUE),
  fmt_wrow(pick_w("R7", 2026L, "heterogeneity_contractor", "h0", "dose:contractor"), "Heterogeneity: H-2ALC contractor", show_holm = TRUE),
  fmt_wrow(pick_w("R7", 2026L, "heterogeneity_large", "h0", "dose:large"), "Heterogeneity: above-median baseline hours", show_holm = TRUE),
  "\\bottomrule",
  "\\end{tabular}",
  "\\begin{minipage}{0.96\\linewidth}",
  "\\vspace{0.4em}\\footnotesize",
  paste0(
    "\\textit{Notes:} Complete FY2026 Q4 disclosure (decisions Oct 1--Sep 30) across all 46 states in the 15 multi-state former AEWR regions. ",
    "Dose is the October 2, 2025 U.S.\\ skill-I statewide AEWR minus the April 1, 2025 regional AEWR (nominal \\$/hr). ",
    "All regressions absorb former AEWR-region fixed effects; within-filer rows additionally absorb filer fixed effects. ",
    "Primary inference is exact delete-one-state jackknife (CV3J, $t(45)$); state-clustered CRV1 SEs appear in parentheses. ",
    "Panel C heterogeneity $p$-values are jointly adjusted across the three interaction tests via Holm step-down. ",
    "R7 identifies filing entities, not worksite businesses; job-level effective legal floors remain unresolved. ",
    "$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$ (unadjusted JK $p$)."
  ),
  "\\end{minipage}",
  "\\end{table}"
)

writeLines(tex_wages, file.path(out, "table_policy_2025_wages.tex"))

# 2. Booktabs Table: table_policy_2025_earnings.tex
pick_e <- function(win_nm, spec_nm, tau_yr = 2026L) {
  r <- e %>% filter(window == !!win_nm, specification == !!spec_nm, tau == !!tau_yr)
  stopifnot(nrow(r) == 1L)
  r[1L, ]
}

fmt_erow <- function(r, lbl) {
  wt_lbl <- if (r$weights == "emp_l") "Prior emp." else "Equal"
  if (r$status != "estimated") {
    return(paste0(
      lbl, " & ", wt_lbl, " & --- & --- & --- & --- & ",
      fmt_i(r$n), " (", fmt_i(r$states), ") \\\\"
    ))
  }
  ci_txt <- paste0("[", fmt_n(r$low, 4L), ", ", fmt_n(r$high, 4L), "]")
  pct_txt <- if (is.finite(r$percent_at_S03_cut_minus4)) paste0(sprintf("%+.2f", r$percent_at_S03_cut_minus4), "\\%") else "---"
  paste0(
    lbl, " & ",
    wt_lbl, " & ",
    fmt_n(r$estimate, 4L), stars(r$p), " & ",
    "[", fmt_n(r$se_jackknife, 4L), "] & ",
    ci_txt, " & ",
    pct_txt, " & ",
    fmt_i(r$n), " (", fmt_i(r$states), ") \\\\"
  )
}

plac_e <- e %>% filter(window == "matched_two_quarter_placebo", specification == "crop_primary")
plac_mean <- mean(plac_e$estimate, na.rm = TRUE)
plac_min <- min(plac_e$estimate, na.rm = TRUE)
plac_max <- max(plac_e$estimate, na.rm = TRUE)

tex_earnings <- c(
  "\\begin{table}[!htbp]",
  "\\centering",
  "\\caption{2025 Statewide AEWR Rule: Covered Agricultural Weekly Earnings in QCEW (Interim 2025Q4--2026Q1)}",
  "\\label{tab:policy_2025_earnings}",
  "\\small",
  "\\begin{tabular}{lcccccc}",
  "\\toprule",
  "Specification / Window & Weight & Slope $\\hat{\\beta}$ & JK SE & 95\\% JK CI & Implied \\% & Counties (States) \\\\",
  "\\midrule",
  "\\multicolumn{7}{l}{\\textit{Panel A: Primary Interim Crop Earnings (NAICS 111), Quarter Splits, and Lead}} \\\\",
  fmt_erow(pick_e("interim_2025Q4_2026Q1", "crop_primary"), "Primary two-quarter interim (2025Q4--2026Q1)"),
  fmt_erow(pick_e("quarter_2025Q4", "crop_primary"), "2025Q4 alone (all eligible counties)"),
  fmt_erow(pick_e("quarter_2025Q4_common_sample", "crop_primary"), "2025Q4 alone (common two-quarter sample)"),
  fmt_erow(pick_e("quarter_2026Q1", "crop_primary"), "2026Q1 alone (all eligible counties)"),
  fmt_erow(pick_e("quarter_2026Q1_common_sample", "crop_primary"), "2026Q1 alone (common two-quarter sample)"),
  fmt_erow(pick_e("prior_window_lead", "crop_primary", 2025L), "Prior-window lead (2024Q4--2025Q2)"),
  "\\midrule",
  "\\multicolumn{7}{l}{\\textit{Panel B: Sensitivities, UI-Coverage Splits, and Comparison Industries (2025Q4--2026Q1)}} \\\\",
  fmt_erow(pick_e("interim_2025Q4_2026Q1", "crop_employment_weighted"), "Crop (NAICS 111), prior-year employment weights"),
  fmt_erow(pick_e("interim_2025Q4_2026Q1", "crop_cash_dose"), "Crop (NAICS 111), H-2A skill-I cash dose"),
  fmt_erow(pick_e("interim_2025Q4_2026Q1", "crop_commuting_zone"), "Crop (NAICS 111), commuting-zone exposure"),
  fmt_erow(pick_e("interim_2025Q4_2026Q1", "crop_no_intensity_control"), "Crop (NAICS 111), omitting $S_c$ control"),
  fmt_erow(pick_e("interim_2025Q4_2026Q1", "crop_UI_ambiguous"), "Crop UI split: ambiguous/unclassified states"),
  fmt_erow(pick_e("interim_2025Q4_2026Q1", "crop_UI_excluded"), "Crop UI split: H-2A excluded states ($G=3$)"),
  fmt_erow(pick_e("interim_2025Q4_2026Q1", "crop_UI_covered"), "Crop UI split: H-2A covered states (insuff.\\ $G=2$)"),
  fmt_erow(pick_e("interim_2025Q4_2026Q1", "animal_primary"), "Comparison: animal production (NAICS 112)"),
  fmt_erow(pick_e("interim_2025Q4_2026Q1", "all_private_primary"), "Comparison: all private industries"),
  "\\bottomrule",
  "\\end{tabular}",
  "\\begin{minipage}{0.96\\linewidth}",
  "\\vspace{0.4em}\\footnotesize",
  paste0(
    "\\textit{Notes:} Exploratory interim evidence using cached QCEW single-file archives through 2026Q1 (2026Q2 pending verified publication). ",
    "Outcome is the county mean year-over-year log change in private average weekly earnings across declared quarters, requiring positive disclosed ",
    "wages and monthly employment $\\ge 10$ in all required periods and $\\ge 2$ eligible counties per state. ",
    "Regressions estimate $\\Delta \\ln w_c = \\beta (S_c \\times \\text{cut}_s) + \\gamma S_c + \\alpha_s + \\varepsilon_c$, where $S_c$ is frozen ",
    "2022--2024 H-2A intensity (winsorized at p99) and $\\text{cut}_s$ is the October 2, 2025 statutory U.S.\\ skill-I cut; positive $\\hat{\\beta}$ ",
    "indicates relative earnings declined more where intensity and statutory cuts were larger. ",
    "Implied \\% evaluates $\\exp(\\hat{\\beta} \\times 0.30 \\times (-4)) - 1$. ",
    "Primary inference is state leave-one-out jackknife ($t(G-1)$). Across the 9 matched Q4--Q1 pre-policy placebo windows (2016--2024), ",
    sprintf("primary crop slopes average %.4f (range [%.4f, %.4f]). ", plac_mean, plac_min, plac_max),
    "Weekly earnings combine wage rates, hours, and workforce composition and do not separately identify domestic-worker wage spillovers. ",
    "$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$ (JK $p$)."
  ),
  "\\end{minipage}",
  "\\end{table}"
)

writeLines(tex_earnings, file.path(out, "table_policy_2025_earnings.tex"))

# 3. Two-Panel Wage Figure: fig_policy_2025_wages.{pdf,png}
fips_abbr <- c(
  "01"="AL","02"="AK","04"="AZ","05"="AR","06"="CA","08"="CO","09"="CT","10"="DE",
  "11"="DC","12"="FL","13"="GA","15"="HI","16"="ID","17"="IL","18"="IN","19"="IA",
  "20"="KS","21"="KY","22"="LA","23"="ME","24"="MD","25"="MA","26"="MI","27"="MN",
  "28"="MS","29"="MO","30"="MT","31"="NE","32"="NV","33"="NH","34"="NJ","35"="NM",
  "36"="NY","37"="NC","38"="ND","39"="OH","40"="OK","41"="OR","42"="PA","44"="RI",
  "45"="SC","46"="SD","47"="TN","48"="TX","49"="UT","50"="VT","51"="VA","53"="WA",
  "54"="WV","55"="WI","56"="WY"
)

p_w <- read_parquet(file.path(out, "wage_filer_state_occupation.parquet")) %>%
  filter(tier == "R7", soc != "unknown", fiscal_year %in% c(2025L, 2026L)) %>%
  mutate(state_fips = state_fips(state_fips), aewr_region_id = aewr_region_id(aewr_region_id))
st_dose <- read_csv(
  file.path(out, "wage_state_dose.csv"),
  col_types = cols(state_fips = "c", aewr_region_id = "c", .default = col_guess())
) %>%
  transmute(state_fips = state_fips(state_fips), aewr_region_id = aewr_region_id(aewr_region_id), dose = dose1_us)

cells_26 <- inner_join(
  p_w %>% filter(fiscal_year == 2025L) %>% select(filer, state_fips, aewr_region_id, soc, wage0 = wage, h0 = h),
  p_w %>% filter(fiscal_year == 2026L) %>% select(filer, state_fips, aewr_region_id, soc, wage1 = wage, h1 = h),
  by = c("filer", "state_fips", "aewr_region_id", "soc"),
  relationship = "one-to-one"
) %>%
  mutate(dw = wage1 - wage0)

st_scatter <- cells_26 %>%
  group_by(state_fips, aewr_region_id) %>%
  summarise(
    state_dw = cd_mean(dw, h0),
    state_h0 = sum(h0, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  inner_join(st_dose, by = c("state_fips", "aewr_region_id")) %>%
  mutate(abbr = unname(fips_abbr[state_fips]))

# Residualize on former-region FE with baseline-hours weights (FWL exact slope match)
m_x_res <- stats::lm(dose ~ factor(aewr_region_id), data = st_scatter, weights = state_h0)
m_y_res <- stats::lm(state_dw ~ factor(aewr_region_id), data = st_scatter, weights = state_h0)
mean_dose_w <- cd_mean(st_scatter$dose, st_scatter$state_h0)
mean_dw_w <- cd_mean(st_scatter$state_dw, st_scatter$state_h0)

st_scatter <- st_scatter %>%
  mutate(
    dose_plot = stats::resid(m_x_res) + mean_dose_w,
    dw_plot = stats::resid(m_y_res) + mean_dw_w
  )

fwl_fit <- stats::lm(dw_plot ~ dose_plot, data = st_scatter, weights = state_h0)
stopifnot(abs(unname(stats::coef(fwl_fit)[["dose_plot"]]) - wp$estimate[[1]]) < 1e-8)

st_scatter <- st_scatter %>%
  mutate(
    fitted_dw = stats::fitted(fwl_fit),
    rank_h0 = rank(-state_h0, ties.method = "first"),
    rank_ext = rank(-abs(dose_plot - mean_dose_w), ties.method = "first"),
    show_lbl = rank_h0 <= 12L | rank_ext <= 6L,
    lbl_vjust = ifelse(dw_plot >= fitted_dw, -0.75, 1.45)
  )

slope_annot <- sprintf(
  "WLS slope = %.3f\n95%% JK CI [%.3f, %.3f]\n46 states, 15 former regions",
  wp$estimate[[1]], wp$low[[1]], wp$high[[1]]
)

p_wA <- ggplot(st_scatter, aes(x = dose_plot, y = dw_plot)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#A0AEC0", linewidth = 0.4) +
  geom_abline(
    intercept = unname(stats::coef(fwl_fit)[[1]]),
    slope = unname(stats::coef(fwl_fit)[[2]]),
    color = "#1B4F72", linewidth = 0.85
  ) +
  geom_point(aes(size = state_h0 / 1e6), color = "#1B4F72", fill = "#5DADE2", shape = 21, alpha = 0.85, stroke = 0.6) +
  geom_text(
    data = st_scatter %>% filter(show_lbl),
    aes(label = abbr, vjust = lbl_vjust),
    size = 2.6, color = "#1A202C", fontface = "bold"
  ) +
  annotate(
    "label",
    x = min(st_scatter$dose_plot) + 0.05,
    y = max(st_scatter$dw_plot) - 0.05,
    label = slope_annot,
    hjust = 0, vjust = 1, size = 2.8, fill = "white", alpha = 0.9, label.size = 0.25
  ) +
  scale_size_continuous(range = c(1.8, 7.5), guide = "none") +
  labs(
    title = "A. State Offered-Wage Change vs. Statutory Cut",
    x = "Oct. 2, 2025 statutory cut ($/hr, region-FE residualized)",
    y = "Continuing R7 mean offer change ($/hr, residualized)"
  ) +
  theme_minimal(base_size = 9.8) +
  theme(
    plot.title = element_text(face = "bold", size = 10.2),
    panel.grid.minor = element_blank(),
    plot.margin = margin(6, 10, 6, 8)
  )

# Panel B: Pre-policy placebos & linkage-tier stability
b_rows <- bind_rows(
  pick_w("R7", 2024L, "occupation_dw", "h0") %>% mutate(lbl = "Placebo: R7 (FY2023-24)", grp = "placebo"),
  pick_w("R7", 2025L, "occupation_dw", "h0") %>% mutate(lbl = "Placebo: R7 (FY2024-25)", grp = "placebo"),
  pick_w("R7", 2026L, "occupation_dw", "h0") %>% mutate(lbl = "Primary: R7 Filer-State-Occ", grp = "main"),
  pick_w("R7", 2026L, "filer_state_fixed_composition", "h0") %>% mutate(lbl = "R7 Fixed Composition", grp = "main"),
  pick_w("R7", 2026L, "within_filer_across_state", "h0") %>% mutate(lbl = "R7 Within Multi-State Filer", grp = "main"),
  pick_w("R7", 2026L, "receipt_aligned_single_state_common_SOC", "h0") %>% mutate(lbl = "R7 Receipt-Aligned Common SOC", grp = "main"),
  pick_w("R5", 2026L, "occupation_dw", "h0") %>% mutate(lbl = "Linkage Tier: R5 Strict", grp = "tier"),
  pick_w("R8", 2026L, "occupation_dw", "h0") %>% mutate(lbl = "Linkage Tier: R8 Loose", grp = "tier"),
  pick_w("FEIN", 2026L, "occupation_dw", "h0") %>% mutate(lbl = "Linkage Tier: FEIN", grp = "tier")
) %>%
  mutate(lbl = factor(lbl, levels = rev(lbl)))

p_wB <- ggplot(b_rows, aes(x = estimate, y = lbl, color = grp)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "#4A5568", linewidth = 0.45) +
  geom_vline(xintercept = 1, linetype = "dotted", color = "#A0AEC0", linewidth = 0.45) +
  geom_errorbarh(aes(xmin = low, xmax = high), height = 0.24, linewidth = 0.7) +
  geom_point(size = 2.4) +
  scale_color_manual(
    values = c("placebo" = "#7F8C8D", "main" = "#1B4F72", "tier" = "#117A65"),
    guide = "none"
  ) +
  labs(
    title = "B. Pre-Policy Placebos & Linkage-Tier Stability",
    x = "Pass-through ($/hr per statutory $/hr; 95% CV3J CI)",
    y = NULL
  ) +
  theme_minimal(base_size = 9.8) +
  theme(
    plot.title = element_text(face = "bold", size = 10.2),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "#EDF2F7", linewidth = 0.35),
    axis.text.y = element_text(size = 8.6, color = "#1A202C"),
    plot.margin = margin(6, 8, 6, 6)
  )

draw_wage_figure <- function() {
  grid::grid.newpage()
  grid::pushViewport(grid::viewport(layout = grid::grid.layout(
    nrow = 3, ncol = 2,
    heights = grid::unit(c(0.55, 1, 0.55), c("in", "null", "in")),
    widths = grid::unit(c(1.02, 1.08), c("null", "null"))
  )))
  grid::pushViewport(grid::viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  grid::grid.text(
    "2025 Statewide AEWR Rule: First-Stage Offered-Wage Pass-Through (Complete FY2026 Q4)",
    x = 0.02, y = 0.68, just = "left",
    gp = grid::gpar(fontsize = 11.8, fontface = "bold", col = "#1A202C")
  )
  grid::grid.text(
    "Continuing H-2A filers across 46 multi-state-region states (baseline certified-hours weighted, former AEWR-region FE)",
    x = 0.02, y = 0.24, just = "left",
    gp = grid::gpar(fontsize = 9.2, col = "#4A5568")
  )
  grid::popViewport()

  print(p_wA, vp = grid::viewport(layout.pos.row = 2, layout.pos.col = 1))
  print(p_wB, vp = grid::viewport(layout.pos.row = 2, layout.pos.col = 2))

  grid::pushViewport(grid::viewport(layout.pos.row = 3, layout.pos.col = 1:2))
  grid::grid.text(
    paste0(
      "Notes: Panel A plots state hours-weighted continuing R7 offered-wage changes (FY2025-FY2026) against Oct. 2, 2025 statutory cuts,\n",
      "residualized on former-region FE (circle area = baseline hours; FWL slope matches 0.623). Panel B shows 95% CV3J intervals (dashed = 0, dotted = 1)."
    ),
    x = 0.02, y = 0.55, just = "left",
    gp = grid::gpar(fontsize = 8.0, col = "#4A5568")
  )
  grid::popViewport()
  grid::popViewport()
}

grDevices::pdf(file.path(out, "fig_policy_2025_wages.pdf"), width = 11.2, height = 5.4)
draw_wage_figure()
grDevices::dev.off()

grDevices::png(file.path(out, "fig_policy_2025_wages.png"), width = 11.2, height = 5.4, units = "in", res = 300)
draw_wage_figure()
grDevices::dev.off()
