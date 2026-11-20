# Identification search, Phase 7 synthesis, step 2: assemble
# explorations/identification_search/ledger.csv, one row per strategy
# (every catalogue entry, every triage proposal, every slotted and new
# strategy). Columns are the declared ledger_row_fields of
# shared/declarations.json, in order; list fields are joined with "; " and
# object fields are written as compact JSON.
#
# Sources:
# - strategies with a ledger_row.json (A1, F1, A2, A3, N1, D2, A6): that file,
#   with the synthesis filling reviewer / review_verdict / review_path from
#   <id>/review.md, p_adjusted from $S/synthesis/multiplicity.csv and the final
#   pooling_eligible from $S/synthesis/pooling_decision.csv. The ledger_row.json
#   files themselves are not modified.
# - triage-only entries: $S/triage/merged.json, credibility.json, power.json.
#
# Usage (repository root, after 01_pooling_multiplicity.R):
#   Rscript --vanilla explorations/identification_search/synthesis/02_ledger.R "$S"
suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1)
S <- normalizePath(args[1])
root <- "explorations/identification_search"
syn <- file.path(S, "synthesis")
today <- "2026-11-06"

decl <- fromJSON(file.path(root, "shared/declarations.json"), simplifyVector = FALSE)
fields <- names(decl$ledger$ledger_row_fields)
merged <- fromJSON(file.path(S, "triage/merged.json"), simplifyVector = FALSE)
cred <- fromJSON(file.path(S, "triage/credibility.json"), simplifyVector = FALSE)
powr <- fromJSON(file.path(S, "triage/power.json"), simplifyVector = FALSE)
pool <- fread(file.path(syn, "pooling_decision.csv"))
mult <- fread(file.path(syn, "multiplicity.csv"))

by_id <- function(lst, id) {
  hit <- Filter(function(x) identical(x$id, id), lst)
  if (length(hit)) hit[[1]] else NULL
}
cell <- function(x) {
  # one CSV cell from a JSON value
  if (is.null(x) || length(x) == 0) return(NA_character_)
  if (is.list(x) && !is.null(names(x))) return(as.character(toJSON(x, auto_unbox = TRUE, null = "null", digits = NA)))
  if (is.list(x)) return(paste(vapply(x, function(e) if (is.null(e)) "null" else as.character(e), ""), collapse = "; "))
  if (length(x) > 1) return(paste(x, collapse = "; "))
  as.character(x)
}
blank_row <- function() setNames(as.list(rep(NA_character_, length(fields))), fields)

triage_mde <- function(id_cred, id_power) {
  c1 <- if (!is.null(id_cred)) by_id(c(cred$scores, cred$new_strategies), id_cred)$rough_mde
  p1 <- if (!is.null(id_power)) by_id(c(powr$scores, powr$new_strategies), id_power)$rough_mde
  paste(c(if (!is.null(c1)) paste0("triage rough MDE, credibility pass: ", c1),
          if (!is.null(p1)) paste0("power pass: ", p1)), collapse = " | ")
}
pass_scores <- function(lst, id) {
  x <- by_id(lst, id)
  if (is.null(x)) return(NULL)
  k <- c("exogeneity", "estimand_distance", "support", "independence", "speed")
  if (all(k %in% names(x))) x[k] else list(total = x$score_total)
}
merged_rubric <- function(id) {
  x <- by_id(merged$seed_notes, id)
  if (!is.null(x)) return(x$merged_rubric)
  list(credibility = pass_scores(cred$scores, id), power = pass_scores(powr$scores, id))
}
reason_from <- function(lst, id) {
  x <- by_id(lst, id)
  if (is.null(x)) NA_character_ else x$reason
}

rows <- list()

# ---- strategies with ledger_row.json -----------------------------------------
for (id in c("A1", "F1", "A2", "A3", "N1", "D2", "A6")) {
  lr <- fromJSON(file.path(root, id, "ledger_row.json"), simplifyVector = FALSE)
  r <- blank_row()
  for (f in intersect(fields, names(lr))) r[[f]] <- cell(lr[[f]])
  rev <- file.path(root, id, "review.md")
  if (file.exists(rev)) {
    v <- grep("^## Verdict", readLines(rev, warn = FALSE), value = TRUE)[1]
    r$review_verdict <- tolower(gsub("^## Verdict: \\*\\*|\\*\\*.*$", "", v))
    r$review_path <- rev
    r$reviewer <- "fresh Claude agent (Claude Opus 5.5), blind Phase 6 review; no Codex cross-review"
    r$phase_reached <- "review"
    r$status <- "reviewed"
  }
  if (id %in% c("D2", "A6")) {
    r$phase_reached <- "triage"
    r$status <- "not_run"
    r$notes <- paste0("ledger_row.json records phase_reached 'not_run', kill_phase 'pre-design', decided by ",
                      lr$decided_by, " on ", lr$date, ".")
    r$rubric_scores <- cell(merged_rubric(id))
    r$proposer <- "catalogue"
  }
  m <- mult[strategy_id == id]
  if (nrow(m)) {
    r$p_adjusted <- paste(sprintf("%s: Holm %s%s", sub(":.*$", "", m$hypothesis),
                                  sprintf("%.4g", m$p_holm_extended_family),
                                  ifelse(is.na(m$p_holm_declared_family), "",
                                         sprintf(" (declared epsilon_cost family: Holm %s)",
                                                 sprintf("%.4g", m$p_holm_declared_family)))),
                          collapse = "; ")
    if (id == "F1") {
      r$p_unadjusted <- paste(sprintf("%s: %s", sub(":.*$", "", m$hypothesis),
                                      sprintf("%.4g", m$p_unadjusted)), collapse = "; ")
      r$p_size_valid <- r$p_unadjusted
    }
  }
  pd <- pool[strategy_id == id]
  if (nrow(pd)) {
    r$pooling_eligible <- tolower(as.character(pd$pooling_eligible))
    syn_reason <- paste0("synthesis (declared inclusion rule): fails ", pd$failed_criteria)
    r$pooling_exclusion_reason <- if (id == "A1")
      paste0(syn_reason, ". Design-stage ledger_row.json recorded pooling_eligible = true (design MDE 0.82 at pi = 1) before review and conversion.")
    else paste(c(r$pooling_exclusion_reason, syn_reason)[!is.na(c(r$pooling_exclusion_reason, syn_reason))], collapse = " | ")
  }
  r$updated <- today
  rows[[id]] <- r
}

# ---- triage-only entries ------------------------------------------------------
tri <- function(id, family, title, status, kill_phase, kill_reason, proposer = "catalogue",
                rubric = merged_rubric(id), mde = triage_mde(id, id), notes = NA_character_,
                assumption_family = NA_character_) {
  r <- blank_row()
  r$strategy_id <- id; r$family <- family; r$title <- title; r$proposer <- proposer
  r$implementer <- "none (triage only)"; r$phase_reached <- "triage"; r$status <- status
  r$kill_phase <- kill_phase; r$kill_reason <- kill_reason; r$rubric_scores <- cell(rubric)
  r$design_mde_by_dgp <- mde; r$assumption_family <- assumption_family
  r$pooling_eligible <- "false"
  r$pooling_exclusion_reason <- if (status %in% c("killed", "not_pursued")) "killed or not pursued at triage" else
    "not a standalone estimator"
  r$falsification_result <- "not_run"; r$split_used <- "false"; r$notes <- notes; r$updated <- today
  r
}
ks <- function(id) reason_from(merged$kill_seeds, id)
np <- function(id) reason_from(merged$not_pursued, id)

rows$A4 <- tri("A4", "A", "Range herder phase-in, 2015-2018", "killed", "triage", ks("A4"), assumption_family = "range_herder_phase_in")
rows$A5 <- tri("A5", "A", "Annual-update timing shocks (2021 delay, 2024 Kansas split)", "not_pursued", "triage", np("A5"))
rows$B1 <- tri("B1", "B", "FLS sampling-composition shocks", "killed", "triage", ks("B1"), assumption_family = "region_year_aewr")
rows$B2 <- tri("B2", "B", "AEWR minus contemporaneous regional wage", "killed", "triage", ks("B2"), assumption_family = "region_year_aewr")
rows$B3 <- tri("B3", "B", "Within-region composition shift-share (leave-own-state-out)", "killed", "triage", ks("B3"), assumption_family = "region_year_aewr")
rows$C1 <- tri("C1", "C", "Housing cost (HUD FMR)", "killed", "triage", ks("C1"), assumption_family = "housing_cost")
rows$C2 <- tri("C2", "C", "Transportation (distance x diesel)", "killed", "triage", ks("C2"), assumption_family = "transport_cost")
rows$C3 <- tri("C3", "C", "Visa and consular processing shocks", "not_pursued", "triage", np("C3"))
rows$C4 <- tri("C4", "C", "Mandated meal and subsistence rates", "not_pursued", "triage", np("C4"))
rows$D1 <- tri("D1", "D", "Revealed binding share x statutory change", "merged", NA_character_,
               "Not killed: folded into A1 as its within-state dose (both triage passes; declarations count it as A1-family). Its outcome is A1's.",
               notes = by_id(merged$seed_notes, "D1")$status, assumption_family = "statutory_2025_statewide")
for (e in 1:5) {
  rows[[paste0("E", e)]] <- tri(paste0("E", e), "E",
    c("Job-order or monthly panels of applications and requested positions",
      "Extensive margin: employer entry and exit, any use",
      "Stable employer x worksite x job-type cells",
      "Aggregate state or region totals",
      "Variance reduction: pre-trend adjustment, synthetic-control or SDID weights")[e],
    "layer", NA_character_,
    "Not a standalone strategy: an outcome or margin layer declared inside each registry (plan catalogue E; triage merge).",
    rubric = list(note = "not scored separately by the triage passes"), mde = NA_character_)
}
rows$F2 <- tri("F2", "F", "Adoption model (discrete choice of H-2A use; A-event holdout)", "deferred", NA_character_,
               paste("Not slotted in execution step 5; kept for the plan's failure branch at synthesis (triage merge).",
                     "Not run: the A-family holdout events it needed (A2, A3) were killed at design, and F1 (its market-wage input) failed review."),
               notes = by_id(merged$seed_notes, "F2")$status)
rows$G <- tri("G", "G", "Geographic contrasts (borders, OES-area borders)", "not_pursued", "triage", np("G"))

prop <- function(id, src, title, status, reason, where = NA_character_) {
  pass <- if (src == "cred") cred$new_strategies else powr$new_strategies
  x <- by_id(pass, id)
  sid <- paste0(if (src == "cred") "cred_" else "power_", id)
  r <- tri(sid, "new", title, status, if (status %in% c("killed", "not_pursued")) "triage" else NA_character_, reason,
           proposer = if (src == "cred") "triage_credibility" else "triage_power",
           rubric = list(total = x$score_total),
           mde = paste0("triage rough MDE (", if (src == "cred") "credibility" else "power", " pass): ", x$rough_mde),
           notes = where)
  r
}
rows$cred_N1 <- prop("N1", "cred", "Predetermined skill class under the 2025 rule", "merged", np("power_P3 (predetermined skill tier)"), "merged into N1")
rows$cred_N2 <- prop("N2", "cred", "Effective floor (max of H-2A skill-I and ag minimum wage)", "merged", np("cred_N2 / power_P2 (effective floor)"), "merged into A1 (dose)")
rows$cred_N2b <- prop("N2", "cred", "Minimum-wage truncation only (role switch)", "not_pursued", np("cred_N2b (truncation only, MW role switch)"), "A1 secondary/falsification layer")
rows$cred_N2b$strategy_id <- "cred_N2b"
rows$cred_N3 <- prop("N3", "cred", "Experience-requirement downgrading", "merged", np("cred_N3 (requirement downgrading)"), "N1 declared secondary (c)")
rows$cred_N4 <- prop("N4", "cred", "Multi-state employer reallocation", "killed", np("cred_N4 / power_P5 (multi-state employer reallocation)"))
rows$cred_N5 <- prop("N5", "cred", "USCIS fee x petition size", "killed", np("cred_N5 (USCIS fee x petition size)"))
rows$power_P1 <- prop("P1", "power", "Within-employer occupation panel over 2023 rise and 2025 cut", "merged", np("power_P1 (2023 rise + 2025 cut, within-employer occupation panel)"), "merged into A3")
rows$power_P2 <- prop("P2", "power", "Effective floor dose", "merged", np("cred_N2 / power_P2 (effective floor)"), "merged into A1 (dose)")
rows$power_P3 <- prop("P3", "power", "Predetermined skill tier", "merged", np("power_P3 (predetermined skill tier)"), "merged into N1")
rows$power_P4 <- prop("P4", "power", "Combined ex-ante effective dose", "merged", np("power_P4 (combined ex-ante effective dose)"), "N1 declared secondary (b)")
rows$power_P5 <- prop("P5", "power", "Multi-state employer reallocation", "killed", np("cred_N4 / power_P5 (multi-state employer reallocation)"))
rows$power_P6 <- prop("P6", "power", "National interrupted time series", "killed", np("power_P6 (national ITS)"))

# Seed design numbers the triage carried into A1/A2/A3/F1/D2 (for reference).
for (id in c("A1", "F1", "A2", "A3", "D2")) {
  tm <- triage_mde(id, id)
  rows[[id]]$notes <- paste(c(rows[[id]]$notes, tm), collapse = " | ")
}

order_ids <- c("A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "C1", "C2", "C3", "C4",
               "D1", "D2", "E1", "E2", "E3", "E4", "E5", "F1", "F2", "G", "N1",
               "cred_N1", "cred_N2", "cred_N2b", "cred_N3", "cred_N4", "cred_N5",
               "power_P1", "power_P2", "power_P3", "power_P4", "power_P5", "power_P6")
stopifnot(setequal(order_ids, names(rows)))
ledger <- rbindlist(lapply(rows[order_ids], as.data.table), fill = TRUE)[, ..fields]
stopifnot(!anyDuplicated(ledger$strategy_id), all(nzchar(ledger$strategy_id)))
fwrite(ledger, file.path(root, "ledger.csv"))
fwrite(ledger[, .(strategy_id, family, phase_reached, status, kill_phase, design_mde_elasticity,
                  review_verdict, pooling_eligible, registry_sha256)], file.path(syn, "ledger_summary.csv"))
print(ledger[, .(strategy_id, family, phase_reached, status, kill_phase, design_mde_elasticity, review_verdict, pooling_eligible)])
cat("Wrote", file.path(root, "ledger.csv"), "with", nrow(ledger), "rows\n")
