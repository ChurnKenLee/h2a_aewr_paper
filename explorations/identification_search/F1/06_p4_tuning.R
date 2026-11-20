# Identification search, strategy F1 (Phase 5): P4 tuning on the 16 exploration
# states only (registry estimands.P4_latent_distribution.tuning_on_exploration_states).
# Candidates: family {normal, logistic} x scale {common, by skill class}. Selection:
# leave-one-state-out predictive log-likelihood summed over the 16 exploration states
# on M26 (fit on 15, evaluate on the held-out state); highest wins; ties within 1
# unit go to the simpler model. The choice is written to f1_p4_tuning.csv and
# f1_p4_choice.json before any confirmation-state fit (07_p4_confirmation.R refuses
# to run without it).
#
# Held-out log-likelihood weights: hours / mean hours over the whole exploration M26
# sample (so fold contributions add up on one scale); training weights normalized to
# mean 1 within each training fold (registry: hours-weighted pseudo-likelihood).
#
# Usage: R_LIBS_USER=... Rscript --vanilla explorations/identification_search/F1/06_p4_tuning.R "$S"
source("explorations/identification_search/F1/03_est_helpers.R")
f1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- f1_out_dir(S)
if (file.exists(file.path(out, "f1_p4_choice.json"))) stop("P4 choice already recorded; tuning is run once")
o <- read_parquet(file.path(out, "f1_orders.parquet"))
doses <- f1_doses(S); ST48 <- sort(unique(doses$state_fips))
o <- f1_attach_doses(o, doses)
EX <- f1_M26(o, ST48) %>% filter(part == "exploration")
ex_states <- sort(unique(EX$state_fips)); stopifnot(length(ex_states) == 16)
hbar <- mean(EX$hours)
res <- list()
for (j in seq_len(nrow(P4_CANDS))) {
  cd <- P4_CANDS[j, ]
  for (s in ex_states) {
    tr <- p4_data(EX %>% filter(state_fips != s))
    fit <- p4_fit(tr, cd$fam, cd$scale_k)
    te <- p4_data(EX %>% filter(state_fips == s), center = tr$center)
    ll <- sum((te$d$hours / hbar) * p4_ll_i(fit$par, te, cd$fam, cd$scale_k))
    res[[length(res) + 1]] <- tibble(model = cd$model, heldout_state = s, ll = ll, conv = fit$conv, n_test = nrow(te$d))
  }
  message("candidate done: ", cd$model)
}
res <- bind_rows(res)
tab <- res %>% group_by(model) %>% summarise(loso_ll = sum(ll), max_conv_code = max(conv), .groups = "drop") %>%
  mutate(simplicity_rank = match(model, P4_SIMPLICITY)) %>% arrange(simplicity_rank)
best <- max(tab$loso_ll)
tab <- tab %>% mutate(within_1_of_best = loso_ll >= best - 1)
chosen <- tab %>% filter(within_1_of_best) %>% arrange(simplicity_rank) %>% slice(1) %>% pull(model)
# Alternative reading of the tie order (family first: normal_common, normal_skill, logistic_common, logistic_skill)
alt_order <- c("normal_common", "normal_skill", "logistic_common", "logistic_skill")
chosen_alt <- tab %>% filter(within_1_of_best) %>% arrange(match(model, alt_order)) %>% slice(1) %>% pull(model)
tab <- tab %>% mutate(chosen = model == chosen)
print(tab)
message("chosen: ", chosen, "; under the family-first tie order: ", chosen_alt)
write_csv(bind_rows(res %>% mutate(level = "fold"), tab %>% mutate(level = "total")), file.path(out, "f1_p4_tuning.csv"))
write_json(list(chosen = chosen, chosen_family_first_order = chosen_alt, loso_ll = setNames(as.list(tab$loso_ll), tab$model),
  exploration_states = ex_states, recorded = format(Sys.time())), file.path(out, "f1_p4_choice.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
message("done ", format(Sys.time()))
