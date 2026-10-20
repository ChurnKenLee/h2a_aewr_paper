# Purpose: Estimate the frozen exploratory redesign and comparable v5 models.
# Owns only the explicitly declared border_discontinuity_redesign artifacts.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(fixest)
library(jsonlite)
library(readr)
source(path_code("designs", "border_discontinuity", "redesign.R"))
fixest::setFixest_nthreads(1)
design <- fromJSON(path_code("designs", "border_discontinuity", "design.json"))
panel <- as.data.frame(read_parquet(path_processed("border_discontinuity_panel.parquet")))
memberships <- as.data.frame(read_parquet(path_int("border_discontinuity_geography_memberships.parquet")))
if (!identical(unique(panel$border_design_id), design$design_id)) stop("Stale border panel version")
legacy <- read_csv(path_tables("border_discontinuity_estimates.csv"), show_col_types = FALSE)
assignment_audit <- aggregate(panel[[design$treatment]], panel[c("aewr_region_id","year")],
  function(z) {
    z <- z[is.finite(z)]
    c(minimum=if(length(z)) min(z) else NA_real_,
      maximum=if(length(z)) max(z) else NA_real_, n_values=length(unique(z)))
  })
assignment_audit <- data.frame(assignment_audit[c("aewr_region_id","year")],assignment_audit$x)
assignment_audit$treatment_range <- assignment_audit$maximum-assignment_audit$minimum
assignment_audit$period <- ifelse(assignment_audit$year>=design$policy_start_year,"policy","preanalysis")
if(any(assignment_audit$treatment_range[assignment_audit$period=="policy"]>1e-10)) stop("Policy AEWR is not region-year constant")
assignment_audit$measurement_status <- ifelse(assignment_audit$year == 2009,
  "statutory_AEWR_unavailable_nonstatutory_mean_quarantined", "recorded_statutory_AEWR")
write_csv(assignment_audit,path_tables("border_discontinuity_redesign_assignment_audit.csv"))


# docs-ground:start border-redesign-estimation-contract
# The original seven-model audit stays in stage04. Every new coefficient is
# exploratory; identification and informative precision are separate gates.
registry <- rbind(
  data.frame(model_id = paste0(design$benchmark_geography_ids, "__base"),
    spec_id = design$benchmark_geography_ids, weight_spec = "equal_county",
    control_spec = "none", role = "v5_benchmark"),
  design$redesign$models[, c("model_id", "spec_id", "weight_spec", "control_spec", "role")]
)
# Treatment, fixed-2011 denominator, 2011--2022 years and county/component-year
# FE are unchanged. All-member support/final locality are geography decisions.
# Fixed 2008--2010 employment changes weights only; baseline-history adjustment
# is the standardized 2008--2010 outcome mean interacted with calendar time.
# Pair WCR11-FNW/Webb and CV3J are comparisons; dyadic inference is a small-node
# sensitivity. Deleting every incident edge is influence, not a node CV3J law.
# docs-ground:end border-redesign-estimation-contract

estimates <- audits <- supports <- deletions <- balances <- placebos <- fwl <- list()
simulation_inputs <- list()
fits <- list()
for (j in seq_len(nrow(registry))) {
  row <- registry[j, ]
  message("Redesign comparison: ", row$model_id)
  prepared <- border_prepare(panel, memberships, row, design)
  fit <- border_fit(prepared)
  d <- prepared$data
  # Exact row/weight/FWL agreement establishes the newly computed benchmark.
  old <- legacy[legacy$model_id == row$model_id, ]
  if (nrow(old)) {
    if (abs(old$estimate - fit$result$estimate) > 1e-8 || old$n_observations != nrow(d) ||
        abs(old$crv1_standard_error-fit$result$crv1_standard_error)>1e-8) stop("v5 benchmark changed")
    wild <- data.frame(wcb_p_value=old$wcb_p_value,wcb_confidence_low=old$wcb_confidence_low,
      wcb_confidence_high=old$wcb_confidence_high,wcb_status="reproduced_v5",wcb_repetitions=old$wcb_repetitions,
      wcb_seed=old$wcb_seed,wcb_method=old$wcb_method)
  } else wild <- border_wild(fit, design$bootstrap)
  estimates[[j]] <- cbind(row, fit$result, border_pair_jackknife(fit), wild,
    data.frame(design_id=design$design_id,outcome=design$outcome,treatment=design$treatment,
      effect_unit="certifications_per_fixed_2011_farm_worker_per_2012_dollar_hour"))
  audits[[length(audits)+1]] <- prepared$audit
  fits[[row$model_id]] <- fit
  gap <- border_gap_check(fit)
  fwl[[j]] <- cbind(data.frame(model_id=row$model_id,fwl_coefficient_error=fit$result$fwl_coefficient_error,
    fwl_score_sum=fit$result$fwl_score_sum),gap)

  # Support shows graph rank, treatment and noise, rather than treating rows as assignments.
  partial_ss <- rowsum(fit$score_loading^2, d$region_pair_id, reorder=TRUE)[,1]
  pair_rss <- rowsum(fit$u^2,d$region_pair_id,reorder=TRUE)[,1]
  pair_rows <- data.frame(model_id=row$model_id,region_pair_id=names(partial_ss),
    partial_treatment_leverage=partial_ss/sum(partial_ss),residual_outcome_ss=pair_rss,
    coefficient_contribution=rowsum(fit$score_loading*fit$y,d$region_pair_id,reorder=TRUE)[,1],
    pair_score=fit$pair_score, row.names=NULL)
  supports[[j]] <- pair_rows

  for (r in sort(unique(d$aewr_region_id))) {
    incident <- unique(d$region_pair_id[d$aewr_region_id==r])
    keep <- !d$region_pair_id %in% incident
    dd <- d[keep,,drop=FALSE]
    z <- tryCatch({
      deleted_fit <- border_fit(list(data=dd,x=prepared$x[keep,,drop=FALSE],y=prepared$y[keep]))
      data.frame(estimate=deleted_fit$result$estimate,status="ok")
    },error=function(e) data.frame(estimate=NA_real_,status=paste0("unidentified: ",conditionMessage(e))))
    deletions[[length(deletions)+1]] <- cbind(data.frame(model_id=row$model_id,excluded_region_id=r,
      excluded_pairs=paste(sort(incident),collapse=";"),n_excluded_pairs=length(incident),
      n_retained_pairs=length(unique(dd$region_pair_id)),full_estimate=fit$result$estimate),z)
  }

  pre <- border_prepare(panel,memberships,row,design,period="placebo",require_treatment=FALSE)
  # The source's 2009 special case is an average certified application wage,
  # not an AEWR schedule. Preserve the v5 audit separately; exclude that year
  # from the NEW wage diagnostic, and calibrate power using FE-only outcomes.
  diag_pre <- border_prepare(panel,memberships,row,design,period="placebo",
    years=design$redesign$preanalysis_wage_audit$new_diagnostic_years)
  pfit <- border_fit(diag_pre)
  audits[[length(audits)+1]] <- diag_pre$audit
  pre$audit$period <- "calibration_preperiod"
  audits[[length(audits)+1]] <- pre$audit
  pcrit <- qt(.975,pfit$result$n_pairs-1)+qnorm(.8)
  placebos[[j]] <- cbind(row,pfit$result,border_pair_jackknife(pfit),
    data.frame(period="2008_and_2010_already_exposed_two_year_diagnostic",approximate_pair_mde=pcrit*pfit$result$crv1_standard_error,
      zero_effect_not_guaranteed=TRUE, excluded_wage_year=2009L))
  simulation_inputs[[j]] <- list(model_id=row$model_id,policy=prepared,pre=pre)

  # Baseline differences and pre-exposure association on the actual weighted population.
  bd <- pre$data[pre$data$year == design$feature_end_year,,drop=FALSE]
  vars <- paste0("baseline_",c(design$outcome,design$baseline_control_columns),"_2008_2010_z")
  wx <- "valid_preanalysis_wage_mean"
  valid_wages <- aggregate(border_x~county_fips,diag_pre$data,mean)
  bd[[wx]] <- valid_wages$border_x[match(bd$county_fips,valid_wages$county_fips)]
  bd[[wx]] <- as.numeric(scale(bd[[wx]]))
  for(v in vars) {
    z <- bd[complete.cases(bd[,c(v,wx)]),,drop=FALSE]
    z$balance_y <- z[[v]]; z$balance_x <- z[[wx]]
    bm <- feols(balance_y~balance_x|stratum_id,z,weights=~estimation_weight,vcov=~region_pair_id,nthreads=1,notes=FALSE)
    balances[[length(balances)+1]] <- data.frame(model_id=row$model_id,variable=v,
      preperiod_wage_association=unname(coef(bm)["balance_x"]),
      pair_standard_error=unname(se(bm)["balance_x"]),pair_p_value=unname(pvalue(bm)["balance_x"]),
      weighted_mean=weighted.mean(z[[v]],z$estimation_weight),weighted_sd=sqrt(weighted.mean((z[[v]]-weighted.mean(z[[v]],z$estimation_weight))^2,z$estimation_weight)),
      n_counties=length(unique(z$county_fips)),wage_baseline_years="2008;2010",excluded_wage_year=2009L)
  }
}
estimates <- do.call(rbind,estimates)

# Common observations isolate an adjustment or weighting change from sample loss.
base <- fits[["screened_h1__base"]]
for(id in c("screened_h1__preemployment_weight","screened_h1__baseline_history","screened_h1__region_disjoint")) {
  alt <- fits[[id]]
  key <- paste(alt$prepared$data$county_fips,alt$prepared$data$year)
  keep <- paste(base$prepared$data$county_fips,base$prepared$data$year) %in% key
  pp <- base$prepared
  pp$data <- pp$data[keep,,drop=FALSE]; pp$x <- pp$x[keep,,drop=FALSE]; pp$y <- pp$y[keep]
  common <- border_fit(pp)
  idx <- estimates$model_id==id
  estimates$base_on_comparable_county_years[idx] <- common$result$estimate
  estimates$comparable_county_years[idx] <- nrow(pp$data)
  estimates$change_from_comparable_base[idx] <- estimates$estimate[idx]-common$result$estimate
}
# Each side of a comparable-sample pair retains its own documented specification.
comparables <- list()
for (ids in list(c("ring_h1_disjoint__base", "screened_h1__base"),
                 c("ring_h0_disjoint__base", "screened_h1__base"),
                 c("screened_h1__base", "screened_h1__preemployment_weight"),
                 c("screened_h1__base", "screened_h1__baseline_history"),
                 c("screened_h1__base", "screened_h1__region_disjoint"))) {
  a <- fits[[ids[1]]]$prepared; b <- fits[[ids[2]]]$prepared
  common <- intersect(paste(a$data$county_fips,a$data$year),paste(b$data$county_fips,b$data$year))
  for (id in ids) {
    pp <- fits[[id]]$prepared
    keep <- paste(pp$data$county_fips,pp$data$year) %in% common
    pp$data <- pp$data[keep,,drop=FALSE]; pp$x <- pp$x[keep,,drop=FALSE]; pp$y <- pp$y[keep]
    cf <- border_fit(pp)
    comparables[[length(comparables)+1]] <- cbind(data.frame(comparison_id=paste(ids,collapse="__versus__"),model_id=id),cf$result)
  }
}
write_csv(do.call(rbind,comparables),path_tables("border_discontinuity_redesign_comparable_samples.csv"))

# Outcome-informed v5 concentration diagnostic is descriptive and never an eligibility rule.
old_d <- fits[["ring_h1_disjoint__base"]]$prepared$data
old_fit <- fits[["ring_h1_disjoint__base"]]
core <- unique(fits[["ring_h0_disjoint__base"]]$prepared$data$county_fips)
old_d$in_old_core <- old_d$county_fips %in% core
old_d$fit_rss <- old_fit$u^2
old_d$fit_treatment_ss <- old_fit$X[,1]^2
old_d$fit_coefficient <- old_fit$score_loading*old_fit$y
county <- aggregate(cbind(fit_rss,fit_treatment_ss,fit_coefficient)~county_fips+in_old_core,old_d,sum)
county$rss_share <- county$fit_rss/sum(county$fit_rss)
county$treatment_ss_share <- county$fit_treatment_ss/sum(county$fit_treatment_ss)
county <- county[order(-county$rss_share),]
write_csv(county,path_tables("border_discontinuity_redesign_noise_diagnosis.csv"))

write_csv(estimates,path_tables("border_discontinuity_redesign_estimates.csv"))
write_csv(do.call(rbind,audits),path_tables("border_discontinuity_redesign_sample_audit.csv"))
write_csv(do.call(rbind,supports),path_tables("border_discontinuity_redesign_support.csv"))
write_csv(do.call(rbind,deletions),path_tables("border_discontinuity_redesign_region_deletion.csv"))
write_csv(do.call(rbind,balances),path_tables("border_discontinuity_redesign_balance.csv"))
write_csv(do.call(rbind,placebos),path_tables("border_discontinuity_redesign_placebo.csv"))
write_csv(do.call(rbind,fwl),path_tables("border_discontinuity_redesign_fwl.csv"))
saveRDS(simulation_inputs,path_int("border_discontinuity_redesign_simulation_inputs.rds"))
message("Retained ",nrow(estimates)," v6 comparisons; no empirical-adequacy claim is inferred from completion.")
