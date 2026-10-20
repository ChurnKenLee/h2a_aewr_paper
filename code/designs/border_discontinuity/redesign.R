# Numerical helpers for the declared v6 local border comparisons.
# These methods do not select specifications using estimated effects.

border_prune_singletons <- function(d) {
  repeat {
    keep <- ave(seq_len(nrow(d)), d$unit_fe_id, FUN = length) > 1 &
      ave(seq_len(nrow(d)), d$stratum_year_id, FUN = length) > 1
    if (all(keep) || !any(keep)) return(d[keep, , drop = FALSE])
    d <- d[keep, , drop = FALSE]
  }
}

border_prepare <- function(panel, memberships, row, design, period = "policy", years = NULL,
                           require_treatment = TRUE) {
  member <- memberships[memberships$spec_id == row$spec_id, c(
    "county_fips", "membership_id", "stratum_id", "region_pair_id",
    "membership_mode", "analysis_weight"
  )]
  d <- merge(panel, member, by = "county_fips", sort = FALSE)
  yrs <- if (period == "policy") c(design$policy_start_year, design$policy_end_year) else
    c(design$placebo_start_year, design$placebo_end_year)
  d <- d[d$year >= yrs[1] & d$year <= yrs[2], , drop = FALSE]
  if (!is.null(years)) d <- d[d$year %in% years, , drop = FALSE]
  d$unit_fe_id <- ifelse(d$membership_mode == "stacked", d$membership_id, d$county_fips)
  d$stratum_year_id <- paste(d$stratum_id, d$year, sep = "__")
  d$estimation_weight <- d$analysis_weight
  if (row$weight_spec == "baseline_farm_employment") {
    d$estimation_weight <- d$estimation_weight * d$border_baseline_farm_employment
  }
  d$border_y <- d[[design$outcome]]
  d$border_x <- d[[design$treatment]]
  controls <- character()
  if (row$control_spec == "baseline_outcome_trend") {
    d$history_trend <- d[[paste0("baseline_", design$outcome, "_2008_2010_z")]] *
      (d$year - design$policy_start_year)
    controls <- "history_trend"
  }
  required <- c("border_y", if (require_treatment) "border_x", controls, "estimation_weight")
  complete <- complete.cases(d[, c(required, "unit_fe_id", "stratum_year_id", "region_pair_id")]) &
    apply(as.matrix(d[, required, drop = FALSE]), 1, function(z) all(is.finite(z))) &
    d$estimation_weight > 0
  n_joined <- nrow(d)
  d <- d[complete, , drop = FALSE]
  n_complete <- nrow(d)
  d <- border_prune_singletons(d)
  d <- d[order(d$stratum_id, d$county_fips, d$year), , drop = FALSE]
  d$estimation_weight <- d$estimation_weight / mean(d$estimation_weight)
  # Identifiers stay character in all persistent artifacts, including RDS.
  for (id in c("county_fips", "aewr_region_id", "region_pair_id", "stratum_id", "unit_fe_id", "stratum_year_id")) {
    d[[id]] <- as.character(d[[id]])
  }
  list(data = d, x = as.matrix(d[, c("border_x", controls), drop = FALSE]), y = d$border_y,
    audit = data.frame(model_id = row$model_id, period = period,
      joined_rows = n_joined, complete_case_rows = n_complete, estimation_rows = nrow(d),
      dropped_missing_rows = n_joined - n_complete, dropped_singleton_rows = n_complete - nrow(d),
      n_counties = length(unique(d$county_fips)), n_strata = length(unique(d$stratum_id)),
      n_pairs = length(unique(d$region_pair_id)), n_regions = length(unique(d$aewr_region_id))))
}

border_fit <- function(prepared) {
  d <- prepared$data
  rhs <- colnames(prepared$x)
  f <- as.formula(paste("border_y ~", paste(rhs, collapse = " + "), "| unit_fe_id + stratum_year_id"))
  m <- fixest::feols(f, d, weights = ~estimation_weight, vcov = ~region_pair_id,
    notes = FALSE, nthreads = 1)
  if (!all(rhs %in% names(coef(m)))) stop("Declared redesign regressors are unidentified")
  within <- fixest::demean(m, as.matrix = TRUE, tol = 1e-10, nthreads = 1)
  X <- within[, rhs, drop = FALSE] * sqrt(d$estimation_weight)
  y <- within[, 1] * sqrt(d$estimation_weight)
  bread <- solve(crossprod(X))
  b <- drop(bread %*% crossprod(X, y))
  if (max(abs(b - coef(m)[rhs])) > 1e-8) stop("Redesign FWL coefficient mismatch")
  u <- drop(y - X %*% b)
  # The partial-treatment score sums reproduce beta and all sandwich variances.
  v <- drop(X %*% bread[, 1])
  pair_score <- rowsum(v * u, d$region_pair_id, reorder = TRUE)[, 1]
  pair_ss <- rowsum(v^2, d$region_pair_id, reorder = TRUE)[, 1]
  endpoints <- strsplit(names(pair_score), "_", fixed = TRUE)
  regions <- sort(unique(unlist(endpoints)))
  incidence <- matrix(0, length(regions), length(endpoints), dimnames = list(regions, names(pair_score)))
  for (j in seq_along(endpoints)) incidence[endpoints[[j]], j] <- 1
  dyadic_raw <- sum((incidence %*% pair_score)^2) - sum(pair_score^2)
  nr <- length(regions)
  # Explicit small-node sensitivity, not an exact finite-sample reference law.
  dyadic_var <- dyadic_raw * nr / (nr - 1)
  dyadic_se <- if (is.finite(dyadic_var) && dyadic_var > 0) sqrt(dyadic_var) else NA_real_
  raw_two <- fixest::vcov_cluster(m, ~region_pair_id + aewr_region_id, vcov_fix = FALSE)
  two <- summary(m, vcov = ~region_pair_id + aewr_region_id, vcov_fix = TRUE)
  ci <- confint(m, "border_x")
  ci_two <- confint(two, "border_x")
  result <- data.frame(estimate = b[1], crv1_standard_error = fixest::se(m)["border_x"],
    crv1_p_value = fixest::pvalue(m)["border_x"], crv1_confidence_low = ci[1,1], crv1_confidence_high = ci[1,2],
    two_way_standard_error = fixest::se(two)["border_x"], two_way_p_value = fixest::pvalue(two)["border_x"],
    two_way_confidence_low = ci_two[1,1], two_way_confidence_high = ci_two[1,2],
    two_way_raw_minimum_eigenvalue = min(eigen(raw_two, symmetric = TRUE)$values),
    two_way_psd_adjustment = max(abs(vcov(two) - raw_two)),
    dyadic_raw_variance = dyadic_raw, dyadic_standard_error = dyadic_se,
    dyadic_confidence_low = b[1] - qt(.975, nr - 1) * dyadic_se,
    dyadic_confidence_high = b[1] + qt(.975, nr - 1) * dyadic_se,
    dyadic_p_value = 2 * pt(-abs(b[1]/dyadic_se), nr - 1),
    dyadic_status = ifelse(is.finite(dyadic_se), "small_node_sensitivity", "nonpositive_variance_unavailable"),
    fwl_coefficient_error = max(abs(b - coef(m)[rhs])),
    fwl_score_sum = sum(v * u), residual_outcome_ss = sum(u^2),
    treatment_partial_ss = 1 / bread[1,1], treatment_raw_sd = sd(d$border_x),
    treatment_within_sd = sd(within[, "border_x"]),
    effective_pair_leverage_count = sum(pair_ss)^2 / sum(pair_ss^2),
    maximum_pair_leverage = max(pair_ss) / sum(pair_ss),
    region_graph_incidence_rank = qr(t(incidence) * rep(c(1), length(endpoints)))$rank,
    n_observations = nrow(d), n_counties = length(unique(d$county_fips)),
    n_strata = length(unique(d$stratum_id)), n_pairs = length(endpoints), n_regions = nr,
    row.names = NULL)
  # Signed incidence rank is the policy contrast ceiling, unlike unsigned incidence.
  signed <- incidence
  for (j in seq_along(endpoints)) signed[endpoints[[j]][2], j] <- -1
  result$region_graph_incidence_rank <- qr(signed)$rank
  list(model = m, prepared = prepared, X = X, y = y, u = u, bread = bread, score_loading = v,
    pair_score = pair_score, pair_ss = pair_ss, incidence = incidence, result = result)
}

border_pair_jackknife <- function(fit) {
  d <- fit$prepared$data
  vals <- sort(unique(d$region_pair_id))
  beta <- vapply(vals, function(g) {
    keep <- d$region_pair_id != g
    X <- fit$X[keep, , drop = FALSE]
    tryCatch(drop(solve(crossprod(X), crossprod(X, fit$y[keep])))[1], error = function(e) NA_real_)
  }, numeric(1))
  se <- if (all(is.finite(beta))) sqrt((length(vals)-1)/length(vals) * sum((beta-mean(beta))^2)) else NA_real_
  b <- fit$result$estimate
  data.frame(crv3_jackknife_standard_error = se, crv3_jackknife_p_value = 2*pt(-abs(b/se), length(vals)-1),
    crv3_jackknife_confidence_low = b - qt(.975, length(vals)-1)*se,
    crv3_jackknife_confidence_high = b + qt(.975, length(vals)-1)*se,
    crv3_status = ifelse(is.finite(se), "ok", "unidentified_delete_pair"))
}

border_wild <- function(fit, config) {
  bdat <- as.data.frame(fit$X)
  names(bdat) <- paste0("x", seq_len(ncol(fit$X)))
  bdat$y <- fit$y
  bdat$region_pair_id <- fit$prepared$data$region_pair_id
  mod <- fixest::feols(reformulate(paste0("x",seq_len(ncol(fit$X))), "y", intercept = FALSE), bdat, nthreads = 1)
  set.seed(config$seed)
  dqrng::dqset.seed(config$seed)
  ans <- tryCatch({
    z <- fwildclusterboot::boottest(mod, param = "x1", B = config$repetitions,
      clustid = "region_pair_id", bootcluster = "region_pair_id", type = config$weights,
      impose_null = TRUE, bootstrap_type = config$bootstrap_type, conf_int = TRUE, engine = "R", nthreads = 1)
    z <- generics::tidy(z)
    data.frame(wcb_p_value = z$p.value, wcb_confidence_low = z$conf.low,
      wcb_confidence_high = z$conf.high, wcb_status = if (any(is.na(c(z$conf.low,z$conf.high)))) "inversion_failed" else if (any(is.infinite(c(z$conf.low,z$conf.high)))) "unbounded" else "ok")
  }, error = function(e) data.frame(wcb_p_value = NA_real_, wcb_confidence_low = NA_real_,
    wcb_confidence_high = NA_real_, wcb_status = paste0("failed: ", conditionMessage(e))))
  ans$wcb_repetitions <- config$repetitions
  ans$wcb_seed <- config$seed
  ans$wcb_method <- config$method
  ans
}

border_gap_check <- function(fit) {
  d <- fit$prepared$data
  if (ncol(fit$X) != 1) return(data.frame(gap_status = "controls_not_collapsed", gap_coefficient = NA_real_, gap_error = NA_real_))
  counts <- table(d$unit_fe_id)
  year_sets <- vapply(split(d$year, d$unit_fe_id), function(z) paste(sort(z), collapse = ","), character(1))
  if (length(unique(counts)) != 1 || length(unique(year_sets)) != 1 || any(tapply(d$estimation_weight, d$unit_fe_id, function(z) diff(range(z))) > 1e-12)) {
    return(data.frame(gap_status = "unbalanced_not_naively_collapsed", gap_coefficient = NA_real_, gap_error = NA_real_))
  }
  groups <- split(seq_len(nrow(d)), paste(d$stratum_id,d$year,sep = "__"))
  rows <- lapply(groups,function(ii) {
    z <- d[ii,]; rr <- sort(unique(z$aewr_region_id))
    if(length(rr)!=2) stop("Gap comparison lacks two observed sides")
    side <- lapply(rr,function(r) {a <- z[z$aewr_region_id==r,]; c(w=sum(a$estimation_weight),y=weighted.mean(a$border_y,a$estimation_weight),x=weighted.mean(a$border_x,a$estimation_weight))})
    data.frame(stratum_id=z$stratum_id[1],year=z$year[1], y=side[[1]]["y"]-side[[2]]["y"],
      x=side[[1]]["x"]-side[[2]]["x"],w=prod(c(side[[1]]["w"],side[[2]]["w"]))/sum(c(side[[1]]["w"],side[[2]]["w"])))
  })
  gaps <- do.call(rbind,rows)
  gm <- fixest::feols(y~x|stratum_id,gaps,weights=~w,nthreads=1,notes=FALSE)
  error <- abs(coef(gm)["x"]-fit$result$estimate)
  if(error>1e-8) stop("Weighted side-gap equivalence failed")
  data.frame(gap_status="exact_balanced_side_gap",gap_coefficient=coef(gm)["x"],gap_error=error,row.names=NULL)
}
