# Scratch-only helpers; no retained artifact writes or specification selection.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "specification.R"))
source(path_code("designs", "panel_iv", "decomposition.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(fixest); library(readr)})
setFixest_nthreads(1)
setFixest_notes(FALSE)
setFixest_ssc(ssc(K.adj = TRUE, K.fixef = "nonnested", G.adj = TRUE))
bite_spec <- panel_iv_read_specification(path_code("designs", "panel_iv", "estimation_spec.json"))
bite_outcomes <- setNames(bite_spec$outcomes$outcome[1:4], c("cert", "hours", "applications", "businesses"))
bite_variants <- c("E_gap", "E_frac", "E_gap_early", "E_gap_ag", "E_rank")
bite_output_dir <- function() {
  arg <- commandArgs(trailingOnly = TRUE)[1]
  if (is.na(arg) || !nzchar(arg)) stop("Supply a scratch output directory outside the repository")
  dir.create(arg, recursive = TRUE, showWarnings = FALSE)
  arg <- normalizePath(arg)
  if (startsWith(paste0(arg, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
  options(device = function(...) grDevices::pdf(file.path(arg, "Rplots.pdf"), ...))
  arg
}
bite_check_keys <- function(d, keys = c("county_fips", "year")) {
  stopifnot(nrow(d) > 0, !anyDuplicated(d[keys]), !anyNA(d[keys]),
    is.character(d$county_fips), identical(d$county_fips, harmonize_county_fips_2010(d$county_fips)))
}
bite_center <- function(x, region) {
  v <- x - ave(x, region, FUN = function(z) mean(z, na.rm = TRUE))
  v / sd(v, na.rm = TRUE)
}
bite_complete <- function(d, cols) {
  d[Reduce(`&`, lapply(d[cols], function(x) if (is.numeric(x)) is.finite(x) else !is.na(x) & nzchar(x))), , drop = FALSE]
}
bite_fit <- function(y, rhs, fe, d, iv = NULL) {
  feols(as.formula(paste(y, "~", rhs, "|", fe, if (!is.null(iv)) paste("|", iv) else "")), d,
    vcov = ~aewr_region_id, demeaned = TRUE, data.save = TRUE)
}
bite_rows <- function(m, d, terms, metadata) {
  d <- d[obs(m), , drop = FALSE]
  bind_rows(lapply(c("state_fips", "aewr_region_id", "aewr_iv_cluster_id"), function(cl) {
    if (!cl %in% names(d) || anyNA(d[[cl]])) return(NULL)
    sm <- summary(m, vcov = as.formula(paste0("~", cl)))
    ct <- coeftable(sm); terms <- intersect(terms, rownames(ct))
    ci <- confint(sm, parm = terms)
    bind_cols(metadata[rep(1, length(terms)), , drop = FALSE], tibble(term = terms,
      cluster = cl, clusters = n_distinct(d[[cl]]), estimate = ct[terms, 1], se = ct[terms, 2],
      p = ct[terms, 4], low = ci[,1], high = ci[,2], n = nrow(d), counties = n_distinct(d$county_fips),
      first_year = min(d$year), last_year = max(d$year), mean_y = mean(d[[all.vars(m$fml_all$linear)[1]]]),
      mde = 2.8 * ct[terms,2]))
  })) %>% mutate(mde_pct_mean = 100 * mde / abs(mean_y))
}
# Restricted Webb bootstrap. Each cluster's null residual is projected through
# ALL FE and nuisance regressors before drawing multipliers. This also handles
# global year FE, which do not commute with region multipliers. Fixed-X/Z IV
# Wald bootstrap is a conditional diagnostic, not weak-instrument-robust inference.
bite_wild <- function(m, d, term, B = 9999L, seed = 20261102L) {
  d <- d[obs(m), , drop = FALSE]
  X <- m$X_demeaned; y <- m$y_demeaned
  if (is.null(X)) stop("Bootstrap requires demeaned design")
  is_iv <- isTRUE(m$is_iv)
  if (is_iv) {
    # fixest stores fitted endogenous columns in X_demeaned. Recover actual X.
    actual <- model.matrix(m, type = "rhs")
    Xactual <- demean(actual, f = m$fixef_id, nthreads = 1)
    Xactual <- Xactual[, sub("^fit_", "", colnames(X)), drop = FALSE]
  } else Xactual <- X
  j <- match(term, colnames(X)); stopifnot(!is.na(j))
  H <- X %*% solve(crossprod(X, Xactual))
  keep <- setdiff(seq_len(ncol(X)), j)
  u0 <- y
  if (length(keep)) u0 <- y - Xactual[,keep,drop=FALSE] %*%
    solve(crossprod(X[,keep,drop=FALSE], Xactual[,keep,drop=FALSE]), crossprod(X[,keep,drop=FALSE], y))
  # FE null residual equals demeaned y minus demeaned structural fitted values.
  cl <- factor(d$aewr_region_id); G <- nlevels(cl)
  U <- matrix(0, nrow(d), G); U[cbind(seq_len(nrow(d)), as.integer(cl))] <- u0
  U <- demean(U, f = m$fixef_id, nthreads = 1, tol = 1e-10)
  db <- crossprod(H, U)
  err <- U - Xactual %*% db
  L <- rowsum(err * H[,j], cl, reorder = TRUE)
  set.seed(seed)
  W <- matrix(sample(c(-sqrt(1.5), -1, -sqrt(.5), sqrt(.5), 1, sqrt(1.5)), G*B, TRUE), G)
  tb <- drop(db[j,,drop=FALSE] %*% W) / sqrt(G/(G-1) * colSums((L %*% W)^2))
  u <- y - Xactual %*% coef(m)
  t0 <- coef(m)[j] / sqrt(G/(G-1) * sum(rowsum(H[,j]*u, cl)^2))
  tibble(term = term, wild_p = (1+sum(abs(tb)>=abs(t0)))/(B+1), wild_B = B,
    wild_regions = G, wild_seed = seed, wild_status = if (is_iv) "fixed_design_IV_Wald_not_weak_robust" else "restricted_Webbb_full_nuisance_projection")
}
bite_residualize <- function(d, columns, controls = "1", fe = "county_fips + year") {
  vapply(columns, function(v) resid(bite_fit(v, controls, fe, d)), numeric(nrow(d)))
}
# Conditional Sanderson-Windmeijer Wald F for two endogenous variables.
# Test residual instrument direction after projecting on the other fitted
# endogenous regressor; covariance uses v_j - delta*v_other, not the naive
# conditional-regression residual. Exactly identified: one restriction.
bite_sw <- function(X, Z, cl) {
  Pi <- solve(crossprod(Z), crossprod(Z,X)); Xhat <- Z %*% Pi; V <- X-Xhat
  G <- n_distinct(cl)
  vapply(seq_len(ncol(X)), function(j) {
    other <- setdiff(seq_len(ncol(X)),j)
    delta <- solve(crossprod(Xhat[,other,drop=FALSE]), crossprod(Xhat[,other,drop=FALSE],Xhat[,j]))
    zres <- Z - Xhat[,other,drop=FALSE] %*% solve(crossprod(Xhat[,other,drop=FALSE]),crossprod(Xhat[,other,drop=FALSE],Z))
    zz <- svd(zres, nu=1, nv=0)$u[,1]
    err <- V[,j] - V[,other,drop=FALSE] %*% delta
    score <- rowsum(zz*drop(err), cl)
    sum(zz*X[,j])^2 / (G/(G-1)*sum(score^2))
  }, numeric(1))
}

# KP rk Wald F, following the installed fixest::kp_stat SVD construction but
# calculating the reduced-form cluster meat directly. The installed helper
# fails when its 4-column score matrix replaces a model with additional
# controls. CRV1 here is G/(G-1), conditional on the nuisance projection.
bite_kp <- function(X,Z,cl) {
  Pi<-solve(crossprod(Z),crossprod(Z,X)); V<-X-Z%*%Pi
  Fm<-chol(crossprod(Z)); Gm<-chol(crossprod(V))
  theta<-Fm%*%Pi%*%solve(Gm); ss<-svd(theta); k<-ncol(X)
  a<-ss$u[,k,drop=FALSE]; b<-ss$v[,k,drop=FALSE]
  direction<-kronecker(t(b),t(a)); lambda<-direction%*%as.vector(theta)
  S<-do.call(cbind,lapply(seq_len(k),function(j)Z*V[,j]))
  S<-rowsum(S,cl); G<-nrow(S); meat<-crossprod(S)*G/(G-1)
  K<-t(kronecker(Gm,Fm)); vv<-solve(K,t(solve(K,meat)))
  as.numeric(lambda^2/(direction%*%vv%*%t(direction))/ncol(Z))
}
