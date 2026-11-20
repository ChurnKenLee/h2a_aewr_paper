# Design-specific scratch helpers; reuse the established geography/fixest setup.
source("explorations/episode_2025/ep_helpers.R")
CD <- "explorations/chapter_directions"
cd_verify <- function(file = file.path(CD, "registry.sha256"), base = ".") {
  for (line in readLines(file)) {
    p <- strsplit(line, "  ", fixed = TRUE)[[1]]
    f <- file.path(base, p[2])
    if (!file.exists(f) || digest::digest(file = f, algo = "sha256") != p[1]) stop("Hash mismatch: ", f)
  }
  invisible(TRUE)
}
cd_first <- function(x) { x <- x[!is.na(x) & trimws(as.character(x)) != ""]; if (length(x)) x[1] else NA }
cd_unique <- function(x) { x <- unique(x[!is.na(x) & trimws(as.character(x)) != ""]); if (length(x) == 1) x else NA }
cd_mean <- function(x, w) {
  if(length(x)!=length(w))stop("Weighted mean requires matching observation and weight lengths")
  k <- is.finite(x) & is.finite(w) & w > 0
  if (any(k)) sum(x[k] * w[k]) / sum(w[k]) else NA_real_
}
cd_keys <- function(d, keys) stopifnot(nrow(d) > 0, !anyDuplicated(d[keys]), !anyNA(d[keys]))
cd_write <- function(x, out, name) readr::write_csv(x, file.path(out, name))
cd_shift <- function(date, years) {
  md <- format(date, "%m-%d"); md[md == "02-29"] <- "02-28"
  as.Date(ifelse(is.na(date), NA_character_, paste0(as.integer(format(date, "%Y")) + years, "-", md)))
}
cd_overlap <- function(a, b, start = as.Date("2013-10-01"), end = as.Date("2013-10-16")) {
  ifelse(is.na(a) | is.na(b) | b < a, NA, a <= end & b >= start)
}
cd_recoded <- function(date) !is.na(date) & date == as.Date("2013-10-18")
# docs-ground:start chapter-earnings-complete-window
cd_window <- function(q, tau) {
  need <- tibble(year = c(tau-2L, tau-1L, tau-1L, tau-1L, tau, tau), qtr = c(4L,1L,2L,4L,1L,2L))
  if (nrow(anti_join(need, distinct(q, year, qtr), by=c("year","qtr")))) stop("Missing quarter: no partial-window estimate")
  cd_keys(q, c("county_fips","year","qtr"))
  z <- q %>% semi_join(need, by=c("year","qtr")) %>% filter(disclosed, emp >= 10, is.finite(avg_wkly_wage), avg_wkly_wage > 0) %>%
    mutate(lw=log(avg_wkly_wage))
  z %>% inner_join(z %>% transmute(county_fips,year=year+1L,qtr,lw_l=lw,emp_l=emp), by=c("county_fips","year","qtr")) %>%
    filter((year==tau-1L & qtr==4L) | (year==tau & qtr %in% 1:2)) %>% group_by(county_fips) %>%
    summarise(nq=n(),change=mean(lw-lw_l),emp_l=mean(emp_l),.groups="drop") %>% filter(nq==3)
}
# docs-ground:end chapter-earnings-complete-window

# Every requested fit, including an identification or deletion failure, gets a row.
cd_fit <- function(d, formula, terms, spec, weight = NULL, track = "wage", tau = 2026L, tier = "R7") {
  f <- as.formula(formula); vars <- unique(c(all.vars(f), weight, "state_fips"))
  d <- as.data.frame(d)
  for (v in vars) d <- d[!is.na(d[[v]]) & if (is.numeric(d[[v]])) is.finite(d[[v]]) else nzchar(as.character(d[[v]])), , drop=FALSE]
  if (!is.null(weight)) d <- d[d[[weight]] > 0, , drop=FALSE]
  meta <- tibble(track=track,specification=spec,tau=tau,tier=tier,weights=if(is.null(weight)) "equal" else weight,formula=formula)
  fail <- function(reason) bind_cols(meta[rep(1,length(terms)),], tibble(term=terms,status=reason,n=nrow(d),states=n_distinct(d$state_fips),estimate=NA_real_,se_jackknife=NA_real_,low=NA_real_,high=NA_real_,p=NA_real_,se_crv1=NA_real_))
  if (nrow(d) < 3 || n_distinct(d$state_fips)<3) return(list(rows=fail("insufficient_support"),loo=tibble()))
  wf <- if (is.null(weight)) NULL else as.formula(paste0("~",weight))
  fit <- function(dd) tryCatch(feols(f, dd, weights=wf, vcov=~state_fips),error=function(e) NULL)
  m <- fit(d)
  if (is.null(m)) return(list(rows=fail("fit_failed"),loo=tibble()))
  used <- d[obs(m),,drop=FALSE]; states <- sort(unique(used$state_fips)); G <- length(states)
  l <- lapply(states, function(s) { mm<-fit(d[d$state_fips!=s,,drop=FALSE]); b<-if(is.null(mm)) numeric() else coef(mm); tibble(omitted_state=s,term=terms,estimate=unname(b[terms])) })
  loo <- bind_rows(l)
  rows <- lapply(terms,function(term) {
    b<-unname(coef(m)[term]); bs<-loo$estimate[loo$term==term]
    ok<-length(bs)==G && all(is.finite(bs)) && is.finite(b)
    sj<-if(ok) sqrt((G-1)/G*sum((bs-mean(bs))^2)) else NA_real_
    bind_cols(meta,tibble(term=term,status=if(!is.finite(b)) "term_not_identified" else if(!ok) "jackknife_not_identified" else "estimated",n=nobs(m),states=G,
      estimate=b,se_jackknife=sj,low=b-qt(.975,G-1)*sj,high=b+qt(.975,G-1)*sj,p=2*pt(-abs(b/sj),G-1),se_crv1=unname(se(m)[term])))
  })
  list(rows=bind_rows(rows),loo=bind_cols(meta[rep(1,nrow(loo)),],loo))
}
