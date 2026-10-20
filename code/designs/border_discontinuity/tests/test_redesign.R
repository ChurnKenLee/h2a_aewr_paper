# Numerical checks with artificial data; no empirical result files are written.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "border_discontinuity", "redesign.R"))
library(fixest)
setFixest_nthreads(1)
set.seed(201103)
d <- expand.grid(year=2011:2016,county_index=1:24)
d$county_fips <- sprintf("%05d",d$county_index)
d$region_pair_id <- rep(c("01_02","02_03","03_04","01_04"),each=36)
d$stratum_id <- paste0("s",d$region_pair_id)
ends <- strsplit(d$region_pair_id,"_",fixed=TRUE)
d$aewr_region_id <- vapply(seq_len(nrow(d)),function(i) as.character(as.integer(ends[[i]][1+(d$county_index[i]%%2)])),character(1))
d$unit_fe_id <- d$county_fips
d$stratum_year_id <- paste(d$stratum_id,d$year)
d$estimation_weight <- 1 + (d$county_index %% 4)/3
d$border_x <- 10 + .1*as.integer(d$aewr_region_id) + sin(d$year*as.integer(d$aewr_region_id))
d$border_y <- .2*d$border_x + .1*d$county_index + cos(d$year) + rnorm(nrow(d),sd=.05)
p <- list(data=d,x=as.matrix(d["border_x"]),y=d$border_y)
f <- border_fit(p)
explicit <- lm(border_y~border_x+factor(unit_fe_id)+factor(stratum_year_id),d,weights=estimation_weight)
stopifnot(abs(coef(explicit)["border_x"]-f$result$estimate)<1e-9)
g <- border_gap_check(f)
stopifnot(g$gap_error<1e-9)
j <- border_pair_jackknife(f)
bs <- vapply(unique(d$region_pair_id),function(pair) {
  keep <- d$region_pair_id!=pair
  border_fit(list(data=d[keep,],x=p$x[keep,,drop=FALSE],y=p$y[keep]))$result$estimate
},numeric(1))
stopifnot(abs(j$crv3_jackknife_standard_error-sqrt(3/4*sum((bs-mean(bs))^2)))<1e-9)
a <- crossprod(f$incidence)>0
expected_meat <- drop(t(f$pair_score)%*%a%*%f$pair_score)
stopifnot(abs(f$result$dyadic_raw_variance-expected_meat)<1e-12)
stopifnot(f$result$region_graph_incidence_rank==3)
# Rescaling the outcome rescales coefficients and uncertainty, not test statistics.
p2 <- p;p2$y <- 1000*p$y;p2$data$border_y <- p2$y
f2 <- border_fit(p2)
stopifnot(abs(f2$result$estimate-1000*f$result$estimate)<1e-8,
 abs(f2$result$crv1_standard_error-1000*f$result$crv1_standard_error)<1e-8,
 abs(f2$result$crv1_p_value-f$result$crv1_p_value)<1e-9)
# The same FE structure cannot identify a region-year-constant treatment with
# region-year FE. A dropped regressor is evidence of nonidentification.
d$region_year <- paste(d$aewr_region_id,d$year)
absorbed <- demean(as.matrix(d["border_x"]),f=list(d$region_year),nthreads=1)
stopifnot(max(abs(absorbed))<1e-10)

# Outcome-only preperiod calibration must retain 2009 when its statutory wage
# is unavailable; wage-regression diagnostics must still exclude those rows.
pre_panel <- expand.grid(year=2008:2010,county_index=1:4)
pre_panel$county_fips <- sprintf("%05d",pre_panel$county_index)
pre_panel$aewr_region_id <- ifelse(pre_panel$county_index <= 2,"01","02")
pre_panel$y <- seq_len(nrow(pre_panel))/100
pre_panel$x <- ifelse(pre_panel$year==2009,NA_real_,10+pre_panel$year/1000)
members <- data.frame(county_fips=sprintf("%05d",1:4),spec_id="test",
 membership_id=paste0("m",1:4),stratum_id="s",region_pair_id="01_02",
 membership_mode="disjoint",analysis_weight=1)
row <- list(spec_id="test",model_id="test",weight_spec="equal",control_spec="none")
registry <- list(outcome="y",treatment="x",placebo_start_year=2008,
 placebo_end_year=2010,policy_start_year=2011,policy_end_year=2022)
noise <- border_prepare(pre_panel,members,row,registry,period="pre",require_treatment=FALSE)
wage <- border_prepare(pre_panel,members,row,registry,period="pre")
stopifnot(nrow(noise$data)==12,nrow(wage$data)==8,
 identical(sort(unique(noise$data$year)),2008:2010),
 all(is.na(noise$data$border_x[noise$data$year==2009])))
cat("Passed weighted FWL, side-gap aggregation, exact pair deletion, dyadic score, rank and scale checks.\n")
