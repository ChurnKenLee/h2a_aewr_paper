source("code/designs/policy_2025/interim_helpers.R")
out <- id_output_dir()
cd_verify()
checks <- list()
check <- function(name, ok) {
  if (!isTRUE(ok)) stop("Policy 2025 validation failed: ", name)
  checks[[length(checks)+1L]] <<- tibble(check=name, status="passed")
}
contract <- jsonlite::read_json(file.path(CD,"design.json"), simplifyVector=TRUE)

# Compare the entire declared ledger, including failures, samples and inference.
for (name in c("wage_estimates.csv","earnings_pre_estimates.csv","earnings_interim_estimates.csv")) {
  old <- read_csv(file.path(CD,"tests/fixtures",name), show_col_types=FALSE)
  current <- read_csv(file.path(out,name), show_col_types=FALSE)
  check(paste0(name,"_columns_and_rows"), identical(names(old),names(current)) && nrow(old)==nrow(current))
  keys <- intersect(c("track","window","tau","tier","specification","weights","term"), names(current))
  cd_keys(current,keys)
  old <- arrange(old,across(all_of(keys)))
  current <- arrange(current,across(all_of(keys)))
  maximum <- 0
  for (column in names(current)) {
    a <- current[[column]]; b <- old[[column]]
    check(paste(name,column,"missingness",sep="_"), identical(is.na(a),is.na(b)))
    if (is.numeric(a)) {
      check(paste(name,column,"finite_support",sep="_"), identical(is.finite(a),is.finite(b)))
      difference <- abs(a[is.finite(a)]-b[is.finite(b)])
      maximum <- max(c(maximum,difference))
      check(paste(name,column,"numeric_equivalence",sep="_"), all(difference<=contract$equivalence_tolerance))
    } else check(paste(name,column,"equivalence",sep="_"), identical(a,b))
  }
  cat(name,": all rows reproduced; largest absolute difference",maximum,"\n")
}
# Validate additive Track 1 quantity ledger (79 rows, 1e-10 tolerance).
q_old <- read_csv(file.path(CD,"tests/fixtures/quantity_estimates.csv"), show_col_types=FALSE)
q_cur <- read_csv(file.path(out,"quantity_estimates.csv"), show_col_types=FALSE)
check("quantity_estimates.csv_columns_and_rows", identical(names(q_old),names(q_cur)) && nrow(q_old)==nrow(q_cur) && nrow(q_cur)==contract$expected_rows$quantity)
q_keys <- c("track","window","contrast","specification","outcome","estimator","treatment")
cd_keys(q_cur,q_keys)
q_old <- arrange(q_old,across(all_of(q_keys)))
q_cur <- arrange(q_cur,across(all_of(q_keys)))
q_max <- 0
for (column in names(q_cur)) {
  a <- q_cur[[column]]; b <- q_old[[column]]
  check(paste("quantity_estimates.csv",column,"missingness",sep="_"), identical(is.na(a),is.na(b)))
  if (is.numeric(a)) {
    check(paste("quantity_estimates.csv",column,"finite_support",sep="_"), identical(is.finite(a),is.finite(b)))
    difference <- abs(a[is.finite(a)]-b[is.finite(b)])
    q_max <- max(c(q_max,difference))
    check(paste("quantity_estimates.csv",column,"numeric_equivalence",sep="_"), all(difference<=contract$equivalence_tolerance))
  } else check(paste("quantity_estimates.csv",column,"equivalence",sep="_"), identical(a,b))
}
cat("quantity_estimates.csv : all rows reproduced; largest absolute difference",q_max,"\n")
qp <- read_parquet(file.path(out,"quantity_county_fy.parquet"))
cd_keys(qp,c("county_fips","year"))
check("quantity_panel_shape_and_geo", nrow(qp)==2921L*15L && is.character(qp$county_fips) && all(grepl("^[0-9]{5}$",qp$county_fips)) && is.character(qp$state_fips) && all(grepl("^[0-9]{2}$",qp$state_fips)))
h2y <- read_parquet(path_int("h2a_data_year.parquet"), col_select=c("county_fips","year","nbr_workers_certified_fiscal_year")) %>%
  mutate(year=as.integer(year)) %>% filter(year>=2012L, year<=2025L)
q_par <- qp %>% filter(year<=2025L) %>% select(county_fips,year,cert_all_fy) %>%
  left_join(h2y, by=c("county_fips","year")) %>% mutate(nbr_workers_certified_fiscal_year=coalesce(nbr_workers_certified_fiscal_year,0))
check("quantity_fy2012_2025_producer_parity", max(abs(q_par$cert_all_fy - q_par$nbr_workers_certified_fiscal_year)) < 1e-10)
q_head <- q_cur %>% filter(window=="full_fy", specification=="B_primary", outcome %in% c("hours","cert"), estimator=="PPML")
q_s5 <- q_cur %>% filter(window=="full_fy", specification %in% c("S5_control_mw_general","S5_control_mw_ag"))
check("quantity_headline_and_s5_all_46_states", nrow(q_head)==2L && all(q_head$states==46L) && all(q_head$status=="estimated") && all(is.finite(q_head$se_jackknife) & q_head$se_jackknife>0) && nrow(q_s5)==8L && all(q_s5$states==46L))
q_indep <- fixest::fepois(hours_full_fy ~ x_us_skill_1 | county_fips + aewr_region_id^year, data=qp %>% filter(year %in% 2025L:2026L), vcov=~state_fips, notes=FALSE)
q_h <- q_cur %>% filter(window=="full_fy", specification=="B_primary", outcome=="hours", estimator=="PPML")
check("independent_quantity_PPML_hours", nrow(q_h)==1L && abs(unname(coef(q_indep)["x_us_skill_1"]) - q_h$estimate)<1e-10 && abs(unname(se(q_indep)["x_us_skill_1"]) - q_h$se_state_crv1)<1e-10)

# Validate additive FY2026 Q4 mechanism heterogeneity ledger (96 rows, 1e-10 tolerance).
m_old <- read_csv(file.path(CD,"tests/fixtures/mechanism_estimates.csv"), show_col_types=FALSE)
m_cur <- read_csv(file.path(out,"mechanism_estimates.csv"), show_col_types=FALSE)
check("mechanism_estimates.csv_columns_and_rows", identical(names(m_old),names(m_cur)) && nrow(m_old)==nrow(m_cur) && nrow(m_cur)==contract$expected_rows$mechanisms)
m_keys <- c("track","panel","contrast","specification","outcome","estimator","term")
cd_keys(m_cur,m_keys)
m_old <- arrange(m_old,across(all_of(m_keys)))
m_cur <- arrange(m_cur,across(all_of(m_keys)))
m_max <- 0
for (column in names(m_cur)) {
  a <- m_cur[[column]]; b <- m_old[[column]]
  check(paste("mechanism_estimates.csv",column,"missingness",sep="_"), identical(is.na(a),is.na(b)))
  if (is.numeric(a)) {
    check(paste("mechanism_estimates.csv",column,"finite_support",sep="_"), identical(is.finite(a),is.finite(b)))
    difference <- abs(a[is.finite(a)]-b[is.finite(b)])
    m_max <- max(c(m_max,difference))
    check(paste("mechanism_estimates.csv",column,"numeric_equivalence",sep="_"), all(difference<=contract$equivalence_tolerance))
  } else check(paste("mechanism_estimates.csv",column,"equivalence",sep="_"), identical(a,b))
}
cat("mechanism_estimates.csv : all rows reproduced; largest absolute difference",m_max,"\n")
mp <- read_parquet(file.path(out,"mechanisms_county_fy.parquet"))
cd_keys(mp,c("county_fips","year"))
check("mechanisms_panel_shape_and_geo", nrow(mp)==2921L*3L && is.character(mp$county_fips) && all(grepl("^[0-9]{5}$",mp$county_fips)) && is.character(mp$state_fips) && all(grepl("^[0-9]{2}$",mp$state_fips)))
check("mechanisms_conservation_identities",
  max(abs((mp$incumbent_1yr_hours + mp$entrant_1yr_hours) - mp$worksite_bus_hours)) < 1e-8 &&
  max(abs((mp$worksite_bus_hours + mp$non_wb_hours) - mp$hours_full_fy)) < 1e-8 &&
  max(abs((mp$h2alc_hours + mp$direct_hours) - mp$hours_full_fy)) < 1e-8 &&
  max(abs((mp$early_start_hours + mp$late_start_hours) - mp$hours_full_fy)) < 1e-8 &&
  max(abs((mp$certified_apps + mp$denied_withdrawn_apps + mp$other_apps) - mp$filed_apps)) < 1e-8)
m_fam <- m_cur %>% filter(in_primary_family)
check("mechanisms_primary_holm_family", nrow(m_fam)==12L && all(m_fam$contrast=="fy2025_2026") && all(is.finite(m_fam$p_holm_family) == is.finite(m_fam$p_jackknife)) && all(m_fam$p_holm_family[is.finite(m_fam$p_holm_family)] >= m_fam$p_jackknife[is.finite(m_fam$p_holm_family)] - 1e-12))
m_indep <- fixest::fepois(incumbent_1yr_hours ~ x_us_skill_1 | county_fips + aewr_region_id^year, data=mp %>% filter(year %in% 2025L:2026L), vcov=~state_fips, notes=FALSE)
m_inc <- m_cur %>% filter(contrast=="fy2025_2026", specification=="wb_1yr_incumbent_hours")
check("independent_mechanisms_PPML_incumbent_hours", nrow(m_inc)==1L && abs(unname(coef(m_indep)["x_us_skill_1"]) - m_inc$estimate)<1e-10 && abs(unname(se(m_indep)["x_us_skill_1"]) - m_inc$se_state_crv1)<1e-10)
p <- read_parquet(file.path(out,"wage_filer_state_occupation.parquet"))
cd_keys(p,c("tier","filer","state_fips","soc","fiscal_year"))
check("wage_geographic_strings",is.character(p$state_fips) && all(grepl("^[0-9]{2}$",p$state_fips)))
check("wage_CDF_complete_and_ordered",all(is.finite(p$cdf0)) && all(p$cdf0<=p$cdf25+1e-12) && all(p$cdf25<=p$cdf100+1e-12))
allocation <- read_csv(file.path(out,"wage_allocation_check.csv"),show_col_types=FALSE)
check("allocated_positions_and_hours_conserved",all(allocation$max_position_error<1e-8) && all(allocation$max_hour_error<1e-6))
floor <- read_csv(file.path(out,"wage_floor_audit.csv"),show_col_types=FALSE)
check("effective_floor_not_invented",all(floor$resolved_effective_floors==0))

# Independent WLS on exactly the continuing R7 occupation sample.
baseline <- filter(p,tier=="R7",fiscal_year==2025,soc!="unknown")
current <- filter(p,tier=="R7",fiscal_year==2026,soc!="unknown")
d <- inner_join(baseline,current,by=c("tier","filer","state_fips","aewr_region_id","soc"),suffix=c("0","1")) %>%
  left_join(read_csv(file.path(out,"wage_state_dose.csv"),col_types=cols(state_fips="c",aewr_region_id="c",.default=col_guess())),by="state_fips",suffix=c("","_dose")) %>%
  mutate(change=wage1-wage0)
independent <- coef(lm(change ~ dose1_us + factor(aewr_region_id),data=d,weights=h0))[["dose1_us"]]
w <- read_csv(file.path(out,"wage_estimates.csv"),show_col_types=FALSE) %>% filter(tier=="R7",tau==2026,specification=="occupation_dw",weights=="h0")
check("independent_wage_WLS",nrow(w)==1 && abs(independent-w$estimate)<1e-10 && nrow(d)==w$n)
check("pre_period_keys",{q<-read_parquet(file.path(out,"earnings_pre.parquet"));cd_keys(q,c("industry_code","county_fips","year","qtr"));max(q$year)<=2024})
check("no_administration_artifacts",!any(grepl("^admin_",list.files(out))))
cd_write(bind_rows(checks),out,"validation.csv")
jsonlite::write_json(list(status="passed",checks=length(checks),tolerance=contract$equivalence_tolerance,
  coefficient_rows=contract$expected_rows,outcomes_previously_inspected=TRUE),file.path(out,"validation.json"),pretty=TRUE,auto_unbox=TRUE)
cat("Policy 2025 migration and artifact checks passed:",length(checks),"\n")
