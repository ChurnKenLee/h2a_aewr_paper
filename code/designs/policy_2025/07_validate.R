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
