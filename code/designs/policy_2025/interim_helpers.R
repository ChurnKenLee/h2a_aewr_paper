source("code/designs/policy_2025/helpers.R")
# This explicit-period helper is separate from the frozen three-quarter helper.
# docs-ground:start policy-2025-interim-window
interim_window <- function(q, target) {
  cd_keys(target,c("year","qtr"));stopifnot(all(target$qtr %in% 1:4))
  need<-bind_rows(target,target %>% mutate(year=year-1L)) %>% distinct()
  if(nrow(anti_join(need,distinct(q,year,qtr),by=c("year","qtr"))))stop("Missing declared interim quarter")
  cd_keys(q,c("county_fips","year","qtr"))
  eligible<-q %>% semi_join(need,by=c("year","qtr")) %>% filter(disclosed,emp>=10,is.finite(avg_wkly_wage),avg_wkly_wage>0) %>% mutate(lw=log(avg_wkly_wage))
  paired<-eligible %>% inner_join(eligible %>% transmute(county_fips,year=year+1L,qtr,prior_log=lw,prior_emp=emp),by=c("county_fips","year","qtr")) %>% semi_join(target,by=c("year","qtr"))
  paired %>% group_by(county_fips) %>% summarise(nq=n(),change=mean(lw-prior_log),emp_l=mean(prior_emp),.groups="drop") %>% filter(nq==nrow(target))
}

# docs-ground:end policy-2025-interim-window
