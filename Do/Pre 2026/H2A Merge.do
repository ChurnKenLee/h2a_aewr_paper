** H2a: Merge
** Hoxie
** 11/25/23

** merge main dataset ** 

/*

Each county pair should show up twice. Once for each county in the pair

*/

cd "$Data"
use border_counties_allmatches, clear // this is the base layer

gen census_period = 2007

count 

append using border_counties_allmatches

replace census_period = 2012 if census_period == .

count 

append using border_counties_allmatches

replace census_period = 2017 if census_period == .

count 

** merge in border pair data ** 

gen state_abbrev = strtrim(state)
gen neighbor_abbrev = strtrim(neighbor_state)  

merge m:1 border_pair census_period using border_pair_treatmentassignment

tab border_pair if _merge != 3 // Just DC

keep if _merge == 3
drop _merge

** line up 1 and 2 w/ state, neighbor 

global treatvar "treat_daewr_b2007 treat_daewr_b2012 treat_daewr_avg treat_daewr_stable"

gen state_1_match = 0
replace state_1_match = 1 if state_abbrev == state_abbrev_1

foreach v in $treatvar {
	rename `v' old_`v' 
	gen `v' = . 
	replace `v' = 1 if state_1_match == 1 & old_`v' == 1
	replace `v' = 0 if state_1_match == 1 & old_`v' == 2
	replace `v' = 1 if state_1_match == 0 & old_`v' == 2
	replace `v' = 0 if state_1_match == 0 & old_`v' == 1
}

** merge in AEWRs ** 

drop state neighbor_state orderpair state_abbrev_1 state_abbrev_2 old_treat_daewr_b2007 old_treat_daewr_b2012 old_treat_daewr_avg old_treat_daewr_stable state_1_match xstate

merge m:1 state_abbrev census_period using aewr_merge

keep if _merge == 3 // HI missing
drop _merge

rename aewr state_aewr 
rename aewr_ppi state_aewr_ppi

rename state_abbrev temp // just for now

rename neighbor_abbrev state_abbrev

merge m:1 state_abbrev census_period using aewr_merge // this one is the neighbor aewr

keep if _merge == 3 // HI missing
drop _merge

rename state_abbrev neighbor_abbrev
rename temp state_abbrev
rename aewr aewr_neighbor 
rename aewr_ppi aewr_ppi_neighbor
rename state_aewr aewr
rename state_aewr_ppi aewr_ppi

sort border_pair census_period fipscounty fipsneighbor

** AEWR region for the original state ** 

merge m:1 state_abbrev using aewr_regions

keep if _merge == 3
drop _merge 

** H2A ** 

cd "$Data"
merge m:1 census_period fipscounty using h2a_censusperiod

keep if _merge != 2 // can't use these, interior counties
drop _merge

global h2afixvars "h2a_man_hours_certified h2a_man_hours_requested h2a_nbr_workers_certified h2a_nbr_workers_requested n_applications"

sum $h2afixvars

foreach v in $h2afixvars {
	replace `v' = 0 if `v' == . // replace NAs with 0s 
}

sum $h2afixvars

** ACS Controls ** 

cd "$Data"
merge m:1 census_period fipscounty using acs_immigrant_cleaned

tab fipscounty if _merge == 1 // 22087 46113 missing census years.. 

tab census_period if _merge == 2 // non-borders

gen panel_issue = 0 
replace panel_issue = 1 if _merge != 3

drop if _merge == 2

drop _merge

** regression vars ** 

* AEWR 

gen absdaewr = abs(aewr - aewr_neighbor)
gen absdaewr_ppi = abs(aewr_ppi - aewr_ppi_neighbor)

* 2007-2012 AER

sort fipscounty border_pair census_period

gen c07aewr = aewr if census_period == 2007
gen c12aewr = aewr if census_period == 2012

egen b07aewr = max(c07aewr), by(fipscounty)
egen b12aewr = max(c12aewr), by(fipscounty)

gen c07aewr_ppi = aewr_ppi if census_period == 2007
gen c12aewr_ppi = aewr_ppi if census_period == 2012

egen b07aewr_ppi = max(c07aewr_ppi), by(fipscounty)
egen b12aewr_ppi = max(c12aewr_ppi), by(fipscounty)

* diff 

gen c07absdaewr = absdaewr if census_period == 2007
gen c12absdaewr = absdaewr if census_period == 2012

egen b07absdaewr = max(c07absdaewr), by(fipscounty)
egen b12absdaewr = max(c12absdaewr), by(fipscounty)

gen c07absdaewr_ppi = absdaewr_ppi if census_period == 2007
gen c12absdaewr_ppi = absdaewr_ppi if census_period == 2012

egen b07absdaewr_ppi = max(c07aewr_ppi), by(fipscounty)
egen b12absdaewr_ppi = max(c12aewr_ppi), by(fipscounty)


drop c07* c12*

gen aewr_avg0712 = (b12aewr + b07aewr) / 2
gen aewr_ppi_avg0712 = (b12aewr_ppi + b07aewr_ppi) / 2

gen absdaewr_avg0712 = (b12absdaewr + b07absdaewr) / 2
gen absdaewr_ppi_avg0712 = (b12absdaewr_ppi + b07absdaewr_ppi) / 2

bysort census_period: sum aewr_avg0712 aewr_ppi_avg0712 // treatment

kdensity aewr_avg0712 if census_period == 2012
kdensity aewr_ppi_avg0712 if census_period == 2012

* post dummy
gen postdummy = 0 
replace postdummy = 1 if census_period > 2012
* interaction with treatment 
foreach v in $treatvar aewr_avg0712 aewr_ppi_avg0712 {
	gen `v'_aewr_ppi = `v' * absdaewr_ppi
	gen `v'_aewr = `v' * absdaewr
	
	gen p_`v' = postdummy * `v'
	gen p_`v'_aewrppi = postdummy * `v'_aewr_ppi
	gen p_`v'_aewr = postdummy * `v'_aewr
}


sum post* p_*

** county fe **

egen county_fe = group(fipscounty)

* state FE 

egen state_fe = group(state_abbrev)

* aewr FE: aewr_region_num

* time FEs 

egen time_fe = group(census_period)

* pair fe 

gen neighborfips_greater = 0 
replace neighborfips_greater = 1 if fipsneighbor > fipscounty

gen f1 = string(fipscounty)
gen f2 = string(fipsneighbor)

gen county_pair_id = ""
replace county_pair_id = f1 + "_" + f2 if neighborfips_greater == 0
replace county_pair_id = f2 + "_" + f1 if neighborfips_greater == 1

drop f1 f2 neighborfips_greater

egen pair_fe = group(county_pair_id)

** pair x year FE **

egen pair_year_fe = group(county_pair_id census_period)

** nice log vars **

global logvars "absdaewr_avg0712 absdaewr_ppi_avg0712 aewr_avg0712 aewr_ppi_avg0712 aewr aewr_ppi aewr_neighbor aewr_ppi_neighbor aewr_region_num h2a_man_hours_certified h2a_man_hours_requested h2a_nbr_workers_certified h2a_nbr_workers_requested n_applications  p_treat_daewr_b2007_aewrppi p_treat_daewr_b2007_aewr  p_treat_daewr_b2012_aewrppi p_treat_daewr_b2012_aewr  p_treat_daewr_avg_aewrppi p_treat_daewr_avg_aewr  p_treat_daewr_stable_aewrppi p_treat_daewr_stable_aewr p_aewr_avg0712 p_aewr_avg0712_aewrppi p_aewr_avg0712_aewr p_aewr_ppi_avg0712 p_aewr_ppi_avg0712_aewrppi p_aewr_ppi_avg0712_aewr pop_female_np_doc pop_female_np_nat pop_female_np_undoc pop_female_p_doc pop_female_p_nat pop_female_p_undoc pop_male_np_doc pop_male_np_nat pop_male_np_undoc pop_male_p_doc pop_male_p_nat pop_male_p_undoc pop_tot pop_prime pop_prime_m pop_prime_f pop_undoc pop_primeundoc share_pop_undoc_prime"

count 
foreach lv in $logvars {
	gen ln_`lv' = log(`lv')
	gen ln1_`lv' = log(`lv' + 1) // handle 0s 
}

sum ln_* ln1_*

gen postln_aewr_ppi_avg0712 = postdummy * ln_aewr_ppi_avg0712
gen postln_aewr_avg0712 = postdummy * ln_aewr_avg0712

gen posttreatln_aewr_ppi_avg0712 = postdummy * ln_aewr_ppi_avg0712 * treat_daewr_avg // NAs from internal Region Counties
gen posttreatln_aewr_avg0712 = postdummy * ln_aewr_avg0712 * treat_daewr_avg // NAs from internal Region Counties

gen posttreatln_absdaewr_ppi_avg0712 = postdummy * ln_absdaewr_ppi_avg0712 * treat_daewr_avg // NAs from internal Region Counties
gen posttreatln_absdaewr_avg0712 = postdummy * ln_absdaewr_avg0712 * treat_daewr_avg // NAs from internal Region Counties

tab aewr_region_diff // full sample : 4506

kdensity ln_aewr_avg0712
reg aewr aewr_avg0712 if census_period == 2017

** extensive margein 

gen any_h2a = 0 
replace any_h2a = 1 if n_applications > 0 

** Normal CI ** p_treat_daewr_avg

global controlvars "ln_pop_tot ln_share_pop_undoc_prime ln_pop_primeundoc"

* dummy * 

reghdfe n_applications p_treat_daewr_avg $controlvars if aewr_region_diff == 1 & census_period > 2007, absorb(i.county_fe i.pair_fe#postdummy) vce(robust)

reghdfe h2a_man_hours_requested p_treat_daewr_avg $controlvars if aewr_region_diff == 1 & census_period > 2007, absorb(i.county_fe i.pair_fe#postdummy) vce(robust)

reghdfe h2a_nbr_workers_requested p_treat_daewr_avg $controlvars if aewr_region_diff == 1 & census_period > 2007, absorb(i.county_fe i.pair_fe#postdummy) vce(robust)

reghdfe any_h2a p_treat_daewr_avg $controlvars if aewr_region_diff == 1 & census_period > 2007, absorb(i.county_fe i.pair_fe#postdummy) vce(robust)

reghdfe h2a_nbr_workers_requested aewr_ppi $controlvars if aewr_region_diff == 1 & census_period > 2007, absorb(i.county_fe i.pair_fe#postdummy) vce(robust)

reghdfe h2a_nbr_workers_requested aewr_ppi $controlvars if aewr_region_diff == 1 & census_period == 2012, absorb(i.pair_fe) vce(cluster aewr_region_pair)

reghdfe h2a_nbr_workers_requested aewr_ppi $controlvars if aewr_region_diff == 1 & census_period == 2017, absorb(i.pair_fe) vce(cluster aewr_region_pair)

reghdfe h2a_nbr_workers_requested aewr_ppi $controlvars if aewr_region_diff == 1 & census_period > 2007, absorb(i.county_fe i.pair_fe#postdummy) vce(cluster aewr_region_pair) // vce(robust)

/* 2 x 2
** do the 2 x 2 DiD by hand ** 

* target: 

reg ln_h2a_nbr_workers_requested p_treat_daewr_avg $controlvars i.time_fe i.pair_fe if aewr_region_diff == 1 & census_period > 2007, robust // -.43 = beta

reg ln_h2a_nbr_workers_requested p_treat_daewr_avg  i.time_fe i.pair_fe if aewr_region_diff == 1 & census_period > 2007, robust // -.43 = beta

** 

collapse (mean) ln_h2a_nbr_workers_requested if aewr_region_diff == 1 & census_period > 2007, by(census_period treat_daewr_avg)

twoway line ln_h2a_nbr_workers_requested census_period if treat_daewr_avg == 1,  legend(label( 1 "T")) || ///
line ln_h2a_nbr_workers_requested census_period if treat_daewr_avg == 0, legend(label( 2 "C")) yline(3.803389 3.359362)

di 3.803389 - 3.359362 // pre diff

di 4.207344 - 3.654376 // post diff

di (4.207344 - 3.654376) - (3.803389 - 3.359362) // DiD

*/
/* ** OLD 20240109 **
* Dummy 

reghdfe ln1_h2a_man_hours_requested p_treat_daewr_avg if aewr_region_diff == 1, absorb(i.time_fe i.county_fe) vce(cluster aewr_region_pair)

reghdfe ln1_h2a_nbr_workers_requested p_treat_daewr_avg if aewr_region_diff == 1, absorb(i.time_fe i.county_fe) vce(cluster aewr_region_pair)

reghdfe ln1_n_applications p_treat_daewr_avg if aewr_region_diff == 1, absorb(i.time_fe i.county_fe) vce(cluster aewr_region_pair)

* cts 

reghdfe ln1_h2a_man_hours_requested postln_aewr_ppi_avg0712 if aewr_region_diff == 1, absorb(i.time_fe i.county_fe) vce(cluster aewr_region_pair)

reghdfe ln1_h2a_nbr_workers_requested postln_aewr_ppi_avg0712 if aewr_region_diff == 1, absorb(i.time_fe i.county_fe) vce(cluster aewr_region_pair)

reghdfe ln1_n_applications postln_aewr_ppi_avg0712 if aewr_region_diff == 1, absorb(i.time_fe i.county_fe) vce(cluster aewr_region_pair)
/*