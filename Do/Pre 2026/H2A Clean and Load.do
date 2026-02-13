** H2a: Load and Clean Datasets
** Hoxie
** 11/25/23

/*
Clean: 
Censu of Ag
H2A
OEWS
USDA_Surveys

*/

** AEWR Regions **

cd "$Data"
import delimited "aewr_regions.csv", clear

save aewr_regions, replace // state_abbrev

** PPI *************************************************************************
*Source: https://fred.stlouisfed.org/series/WPU01

cd "$Data"
import delimited "WPU01.csv", clear

split date, p(-)
gen year = real(date1)

collapse (mean) wpu01, by(year)

twoway line wpu01 year

gen wpu01_12 = wpu01 if year == 2012
egen b12_wpu01 = max(wpu01_12)

gen ppi_ag_b2012 = wpu01 / b12_wpu01

twoway line ppi_ag_b2012 year if year > 2002 & year <= 2022, ytitle("Producer Price Index, Farm Products (Base = 2012)") xtitle("Year") graphregion(fcolor(white)) xscale(range(2003 2022)) xlabel(2004 2008 2012 2016 2020)

cd "$Output"
graph export "fig_line_ppi_farmproducts_2012.png", replace as(png)

keep year ppi_ag_b2012

cd "$Data"
save ppi_ag, replace

** Raw H2A Data *****************************************************************
/*
cd "$Data"
import delimited "H2A Final Worksite County Data.csv", clear

* should be numbers: nbr_workers_requested nbr_workers_certified basic_number_of_hours basic_rate_of_pay
* postal codes: worksite_postal_code employer_postal_code
* dates are messed up


*duplicates tag , gen(dupobs)

* we want the
*/

** County Boundary *************************************************************

cd "$Data"
import delimited "county_adjacency2010.csv" , clear

split countyname, parse(,)

rename countyname2 state
drop countyname1

split neighborname, parse(,)

rename neighborname2 neighbor_state
drop neighborname1

gen xstate = 0 
replace xstate = 1 if neighbor_state != state

gen statefips = substr(string(fipscounty, "%05.0f"), 1, 2)
gen neighborstatefips = substr(string(fipsneighbor, "%05.0f"), 1, 2)

gen orderpair = ""
replace orderpair = "state first" if real(statefips) <= real(neighborstatefips)
replace orderpair = "neighbor first" if real(statefips) > real(neighborstatefips)

gen border_pair = ""
replace border_pair = strtrim(state) + "_" + strtrim(neighbor_state) if orderpair == "state first" & xstate == 1
replace border_pair = strtrim(neighbor_state) + "_" + strtrim(state) if orderpair == "neighbor first" & xstate == 1

keep if border_pair != "" // 1308 counties per year 

save border_counties_allmatches, replace
export delimited "border_counties_allmatches.csv", replace

drop if orderpair == "neighbor first"

gen samp_count = 1

sort border_pair

duplicates tag border_pair fipscounty, gen(dup_fipscounty)
duplicates tag border_pair fipsneighbor, gen(dup_fipsneighbor)

gen dup_county_sum = 0
replace dup_county_sum = dup_fipscounty / (1 + dup_fipscounty)

gen dup_neighbor_sum = 0
replace dup_neighbor_sum = dup_fipsneighbor / (1 + dup_fipsneighbor)

collapse (sum) samp_count dup_county_sum dup_neighbor_sum, by(border_pair)


gen unique_counties = samp_count - dup_county_sum
gen unique_neighbors = samp_count - dup_neighbor_sum

kdensity samp_count
kdensity unique_neighbors
kdensity unique_counties

gen unique_counts_tot = unique_counties + unique_neighbors

twoway kdensity samp_count, color(dknavy) legend(label(1 "Pairs")) || ///
	kdensity unique_counts_tot, color(maroon) lpattern("longdash") legend(label(2 "Counties")) ytitle("Density") xtitle("Observations within State Border Pair") graphregion(fcolor(white))

cd "$Output"
graph export "fig_dense_stateborder_samplecounts.png", replace as(png)

split border_pair, p(_)

rename border_pair1 stateabbrev_1
rename border_pair2 stateabbrev_2

keep border_pair samp_count stateabbrev_1 stateabbrev_2 unique_counts_tot

count 


cd "$Data"	
save state_border_pairs, replace
export delimited "state_border_pairs.csv", replace


** Census of Ag ****************************************************************

cd "$Data"
use census_of_agriculture, clear
* COMMODITY TOTALS-ALL CLASSES, sales

* three years so far: 2007, 2012, 2017
* want a county-year level dataset

merge m:1 year using ppi_ag

keep if _merge == 3
drop _merge

rename year census_period

** h2a *************************************************************************

cd "$Data"
use h2a, clear

* FY 2006 to 2022 *

* create year bins *
count 

gen census_period = . 
replace census_period = 2007 if year == 2007
replace census_period = 2012 if year > 2007 & year <= 2012
replace census_period = 2017 if year > 2012 & year <= 2017

keep if census_period != .

sum h2a_man_hours_certified h2a_man_hours_requested h2a_nbr_workers_certified h2a_nbr_workers_requested n_applications

count 

collapse (sum) h2a_man_hours_certified h2a_man_hours_requested h2a_nbr_workers☺_certified h2a_nbr_workers_requested n_applications, by(census_period state_fips_code county_fips_code)

tab census_period

keep if county_fips_code != "" & state_fips_code != "72"

gen fipscounty = real(state_fips_code + county_fips_code)

drop state_fips_code county_fips_code

save h2a_censusperiod, replace // fipscounty (#)

** make a line graph ** 

cd "$Data"
use h2a, clear

collapse (sum) h2a_man_hours_certified h2a_man_hours_requested h2a_nbr_workers_certified h2a_nbr_workers_requested n_applications, by(year)

keep if year >= 2007 & year <= 2022

gen certgap_manhrs = h2a_man_hours_certified - h2a_man_hours_requested
gen certgap_workers = h2a_nbr_workers_certified - h2a_nbr_workers_requested

gen denial_rate_manhrs = ( h2a_man_hours_requested - h2a_man_hours_certified) / h2a_man_hours_requested 
gen denial_rate_workers = ( h2a_nbr_workers_requested - h2a_nbr_workers_certified) / h2a_nbr_workers_requested 

** idex ** 

global indexvars "h2a_man_hours_certified h2a_man_hours_requested h2a_nbr_workers_certified h2a_nbr_workers_requested n_applications"

sort year

foreach v in $indexvars {
	gen v07_`v' = `v' if year == 2007
	egen b07_`v' = max(v07_`v')
	gen indx07_`v' = `v' / b07_`v'
}

foreach v in $indexvars {
	gen v12_`v' = `v' if year == 2012
	egen b12_`v' = max(v12_`v')
	gen indx12_`v' = `v' / b12_`v'
}

foreach v in $indexvars {
	gen v14_`v' = `v' if year == 2014
	egen b14_`v' = max(v14_`v')
	gen indx14_`v' = `v' / b14_`v'
}

twoway line h2a_man_hours_certified year, xline(2007 2012 2017)
twoway line h2a_man_hours_requested year, xline(2007 2012 2017)
twoway line h2a_nbr_workers_certified year, xline(2007 2012 2017)
twoway line h2a_nbr_workers_requested year, xline(2007 2012 2017)
twoway line n_applications year, xline(2007 2012 2017)

twoway line certgap_manhrs year, xline(2007 2012 2017 2022)
twoway line certgap_workers year, xline(2007 2012 2017 2022)

twoway line denial_rate_manhrs year, xline(2007 2012 2017 2022)
twoway line denial_rate_workers year, xline(2007 2012 2017 2022)

* indexed graphs *

twoway line indx14_h2a_man_hours_certified year, color(dknavy) legend(label(1 "Certified")) || ///
	line indx14_h2a_man_hours_requested year, color(maroon) legend(label(2 "Requested"))

twoway line indx14_h2a_nbr_workers_certified year, color(dknavy) legend(label(1 "Certified")) || ///
	line indx14_h2a_nbr_workers_requested year, color(maroon) legend(label(2 "Requested"))

twoway line indx12_h2a_man_hours_certified year, color(dknavy) legend(label(1 "Man-Hours Certified")) || ///
	line indx12_h2a_nbr_workers_certified year, color(maroon) legend(label(2 "Workers Certified")) lpattern("longdash") ytitle("Indexed Values (2012 = 1)") xtitle("Year") xscale(range(2007 2022)) xlabel(2008 (2) 2022) ylabel(0 1 2 3 4 5 6) graphregion(fcolor(white))
	
twoway line indx12_h2a_man_hours_certified year if year > 2007 & year <= 2017, color(dknavy) legend(label(1 "Man-Hours Certified")) || ///
	line indx12_h2a_nbr_workers_certified year if year > 2007 & year <= 2017, color(maroon) legend(label(2 "Workers Certified")) lpattern("longdash") || /// 
	line indx12_n_applications year if year > 2007 & year <= 2017, color(teal) legend(label(3 "Number of Applications")) lpattern("shortdash") ytitle("Indexed Values (2012 = 1)") xtitle("Year") xscale(range(2008 2017)) xlabel(2008 (2) 2016) graphregion(fcolor(white)) xline(2012)

cd "$Output"
graph export "fig_line_h2a_certified_manhrs_workers_year.png", replace as(png)
	
** ACS & QCEW **************************************************************************

cd "$Data"
use acs_qcew, clear
* counts of workers 

gen census_period = year

gen acs_agemp = acs_crop_emp + acs_animal_emp

gen empratio_crop_animal_acs = acs_crop_emp / acs_animal_emp
gen empratio_crop_animal_qcew = qcew_crop_emp / qcew_animal_emp

corr empratio_crop_animal_acs empratio_crop_animal_qcew

gen ln_empratio_crop_animal_acs = log(empratio_crop_animal_acs)
gen ln_empratio_crop_animal_qcew = log(empratio_crop_animal_qcew)

binscatter ln_empratio_crop_animal_acs ln_empratio_crop_animal_qcew [aw = acs_agemp]

count if acs_crop_emp != . & acs_animal_emp == .
count if acs_animal_emp != . & acs_crop_emp == .

count if qcew_crop_emp != . & qcew_animal_emp == .
count if qcew_animal_emp != . & qcew_crop_emp == .

gen qcew_crop_emp_clean = qcew_crop_emp 
replace qcew_crop_emp_clean = 0 if qcew_animal_emp != . & qcew_crop_emp == . 

gen acs_crop_emp_clean = acs_crop_emp 
replace acs_crop_emp_clean = 0 if acs_animal_emp != . & acs_crop_emp == . 

gen qcew_animal_emp_clean = qcew_animal_emp 
replace qcew_animal_emp_clean = 0 if qcew_crop_emp != . & qcew_animal_emp == . 

gen acs_animal_emp_clean = acs_crop_emp 
replace acs_animal_emp_clean = 0 if acs_crop_emp != . & acs_animal_emp == . 

gen acs_agemp_clean = acs_crop_emp_clean + acs_animal_emp_clean

drop year

cd "$Data"
save acs_qcew_cleaned, replace // census period, state fips (01) county fips (001)
 
** nawspad **********************************************************************

cd "$Data"
use nawspad, clear

** OEWS *************************************************************************

cd "$Data"
use oews, clear



** AEWR ************************************************************************

cd "$Data"
use aewr, clear

global fixvars "year aewr"

foreach v in $fixvars {
	
	rename `v' `v'old
	gen `v' = real(`v'old)
	drop `v'old 

}

gen census_period = . 
replace census_period = 2007 if year > 2002 & year <= 2007 
replace census_period = 2012 if year > 2007 & year <= 2012 
replace census_period = 2017 if year > 2012 & year <= 2017 

keep if census_period != .

* real AEWR 

merge m:1 year using ppi_ag

keep if _merge == 3

drop _merge

gen aewr_ppi = aewr / ppi_ag

collapse (mean) aewr aewr_ppi, by(census_period state_fips_code)

gen fips = real(state_fips_code)

merge m:1 fips using fips_codes

keep if _merge == 3

drop _merge

save aewr_cleaned, replace // census_period state (state_abbrev)

keep  aewr aewr_ppi census_period state_abbrev

save aewr_merge, replace // census_period state (state_abbrev) , only the basincs


**********************************************************************************

** merge to make border county pair dataset (state level) **

cd "$Data"
use state_border_pairs, clear

gen census_period = 2007 

append using state_border_pairs

replace census_period = 2012 if census_period == . 

append using state_border_pairs

replace census_period = 2017 if census_period == . 

**

rename stateabbrev_1 state_abbrev

count 

merge m:1 census_period state_abbrev using aewr_cleaned

tab border_pair if _merge == 1 // only DC

tab state_abbrev if _merge == 2 // will match in other pair 

keep if _merge == 3
drop _merge

* aewr regions

merge m:1 state_abbrev using aewr_regions

keep if _merge == 3
drop _merge

rename aewr_region_num aewr_region_num_1
rename state_abbrev state_abbrev_1
rename state_fips_code state_fips_code_1
rename aewr aewr_1
rename aewr_ppi aewr_ppi_1
rename fips fips_1
rename state state_1

rename stateabbrev_2 state_abbrev

merge m:1 census_period state_abbrev using aewr_cleaned

tab border_pair if _merge == 1 // only DC

tab state_abbrev if _merge == 2 // will match in other pair 

keep if _merge == 3
drop _merge

merge m:1 state_abbrev using aewr_regions

keep if _merge == 3
drop _merge

rename aewr_region_num aewr_region_num_2

rename state_abbrev state_abbrev_2
rename state_fips_code state_fips_code_2
rename aewr aewr_2
rename fips fips_2
rename state state_2
rename aewr_ppi aewr_ppi_2

sort border_pair census_period
bysort border_pair: gen lag_aewr_1 = aewr_1[_n - 1]
bysort border_pair: gen lag_aewr_2 = aewr_2[_n - 1]
bysort border_pair: gen lag_aewr_ppi_1 = aewr_ppi_1[_n - 1]
bysort border_pair: gen lag_aewr_ppi_2 = aewr_ppi_2[_n - 1]

gen daewr = aewr_2 - aewr_1
gen daewr_ppi = aewr_ppi_2 - aewr_ppi_1

gen lag_daewr = lag_aewr_2 - lag_aewr_1
gen lag_daewr_ppi = lag_aewr_ppi_2 - lag_aewr_ppi_1

gen adaewr = abs(aewr_2 - aewr_1)
gen adaewr_ppi = abs(aewr_ppi_2 - aewr_ppi_1)



gen lag_adaewr = abs(lag_aewr_2 - lag_aewr_1)
gen lag_adaewr_ppi = abs(lag_aewr_ppi_2 - lag_aewr_ppi_1)

gen chang_daewr = daewr - lag_daewr
gen chang_daewr_ppi = daewr_ppi - lag_daewr_ppi

bysort border_pair: gen lag_chang_daewr = chang_daewr[_n - 1]
bysort border_pair: gen lag_chang_daewr_ppi = chang_daewr_ppi[_n - 1]

sort census_period daewr border_pair

gen aewr_region_diff = 0 
replace aewr_region_diff = 1 if aewr_region_num_1 != aewr_region_num_2

gen aewr_region_1_larger = . 
replace aewr_region_1_larger = 1 if aewr_region_num_1 > aewr_region_num_2
replace aewr_region_1_larger = 0 if aewr_region_num_1 < aewr_region_num_2

gen reg1 = string(aewr_region_num_1)
gen reg2 = string(aewr_region_num_2)

gen aewr_region_pair = ""
replace aewr_region_pair = reg1 + "_" + reg2 if aewr_region_1_larger == 0
replace aewr_region_pair = reg2 + "_" + reg1 if aewr_region_1_larger == 1

drop reg1 reg2 aewr_region_num_1 aewr_region_num_2 aewr_region_1_larger 

twoway kdensity adaewr if census_period == 2007 & aewr_region_diff == 1, color(dknavy) legend(label(1 "2007")) || ///
 kdensity adaewr if census_period == 2012 & aewr_region_diff == 1, color(maroon) legend(label(2 "2012")) lpattern("longdash") || /// 
 kdensity adaewr if census_period == 2017 & aewr_region_diff == 1, color(teal) legend(label(3 "2017")) lpattern("shortdash") ytitle("Density") xtitle("Absolute Difference in AEWR (Nominal USD)") graphregion(fcolor(white)) legend(rows(1))


cd "$Output"
graph export "fig_density_borderstatepairs_absdaewr.png", replace as(png)


twoway kdensity daewr if census_period == 2007 & aewr_region_diff == 1, color(dknavy) legend(label(1 "2007")) || ///
 kdensity daewr if census_period == 2012 & aewr_region_diff == 1, color(maroon) legend(label(2 "2012")) lpattern("longdash") || /// 
 kdensity daewr if census_period == 2017 & aewr_region_diff == 1, color(teal) legend(label(3 "2017")) lpattern("shortdash") ytitle("Density") xtitle("Absolute Difference in AEWR (Nominal USD)") graphregion(fcolor(white)) legend(rows(1)) xline(0)

cd "$Output"
graph export "fig_density_borderstatepairs_daewr.png", replace as(png)

	
twoway kdensity adaewr_ppi if census_period == 2007 & aewr_region_diff == 1, color(dknavy) legend(label(1 "2007")) || ///
 kdensity adaewr_ppi if census_period == 2012 & aewr_region_diff == 1, color(maroon) legend(label(2 "2012")) lpattern("longdash") || /// 
 kdensity adaewr_ppi if census_period == 2017 & aewr_region_diff == 1, color(teal) legend(label(3 "2017")) lpattern("shortdash") ytitle("Density") xtitle("Absolute Difference in AEWR (2012 USD)") graphregion(fcolor(white)) legend(rows(1))
 
cd "$Output"
graph export "fig_density_borderstatepairs_absdaewr_ppi2012.png", replace as(png)

twoway scatter daewr_ppi lag_daewr_ppi if census_period == 2012 & aewr_region_diff == 1, msymbol(Dh) color(navy%50) legend(label(1 "2012")) || ///
	scatter daewr_ppi lag_daewr_ppi if census_period == 2017 & aewr_region_diff == 1, msymbol(O) color(maroon%50) legend(label(2 "2017")) ytitle("Diff. in AEWR, Current Census Period (2012 USD)") xtitle("Diff. in AEWR, Previous Census Period (2012 USD)") graphregion(fcolor(white)) 
	cd "$Output"
graph export "fig_scat_daewr_ppi_lag_daewr_ppi.png", replace as(png)


twoway scatter adaewr_ppi lag_adaewr_ppi if census_period == 2012 & aewr_region_diff == 1, msymbol(Dh) color(navy%50) legend(label(1 "2012")) || ///
	scatter adaewr_ppi lag_adaewr_ppi if census_period == 2017 & aewr_region_diff == 1, msymbol(O) color(maroon%50) legend(label(2 "2017")) ytitle("Abs. Diff. in AEWR, Current Census Period (2012 USD)") xtitle("Abs. Diff. in  AEWR, Previous Census Period (2012 USD)") graphregion(fcolor(white)) 
	cd "$Output"
graph export "fig_scat_adaewr_ppi_lag_adaewr_ppi.png", replace as(png)

twoway scatter adaewr lag_adaewr if census_period == 2012 & aewr_region_diff == 1, msymbol(Dh) color(navy%50) legend(label(1 "2012")) || ///
	scatter adaewr lag_adaewr if census_period == 2017 & aewr_region_diff == 1, msymbol(O) color(maroon%50) legend(label(2 "2017")) ytitle("Abs. Diff. in AEWR, Current Census Period (Nominal USD)") xtitle("Abs. Diff. in  AEWR, Previous Census Period (Nominal USD)") graphregion(fcolor(white)) 
	cd "$Output"
graph export "fig_scat_adaewr_lag_adaewr.png", replace as(png)

twoway scatter daewr lag_daewr if census_period == 2012 & aewr_region_diff == 1, msymbol(Dh) color(navy%50) legend(label(1 "2012")) || ///
	scatter daewr lag_daewr if census_period == 2017 & aewr_region_diff == 1, msymbol(O) color(maroon%50) legend(label(2 "2017")) ytitle("Diff. in AEWR, Current Census Period (Nominal USD)") xtitle("Diff. in  AEWR, Previous Census Period (Nominal USD)") graphregion(fcolor(white)) xline(0) yline(0)
	cd "$Output"
graph export "fig_scat_daewr_lag_daewr.png", replace as(png)

reg adaewr_ppi lag_adaewr_ppi if census_period == 2012, robust
reg adaewr_ppi lag_adaewr_ppi if census_period == 2017, robust

reg daewr_ppi lag_daewr_ppi if census_period == 2012, robust 
reg daewr_ppi lag_daewr_ppi if census_period == 2017, robust

reg daewr lag_daewr if census_period == 2012 & lag_daewr_ppi != 0, robust 
reg daewr lag_daewr if census_period == 2017 & lag_daewr_ppi != 0, robust

reg daewr_ppi lag_daewr_ppi if census_period == 2012 & lag_daewr_ppi != 0, robust 
reg daewr_ppi lag_daewr_ppi if census_period == 2017 & lag_daewr_ppi != 0, robust

twoway kdensity chang_daewr_ppi if census_period == 2012 & aewr_region_diff == 1, color(dknavy) legend(label(1 "2007-2012")) || ///
 kdensity chang_daewr_ppi if census_period == 2017 & aewr_region_diff == 1, color(maroon) legend(label(2 "2012-2017")) lpattern("longdash") ytitle("Density") xtitle("Change in Diff. in AEWR (2012 USD)") graphregion(fcolor(white)) legend(rows(1))
 
cd "$Output"
graph export "fig_density_borderstatepairs_change_daewr_ppi2012.png", replace as(png)

twoway kdensity chang_daewr if census_period == 2012 & aewr_region_diff == 1, color(dknavy) legend(label(1 "2007-2012")) || ///
 kdensity chang_daewr if census_period == 2017 & aewr_region_diff == 1, color(maroon) legend(label(2 "2012-2017")) lpattern("longdash") ytitle("Density") xtitle("Change in Diff. in AEWR (Nominal USD)") graphregion(fcolor(white)) legend(rows(1))
 
cd "$Output"
graph export "fig_density_borderstatepairs_change_daewr.png", replace as(png)

scatter chang_daewr lag_chang_daewr if census_period == 2017 & aewr_region_diff == 1
reg chang_daewr lag_chang_daewr if census_period == 2017 & aewr_region_diff == 1, robust

keep aewr_region_pair aewr_region_diff census_period border_pair state_abbrev_1 aewr_1 aewr_ppi_1  lag_aewr_1 lag_aewr_ppi_1 state_abbrev_2  aewr_2 aewr_ppi_2  lag_aewr_2 lag_aewr_ppi_2 daewr daewr_ppi

** identify treatment and control states ** 

sort border_pair census_period

global treatvars "daewr"

foreach var in $treatvars { // Diffs are state2 - state1
	gen c07`var' = `var' if census_period == 2007
	gen c12`var' = `var' if census_period == 2012
	egen b07`var' = max(c07`var'), by(border_pair)
	egen b12`var' = max(c12`var'), by(border_pair)
	gen avg_b_`var' = (b07`var' + b12`var') / 2
	gen treat_`var'_b2007 = . // identifies high AEWR state in pair
	replace treat_`var'_b2007 = 2 if b07`var' > 0 
	replace treat_`var'_b2007 = 1 if b07`var' < 0 
	gen treat_`var'_b2012 = . // identifies high AEWR state in pair
	replace treat_`var'_b2012 = 2 if b12`var' > 0 
	replace treat_`var'_b2012 = 1 if b12`var' < 0 
	gen treat_`var'_avg = . // identifies high AEWR state in pair
	replace treat_`var'_avg = 2 if avg_b_`var' > 0 
	replace treat_`var'_avg = 1 if avg_b_`var' < 0 
}

sort border_pair census_period

* check dists: https://www.fb.org/market-intel/examining-the-2023-aewr

sum treat_*

tab treat_daewr_b2007 treat_daewr_b2012
tab treat_daewr_avg treat_daewr_b2012
tab treat_daewr_b2007 treat_daewr_avg

drop c07* c12*

** stable assignment ** 

gen treat_daewr_stable = . 
replace treat_daewr_stable = 1 if treat_daewr_b2007 == 1 & treat_daewr_b2012 == 1 & treat_daewr_avg == 1
replace treat_daewr_stable = 2 if treat_daewr_b2007 == 2 & treat_daewr_b2012 == 2 & treat_daewr_avg == 2

bysort census_period: tab treat_daewr_stable // 55 stable groups

keep aewr_region_pair aewr_region_diff census_period border_pair state_abbrev_1 state_abbrev_2 treat_*

cd "$Data"
save border_pair_treatmentassignment, replace // state_abbrev_1 census_period

** border pair / region diff **

keep if census_period == 2017 // doesn't matter the year

gen pair_count = 1

collapse (sum) pair_count, by(border_pair aewr_region_pair aewr_region_diff)

sort aewr_region_pair border_pair

keep aewr_region_pair border_pair

save border_pair_aewr_borders, replace

** US County Data from BEA *****************************************************

cd "$Data/CAEMP25N" // Employmebt
import delimited "CAEMP25N__ALL_AREAS_2001_2022_trim.csv", clear
* save farm employment, private non-farm, and total emp lines for each, make into long form dataset

* clean fips 

gen countyfips = real(subinstr(geofips, `"""', "",.))

drop geofips 
* 10 50 70 80 90

keep if linecode == 10 | linecode == 50 | linecode == 70 | linecode == 80 | linecode == 90

gen varname = ""
replace varname = "emp_tot" if linecode == 10
replace varname = "emp_farm_propr" if linecode == 50
replace varname = "emp_farm" if linecode == 70
replace varname = "emp_nonfarm" if linecode == 80
replace varname = "emp_privnonfarm" if linecode == 90

keep y* countyfips varname geoname

global fixvars "y2001 y2002 y2003 y2004 y2005 y2006 y2007 y2008 y2009 y2010 y2011 y2012 y2013 y2014 y2015 y2016 y2017 y2018 y2019 y2020 y2021 y2022"

foreach v in $fixvars {
	rename `v' old 
	gen `v' = real(old)
	drop old
}

** reshape ** 

reshape long y, i(countyfips geoname varname) j(year) s

rename y value

reshape wide value, i(countyfips geoname year) j(varname) s

rename year old
gen year = real(old)
drop old

global fixvars "emp_farm emp_farm_propr emp_nonfarm emp_privnonfarm emp_tot"

foreach v in $fixvars {
	rename value`v' `v'
}

rename countyfips fipscounty

gen census_period = . 
replace census_period = 2007 if year > 2002 & year <= 2007
replace census_period = 2012 if year > 2007 & year <= 2012
replace census_period = 2017 if year > 2012 & year <= 2017

keep if census_period != .

collapse (mean) emp_farm emp_farm_propr emp_nonfarm emp_privnonfarm emp_tot, by(fipscounty geoname census_period)


cd "$Data"
save bea_caemp25n_employment, replace

**** 

cd "$Data/CAINC45" // Farm Data
import delimited "CAINC45__ALL_AREAS_1969_2022_trim.csv", clear


* clean fips 

gen countyfips = real(subinstr(geofips, `"""', "",.))

drop geofips 
/*

line descr
60 Cash receipts: Crops
20 Cash receipts: Livestock and products
130 Government payments
210 Hired farm labor expenses
270 Cash receipts and other income
150 Production expenses
*/

keep if linecode == 60 | linecode == 20 | linecode == 130 | linecode == 210 | linecode == 270 | linecode == 150
tab description 

gen varname = ""
replace varname = "farm_cashcrops" if linecode == 60
replace varname = "farm_cashanimal" if linecode == 20
replace varname = "farm_govpayments" if linecode == 130
replace varname = "farm_laborexpense" if linecode == 210
replace varname = "farm_cashandinc" if linecode == 270
replace varname = "farm_prodexp" if linecode == 150

keep y* countyfips varname geoname region

global fixvars "y1969	y1970	y1971	y1972	y1973	y1974	y1975	y1976	y1977	y1978	y1979	y1980	y1981	y1982	y1983	y1984	y1985	y1986	y1987	y1988	y1989	y1990	y1991	y1992	y1993	y1994	y1995	y1996	y1997	y1998	y1999	y2000	y2001	y2002	y2003	y2004	y2005	y2006	y2007	y2008	y2009	y2010	y2011	y2012	y2013	y2014	y2015	y2016	y2017	y2018	y2019	y2020	y2021	y2022"

foreach v in $fixvars {
	rename `v' old 
	gen `v' = real(old)
	drop old
}

** reshape ** 

reshape long y, i(countyfips geoname region varname) j(year) s

rename y value

reshape wide value, i(countyfips geoname region year) j(varname) s

rename year old
gen year = real(old)
drop old

global fixvars "farm_cashandinc farm_cashanimal farm_cashcrops farm_govpayments farm_laborexpense farm_prodexp"

foreach v in $fixvars {
	rename value`v' `v'
}

rename countyfips fipscounty

gen census_period = . 
replace census_period = 2007 if year > 2002 & year <= 2007
replace census_period = 2012 if year > 2007 & year <= 2012
replace census_period = 2017 if year > 2012 & year <= 2017

keep if census_period != .

* real 

cd "$Data"
merge m:1 year using ppi_ag

keep if _merge == 3
drop _merge

foreach v in $fixvars {
	gen `v'ppi = `v' / ppi_ag_b2012
}

collapse (mean) farm_cashandinc farm_cashanimal farm_cashcrops farm_govpayments farm_laborexpense farm_prodexp farm_cashandincppi farm_cashanimalppi farm_cashcropsppi farm_govpaymentsppi farm_laborexpenseppi farm_prodexpppi, by(fipscounty geoname region census_period)

cd "$Data" // CAINC45
save bea_cainc45_farmexpenses, replace

** ACS Immigration Imputation ***************************************************

cd "$Data"
use acs_immigrant_imputed, clear

** reshape to wide ** 

gen prime_cat = "np"
replace prime_cat = "p" if prime_age == 1

gen imm_cat = "nat"
replace imm_cat = "doc" if immigrant_type == "documented_immigrant"
replace imm_cat = "undoc" if immigrant_type == "undocumented_immigrant"

gen obcat = "_" + sex + "_" + prime_cat + "_" + imm_cat

keep year obcat pop state_fips_code county_fips_code

reshape wide pop, i(year state_fips_code county_fips_code) j(obcat) s

keep if state_fips_code != ""

** Totals ** 

egen pop_tot = rowtotal(pop_female_np_doc pop_female_np_nat pop_female_np_undoc pop_female_p_doc pop_female_p_nat pop_female_p_undoc pop_male_np_doc pop_male_np_nat pop_male_np_undoc pop_male_p_doc pop_male_p_nat pop_male_p_undoc)

* prime aged

egen pop_prime = rowtotal(pop_female_p_doc pop_female_p_nat pop_female_p_undoc pop_male_p_doc pop_male_p_nat pop_male_p_undoc)

* prime aged men

egen pop_prime_m = rowtotal(pop_male_p_doc pop_male_p_nat pop_male_p_undoc)

* prime aged women 

egen pop_prime_f = rowtotal(pop_female_p_doc pop_female_p_nat pop_female_p_undoc)

* undocumented (all)

egen pop_undoc = rowtotal(pop_female_np_undoc pop_female_p_undoc pop_male_np_undoc pop_male_p_undoc)

* undocumented (prime aged)

egen pop_primeundoc = rowtotal( pop_female_p_undoc  pop_male_p_undoc)

gen share_pop_undoc_prime = pop_primeundoc / pop_prime

label variable share_pop_undoc_prime "ACS Share of prime age workers that are undocumented"

label variable pop_tot "ACS total pop"
label variable pop_prime "ACS prime age pop (25-64)"
label variable pop_undoc "ACS undocumented immigrant pop"
label variable pop_primeundoc "ACS prime age undocumented immigrant pop"

twoway kdensity share_pop_undoc_prime  if year == 2007 , color(dknavy) legend(label(1 "2007")) || ///
	kdensity share_pop_undoc_prime  if year == 2012, color(maroon) legend(label(2 "2012")) lpattern("longdash") || /// 
	kdensity share_pop_undoc_prime  if year == 2017, color(teal) legend(label(3 "2017")) lpattern("shortdash")   graphregion(fcolor(white)) ytitle("Density") xtitle("Share of prime age workers that are undocumented") legend(rows(1))
	
	cd "$Output"
graph export "fig_desnity_share_pop_undoc_prime_year.png", replace as(png)	

* Clean county fips *

gen fipscounty = real(state_fips_code + county_fips_code)

duplicates list fipscounty year

drop state_fips_code county_fips_code

rename year census_period

cd "$Data"
save acs_immigrant_cleaned, replace