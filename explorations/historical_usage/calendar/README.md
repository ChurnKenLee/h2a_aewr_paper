# Where in the agricultural season did H-2A expand?

The growth in certified capacity kept a measurable relationship with local crop seasons. Among counties with a measurable crop-employment peak before 2008 and **zero county-matched certification position-days in 2009–11**, 460 of 487 had positive capacity in at least one of four reference months in 2022–24. Their historical crop peak accounted for 36.60% of their capacity across those months. Exchanging peak labels among counties in the same state and historical crop-employment size group gives 33.82%. These counties accounted for 12.2% of national capacity across the four reference months in the later period.

## Predetermined crop calendars

For each county, use January, April, July and October reported QCEW NAICS 111 crop employment in 2003–07. Retain years with four disclosed observations and positive annual summed employment. Average the four employment shares across complete years with equal year weights. At least three years and a unique peak are required. The resulting 1,293 counties are selected entirely on historical QCEW data. The peak can be identified only among those four months. Later H-2A capacity includes all occupations because the worksite calendar pools contracts.

Each period's observed share is the ratio of summed mean daily certification capacity in each county's historical peak month to summed capacity across all four reference months. Monthly means divide inclusive position-days by the calendar month's length. The comparison is the exact expected share upon swapping peak labels among all eligible counties of the same state and historical mean crop-employment tercile. This preserves group-level seasonality and the later capacity of each county. The period uses all three years. Unmatched capacity stays in the national coverage denominator.

| Calendar years | Active / eligible counties | National four-month capacity covered | Own peak share | Within-state, size-group benchmark | Difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2010–12 | 778 / 1,293 | 73.17% | 36.91% | 33.82% | +3.08 pp |
| 2017–19 | 1,085 / 1,293 | 80.06% | 38.13% | 35.04% | +3.09 pp |
| 2022–24 | 1,264 / 1,293 | 80.04% | 37.53% | 34.52% | +3.01 pp |
| 2022–24, no measured 2009–11 capacity | 460 / 487 | 12.20% | 36.60% | 33.82% | +2.78 pp |

The own-county advantage changed little across these windows as coverage broadened. In 2022–24 it was 2.17 percentage points under equal active-county weights and 2.87 points using capacity on each reference month's 12th day. Omitting one state code at a time yielded 2.66–3.55 points under the primary weighting. Among the 783 counties with the same peak in 2000–02 and 2003–07, the late-period advantage was 3.65 points. An earlier peak could be measured for 1,051 counties. The disclosure and stability restrictions favor counties where public records capture a clear crop cycle.

A swapping benchmark describes how much county-specific timing adds to state and size-group averages. It supplies no randomized assignment or sampling confidence interval. Common crop technology, climate, and persistent operation mix can explain the relationship. Observed zero county capacity allows unidentified geography and omitted records. Neither crop-sector QCEW nor the calendar separates domestic and H-2A workers in realized employment. A historical crop peak is useful for diagnosing season-specific pressure or defining a pre-period exposure in a separately identified design. Its direct connection to labor demand limits its use as a wage instrument.

## Duration, winter use and geographic composition

FY2008 contributed 90,082 positions with a valid interval and 18.58 million inclusive allocated certification position-days. FY2025 contributed 398,059 timed positions and 68.54 million days. The position count grew by a factor of 4.42 and the position-day total by 3.69. Mean duration weighted by allocated positions fell from **206.2 to 172.2 days**. It had already reached 172.0 days in FY2015 under the same broad source date regime as FY2008. The endpoint difference shows why a certification count alone overstates the proportional growth of calendar capacity. All fiscal-cohort duration calculations include the contract's full interval even when it crosses a calendar-year boundary. The FY2008 local position total exceeds the published federal count by 9.8%, so the 2008 growth factors describe this local record vintage. The [raw-workbook reconciliation](../business/README.md) traces most of the difference to records preceding worksite allocation.

| Calendar window, matched geography | Winter share of position-days | Winter / summer capacity per day | Peak month | Mean daily capacity in peak month | National peak / sum of state peaks |
| --- | ---: | ---: | --- | ---: | ---: |
| 2010–12 | 13.82% | 0.438 | September | 57,445 | 0.810 |
| 2017–19 | 12.66% | 0.406 | September | 147,959 | 0.815 |
| 2020–22 | 12.91% | 0.424 | September | 193,366 | 0.807 |
| 2022–24 | 12.97% | 0.428 | September | 229,520 | 0.805 |

Winter is December, January and February of each selected calendar year. Summer is June through August. Daily capacity divides each month's position-days by its actual days. National and state peaks use month-of-year profiles pooled over the three calendar years. The geographic distribution of peaks allowed the nationwide September peak to sit about one-fifth below the sum of each state's own peak in both endpoint windows.

The winter share decreased by 0.85 percentage points from 2010–12 to 2022–24. An exact symmetric state decomposition attributes −1.40 points to within-state changes, +0.51 points to changes in the shares of states present in both windows, and +0.04 points to newly represented state codes. Capacity expanded within a similar national seasonal profile, with a modest shift toward the warmer months within states.

Separate site dates are unavailable before FY2020. Rules for canonical case dates also change in FY2016, FY2018, and FY2020. The share of calendar capacity using site dates rose from 0.08% in 2019 to 54.08% in 2020 because fiscal years can cross calendar years. Replacing selected site/case dates with case dates alone on common support moves the national winter share by at most 0.09 percentage points in any complete interior calendar year. Using case dates alone gives 174.3 days per timed position in FY2025. Raw certification records can describe allocated capacity even when individual workers rotate among sites. Actual days worked, admissions and unique workers require other measurements.

The scripts use fiscal disclosure partitions FY2008 through FY2025. Calendar 2008 may miss FY2007 carry-in and calendar 2025 may miss FY2026 decisions covering late 2025. The seasonal comparisons use interior calendar years, which have all adjoining eligible fiscal partitions under the producer's one-year date guard. The endpoint duration comparison instead uses the complete FY2008 and FY2025 partitions. Coverage checks retain unresolved county mass nationally. The peak and state decomposition exclude that mass.

## Reproduction

From the repository root, run:

```sh
source scripts/pipeline_helpers.sh
configure_r_library
Rscript --vanilla explorations/historical_usage/calendar/season.R /tmp/h2a-historical-usage-calendar
Rscript --vanilla explorations/historical_usage/calendar/crop_peak.R /tmp/h2a-historical-usage-calendar
```

Inputs are the B01 monthly worksite partitions, allocation audit and coverage totals under `data/intermediate`, plus the A01 county-level QCEW reference-month file. The scripts write `annual_calendar.csv`, `seasonal_windows.csv`, `winter_decomposition.csv`, `filing_cohorts.csv`, `common_case_sensitivity.csv`, `crop_peak_alignment.csv` and `crop_peak_support.csv` to the supplied directory.

The season script checks position-days and unmatched mass against every FY source total, independently reconstructs daily capacity from the audit's interval endpoints, and verifies the exact seasonal decomposition. The crop script checks keys, exposure support and coverage. An independent DuckDB aggregation reproduced the 2022–24 four-month coverage and peak share to rounding. These are exploratory descriptions conditional on this source vintage. [Calendar contract](../../../content/contracts/h2a-worksite-exposure.md) · [QCEW source producer](../../../code/a01_sources/04_02_qcew_quarterly_employment.py)
