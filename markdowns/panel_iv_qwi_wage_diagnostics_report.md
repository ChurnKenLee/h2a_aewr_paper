# Panel IV: QWI wage-granularity diagnostic

Written 2026-11-09 from `outputs/tables/iv_qwi_wage_diagnostics_*.csv`
(stage `11_qwi_wage_diagnostics.R`). Diagnostic only: the design instrument,
weights and estimates are unchanged, and no outcome equation is estimated.

## Comparison

OEWS wages with the design weights, against OEWS wages adjusted by annual
county QWI factors with weights recalibrated on the adjusted wage.

| | OEWS (design) | QWI-adjusted, recalibrated |
| --- | --- | --- |
| Weight share | 4.8% | 38.1% |
| Wage share | 89.2% | 53.2% |
| Interaction share | 6.0% | 8.7% |
| First-stage coefficient (SE) | 0.170 (0.025) | 0.141 (0.023) |
| First-stage F | 44.7 | 36.9 |
| Weight component alone: coefficient (SE) | 0.245 (0.138) | 0.006 (0.035) |
| Weight component alone: F | 3.1 | 0.0 |
| Wage component alone: F | 82.6 | 23.3 |

Shares are of the balanced-donor instrument's variance after county and year
fixed effects and controls. First stages use the preferred column. Both
versions use 36,239 observations and 80 clusters; component rows use the
34,060 observations (74 target clusters) where the decomposition exists.

## Finding

- The weight share rises from 5% to 38%.
- The first stage weakens moderately: F 44.7 to 36.9.
- The weight component alone still does not predict the AEWR.
- The wage component alone weakens (F 82.6 to 23.3): county-relative QWI
  movements add noise to it.

## Why the weight share rises

| | OEWS (design) | QWI-adjusted, recalibrated |
| --- | --- | --- |
| Wage levels per donor set (median) | 8 | 27.5 |
| Spread of wage levels across donors (SD) | $0.92 | $2.87 |
| Donor mass moved over time | 13.9% | 13.8% |
| Weight component variance | 0.024 | 0.291 |
| Wage component variance | 0.317 | 0.342 |

- County factors triple the wage spread across donors.
- The weights move about as much as before; each move now shifts more wage.

## Recalibration

- Recalibrated weights differ from the design weights by 1.9% of mass per
  region-year on average.
- Fit and concentration are close to the design: moment gap closed 16.0% vs.
  15.7%; median effective counties 51.3 vs. 51.1; largest county weight 0.607
  vs. 0.585.
- Calibrating on the adjusted wage pulls the weighted wage toward the FLS wage
  target, which also sets the AEWR. Part of the retained first stage may be
  mechanical.

## Measure

- QWI average monthly earnings of stable NAICS 111/112 jobs, published values
  only, stable-employment weighted across quarters and industries, PPI
  deflated.
- Annual factor: the county's QWI/OEWS ratio in that year over its AEWR
  region's weighted mean ratio. The region's wage path stays OEWS; county
  positions follow QWI.
- A missing county-year takes the nearest earlier year, else the nearest
  later. A county never observed keeps factor one.
- The factor keeps each region-year's prior-mean wage within 3% of the OEWS
  value, so recalibration responds to county dispersion, not to a level shift.
- Recalibration: the factor is applied to the calibration's county wage
  feature; stage 04's moments, solver and `rho = 0.10` are rerun.

## Coverage

- 26,144 eligible donor county-years: 77% have an observed factor, 9% a
  filled one, 14% keep OEWS.
- Calibrated counties with an observed factor hold 92% of prior mass on
  average.
- Factor: 25th--75th percentile 0.84--1.13; 5th--95th 0.64--1.44.
- QWI and OEWS levels correlate at 0.41.

## Caveats

- QWI earnings mix pay with hours; they are not an hourly wage.
- One measure only. A clean county hourly wage is untested.
- Filled factors can use later years' data.
- Standard errors cluster on subregions; the AEWR varies across 17 regions.
- No formal test of the difference in first-stage strength.
- Other QWI versions were examined before this one. Adopting it because its
  first stage held up would be specification search.

## Conclusion

QWI-adjusted wages with recalibrated weights raise the weight share to 38%
and keep most of the first stage. The weights still carry no detectable
relevance; the strength comes from wage movements and possibly from
calibration to the FLS wage.
