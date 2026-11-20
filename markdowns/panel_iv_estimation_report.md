# Panel IV estimation: short report

Written 2026-11-09 from the outputs of `./scripts/run_panel_iv.sh` stages 07--09
(`outputs/tables/iv_preferred_*.csv`, `iv_static_propensity_trend_diagnostic.csv`).
Numbers marked "ad hoc" were computed for this note from
`data/processed/panel_iv_county_year.parquet` and are not pipeline outputs.

## Bottom line

The first stage is strong and the three H-2A quantity outcomes are negative,
but the evidence is weaker than "significant negative elasticities":

- only certified hours clears 5%; certified workers and applications are
  significant at 10% only;
- the coefficients are level effects per real dollar of AEWR, not
  elasticities, and the elasticities they imply are too large to take at face
  value;
- standard errors are clustered on 80 subregions while the AEWR varies only
  across 17 regions, so the reported precision is probably overstated; and
- farm income per worker rises sharply with the AEWR, which is hard to read as
  a causal effect and is a warning for the exclusion restriction.

## Specification

- Outcome on real AEWR (2012 dollars), instrumented by
  `z_wage_seasonal_composition_real`, with county and year fixed effects and
  equal county-year weights.
- Column 1 has no controls. Column 2 (preferred) adds lagged log population,
  farm-employment share, employment/population, real p10 wage, and the static
  H-2A propensity interacted with a linear trend.
- Policy years 2011--2023; 36,239 county-years, 2,848 counties, 80 inference
  clusters in the first-stage sample.

## First stage

| | Column 1 | Column 2 (preferred) |
| --- | --- | --- |
| Instrument coefficient | 0.185 (0.030) | 0.170 (0.025) |
| Excluded-instrument F | 38.0 | 44.7 |

The instrument is relevant by the usual thresholds. The design page already
notes that the calibration targets FLS wages, which also set the AEWR, so some
of this relevance may be mechanical.

## Second stage, preferred column

Coefficients are the change in the outcome per one-dollar increase in the real
AEWR. Clustered standard errors are in parentheses.

| Outcome | Coefficient | p-value | N |
| --- | --- | --- | --- |
| H-2A certified workers / 2011 farm employment | -0.0893 (0.0486) | 0.070 | 36,239 |
| H-2A certified hours / 2011 farm employment | -65.5 (31.6) | 0.042 | 36,239 |
| H-2A applications / 2011 farm employment | -0.00228 (0.00131) | 0.087 | 36,239 |
| H-2A linked worksite businesses / 2011 farm employment | -0.0005 (0.0038) | 0.891 | 36,239 |
| Certified positions / application | -2.64 (2.67) | 0.324 | 23,827 |
| Certified hours / position | -43.4 (61.6) | 0.483 | 23,550 |
| Real Fisher crop price index | -3.54 (5.50) | 0.521 | 36,176 |
| Farm employment | -10.1 (9.3) | 0.285 | 33,497 |
| Production expenses / cash receipts and other income | 0.0016 (0.0316) | 0.960 | 33,497 |
| Farm income per farm worker (2012 dollars) | 38,500 (12,970) | 0.004 | 33,497 |
| Hired-labor share of production expenses | 0.0055 (0.0029) | 0.060 | 33,497 |
| Fisher crop output-quantity index | 192 (281) | 0.497 | 36,176 |

Column 1 is close to column 2 for every outcome (certified workers -0.0906,
p = 0.068; hours -67.0, p = 0.036; applications -0.00267, p = 0.056). Adding the
static-propensity trend to the four lagged controls also changes little
(certified workers -0.0918 to -0.0893).

## Reading the H-2A results

- **Direction.** Certified workers, certified hours and applications all fall
  when the AEWR rises. Linked businesses, positions per application and hours
  per position are not distinguishable from zero, so the response shows up in
  total volume rather than in a clear intensive margin.
- **Magnitude.** Evaluated at the summary-statistics means (AEWR $12.86;
  certified share 0.089), the implied elasticities are about -13 for certified
  workers, -10 for hours and -5 for applications. A one-dollar increase would
  remove roughly the entire mean certified share.
- **Why the scale is fragile.** The outcomes are heavily skewed: in the
  estimation sample 35% of county-years have zero certifications and the
  median certified share is 0.008 against a mean of 0.091 (ad hoc). A dollar
  is also a large change here: after county and year fixed effects the AEWR
  has a standard deviation of $0.37 (ad hoc), so a one-standard-deviation move
  lowers the certified share by about 0.033.

## Reading the farm results

Prices, farm employment, the expense share and output quantities show no
detectable response. Two results need care:

- Farm income per farm worker rises by about $38,500 per AEWR dollar, a fifth
  of its mean, with p = 0.004. A higher wage floor raising income per worker
  by that much is not plausible as a direct effect. It may reflect the
  denominator (current farm employment), or donor-wage movements that track
  regional farm conditions.
- The hired-labor expense share rises by 0.0055 (p = 0.060), which is the
  expected sign for a binding wage floor.

## Caveats on inference

- The AEWR takes 221 distinct region-year values in the sample (17 regions by
  13 years, ad hoc count). Clustering on 80 subregions treats counties in
  different subregions of one region as independent draws of the treatment.
  Region-level clustering has not been estimated.
- Twelve outcomes are tested and no multiple-testing adjustment is applied.
  The H-2A p-values of 0.04--0.09 would not survive one.
- The recovered county weights and the instrument are generated inputs, and
  the standard errors condition on them.

## Suggested next checks

1. Re-estimate the preferred column with standard errors clustered on the 17
   AEWR regions (wild-cluster bootstrap, given 17 clusters).
2. Report the H-2A outcomes on a scale that handles the zeros, such as PPML or
   an extensive/intensive split, before quoting elasticities.
3. Investigate the farm-income result, starting with the numerator and
   denominator separately.
