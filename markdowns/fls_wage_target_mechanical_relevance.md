# The FLS wage target makes weight variation copy the AEWR

## Question

Can the panel IV get a strong first stage from variation in the calibrated
county weights alone, and would richer within-region wage data (for example,
Ray et al. 2026) help?

## Findings

1. **The wage target is the AEWR.** The wage-only calibration target for source
   year *t* is the FLS annual field-and-livestock hourly wage. It equals the
   policy-year *t+1* AEWR in all 238 region-years (correlation 1.000, median
   absolute difference $0.00).

2. **The weights barely respond to that target.** With one standardized moment,
   soft entropy calibration closes a share ρ/(1+ρ) of the gap between the
   target and the prior-implied moment. At ρ = 0.10 that is 9.1%; the observed
   median is 9.1% (range 8.3–10.3%). The median wage-only calibration moves
   1.3% of county weight (maximum 8%).

3. **The gap itself is not small.** After region and year effects, the
   standardized gap has an SD of 0.39 prior SDs, about $0.49 in the target. The
   tilt is small because of ρ, not a lack of signal.

4. **The weight response is a scaled copy of the AEWR.** Linearizing the tilt,

   z_weight,t ≈ k(ρ) · (AEWR_{t+1} − prior OEWS mean_t) / s_t,

   with k increasing in ρ. After region and year effects, the standardized gap
   correlates 0.57 with the AEWR. Any first-stage strength that a higher ρ
   buys through this channel would be mechanical relevance: the instrument
   would inherit the regional labor-demand shocks that make the AEWR
   endogenous. The comparison below shows that, in practice, raising ρ does
   not strengthen the weights-only first stage at all.

5. **More wage dispersion would shrink the tilt.** Moments are standardized
   by the prior SD of the county feature, so a wider spread makes the same FLS
   movement a smaller standardized gap.

6. **Ray et al. (2026) do not supply sub-regional farm wages.** Their hourly
   wages are NAWS (9 Farm Resource Regions; 2002, 2010, 2017) or regional FLS.
   County and CZ wages are the regional wage times a markup built from the
   contract-labor payroll share, HUD Fair Market Rent and county H-2A
   wages. The H-2A component is the treatment, and the rest has only 2002 and
   2017 cross-sections.

## Implication

Letting the wage moment shape the weights is consistent with the design, but
weight variation driven by it is not independent identifying variation. A
weights-only instrument is informative only if its relevance survives when the
calibration never sees the AEWR. That means non-wage targets only: FLS seasonal
worker counts, and possibly field/livestock composition. The composition
moment still uses FLS field, livestock and combined wage *differences* as
coefficients, so seasonal-only calibration is the cleanest test.

## Comparison: weights-only first stage by calibration target and ρ

Scratch diagnostic, not a pipeline product. Each region-year was recalibrated
with the retained solver and prior. The weight component
z_weight = Σ (p − p̄) w̄ uses the stage-10 balanced donor set and fixed
2008–2010 donor wages. First stages use the column 3/4 specification
(controls, county and year FE, CRV1 by target subregion) on 34,060 county-years
(2011–2023). At ρ = 0.10 the wage-only and preferred components reproduce the
retained stage-10 values exactly (2,072 cluster-years, maximum difference 0).
"Corr. with AEWR gap" is the correlation of residualized z_weight with the
residualized standardized FLS-wage gap.

| Calibration targets | ρ | Median weight moved | Share of gap closed | z_weight F | Corr. with AEWR gap |
|---|---|---|---|---|---|
| Wage only | 0.10 | 1.3% | 9% | 1.00 | 0.20 |
| Wage only | 1 | 7.1% | 50% | 0.24 | 0.36 |
| Wage only | 10 | 13.2% | 91% | 0.02 | 0.35 |
| Wage + seasonal + composition | 0.10 | 9.7% | 14% | 3.14 | 0.14 |
| Wage + seasonal + composition | 1 | 44.9% | 58% | 0.66 | 0.16 |
| Seasonal + composition | 0.10 | 9.3% | 15% | 3.04 | 0.07 |
| Seasonal + composition | 1 | 43.8% | 60% | 2.40 | 0.08 |
| Seasonal + composition | 10 | 70.9% | 86% | 0.15 | 0.05 |
| Seasonal only | 0.10 | 6.8% | 11% | 2.12 | 0.06 |
| Seasonal only | 1 | 40.3% | 61% | 1.21 | 0.06 |
| Seasonal only | 10 | 64.3% | 92% | 1.72 | 0.08 |

Medians are over the 238 region-years. Weight moved is the total-variation
distance from the prior.

Results:

- **No weights-only instrument is relevant.** Every first-stage F is at most
  3.1, under every target set and ρ.
- **Larger weight shifts add noise, not signal.** Raising ρ from 0.10 to 10
  moves about two-thirds or more of county weight (median 64–71%), but the first-stage F falls or
  stays near zero.
- **Dropping the wage target costs almost nothing.** Without it, the weight
  component is nearly uncorrelated with the AEWR gap (0.05–0.08), and its F
  (3.0 seasonal + composition, 2.1 seasonal only) is close to the retained
  preferred value (3.1). The little relevance the weights have does not come
  from the AEWR.
- **The mechanical channel shows up in correlation, not in the first stage.**
  With the wage target, z_weight's correlation with the AEWR gap rises from
  0.20 to 0.36 as ρ rises, but the F falls. Donor wages relate to the
  calibration feature too weakly for the copied AEWR to survive the fixed
  effects and controls.
- **The full instrument's strength comes from donor wages.** Full balanced
  instruments have F of 67–73 at ρ = 0.10 without the wage target, against
  97 with it. Raising ρ lowers F for the non-wage versions (to 15–26) and
  raises it for wage-only (to 136).

Calibrated county weights, as constructed from FLS regional targets, do not
supply usable identifying variation on their own. A design that needs strong
weight-only relevance would need targets with real county-level information,
which the public FLS does not publish.

## Sources

- `data/intermediate/panel_iv_fls_county_moment_diagnostics.parquet`
- `data/intermediate/panel_iv_fls_county_weight_summary.parquet`
- `data/intermediate/panel_iv_fls_county_calibration_diagnostics.parquet`
- `outputs/tables/iv_instrument_decomposition_variance.csv`
- `replications/ray_et_al/Ray26AJAE_Replication/scripts/0a_PRESTAGE.R`
  (wage and hiring-cost construction)
