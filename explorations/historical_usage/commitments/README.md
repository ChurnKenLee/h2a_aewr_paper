# Did prior commitments change the response to annual AEWR updates?

A county with a 25-percentage-point higher historical share of already-certified capacity had an estimated **0.64% fewer certified hours the next fiscal year** after a 0.05 log-point larger regional AEWR increase relative to its recent history. The approximate 95% interval runs from **−3.76% to +2.58%**. Dropping Florida changes the estimate to +1.07%. These data support an audit of exposure to wage-update timing. The response estimate is too dependent on a few regions to anchor a causal claim.

## Timing and exposure

1. Freeze the county's certified calendar for service FY2012 through FY2015, October 2011 through September 2015. The denominator is all county-matched allocated certification position-days across those four fiscal years. The numerator is January–September position-days on contracts whose certification decision was recorded by October 31 immediately before that January. The intervening October–December months remain in the denominator. A fiscal disclosure partition and a service fiscal year can differ. The one-year date guard requires reading FY2010 through FY2016 to cover every eligible service month.
2. Match cases on `fiscal_year, case_number`. Require a unique parseable decision date falling within the case's disclosure fiscal year. An ambiguous or unavailable decision leaves timing unknown. A county with positive unknown January–September mass is excluded. In this vintage, 74.37 million national county-matched position-days included 56.72 million in January–September, 9.18 million of which were already certified. All positive January–September mass was classifiable. These national totals include locations outside the shared panel geography.
3. For each of the 17 AEWR regions and reference years 2017–24, take the nominal log change from last year's annual reference minus the arithmetic mean of the preceding three annual log changes. Each observation uses only earlier reference rates for its forecast. The notices for all eight current references were published after October 31 preceding January of the reference year. Wage growth may be predictable from other public or private information.
4. Estimate PPML for the owning panel's total certified contract hours by fiscal filing cohort, including ordinary zero cells. The primary outcome for reference year *t* is the entire fiscal year ending in September *t*+1. For example, the 2017 rate predicts FY2018 hours. This lag allows a full subsequent filing cycle and exposes the outcome to other wage updates. The contemporaneous comparison uses FY*t* and includes an October–December quarter that can precede the new notice. The placebo uses FY*t*−1, which ends before publication. All models use county and state-by-reference-year effects, equal row weights, and region-clustered standard errors. The continuous regressor multiplies the baseline commitment share by excess reference-rate growth. A county effect absorbs the baseline share, and state-year effects absorb the annual region-wide reference shock.
5. The shared panel includes 2,008 counties with observed positive baseline capacity and classifiable timing. Of these, 1,290 had zero early-commitment share. Its median is zero, its 90th percentile is 19.64%, and its position-day-weighted mean is 12.32%. The primary PPML retains 1,993 counties, 15,944 rows and 17 regions after fixed-effect separation. The percent contrast below scales the coefficient by a 25-point commitment difference and 0.05 log points of excess reference growth. The 25-point contrast lies above the sample's 90th percentile relative to zero. Observed innovations range from −0.077 to +0.226 log points with unweighted region-year SD 0.041.

## What changes under the alternatives?

| Contrast for 25 points more precommitment and a 0.05 log-point larger innovation | Expected hours difference | Approximate region-clustered 95% interval |
| --- | ---: | ---: |
| Next fiscal year | −0.64% | [−3.76%, +2.58%] |
| Fiscal year before the notice, placebo | −1.21% | [−5.44%, +3.21%] |
| Fiscal year containing the update | −1.05% | [−2.57%, +0.50%] |
| Next year, separate national commitment slope by reference year | +0.22% | [−2.14%, +2.64%] |
| Next year, commitment share relative only to January–September capacity | −0.45% | [−2.85%, +2.00%] |
| Next year, counties with capacity in all four baseline fiscal years | +0.76% | [−1.72%, +3.31%] |
| Next year, also interact baseline January–March and January–September capacity shares with the innovation | −0.34% | [−2.06%, +1.42%] |

Nearly two-thirds of eligible counties had zero early exposure. Three regions contributed most of the remaining identifying variation. Florida accounts for 42.6% of the fitted-mean-weighted, fixed-effect-residualized squared interaction; Arizona/New Mexico accounts for 23.2%, and Colorado/Nevada/Utah for 15.5%. The inverse sum of squared regional information shares is 3.78. Across 17 region deletions the primary contrast ranges from −1.98% to +1.07%. The leave-region jackknife yields a 95% interval of [−5.04%, +3.96%]. These calculations describe leverage and fragility. Neither interval has been calibrated for finite-sample size.

Using the primary clustered standard error and a two-sided t critical value with 16 degrees of freedom gives an approximate 80%-power detectable increase of 4.56% for the stated contrast. It is based on realized residuals. No design-stage or size-validated power calculation was performed. The placebo's point estimate is more negative than the primary estimate, and its wide interval limits its informativeness.

An already-certified contract can remain subject to updated wage obligations, and a certification is an authorization for capacity. The commitment measure is a filing-timing characteristic. A causal interpretation requires a larger regional wage innovation to have no other differential effect on counties with different baselines, apart from the channel being studied. Persistent crop seasons, employer composition, correlated forecast errors, and regional labor demand all threaten that assumption. Annual references also have occupation and competing-floor limitations. The model measures a differential change in certified hours. Converting its coefficient into an overall labor-demand elasticity would require additional identification and wage measurement.

## Reproduction

From the repository root:

```sh
source scripts/pipeline_helpers.sh
configure_r_library
Rscript --vanilla explorations/historical_usage/commitments/commitments.R /tmp/h2a-historical-usage-commitments
```

The script reads the B01 worksite-month partitions, A01 disclosure decisions, the shared county-year panel and the tracked wage-notice manifest. It writes baseline timing totals, region-year shocks, every fit, deletion estimates, jackknife interval, regional information shares and support diagnostics to the supplied directory. An independent DuckDB sum reproduced the baseline denominator and early-commitment numerator. [Calendar definitions](../../../content/contracts/h2a-worksite-exposure.md) · [Rate applicability and notice sources](../../../content/contracts/aewr-inputs.md) · [Shared outcomes](../../../content/contracts/shared-panel.md)
