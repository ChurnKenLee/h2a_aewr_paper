# FLS Realized-Geography Composition Analog

## Scope

The public data do not identify NASS's confidential operation-level FLS
sample, response adjustments, or survey weights. This procedure therefore
constructs:

- a soft-calibrated central OEWS-area geographic distribution;
- a reproducible ensemble of plausible realized sampling variation; and
- descriptive simulation envelopes for that predeclared ensemble.

It does not claim to recover the original survey weights. Estimation targets
the published FLS worker composition and annual combined field-and-livestock
wage using public area analogs. This is the wage used to set the following
year's AEWR. It does not use a contemporaneous AEWR policy value, first-stage
estimate, or outcome. The supported source years are 2011–2021, and every FLS
region-year is estimated independently.

The implementation is:

```sh
./scripts/run_panel_iv.sh
```

## Expected geographic prior

Let \(M_{ct}\) be the county mass from
`weight_spec = "census_hired_workers_qcew_updated"`. It anchors 2007, 2012,
2017, and 2022 to Census of Agriculture directly hired workers, updates annual
employment with strict QCEW, QWI, and BEA sources, applies a two-sided path
correction, and rakes county mass to the interpolated state Census total.

Let \(h_{cit}\) be OEWS area \(i\)'s share of distinct township codes mapped
within county \(c\), where

$$
\sum_i h_{cit}=1.
$$

The allocated area mass is

$$
A_{irt}=\sum_{c\in r}h_{cit}M_{ct}.
$$

An OEWS area is identified by the pair
\((\text{FLS region},\text{OEWS area code})\), so an area spanning regions
remains separated. The expected geographic prior is

$$
q_{irt}
=
\frac{A_{irt}}{\sum_{j\in r}A_{jrt}}
$$

for positive area mass. Zero-mass areas remain outside the supported frame.
Wage availability and OEWS employment never determine this support or prior.
The county allocation is retained so an area draw can later be expanded with
the fixed baseline conditional county mass.

## FLS joint targets

For each observed reference quarter \(q\), the target has two duration cells:

- workers employed 150 days or more (`long`); and
- workers employed fewer than 150 days (`short`).

Let \(N_{rtqd}\) denote the published count. The normalized target is

$$
\tau_{rtqd}
=
\frac{N_{rtqd}}
{\sum_{q,d}N_{rtqd}}.
$$

The denominator is the sum of the long and short counts. Differences from the
separately published all-worker count are retained as rounding/source
diagnostics.

Targets are never imputed. A region-year requires at least three usable
quarters and a positive duration-cell total. April duration counts are absent
in 2011, so that year uses January, July, and October: six joint cells and
five independent contrasts. Each year from 2012 through 2021 uses four
quarters: eight joint cells and seven independent contrasts.

## Public area features

### Seasonal employment intensity

For each county-year-quarter, private QCEW NAICS 111 and 112 employment is
valid only when both industries are disclosed. If that strict QCEW cell is
incomplete, private QWI beginning-of-quarter employment fills it when both
industries are available.

County employment is allocated with the same township shares used for the
frame prior. An area-quarter aggregate is usable only when its observed
county inputs cover at least 90 percent of the area's frame mass. Partial
totals below that threshold remain missing.

Let \(E_{itq}\) be the accepted area-quarter employment and \(A_{it}\) the
annual area frame mass. First form

$$
I_{itq}=\frac{E_{itq}}{A_{it}}.
$$

Over the quarters observed in the corresponding FLS target, define

$$
S_{itq}
=
\frac{I_{itq}}{\sum_{q'\in Q_{rt}}I_{itq'}}.
$$

### Duration analog

QWI persistence is stable employment divided by beginning-of-quarter
employment. In matched Census benchmark area-quarter cells, estimate the
frame-weighted median odds ratio

$$
R
=
\operatorname{weighted\ median}
\left[
\frac{\operatorname{odds}(\text{QWI persistence})}
{\operatorname{odds}(\text{Census 150-plus-day share})}
\right].
$$

The implementation estimates one public bridge from matched cells and records
the number of matched area-quarter and area-year cells. It uses no FLS data.
Translate QWI persistence to the long-duration analog:

$$
D_{itq}
=
\operatorname{logistic}
\left[
\operatorname{logit}(\text{QWI persistence}_{itq})-\log R
\right].
$$

When QWI duration is unavailable, use the area's Census 150-plus-day share,
interpolated between benchmark years on the logit scale.

### Public-only imputation hierarchy

Any remaining public covariate gap follows this fixed hierarchy:

1. same-area, same-quarter linear interpolation over time, using the log
   scale for employment intensity and the logit scale for duration;
2. the nearest available same-area, same-quarter value within two years;
3. the frame-prior-weighted FLS-region-year mean for the quarter; and
4. failure of the region-year if no regional value exists.

The artifact records the source, coverage, and imputation flag for each
feature, plus employment, duration, and combined imputed prior mass. This
borrowing fills public covariates only; it never smooths recovered weights
across years.

## Joint compositions and Helmert moments

Each area has the joint composition

$$
X_{itq,\mathrm{long}}=S_{itq}D_{itq},
$$

$$
X_{itq,\mathrm{short}}=S_{itq}(1-D_{itq}).
$$

The cells are normalized to sum to one for every supported area. Both the
area compositions and FLS target are projected onto a \(K\times(K-1)\)
orthonormal Helmert basis. This avoids choosing an omitted category.

For contrast \(j\), let \(Z_{ij}\) be the area contrast and \(T_j\) the target
contrast. Standardize using the deterministic frame prior:

$$
\bar Z_j^q=\sum_i q_iZ_{ij},
$$

$$
s_j^q
=
\sqrt{\sum_iq_i(Z_{ij}-\bar Z_j^q)^2},
$$

$$
\widetilde Z_{ij}
=
\frac{Z_{ij}-\bar Z_j^q}{s_j^q},
\qquad
\widetilde T_j
=
\frac{T_j-\bar Z_j^q}{s_j^q}.
$$

Only contrasts with effectively zero cross-area variation are dropped, and
each drop is flagged.

## Wage moment

Within each OEWS area, the retained agricultural occupations are combined
using their reported employment and mean hourly wage. For region-year
\((r,t)\), the area wages and the annual FLS combined field-and-livestock wage
are centered and scaled by the deterministic frame prior exactly as above,
then appended as one additional column of \(\widetilde{\mathbf Z}\) and one
additional element of \(\widetilde{\mathbf T}\). The preliminary FLS release
used by DOL to set the following year's AEWR is preferred; the revised wage is
used only when a preliminary value is unavailable. Areas without a usable
OEWS wage receive the observed prior-weighted region-year mean and are
flagged.

## Soft entropy recovery

For a prior vector \(p\), solve

$$
\min_{\mathbf w}
D_{KL}(\mathbf w\Vert\mathbf p)
+
\frac{\rho}{2}
\left\|
\widetilde{\mathbf Z}'\mathbf w-\widetilde{\mathbf T}
\right\|_2^2
$$

subject to

$$
w_i\geq0,\qquad\sum_iw_i=1.
$$

The dual has the exponential-tilting weights

$$
w_i(\boldsymbol\lambda)
=
\frac{
p_i\exp(\widetilde{\mathbf Z}_i'\boldsymbol\lambda)
}{
\sum_kp_k\exp(\widetilde{\mathbf Z}_k'\boldsymbol\lambda)
}
$$

and objective

$$
\log\left[
\sum_i p_i
\exp(\widetilde{\mathbf Z}_i'\boldsymbol\lambda)
\right]
-
\boldsymbol\lambda'\widetilde{\mathbf T}
+
\frac{1}{2\rho}
\boldsymbol\lambda'\boldsymbol\lambda.
$$

The implementation uses a damped Newton solver with log-sum-exp evaluation.
The composition and wage targets are soft rather than exact constraints.
Larger \(\rho\) pursues their joint fit more strongly. The fixed values are

$$
\rho\in\{0.01,0.03,0.10,0.30,1.00\},
$$

with \(\rho=0.10\) primary. A deterministic center is solved from the original
\(\mathbf q\) for every rho.

## Dirichlet ensembles

For each region-year,

$$
N^{eff}_{rt}
=
\frac{1}{\sum_iq_{irt}^2},
$$

$$
\kappa_{rt}=mN^{eff}_{rt},
$$

and

$$
\mathbf q^{(b)}_{rt}
\sim
\operatorname{Dirichlet}(\kappa_{rt}\mathbf q_{rt}).
$$

The fixed dispersion multipliers are

$$
m\in\{2,5,10,20\},
$$

with \(m=10\) primary. Each prior draw is passed through the same soft entropy
solver.

The primary specification is
`fls_geo_field_livestock_dirichlet_m10_rho010` with 999 draws. Every
non-primary rho at
\(m=10\) has 199 common draws, and every non-primary \(m\) at \(\rho=0.10\)
has 199 draws. The base seed is `20260726`, with deterministic offsets by
region, year, and multiplier. Rho is omitted from the prior-draw seed so its
sensitivity path uses identical priors. Partial and repeated builds therefore
reproduce the same draw IDs and weights.

These dispersion and penalty values are declared sensitivity parameters. They
are not estimated sampling variances and are never selected using wage fit,
first-stage strength, or outcomes.

## Outputs

The recovery produces:

- `panel_iv_fls_geography_features.parquet`;
- `panel_iv_fls_geography_county_area_prior.parquet`;
- `panel_iv_fls_geography_area_prior.parquet`;
- `panel_iv_fls_geography_wage_features.parquet`;
- `panel_iv_fls_geography_draws/`, partitioned by region, source year, and
  specification;
- `panel_iv_fls_geography_weight_summary.parquet`;
- `panel_iv_fls_geography_diagnostics.parquet`; and
- target, duration-bridge, and feature diagnostic artifacts.

The summary contains the frame prior, deterministic center, draw mean,
standard deviation, and 2.5th, 50th, and 97.5th simulation-envelope
percentiles. The draw-level diagnostics contain standardized moment
imbalance, KL divergence, effective area count, maximum area share, imputed
prior mass, and solver status.

Every realized output identifies

- `weight_spec = "fls_realized_geography_dirichlet_entropy"`;
- `baseline_weight_spec = "census_hired_workers_qcew_updated"`;
- `moment_spec = "fls_joint_quarter_duration_plus_field_livestock_wage"`;
- `wage_target_used = TRUE`;
- `rho`;
- `kappa_multiplier`;
- `weight_draw_id`;
- `is_primary`; and
- `simulation_seed`.

If county weights are needed later, the fixed expansion is

$$
w^{(b)}_{crt}
=
w^{(b)}_{irt}
\frac{A_{cit}}{A_{irt}}.
$$

No additional within-area county randomness is estimated.

## Use in the dissimilarity IV

The IV construction uses only the deterministic center from the predeclared
primary specification, `fls_geo_field_livestock_dirichlet_m10_rho010`. For
each target
cluster, the area center is expanded to counties with the fixed conditional
shares above, restricted to the two selected donor clusters, and then
re-aggregated to unique OEWS areas. Any OEWS area touching the target cluster
is excluded. The remaining weights are normalized over areas with an observed
OEWS agricultural wage.

The Census-frame version is retained as a benchmark on identical donor and
wage support. Dirichlet draws and sensitivity specifications remain recovery
diagnostics; they do not generate draw-specific instruments.

## Acceptance checks

The implementation checks:

1. area priors, deterministic centers, and successful draws sum to one within
   \(10^{-10}\);
2. target and area joint-cell vectors sum to one;
3. the 2011 and later-year cell/contrast counts follow the published target
   availability;
4. township allocation conserves county frame mass;
5. fixed county expansion reproduces every area weight;
6. every positive-prior area survives public-feature imputation;
7. calibration does not increase the targeted standardized-imbalance norm;
8. increasing rho weakly reduces that norm for common prior draws;
9. seeds, draw IDs, and weights are reproducible in partial builds;
10. the recovery runs without wage, policy, first-stage, or outcome artifacts;
11. every primary deterministic center succeeds; and
12. at least 99 percent of primary draws succeed in every region-year.

The FLS panel structure and confidential dual-frame design remain
unidentified. Region-years are treated independently, and later Census
benchmarks can revise the expected frame consistently with its two-sided
construction.
