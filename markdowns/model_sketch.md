# A parsimonious model for AEWR welfare counterfactuals

## Recommendation in one paragraph

The paper should use a **static, horizon-specific partial-equilibrium envelope model** as its core welfare framework. The model should take the paper's estimated response of H-2A **contract hours** to the AEWR as the central behavioral object and integrate those hours over a counterfactual wage path to recover the change in farm profit. This calculation does not require choosing a production function, estimating farm productivity, specifying a CES nest between domestic and H-2A labor, or calibrating the fixed cost of entering H-2A. Those margins are already reflected in the observed aggregate response. A domestic-worker block and a farm-to-retail consumer-incidence block can then be added as transparent extensions, with estimates reported separately when their additional moments are credible and as bounds or scenarios otherwise. This is the most defensible bridge from the paper's reduced-form design to welfare.

The proposed hierarchy is:

```text
AEWR policy path
    -> wage-floor exposure and H-2A contract-hours response
    -> farm-profit envelope calculation                         [core]
    -> H-2A earnings and worker-surplus accounting              [core, with welfare-boundary scenarios]
    -> domestic employment and wage effects                     [extension]
    -> farm-gate/retail price incidence and household welfare   [extension]
```

This follows the logic in the [shared welfare-methods discussion](https://chatgpt.com/share/6a32bc5d-8604-83e8-ae7e-f2e56dc1127b): use a small set of well-identified sufficient statistics, make extrapolation visible, and add general-equilibrium structure only where it changes the answer.

## 1. Why this model, rather than a conventional structural farm model?

The current draft begins with heterogeneous farms, a concave production function, domestic labor, H-2A labor, and an H-2A participation cost. That is economically intuitive, but taking it literally would require parameters that the current data do not separately identify:

- the farm productivity distribution;
- the production-function curvature;
- substitution between H-2A workers, domestic workers, machinery, and materials;
- the distribution of H-2A participation costs;
- the number of domestic workers in corresponding employment at H-2A employers;
- crop-specific output demand and price pass-through;
- the difference between certified positions and labor actually used.

A calibrated CES or Cobb-Douglas version would therefore look more structural without necessarily being more credible. The proposed model instead leaves farm technology and participation heterogeneity latent. The envelope theorem makes that possible for an AEWR-only counterfactual.

The core model can answer:

1. How many H-2A positions and contract hours are lost or gained under an AEWR change?
2. How much farm quasi-rent changes, allowing farms to adjust participation, hours, scale, and other inputs?
3. How H-2A wage income changes?
4. What the implied global surplus change is under explicit assumptions about foreign workers' opportunity cost?

The extensions can answer:

5. Whether domestic workers gain through corresponding-employment coverage or broader labor-market effects.
6. How much of the farm cost shock reaches retail food prices.
7. How consumer incidence varies across income groups without imposing a named non-homothetic utility function.

The model should not claim to answer long-run crop switching, mechanization, migration, or nationwide agricultural reallocation unless those margins are later estimated. They can be discussed as omitted long-run responses.

## 2. Units, timing, and counterfactual horizon

### 2.1 Economic unit

Let \(m\) index a local agricultural market. The empirical unit can initially be a county or a commuting-zone-by-AEWR-region cell. Crop groups should be introduced only in the consumer-incidence extension, because crop assignment for H-2A contracts is incomplete.

Let \(r(m)\) denote the AEWR region applying to market \(m\). The core quantity is worker-hours, not headcount:

- \(H^{c}_{mt}\): certified H-2A contract hours;
- \(H_{mt}=\rho_{mt}H^{c}_{mt}\): paid or realized H-2A hours;
- \(\rho_{mt}\): the realization factor converting certified hours to paid hours.

The existing pipeline constructs certified contract hours from positions, stated weekly hours, and the contract date range. It does **not** observe payroll hours actually worked. The paper should therefore report results in two units:

1. a directly supported **certification-space** result per certified contract hour; and
2. an **employment-space** result after applying a transparent realization-factor range.

### 2.2 Policy timing

Let \(a_{rt}\) be the applicable AEWR. The primary counterfactual should be an AEWR announced before employers file job orders, so both participation and requested labor can respond. This matches the intended labor-demand elasticity.

An unexpected AEWR change imposed after contracts have been planned is a different experiment: hours may be partly fixed and the response will be smaller. The paper should not mix the two. If the empirical design uses a lagged AEWR because applications are filed in advance, the model should align the counterfactual timing with that decision date while using the contemporaneous legally applicable wage for the wage bill.

### 2.3 Numeraire

All wage and welfare amounts must use one documented constant-dollar numeraire. The current pipeline deflates several monetary variables with the farm-products PPI (`WPU01`). That index can be useful as an agricultural output-price deflator, but it is not a general purchasing-power deflator. Before reporting money-metric welfare, reconstruct the required values with a CPI, PCE, or GDP deflator and retain the farm-products PPI separately as an output-price control or relative-price measure.

### 2.4 Horizon

The preferred presentation should report at least two empirical horizons:

- a short-run response, after employers can change applications and contract size but before substantial capital or crop adjustment; and
- a medium-run response, estimated from an appropriate distributed-lag or event-time design.

Calendar-year event-study coefficients are not automatically impulse responses. A horizon-specific welfare calculation requires a horizon-specific causal dose response.

## 3. Farm problem and the envelope result

### 3.1 A deliberately general farm problem

Farm \(i\) in market \(m\) chooses output, H-2A labor, domestic labor, materials, capital services, and whether to use H-2A. Write its optimized profit as

\[
\pi_{im}(a_m,p_m,w^D_m)
=\max_{z_{im}}
\left\{
R_{im}(z_{im};p_m)
-C^{O}_{im}(z_{im};w^D_m)
-w^{H}_{im}(a_m)h_{im}
-w^{H}_{im}(a_m)c_{im}
-F_{im}\mathbf 1\{h_{im}>0\}
\right\}.
\]

Here:

- \(z_{im}\) contains every farm choice, including \(h_{im}\);
- \(h_{im}\) is paid H-2A labor;
- \(c_{im}\) is domestic labor in **corresponding employment** whose pay is directly tied to the H-2A job-order wage;
- \(C^O_{im}\) includes other domestic labor paid \(w^D_m\), materials, capital services, and all variable non-AEWR program costs;
- \(F_{im}\) is the fixed private cost of using the program;
- \(R_{im}\) may embody any production function and, in partial equilibrium, a fixed output price \(p_m\).

This formulation corrects an important simplification in the current draft: the AEWR does not mechanically apply to every domestic employee on an H-2A farm. It applies to workers in [legally defined corresponding employment](https://www.dol.gov/agencies/whd/workers/h2a). That group is not observed in the current public data and should not be assigned a point value without evidence.

The applicable job-order wage is

\[
w^H_{im}(a_m)=\max\{a_m,\widetilde w_{im}\},
\]

where \(\widetilde w_{im}\) collects another applicable wage floor or a wage the employer would offer above the AEWR. A more general specification can allow an above-floor wage to move partially with the AEWR.

### 3.2 AEWR-exposed hours

Define the number of paid hours whose wage changes at the margin when the AEWR changes:

\[
X_m(a)
=\sum_i \left[h_{im}(a)+c_{im}(a)\right]
\frac{\partial w^H_{im}(a)}{\partial a}.
\]

With the max rule, the derivative is an indicator that the AEWR binds. Thus \(X_m\) is not automatically total certified H-2A hours. It incorporates:

- the certified-to-paid-hours realization factor;
- the hour-weighted binding share;
- any directly covered corresponding-employment hours.

### 3.3 The sufficient-statistics result

Holding the local output price and the wage of other domestic labor fixed, the envelope theorem gives

\[
\frac{d\Pi_m(a)}{da}=-X_m(a),
\qquad
\Pi_m(a)=\sum_i \pi_{im}(a),
\]

and therefore

\[
\boxed{
\Delta \Pi_m^{PE}
=-\int_{a_{m0}}^{a_{m1}} X_m(s)\,ds
}.
\]

This is the core welfare bridge. It is valid even when farms differ in productivity, some farms enter or exit H-2A, contract sizes change, domestic labor and machinery substitute for H-2A labor, and production is not CES. Fixed and variable application, housing, transportation, and recruitment costs affect the farm's decision and therefore the observed hours response, but they do not need to be separately calibrated for a counterfactual that changes only the AEWR.

Those non-wage costs do need to be modeled if the counterfactual changes program fees, housing rules, transportation obligations, processing delays, or recruitment requirements. They are not required for the first paper's AEWR-only welfare exercise.

## 4. The empirical dose response used inside the model

### 4.1 Preferred outcome

The welfare calculation should be calibrated to the response of certified contract hours, with positions and applications reported as decompositions:

\[
H^c=N\times q,
\]

where \(N\) is certified positions and \(q\) is scheduled hours per position. Estimate AEWR responses for:

- certified positions;
- certified contract hours;
- applications;
- positions per application;
- weekly hours;
- contract duration.

This identifies whether adjustment occurs through program entry, contract scale, or contract length. The headcount response remains a central result, but an hourly wage counterfactual cannot be monetized credibly with headcount alone.

### 4.2 A positive local response curve

For the primary finite counterfactual, a convenient local representation is a constant dollar semi-elasticity:

\[
H^c_m(a)
=H^c_{m0}\exp[-\psi_m(a-a_{m0})],
\qquad \psi_m\geq 0.
\]

This maps naturally to the paper's dollar-valued AEWR-bite design, preserves positive hours, and can be estimated with a log-link count model or recovered locally from a linear slope. It is an approximation within the observed support, not a claim about the demand curve at arbitrarily low or high wages.

The main functional-form robustness checks should be:

1. locally linear hours, truncated only outside the reported support;
2. the exponential semi-elasticity above;
3. constant elasticity,
   \[
   H^c_m(a)=H^c_{m0}(a/a_{m0})^{-\varepsilon_m};
   \]
4. a monotone spline or binned dose response estimated over observed AEWR variation.

The welfare object is an integral, so a flexible estimated curve can be integrated numerically. There is no computational need to choose one global functional form. For larger counterfactuals, report the range across admissible curves or curvature bounds rather than a single extrapolated number.

### 4.3 Closed-form illustration

If the realization factor, binding share, and corresponding-employment exposure are constant along a small policy path, write \(X_m(a)=X_{m0}\exp[-\psi_m(a-a_{m0})]\). Then

\[
\Delta\Pi_m^{PE}
=-X_{m0}\frac{1-\exp[-\psi_m\Delta a_m]}{\psi_m},
\]

with the continuous limit \(-X_{m0}\Delta a_m\) when \(\psi_m=0\).

For the constant-elasticity curve, with elasticity \(\varepsilon_m>0\),

\[
\Delta\Pi_m^{PE}
=-X_{m0}a_{m0}
\frac{(a_{m1}/a_{m0})^{1-\varepsilon_m}-1}{1-\varepsilon_m},
\]

and the expression becomes \(-X_{m0}a_{m0}\log(a_{m1}/a_{m0})\) when \(\varepsilon_m=1\).

In implementation, the numerical integral should be the canonical calculation; these formulas are useful for exposition and checks.

### 4.4 Mapping the current regression coefficient

The current main outcome is

\[
Y_{ct}=\frac{N^c_{ct}}{E_{c,2011}},
\]

and the treatment is a real-dollar AEWR premium relative to a local wage. If \(\delta_N=\partial Y/\partial B\) is interpreted causally and the local comparison wage is held fixed, then

\[
\frac{\partial N^c_{ct}}{\partial a}=\delta_N E_{c,2011},
\qquad
\varepsilon^N_{ct}
=-\frac{a_{ct}}{N^c_{ct}}\delta_N E_{c,2011}.
\]

This conversion is useful for checking units, but it should not be the final calibration for four reasons:

1. it is a position response rather than an hours response;
2. the current coefficient is a post-2011 interaction, not automatically a policy derivative at every date;
3. elasticities are undefined or unstable in zero-use counties;
4. the treatment combines the AEWR with a local wage, so an AEWR counterfactual requires the identifying variation to isolate the AEWR component or explicitly hold the local wage fixed.

The preferred model input is therefore a counterfactual-ready predicted hours curve from the paper's final IV or other preferred design. A fixed-effects PPML or a two-stage/control-function analogue is attractive because it handles zeros and delivers a semi-elasticity directly, but the model does not require one particular estimator.

There is also a unit issue to resolve before calibration: a coefficient of \(-0.0061\) when the outcome is a share is a change of \(-0.61\) percentage points per dollar, not \(-0.0061\) percentage points. The abstract, table text, and counterfactual code should use one consistent convention.

## 5. Incidence and welfare boundaries

An AEWR is both a cost to farms and a transfer to workers. A welfare estimate is not well defined until the population whose welfare counts is stated. The paper should report an incidence vector first and aggregate it under more than one welfare boundary.

### 5.1 Farmers

The core farm-profit effect is \(\Delta\Pi^{PE}\) above. It includes optimal behavioral adjustment and is not merely a wage-bill calculation.

For scale, report it as:

- dollars per certified position;
- dollars per certified or realized H-2A hour;
- a share of crop cash receipts;
- a share of hired-labor expense;
- a share of net farm income or farm proprietor income.

The last denominator should be added from the raw BEA CAINC45 lines rather than inferred as cash receipts minus the limited expense variables currently retained.

### 5.2 H-2A workers

The change in H-2A wage earnings is directly interpretable incidence:

\[
\Delta E_H
=w^H_1H_1-w^H_0H_0.
\]

It is not the same as worker welfare. If \(b_H\) is the per-hour value of the worker's outside option net of migration costs, a perfectly elastic foreign labor supply gives

\[
\Delta WS_H
=(w^H_1-b_H)H_1-(w^H_0-b_H)H_0.
\]

The project does not currently observe \(b_H\). The paper should therefore make H-2A earnings a headline incidence result and report worker surplus over a transparent outside-option range. A single calibrated foreign outside wage would create more precision than the data warrant.

### 5.3 Domestic workers

There are two conceptually different domestic-worker effects.

First, workers in corresponding employment may receive a direct wage increase. Their covered hours \(C^{corr}\) are not observed in the public OFLC disclosure data. Use a bound or scenario unless a defensible employer-level source is obtained.

Second, changes in H-2A use may affect the broader domestic agricultural labor market. Estimate with the same preferred design:

\[
\gamma_w=\frac{d\log w_D}{d\log a},
\qquad
\gamma_D=\frac{d\log D}{d\log a}.
\]

QCEW, ACS, and OEWS provide complementary wage and employment measures, each with different coverage limitations. The BLS [QCEW overview](https://www.bls.gov/cew/overview.htm) documents its unemployment-insurance-based coverage. If domestic worker surplus is desired, impose a labor-supply elasticity only in this extension. With isoelastic supply \(D=D_0(w_D/w_{D0})^{\eta_D}\), worker surplus at an equilibrium point is

\[
WS_D=\frac{w_DD}{1+\eta_D},
\]

so the change can be computed from the predicted wage and employment endpoints. Because \(\eta_D\) is not separately identified by the AEWR design, report a literature-informed range rather than treating it as an estimated project parameter.

If both domestic channels are reported, treat corresponding-employment workers as a separate group or exclude them from the broader \(D\) block so their wage gain is not counted twice.

### 5.4 When domestic wages or output prices move

The partial-equilibrium formula above holds \(w^D_m\) and the farm-gate output price \(p_m\) fixed. If either is allowed to respond in the domestic-worker or consumer extension, producer incidence must use the expanded envelope rather than adding price effects to \(\Delta\Pi^{PE}\) informally. Let \(D^O_m\) be other domestic labor and let \(Q_m=\partial R_m/\partial p_m\) be marketed output. Along a counterfactual equilibrium path,

\[
\frac{d\Pi_m}{da}
=-X_m(a)
-D^O_m(a)\frac{dw^D_m(a)}{da}
+Q_m(a)\frac{dp_m(a)}{da},
\]

and hence

\[
\Delta\Pi_m
=\int_{a_{m0}}^{a_{m1}}
\left[
-X_m(s)
-D^O_m(s)\frac{dw^D_m(s)}{ds}
+Q_m(s)\frac{dp_m(s)}{ds}
\right]ds.
\]

Use \(\Delta\Pi_m^{PE}\) for the fixed-price core and this expanded expression whenever domestic wages or farm-gate prices are endogenous. This distinction prevents double-counting and ensures that a price transfer from consumers to producers appears on both sides of the incidence ledger. It also keeps the one-way retail-price bridge honest: if it implies a meaningful farm-gate price change, that change belongs in producer profit as well as consumer equivalent variation.

### 5.5 Welfare aggregates

At minimum, report:

\[
\Delta W^{US}
=\Delta\Pi+\Delta WS_D+\Delta CS+\Delta G,
\]

where foreign-worker surplus is excluded, and

\[
\Delta W^{Global}
=\Delta\Pi+\Delta WS_D+\Delta WS_H+\Delta CS+\Delta G.
\]

\(\Delta G\) is government fee revenue net of administrative resource costs. Fees are transfers within the relevant welfare boundary; staff time and other real administrative resources are costs. For a pure AEWR change, this term is likely secondary and can be omitted from the core if application volume effects cannot be costed.

It is also useful to report a distributionally weighted index,

\[
\Delta\mathcal W
=\omega_F\Delta\Pi
+\omega_D\Delta WS_D
+\omega_H\Delta WS_H
+\sum_k\omega_k\Delta EV_k
+\omega_G\Delta G,
\]

but the unweighted incidence components should always accompany it. Otherwise the social weights, rather than the evidence, can determine the conclusion.

## 6. Optional national price and consumer-incidence extension

### 6.1 Why this is an extension

The local empirical design is close to partial equilibrium: a single county's H-2A response is unlikely to change a national crop price. A nationwide AEWR counterfactual can affect farm-gate and retail prices, however. Those price changes alter incidence and may feed back into H-2A demand.

The first paper should use a one-way, first-order incidence bridge and show that the feedback is small before solving a full product-market equilibrium. If the implied retail effects are material, the model can be expanded later.

### 6.2 Crop cost exposure

For broad crop group \(g\), construct the realized H-2A wage-bill share at the farm gate:

\[
s^H_g=\frac{w^H_gH_g}{R^{farm}_g}.
\]

For a small wage change, the first-order farm cost shock is approximately

\[
d\log c^{farm}_g
\simeq s^{X}_g\,d\log a,
\]

where \(s^X_g\) uses AEWR-exposed, realized hours rather than all certified hours. This is an application of the cost-function envelope theorem. Substitution matters at second order and is already connected to the estimated H-2A response.

Crop allocation should use a hierarchy:

1. crop or activity reported on the H-2A job order when available;
2. employer-worksite crop information recoverable from Addendum B or linked records;
3. predetermined county crop shares from the CDL and NASS;
4. broad labor-intensity weights from Census of Agriculture or ARMS;
5. explicit unallocated/unknown mass rather than forced assignment.

### 6.3 From farm costs to retail prices

Keep farm-market incidence separate from downstream pass-through. A transparent local approximation is

\[
\Delta\log p^{farm}_g
=\kappa_g s^X_g\Delta\log a
\]

and

\[
\Delta\log P^{retail}_g
=\mu_g f_g\Delta\log p^{farm}_g
=\mu_g f_g\kappa_g s^X_g\Delta\log a.
\]

Here \(\kappa_g\) is the fraction of the farm unit-cost shock appearing in the farm-gate price, \(f_g\) is the farm share of the retail food dollar, and \(\mu_g\) scales dollar pass-through from the farm gate to retail. Under full dollar pass-through downstream, \(\mu_g=1\). The [USDA ERS Food Dollar](https://www.ers.usda.gov/data-products/food-dollar) provides detailed food accounts and farm shares. Product-level farm, wholesale, and retail price series can be built from the NASS/AMS data already in the project and BLS/ERS food price series.

The farm-price response \(\kappa_g\) can be estimated directly where the crop price data and policy variation are adequate, or bounded using crop supply and demand elasticities. The retail pass-through parameter \(\mu_g\) can be estimated from distributed-lag farm-to-retail price responses. Until either is estimated credibly, report low, central, and high values or bounds. The county Fisher index currently in the project should not be treated as a retail food price: it assigns state or national farm prices to county crop baskets and also reflects crop composition. It is useful for farm-price diagnostics, not direct consumer welfare.

### 6.4 Household welfare without a named utility function

Let \(s_{kg}\) be food category \(g\)'s expenditure share for household group \(k\). For a small vector of retail price changes, first-order equivalent variation is

\[
\frac{\Delta EV_k}{Y_k}
\simeq-\sum_g s_{kg}\Delta\log P^{retail}_g.
\]

Using group-specific expenditure shares permits arbitrary local non-homotheticity: low- and high-income households can have different food shares and different baskets without imposing Stone-Geary, AIDS, PIGLOG, or non-homothetic CES preferences.

With the sign convention above, summing \(\Delta EV_k\) over households supplies the consumer-welfare term \(\Delta CS\) in the aggregate ledger.

The [BLS Consumer Expenditure Survey public-use microdata](https://www.bls.gov/cex/pumd-getting-started-guide.htm) can supply these shares by income quintile and household type. If retail price changes are large enough for substitution to matter, add a second-order term using compensated price derivatives and draw elasticity matrices from the [USDA ERS Commodity and Food Elasticities database](https://www.ers.usda.gov/data-products/commodity-and-food-elasticities/documentation). Enforce adding-up, homogeneity, symmetry, and negative semidefiniteness rather than combining unrelated elasticities mechanically.

## 7. Moments and their empirical analogues

### 7.1 Moments required for the core model

| Model object | Empirical analogue and construction | Existing source | Treatment in the model |
|---|---|---|---|
| \(H^c_{m0}\): baseline certified hours | Certified positions × stated weekly hours × contract weeks, allocated to worksite geography | `data/intermediate/h2a_aggregated.parquet`; constructed in `code/b01_derived/01_h2a_aggregation_nodupes.R` | Required. Rebuild from a unique contract-worksite artifact and retain the full distribution. |
| \(N^c_{m0}\): certified positions | Adjusted certified worker count by worksite and contract start year | Same H-2A files | Required for headline quantity effects and hours-per-position decomposition. |
| \(A_{m0}\): applications | Fractionally allocated unique case count | Same H-2A files | Validation and participation-margin decomposition. |
| \(q_{m0}\): hours per position | \(H^c/N^c\); also weekly hours and contract duration separately | Raw OFLC fields already ingested | Required to translate headcount into contract hours and diagnose the response margin. |
| \(\psi_m\) or \(\varepsilon_m\): hours response | Causal semi-elasticity or elasticity of certified hours with respect to the AEWR, using the preferred IV/design | Must be estimated; current main regression uses positions normalized by 2011 farm employment | Required. This is the central behavioral sufficient statistic. |
| \(b_m(a)\): binding exposure | Share of certified contract hours whose offered wage is at the AEWR (within a documented tolerance), accounting for other applicable floors | Raw `wage_rate`, wage unit, AEWR, dates in `data/intermediate/h2a_with_fips.parquet`, and source files | Required for the mechanical wage exposure. The currently saved county mean wage is not enough. |
| \(\partial w^H/\partial a\) above the floor | Contract-level covariance of offered wages with AEWR among jobs above the floor, or two polar rules: fixed premium versus fixed market wage | Raw contract panel | Sensitivity for finite counterfactuals. |
| \(\rho_{mt}\): realized/certified hours | Paid hours divided by certified hours; not publicly observed at county level | Not in project | Partially identified. Report certification-space results and a national/state realization range. DOL certifications, USCIS approved workers, State Department visas, and the three-fourths guarantee inform but do not point-identify paid hours. |
| \(a_{rt}\): policy wage path | Legally applicable AEWR by region, occupation, and effective date | Existing AEWR extraction and Federal Register notices | Required. Use nominal legal values for payment, then deflate with the chosen welfare numeraire. |
| Baseline offered wage bill | Hour-weighted offered wage × realized hours | Raw H-2A records and `data/intermediate/h2a_with_fips.parquet` | Required for earnings incidence and validation. The current mean is worker-weighted rather than hour-weighted. |
| Farm financial scale | Hired labor expense, total production expense, crop cash receipts, and net farm income | `data/intermediate/bea_cainc45_data_year.parquet`; raw BEA CAINC45; Census of Agriculture | Required for validation and scale ratios, not for the envelope integral itself. |

The public [DOL OFLC performance-data page](https://www.dol.gov/agencies/eta/foreign-labor/performance) supplies annual H-2A disclosures and record layouts. The [USCIS H-2A Employer Data Hub](https://www.uscis.gov/tools/reports-and-studies/h-2a-employer-data-hub) provides approved-petition information from FY2015 onward. State Department visa issuances and DHS admissions can discipline national utilization, but visas, admissions, positions, and unique workers are different concepts and must not be equated.

### 7.2 Additional moments for domestic-worker welfare

| Model object | Empirical analogue | Available or straightforward source | Identification status |
|---|---|---|---|
| \(w^D_{mt}\) | Agricultural hourly wage or annual pay | `data/intermediate/acs_state_ag_wage.parquet`, `acs_czone_wage_quantiles.parquet`, `qcew.parquet`, and `oews_county_aggregated.parquet`; FLS region wages | Existing data are available. QCEW excludes some farm employment; OEWS excludes most farms; ACS is noisy locally; FLS is region-level and helps set the AEWR. Triangulate rather than select by fit. |
| \(D_{mt}\) | Domestic hired agricultural employment or hours | QCEW NAICS 111 and 1151; ACS workers and usual hours; Census of Agriculture workers by days worked; `data/intermediate/nawspad.parquet` annual hours | Mostly available. Construct hours consistently and avoid BEA farm employment when a hired-worker concept is required because BEA includes proprietors/jobs with different coverage. |
| \(\gamma_w,\gamma_D\) | AEWR causal effects on domestic wages and employment | Re-estimate the preferred design with wage and employment outcomes | Not yet estimated. Required for a point estimate of broad domestic benefits. |
| \(C^{corr}\) | Paid hours of domestic workers in corresponding employment at H-2A employers | Employer payroll/job-order records | Not observed in current public data. Use bounds or a clearly labeled scenario. Do not set equal to all domestic labor. |
| \(\eta_D\) | Domestic agricultural labor-supply elasticity | External estimates or a range | Calibration/sensitivity parameter; not identified by the H-2A demand regression. |

The USDA [Farm Labor Survey](https://www.nass.usda.gov/Surveys/Guide_to_NASS_Surveys/Farm_Labor/) reports directly hired worker counts, hours, and wage rates at AEWR-region frequency. The [2022 Census of Agriculture farm-labor tables](https://data.nass.usda.gov/Publications/AgCensus/2022/Full_Report/Volume_1%2C_Chapter_2_US_State_Level/st99_2_007_007.pdf) report hired workers, payroll, workers by days worked, and migrant workers. These are useful validation totals and hours inputs; neither directly reveals corresponding-employment hours.

### 7.3 Additional moments for price and consumer incidence

| Model object | Empirical analogue | Existing or straightforward source | Treatment |
|---|---|---|---|
| \(s^X_g\) | AEWR-exposed realized wage bill divided by crop farm revenue | H-2A contract data + NASS prices/yields + CDL acreage + BEA crop receipts | Construct for broad crop groups; show unallocated contracts. |
| Crop allocation | Contract crop/activity or predicted allocation using predetermined local crop shares | H-2A records; `data/intermediate/croplandcros_county_crop_acres.parquet`; NASS/CDL crosswalk; RMA and AMS data | Use multiple allocation rules and report sensitivity. |
| Farm-gate prices and quantities | State crop prices/yields/production and county crop acreage | Existing NASS Quick Stats, synthetic state price/yield, CDL, AMS My Market News, RMA | Available. Current county Fisher index is a diagnostic farm-price/composition measure. |
| \(\kappa_g\): farm-market incidence | Farm-gate price response to an H-2A unit-cost shock | NASS/AMS farm prices and the estimated AEWR shock; alternatively crop supply and demand elasticities | Estimate for broad groups where powered; otherwise bound. Required only for the endogenous-price extension. |
| \(f_g\): farm share | Farm commodity value as a share of retail expenditure by detailed food account | USDA ERS Food Dollar | Straightforward external download. |
| \(\mu_g\): downstream dollar pass-through | Dynamic response of wholesale/retail food price to farm-gate price | NASS/AMS farm prices; ERS/BLS PPI and CPI food categories | Estimate with lags where feasible; otherwise report bounded scenarios. |
| \(s_{kg}\): household shares | Food-category spending divided by total expenditure or income, by income/household group | BLS CEX PUMD or published CEX tables | Straightforward external data. Required for first-order distributional incidence. |
| Hicksian derivatives | Compensated own/cross-price elasticities by food category | ERS elasticity database or an estimated demand system | Optional second-order robustness only. |

### 7.4 Objects that should remain bounds or scenarios in the first version

The following are not credibly point-identified with current data:

- paid H-2A hours by county;
- H-2A workers' foreign outside option;
- domestic corresponding-employment hours;
- crop assignment for every H-2A contract;
- crop-specific retail pass-through;
- a long-run capital/mechanization elasticity;
- nationwide product-price feedback into local H-2A demand.

Treating these as sensitivity parameters is a feature of the design, not a failure. The paper should separate statistical confidence intervals for estimated moments from identified sets or scenario ranges for unobserved ones.

## 8. Counterfactuals worth reporting

### 8.1 Primary local counterfactual

Report a one-dollar increase in the real AEWR, evaluated at observed market-year baselines. This is closest to the current empirical variation and makes the profit formula easy to audit. Report:

- change in certified positions;
- change in certified contract hours;
- change in applications and hours per position;
- farm-profit change per market and nationally;
- H-2A wage-income change;
- global surplus over the foreign-outside-option range.

### 8.2 Policy-sized counterfactuals

Report uniform \(+5\%\), \(+10\%\), \(-5\%\), and \(-10\%\) changes only where they remain within or close to empirical support. Use the linear, exponential, isoelastic, and flexible dose curves to show curvature sensitivity.

### 8.3 Historical rule counterfactual

Once the legal wage series is constructed carefully, compare the observed AEWR path with a clearly defined alternative methodology, such as the wage path implied by an earlier rule or another published benchmark. This counterfactual is more policy-relevant than setting the AEWR to zero, but it requires a legally and temporally correct alternative series.

### 8.4 Counterfactuals not suitable as headline estimates

Eliminating the AEWR or the H-2A program is far outside the local support and changes worker supply, recruitment costs, domestic labor markets, crop choice, and potentially immigration enforcement. It can be shown only as a deliberately model-dependent stress test, not as the paper's main welfare number.

## 9. Estimation and calibration workflow

### Step 0: establish a data contract

Create one contract-worksite-year artifact before aggregation with:

- case and worksite identifiers;
- county/CZ/AEWR region;
- decision, start, and end dates;
- positions requested and certified;
- weekly hours and certified contract hours;
- offered hourly wage and wage unit;
- applicable AEWR and other known wage floor;
- employer/agent/association type;
- crop/activity when reported;
- flags for imputation, multi-worksite allocation, and binding status.

Preserve this artifact rather than retaining only county means. The binding share, wage distribution, hours weights, and finite counterfactual wage rule all require contract-level information.

### Step 1: estimate a quantity-response vector

Using the final preferred identification strategy, estimate the response to the AEWR of:

\[
(\text{positions},\ \text{contract hours},\ \text{applications},\
\text{positions/application},\ \text{hours/position}).
\]

Estimate heterogeneity only where it is well powered—for example by broad crop labor intensity, baseline H-2A use, or contract size. A separate elasticity for every county or crop is unnecessary and will add noise.

### Step 2: construct policy exposure

For each contract and counterfactual wage:

1. determine whether the AEWR binds;
2. calculate the counterfactual offered wage under each admissible wage rule;
3. predict certified hours;
4. apply the realization-factor draw;
5. add corresponding-employment exposure only under the reported scenario.

### Step 3: integrate farm profit

Numerically evaluate

\[
-\int X_m(a)\,da
\]

over the policy path for every market and parameter draw. Aggregate after calculating market-level responses; do not apply an elasticity to a national average wage and average quantity when exposure and elasticities vary.

### Step 4: add incidence modules

Compute H-2A earnings mechanically. Add domestic worker surplus only when \(w_D(a)\), \(D(a)\), and a labor-supply range are specified. Add household equivalent variation only when crop allocation, farm-to-retail mapping, and CEX shares are available.

### Step 5: propagate uncertainty

Each counterfactual draw should vary:

- the estimated hours response, respecting its clustered sampling distribution;
- dose-response curvature;
- the certified-to-paid realization factor;
- binding-wage exposure and finite wage-setting rule;
- corresponding-employment exposure;
- domestic labor-supply elasticity, if used;
- crop allocation and pass-through;
- household expenditure shares and demand elasticities, if used;
- the foreign-worker outside option and welfare weight.

Report sampling uncertainty and assumption uncertainty separately. A tight confidence interval conditional on a wide realization/pass-through range is not a tight welfare estimate.

The initial implementation can be written transparently in R or ordinary array code. JAX is not necessary for this model; it becomes useful only if the project later evaluates very large grids, differentiates a richer equilibrium, or estimates a high-dimensional demand system.

## 10. Validation and credibility checks

The model should match moments it was not directly targeted to match:

1. **Position-hours decomposition:** predicted position, duration, and weekly-hours effects should aggregate to the predicted contract-hours effect.
2. **Wage binding:** the model should reproduce the hour-weighted distribution of offered wage minus AEWR.
3. **Wage-bill feasibility:** predicted H-2A wage payments should not exceed BEA/Census hired and contract labor expense after concept and coverage adjustments.
4. **Program scale:** aggregate positions and contract duration should reconcile with published DOL/ERS totals.
5. **Realization sensitivity:** national employment-space totals should be compatible with USCIS worker approvals, State Department visa issuances, and the possibility that one visa fills multiple positions.
6. **Domestic outcomes:** any predicted domestic wage/employment effect should be compared across ACS, QCEW, OEWS, and FLS rather than selected from the preferred source ex post.
7. **Price bridge:** predicted farm and retail price effects should be compared with direct reduced-form estimates but should not be calibrated to the same noisy price coefficient they are meant to validate.
8. **Support:** display the share of baseline hours for which each counterfactual lies inside the empirical AEWR support.
9. **Leave-region-out results:** no single AEWR region or large commuting zone should dominate national welfare.
10. **Curvature robustness:** report the integral under multiple demand curves and, for larger changes, the curvature needed to reverse the welfare ranking.

## 11. Data issues to resolve before any welfare calibration

### 11.1 County-year duplication

A read-only audit on July 17, 2026 found that `data/processed/county_df_analysis_year.parquet` has 198,892 rows but only 46,615 unique county-year keys. Most county-years appear four times and 2008 often appears eight times. The duplications are multiplicative: `c03_build_03_lags_classification_write.R` constructs 2008 classification tables without enforcing one row per county and then merges them back by county; the upstream build file already contains some duplicated 2008 keys.

This does not merely affect file size. It can change year weights, lags, summary totals, and inference. Before estimating moments or aggregating welfare:

- assert uniqueness at every intended merge key;
- identify the upstream source of the 2008 duplicate;
- use `distinct(countyfips, .keep_all = TRUE)` only after verifying that supposedly duplicate rows agree on all substantive fields;
- rebuild lags after uniqueness is restored;
- regenerate the regression tables on the unique panel;
- never sum H-2A hours or BEA dollars from the duplicated final panel.

The cleanest welfare inputs should come from dedicated unique artifacts, not from a wide regression panel.

### 11.2 Certified is not employed

`man_hours_certified_*` is scheduled contract labor: the pipeline multiplies certified positions by stated weekly hours and contract length, imputing missing weekly hours. It is not actual hours worked. This distinction must appear in variable names, tables, and welfare text.

### 11.3 Wage weighting and binding

The current county mean offered wage is weighted by certified workers, not contract hours. Welfare needs an hour-weighted wage and the full offered-wage-minus-AEWR distribution. Wage-unit harmonization should be validated at the contract level rather than relying only on a rule that drops reported rates above $100.

### 11.4 Outcome scaling

Keep share units, percentage points, percent changes, semi-elasticities, and elasticities separate. Include a unit-test example in the counterfactual code showing how a one-dollar change maps from the regression coefficient to positions, hours, and percent changes.

### 11.5 Price concepts

Separate:

- the AEWR in nominal legal dollars;
- a real employer wage cost;
- a farm-gate output price;
- a wholesale food price;
- a retail consumer price;
- the general welfare numeraire.

The same farm-products PPI should not silently serve all five roles.

## 12. Suggested paper presentation

The main text can present the model compactly:

1. a general optimized farm-profit problem;
2. the definition of AEWR-exposed hours;
3. the envelope formula \(\Delta\Pi=-\int X(a)da\);
4. the estimated hours dose response;
5. H-2A earnings and U.S./global welfare boundaries;
6. a short explanation that technology, substitution, and program entry are summarized by the empirical hours response.

The appendix can contain:

- the exact mapping from regression coefficients to the dose curve;
- binding-wage and realization-factor construction;
- curvature and support checks;
- domestic-worker surplus formulas;
- farm-to-retail and CEX incidence calculations;
- uncertainty propagation;
- the full moment-to-data crosswalk.

Recommended output tables and figures are:

1. positions, contract hours, applications, and hours-per-position elasticities;
2. hour-weighted wage-floor binding by region and year;
3. a one-dollar AEWR counterfactual: quantity, farmer profit, and H-2A earnings;
4. \(\pm5\%\) and \(\pm10\%\) results across admissible dose curves;
5. U.S. versus global welfare over the foreign outside-option range;
6. domestic-worker effects or bounds;
7. retail price and household equivalent variation by income quintile;
8. a specification curve showing which realization, pass-through, and curvature assumptions materially change the conclusion.

## 13. Bottom line

The paper does not need a large structural model to produce serious welfare counterfactuals. Its strongest model is the one that exploits the economic content of the elasticity it is already designed to estimate:

\[
\text{causal H-2A hours response}
\quad+\quad
\text{wage-floor exposure}
\quad\Longrightarrow\quad
\text{farm-profit change by envelope integration}.
\]

That core result remains valid under a broad class of production technologies, substitution patterns, farm heterogeneity, and H-2A participation decisions. The price of this robustness is a deliberately narrow counterfactual: an anticipated AEWR change over an empirically supported range, initially holding other market prices fixed. Domestic-worker and consumer welfare should be layered on only with their own empirical moments and clearly labeled uncertainty. This gives the project a credible set of welfare estimates without asking the reader to believe an unnecessarily elaborate macro or farm structure.

## Methodological references

- Raj Chetty, [“Sufficient Statistics for Welfare Analysis: A Bridge Between Structural and Reduced-Form Methods”](https://doi.org/10.1146/annurev.economics.050708.142910).
- David S. Lee, Pauline Leung, Christopher J. O'Leary, Zhuan Pei, and Simon Quach, [“Are Sufficient Statistics Necessary? Nonparametric Measurement of Deadweight Loss from Unemployment Insurance”](https://doi.org/10.1086/711594).
- Zi Yang Kang and Shoshana Vasserman, [“Robustness Measures for Welfare Analysis”](https://www.aeaweb.org/articles?id=10.1257/aer.20220673).
