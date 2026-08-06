# Predetermined AEWR-bite design

## Question and estimand

Estimate whether the same regional AEWR change has a larger effect on H-2A demand in local labor markets where the wage floor was more likely to bind.

The design identifies a differential response to predetermined AEWR exposure. It does not, without additional structure, identify the aggregate elasticity to the region-wide AEWR level.

## Why the earlier bite design was discarded

The earlier treatment,

\[
AEWR_{rt}-w^{25}_{ct},
\]

used a time-varying local wage quantile. That wage is itself affected by local labor-demand shocks and may respond to the AEWR. The resulting bite therefore combines the policy with an endogenous outcome. Interacting it with a post-2011 indicator does not create an external policy shock.

## Revised exposure

Freeze the local wage distribution \(F_c^0(w)\) using a pooled pre-period, preferably 2007--2008. Never update it using post-treatment wages.

Construct predicted per-worker exposure:

\[
B^0_{ct}
=
\int
\left[
AEWR_{rt}-\max\{w,M_{ct}\}
\right]_+
dF_c^0(w),
\]

where \(M_{ct}\) is the other applicable wage floor. Simpler alternatives are the fixed baseline share below each year's AEWR, or the baseline gap interacted with the annual AEWR change. Use the full distribution as the preferred measure.

The baseline distribution can use all wage-and-salary workers, matching the old design, with agricultural and low-wage-worker distributions as sensitivity analyses.

## Empirical design

Estimate

\[
\log E[Y_{ct}]
=
\alpha_c+\lambda_{st}
+\beta B^0_{ct}
+X_{ct}'\gamma+\varepsilon_{ct},
\]

where \(\lambda_{st}\) is a state-by-year fixed effect. Identification therefore compares counties or CZs facing the same AEWR and state-level shocks in the same year but having different frozen baseline exposure. AEWR-region-by-year effects are a less restrictive alternative.

An equivalent specification interacts a fixed baseline exposure measure with annual changes in \(\log AEWR_{rt}\). Use a zero-filled county- or CZ-year panel and PPML for applications and requested positions.

## Identifying assumption

Conditional on state-by-year effects and controls, locations with different baseline exposure would have followed parallel H-2A demand trends absent the AEWR change. The main threat is that initially low-wage agricultural areas have different secular changes in crops, technology, labor supply, or H-2A adoption.

## Required controls and diagnostics

- Baseline-exposure-bin trends or baseline covariates interacted with year.
- Crop-by-year and harvest-calendar-by-year effects.
- Event-study coefficients by baseline exposure.
- Placebo interactions using pre-period AEWR changes.
- Alternative frozen baseline years and wage distributions.
- No specifications using contemporaneous local wages.
- AEWR-region wild-bootstrap and few-cluster-robust inference.
- Results excluding border areas and large shared employer/FLC networks.
- Reweighting high- and low-exposure places to common baseline support.
- Employer-level estimates among continuing employers, alongside local-market entry and exit.

Interpret \(\beta\) as the response to predicted relative wage-cost exposure. Translating it into an overall AEWR elasticity requires an explicit assumption about how this exposure maps into each employer's marginal H-2A cost.
