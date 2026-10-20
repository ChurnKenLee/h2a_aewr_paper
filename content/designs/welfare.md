+++
title = "Narrow certification-space welfare calculation"
description = "Frozen elasticity input, real one-dollar AEWR counterfactual, exposure rules, and producer-incidence outputs."

[extra]
scopes = ["code/welfare", "scripts/run_welfare.sh"]
+++

The narrow welfare branch is a static, partial-equilibrium calculator, not an
additional causal estimator. It applies one frozen certified-hours elasticity
to the design-neutral shared county-year panel. The elasticity contract must
record the selected estimate, standard error, inference method, source
artifact and SHA-256 hash, source-row selector, source population and
weighting, target years, and estimator-matched real-AEWR units.

{{ <grounding path="code/welfare/01_narrow_welfare.py" anchor="welfare-design-contract" sha256="c4a7ce0e74ea871db28ee875dbc2a3ce684a61cfe18280614ad9603351b40339" /> }}

The runner refuses to generate real outputs while the contract is unfrozen or
when its source-artifact hash changes. Freezing the contract is therefore an
explicit empirical-specification decision; the welfare code does not search
the retained estimator registry or choose a favorable result.

## Counterfactual and application population

For baseline real AEWR $a_0$, certified hours $H^c_0$, and elasticity
$\eta=d\log H^c/d\log a$, the calculator uses the local constant-elasticity
curve $H^c(a)=H^c_0(a/a_0)^\eta$ and integrates exposed certified hours from
$a_0$ to $a_0+1$. It imposes no sign restriction on $\eta$ and uses the
logarithmic limit when $\eta=-1$.

The elasticity is applied to every shared-panel county-year in the source
estimand's declared target window. This is an explicit external-validity
extrapolation when the source estimand uses a narrower sample. Annual national
totals are retained, and the headline is their unweighted arithmetic mean. The
sum across target years is not presented as an annual effect.

## AEWR-exposed certified hours

The observed calculation uses `cert_hours_at_aewr_start_year`, which classifies
explicit hourly offers within five cents of the contract start-year or
prior-year AEWR. This is a wage-offer exposure proxy, not proof that the AEWR
was the legally binding wage floor.

The coverage-adjusted sensitivity applies the observed binding share among
comparable hours to all certified hours. A positive-hours county-year without
a comparable offer uses its AEWR-region-year hour-weighted share and then the
national-year share if the region-year has no support. The calculator stops if
neither fallback exists.

## Interpretation and exclusions

The reported monetary object is certification-space farm quasi-rent:
producer incidence from the AEWR-exposed certified-hours curve, holding output
prices and other domestic-labor wages fixed. If paid hours are a constant
fraction $\rho$ of certified hours, paid-hours incidence is $\rho$ times the
reported result. The branch does not choose a value for $\rho$.

The calculation excludes corresponding-employment hours, H-2A worker surplus,
broader domestic labor-market effects, consumer incidence, administrative
costs, and general-equilibrium price responses. Confidence intervals propagate
only the frozen elasticity's reported standard error; they are conditional on
the exposure construction and maintained partial-equilibrium assumptions.

Run from the repository root:

```sh
./scripts/run_welfare.sh
```

Inputs, output paths, and contract-freezing instructions are listed in
`code/welfare/README.md`.
