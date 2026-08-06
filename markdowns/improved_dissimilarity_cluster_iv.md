# Improved dissimilarity-cluster IV

## Question and estimand

Estimate the elasticity of H-2A demand with respect to the AEWR. The design uses wage innovations in distant, economically disconnected parts of the same AEWR region to predict the region-wide AEWR faced by a target cluster.

The estimand is local to AEWR movements predicted by the retained donor wage shocks.

## Main changes to the current design

| Current design | Revised design |
|---|---|
| Five clusters and two donors in every region | Region-specific constrained clusters |
| No minimum cluster size or mass | At least 3 CZ-region units and 5% of baseline hired-worker hours |
| Donors selected only by agro-dissimilarity | Donors must also be geographically and economically disconnected |
| Annual weights calibrated to the realized FLS wage | Frozen pre-period weights; no FLS or AEWR calibration |
| Annually updated frame | Fixed baseline worker-hour frame |
| County-year estimation | Target-cluster-year estimation |
| Region-by-cluster standard errors | AEWR-region inference with few-cluster and weak-IV procedures |

## Cluster and donor construction

1. Define units as CZ-by-AEWR-region intersections.
2. Use only pre-sample crop mix, harvest timing, climate, soil, and farm structure.
3. Construct an exclusion graph. A target-donor pair is inadmissible if it:
   - lies within a pre-specified distance;
   - shares important employers, farm labor contractors, or associations;
   - has substantial crop-by-harvest-calendar overlap; or
   - has strong pre-period residual wage or H-2A co-movement.
4. Form clusters subject to minimum unit and baseline worker-hour mass constraints. Let the number of clusters vary by region.
5. Choose donors only from admissible clusters. Require an effective donor-area count of at least five and cap any area's weight at 30%.
6. Freeze the cluster map, donor map, and weights before examining the first or second stage.

Exclude a region if these constraints are infeasible; do not relax them after examining results. Cluster stability under alternative feature weights and small data perturbations is a required diagnostic.

## Instrument

For target cluster \(g\) in region \(r\), define

\[
Z_{grt}
=
\sum_{a\in D_{gr}}s^0_{gra}\,
\log w^{OEWS}_{a,t-1},
\]

where \(D_{gr}\) is the fixed admissible donor set and \(s^0_{gra}\) is a frozen baseline hired-worker-hour share.

OEWS is an external predictor of the FLS wage, not a mechanical component of it. The primary instrument must not use realized FLS wages, AEWRs, quarterly FLS targets, H-2A outcomes, future Census benchmarks, or annual endogenous frame updates.

Use \(\Delta Z_{grt}\) to instrument \(\Delta\log AEWR_{rt}\) in a first-difference robustness specification.

## Estimation

\[
\log Y_{grt}
=
\alpha_g+\lambda_t
+\beta\log AEWR_{rt}
+X_{grt}'\gamma+\varepsilon_{grt},
\]

instrumenting \(\log AEWR_{rt}\) with \(Z_{grt}\). The primary outcome is requested positions or applications; certified positions and contract hours are secondary. Use a fixed baseline farm-employment offset where appropriate.

Estimate at the target-cluster-year level. Cluster at the AEWR-region level and report CR2 or restricted wild-cluster-bootstrap inference together with Anderson--Rubin or CLR confidence sets.

## Identifying assumption

Conditional on fixed effects and controls, retained donor wage innovations affect target-cluster H-2A demand only through the AEWR. Agro-dissimilarity alone is insufficient; the exclusion graph is intended to remove the main geographic, commodity, employer-network, and worker-mobility channels.

## Required diagnostics

- Region-clustered first stage and weak-IV-robust confidence sets.
- Instrument concentration and effective donor count.
- Leads of the instrument and pre-period outcome trends.
- Leave-one-region, donor-cluster, and OEWS-area-out estimates.
- Separate distant-donor and crop-dissimilar-donor instruments.
- Results excluding all shared employer/FLC networks.
- Results for 2011--2019 separately from 2020--2022.
- Permutations over admissible donor maps and plausible-exogeneity bounds.

The design should remain supporting evidence unless these diagnostics show both a strong first stage and limited sensitivity to donor composition.
