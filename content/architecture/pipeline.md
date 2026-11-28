+++
title = "Pipeline architecture"
description = "Supported execution order and branch structure."

[extra]
scopes = ["code", "scripts"]
+++

The supported workflow separates source preparation, reusable derived data, a
design-neutral county-year panel, and empirical-design branches.

Within the source/derived path, the role-aware H-2A matcher uses FY2008--2026 Q4
identity evidence (manual review of the FY2026 Q4 extension pending) before B01 constructs the three
positive-certified county-year worksite-business counts. Its linkage contract
is documented separately from any employer-year panel or estimator.

```text
code/a01_sources -> code/b01_derived -> code/c01_clean -> code/c02_build
                                                               |
             +--------+-------+--------+----------+---------+--+-------+
             |        |       |        |          |         |          |
      descriptives   DiD    bite   panel IV    MC dose  border ring  welfare
```

The optional `code/welfare` branch consumes the shared panel, the Oct. 2, 2025
occupation/skill schedule, and the SHA-256-pinned v2 scenario-bounds contract
(`elasticity_contract.toml`, anchored to the FY2026 Q4 Track 1 lower bound and
offered-wage pass-through). Its standalone runner (`scripts/run_welfare.sh`)
remains outside the complete pipeline and fails closed on contract or hash drift.

The optional `code/designs/bite` branch (`scripts/run_bite.sh`) consumes the
unified shared panel directly without altering `code/c01_clean`, `code/c02_build`,
or the retained DiD branch. It owns the standalone full-panel FE-PPML bite
estimates, CV3J jackknife diagnostics, standalone 48-state domestic-floor margin,
and validation report, and remains outside the complete pipeline.

The optional `code/designs/border_discontinuity` branch consumes the shared
panel and upstream geographic, crop, climate, and soil artifacts. It owns all
similarity features, border-ring windows, treatments, fixed effects, weights,
inference objects, and robustness-audit outputs used by the experiment. Its
standalone runner is not part of the complete pipeline or manuscript workflow.

Panel IV runs a single stages01--11 R pipeline for the original moving-wage
estimator. Calibration is deterministic. The separate experimental extensions
and diagnostics runner were retired; the branch validator is stage09, followed
by the stage10 Kitagawa instrument-decomposition diagnostic and the stage11
QWI wage-granularity diagnostic.
The border branch retains the v5 audit and adds a bounded local redesign with
shared-region dependence, signed power/MDE diagnostics and a generated report;
its runner ends with validation after the new estimation/report stages.
The standalone border input-upgrade runner first corrects cached wage and
Census sources, preserves worksite allocation evidence, builds the reusable
monthly calendar, and prepares point-geocoding requests. It rebuilds the shared
panel and affected designs before the separate denominator/report/validation
stages. The normal B01 runner includes the monthly calendar after allocation;
public geocoding acquisition requires an explicit flag. This flow is documented
in [Border data-input comparisons](@/designs/border-data-inputs.md).
Neither branch moves treatment or calibration into the shared panel. Both
[design contracts](@/contracts/causal-outcomes-and-model-inputs.md) distinguish
executable completion from empirical identification and useful precision.

The complete runner executes sources, derived data, the shared panel,
descriptives, DiD, panel IV, and Mundlak–Chamberlain in that order.

The standalone [2025 policy-shift design](@/designs/policy-2025.md) consumes
cached H-2A source records, the shared panel, wage references, CDL crop acres,
and quarterly earnings files. It owns offered-wage, complete FY2026 Q4 Track 1
county quantity, five-panel mechanism-heterogeneity, and interim covered-earnings
analyses along with their code-generated manuscript tables and figures. Its
runner validates all 703 ledger rows and publishes 52 retained products from
isolated temporary work. It performs no source acquisition, excludes the 2013
shutdown analysis, and remains outside the complete runner.

The standalone [historical wage-reform investigation](@/designs/wage-reforms.md)
reads the owning H-2A source and reviewed business-membership microdata directly;
individual filing dates and offered wages are absent from the county aggregate.
Its four Marimo stages build design panels, estimate diagnostic comparisons,
generate a report and validate retained evidence. It is outside the complete
runner and does not modify shared panels, allocated outcomes or source inputs.

{{ <grounding path="scripts/run_all.sh" anchor="pipeline-order" sha256="6ce943803a89bf2ce24018b67f7d970cdafdaa846a5ecf3eed14c0c64d8ecccf" /> }}

Run the complete chain from the repository root:

```sh
./scripts/run_all.sh
```

Set `DRY_RUN=1` to print the declared order without executing analysis code.
Branch-specific commands remain documented in the repository root and branch
READMEs.
