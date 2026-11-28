# Bite-Design Statistical-Power Rescue Investigation

Isolated two-phase, outcome-blind statistical-power investigation testing
whether any size-valid bite-based design built strictly from cached local
repository data can materially tighten precision relative to
`code/designs/policy_2025/` and `code/designs/bite/`.

- **Plan:** [`markdowns/bite_rescue_power_plan.md`](../../markdowns/bite_rescue_power_plan.md)
- **Final report:** [`markdowns/bite_rescue_power_report.md`](../../markdowns/bite_rescue_power_report.md)
- **Locked Phase 2 registry:** `phase2_registry.json` (`phase2_registry.sha256`)

## Status and verdict

Completed November 27, 2026. **No candidate is promoted to production.**

- **Phase 1 (`59` specifications across `R1`–`R6`):** `55/59` killed strictly
  outcome-blind (`0` Tier 1 survivors; `0` 2025-rule survivors, preserving 100%
  `FY2026` quantity blindness).
- **Phase 2 (`4` Tier 2 pre-2025 FLS survivors):**
  - `R4a` (`2011–2025` baseline contract-duration bite, `P1`/`P2`): **killed**
    in the adversarial referee audit due to mechanical duration mean reversion,
    `96.5%` reliance on national `E_dur x Year` trends, and uninformative
    structural `eps_cost` MDE (`4.28–4.65`).
  - `R2a` (`2020–2025` post-Addendum-B direct-grower `E_p25` bite, `P3`/`P4`):
    **diagnostics-only** clean null (`+0.061 [-0.104, +0.225]` in `1b-PPML`,
    `-0.299 [-0.707, +0.108]` in `1a-PPML`) with tighter reduced-form MDEs
    (`0.463` 2-SD / `0.573` 1-SD) but near-zero differential wage pass-through.

## Execution order

Run from the repository root with `Rscript --vanilla`:

| Stage | Script | Role |
| --- | --- | --- |
| 0 | `00_data_audit.R`, `00b_verify_engine.R` | Audit cached inputs (`GEO-001`, `AEWR-001`) and verify exact 2-period binomial-logit CV3J parity against `fixest::fepois` |
| 1 | `01_phase1_r1_r5.R` | Phase 1 outcome-blind triage for `R1` (min-wage floor truncation & sub-state tiers) and `R5` (14 within-state local-wage crossover first stages) |
| 2 | `02_phase1_r2.R`, `02b_finish_r2c_bite.R` | Phase 1 triage for `R2` (direct-grower restriction, `CDL x Year` & NOAA climate PCs, ANCOVA, SDID, Bai IFE) |
| 3 | `03_phase1_r3.R` | Phase 1 triage for `R3` (fixed-cohort balanced direct-grower businesses and within-multi-state-filer cross-state reallocation) |
| 4 | `04_phase1_r4_r6.R`, `04b_fix_r4_bus_dur.R` | Phase 1 triage for `R4` (duration-amortization & farm labor-share bite) and `R6` (stacked `2009–10` + `2025` reversals) |
| 5 | `05_phase1_deep_audit_borderline.R`, `05b_phase1_deep_audit_bc.R`, `06_compile_phase1_triage.R` | Exact nonlinear CV3J calibration on borderline candidates and compilation of `phase1_triage.csv` |
| 6 | `07_phase2_estimate_and_audit.R` | Fail-closed SHA-256 verification of `phase2_registry.json`, Phase 2 `CV3J` estimation (`P1`–`P4`), Holm adjustment, matched placebos, 17-region LOO influence, and 45-spec adversarial referee audit |

Shared helpers live in `helpers_phase1.R`. Generated `.csv` outputs under this
directory are gitignored by the repository's global `*.csv` rule and are fully
reproducible from the scripts above against cached local data.
