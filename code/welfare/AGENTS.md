# Narrow welfare branch instructions

- Read `README.md`, `elasticity_contract.toml`, and `content/designs/welfare.md`
  before changing the contract, envelope math, exposures, or outputs.
- Keep this branch standalone (`scripts/run_welfare.sh`, outside `run_all.sh`)
  and strictly separate from causal estimation; never search estimator registries
  for a favorable elasticity or misrepresent the scenario-bounds grid as a single
  precision-identified point estimate.
- Keep `01_narrow_welfare.py` and `02_validate_welfare.py` imperative Marimo
  notebooks (`marimo.App`) and `envelope.py` plain functions with zero classes.
- Preserve the `WELFARE-001` contract constants (`ELASTICITY_DEFINITION`,
  `APPLICATION_POPULATION`, `COUNTERFACTUAL_DELTA`), string geographic keys
  (`GEO-001`), fail-closed SHA-256 contract pinning, certification-space units
  with explicit `rho` paid-hours scaling, and first-order vs second-order
  envelope decomposition.
- Run `code/welfare/tests/test_welfare.py`, `ruff check`, and `./scripts/run_welfare.sh`
  (including `02_validate_welfare.py`) after any edit, and review grounded prose
  in `content/designs/welfare.md` whenever `#welfare-design-contract` changes.
