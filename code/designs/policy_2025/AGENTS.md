# 2025 policy design instructions

- Read `README.md`, `design.json`, and the canonical policy-2025 page before
  editing. The November 16 migration is after outcome inspection.
- Preserve the frozen exposure and source vintage. A refresh or specification
  change needs a dated amendment and preservation of the previous outputs;
  never update hashes just to make checks pass.
- The current owner includes wage offers and covered earnings only. Do not
  add the 2013 shutdown or import another design's configuration implicitly.
- Preserve all declared fits and failed estimates, state-deletion failures,
  applicability unknowns, UI uncertainty and partial-window labels.
- Migration checks compare the entire result ledgers at absolute tolerance
  1e-10 and independently reproduce primary wage WLS and earnings OLS.
- Use the standalone runner. Default execution reads local sources and must
  not download data or call the full-window acquisition path.
- Write retained artifacts only after validation, with the destination map
  and code/input/output provenance owned by `12_publish.R`.
