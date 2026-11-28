# 2025 policy design instructions

- Read `README.md`, `design.json`, and the canonical policy-2025 page before
  editing. All work here is after outcome inspection.
- The baseline is the complete FY2026 H-2A disclosure (Q4 release, user
  instruction of November 21, 2026): full-fiscal-year wage window and an
  exposure table rebuilt from that vintage. A further refresh or specification
  change needs a dated record under `history/`; never update hashes just to
  make checks pass.
- The current owner includes wage offers and covered earnings only. Do not
  add the 2013 shutdown or import another design's configuration implicitly.
- Preserve all declared fits and failed estimates, state-deletion failures,
  applicability unknowns, UI uncertainty and partial-window labels.
- Validation compares the entire result ledgers with the current vintage's
  reference ledgers at absolute tolerance 1e-10 and independently reproduces
  primary wage WLS and earnings OLS.
- Use the standalone runner. Default execution reads local sources and must
  not download data or call the full-window acquisition path.
- Write retained artifacts only after validation, with the destination map
  and code/input/output provenance owned by `12_publish.R`.
