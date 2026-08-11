# Shared-helper instructions

- This directory is design-neutral. It may normalize shared geographic or
  source definitions but cannot encode treatments, post periods, target
  clusters, instruments, estimators, or design samples.
- Helper changes have a wide blast radius. Search callers in `c01_clean`,
  `c02_build`, `descriptives`, and every design branch, then validate direct
  behavior and persistent schemas.
- Preserve the 2010 county-vintage mapping and string-identifier contract
  unless the canonical geographic contract is explicitly revised.
