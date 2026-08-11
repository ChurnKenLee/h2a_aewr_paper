# Shared-helper instructions

- This directory is design-neutral. It may normalize shared geographic or source definitions but must not encode treatment, post periods, target clusters, instruments, estimators, or design samples.
- A helper change has a wide blast radius. Search all callers in `c01_clean`, `c02_build`, `descriptives`, and every design branch; validate both direct unit behavior and persistent artifact schemas.
- Preserve the 2010 county-vintage mapping and string-identifier contract exactly unless the project contract is explicitly revised.
