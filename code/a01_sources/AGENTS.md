# Source-stage instructions

- Each producer owns one external source family or one explicit
  geocoding/linkage pass. Preserve raw audit fields before publishing canonical
  fields.
- Network calls can consume money or quotas and can return revised schemas.
  Inspect caches first; do not refresh a source family merely to test code.
- Keep downloads resumable and deterministic. Document provider, coverage,
  access method, credentials, cache, and output in this stage's README.
- H-2A location processing intentionally runs `02_01`, `02_02`, `02_03`, then
  `02_01` again before employer matching. The second pass is not a duplicate.
- Scripts `10` through `12` are optional refreshes and stay outside
  `run_all.sh` unless explicitly invoked.
- Prefer fixtures or cached partitions. Before a live request, identify the
  endpoint, expected request count, retry policy, and overwrite behavior.
