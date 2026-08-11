# A01 source-stage instructions

- Every file owns one external source family or one explicit geocoding/linkage pass. Preserve raw fields needed for audit before publishing canonical fields.
- Assume network calls can cost money, consume quotas, change upstream state, or return revised schemas. Inspect caches and manifests first; never refresh a whole family merely to test a code edit.
- Keep downloads resumable and deterministic. Record provider, coverage, URL/API method, access requirements, and output in `documentation/raw_data_sources.yaml` and the stage README.
- H-2A location processing intentionally runs `02_01`, `02_02`, `02_03`, then `02_01` again before employer matching. Do not remove the second pass as an apparent duplicate.
- `10` through `12` are optional source refreshes and are excluded from `run_all.sh` unless explicitly invoked.
- Use fixtures or cached small partitions for tests. Before a live call, identify the credential, endpoint, expected request count, cache target, retry policy, and overwrite behavior.
