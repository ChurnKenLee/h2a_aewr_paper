# Runner and validation instructions

- Runner order is a public contract. Every `run_step` target must exist, be documented, and own the artifact attributed to it.
- Preserve `set -euo pipefail`, repository-root execution, UTC/default locale stabilization, isolated temporary directories, and cleanup traps.
- Keep `DRY_RUN=1` free of R/Python imports, network calls, credential requirements, and data reads so it remains a reliable ordering test.
- Python Marimo detection/export and ordinary-Python execution are intentionally different paths. Test both when changing `pipeline_helpers.sh`.
- A new supported stage requires a runner entry, README contract, nested AGENTS coverage if semantics differ, assumption/source updates, and regenerated agent docs.
- `run_tests.sh` is the fast static/freshness gate, not evidence that empirical stages completed. Keep strict-tooling behavior suitable for `devenv test` and CI.
- `agent_grounding.py` must independently re-extract every registered code snippet, enforce its reviewed digest, reject unclassified grounding fences, and refuse ordinary generation on excerpt drift. Keep drift acceptance an explicit review-only subcommand.
