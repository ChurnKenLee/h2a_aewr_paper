# Runner and agent-documentation instructions

- Runner order is a public contract. Every `run_step` target must exist, be
  documented, and own its attributed artifact.
- Preserve `set -euo pipefail`, repository-root execution, deterministic locale
  and time behavior, isolated temporary directories, and cleanup traps.
- Keep `DRY_RUN=1` free of R/Python imports, network calls, credentials, and
  data reads.
- Marimo export and ordinary Python execution are separate paths. Test both
  when changing `pipeline_helpers.sh`.
- `agent_docs.py` must independently extract every Zola grounding region,
  enforce reviewed digests, expose scoped context, and require a prose change
  before accepting source drift.
- Generated agent context is a static/freshness check, not evidence that an
  empirical stage completed.
