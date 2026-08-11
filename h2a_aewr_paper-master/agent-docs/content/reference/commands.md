+++
title = "Command reference"
description = "Grounding, documentation, validation, dry-run, and branch entry points."
weight = 1
+++

| Intent | Command |
|---|---|
| Scope context | `python scripts/agent_grounding.py snapshot --scope <path>` |
| Search agent docs | `python scripts/agent_grounding.py query <terms>` |
| Verify freshness | `python scripts/agent_grounding.py verify` |
| Regenerate projections | `python scripts/agent_grounding.py generate` |
| Review one changed excerpt | `python scripts/agent_grounding.py accept-snippet-drift --id <id>` |
| Accept reviewed excerpt digest | `python scripts/agent_grounding.py accept-snippet-drift --id <id> --write` |
| Fast repository gate | `./scripts/run_tests.sh` |
| Pipeline order only | `DRY_RUN=1 ./scripts/run_all.sh` |
| Zola internal build/link check | `zola --root agent-docs check --skip-external-links` |
| Zola strict external link check | `zola --root agent-docs check` |
| Serve guide | `zola --root agent-docs serve` |
| Shared panel | `./scripts/run_shared_panel.sh` |
| DiD | `./scripts/run_did.sh` |
| Panel IV | `./scripts/run_panel_iv.sh` |
| Mundlak–Chamberlain | `./scripts/run_mundlak_chamberlain.sh` |

Run data-bearing commands only after checking prerequisites and side effects.
