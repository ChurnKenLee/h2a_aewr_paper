+++
title = "Operating protocol"
description = "The evidence sequence an agent follows before editing, testing, or interpreting the project."
weight = 1
+++

## Before planning

1. Run `python scripts/agent_grounding.py snapshot --scope <target>`.
2. Read every active `AGENTS.md` named in the snapshot.
3. Run `python scripts/agent_grounding.py verify`.
4. Open the named stage/design page and its canonical README/code contract.
   Inspect the relevant entries in [source-linked code grounding](@/generated/code-grounding.md); do not substitute a remembered implementation.
5. Check whether data, credentials, restored environments, network, GPU, and
   sufficient resources actually exist.
6. State the intended artifact, its owner, downstream consumers, and the
   narrowest useful validation.

## Evidence labels

Use precise completion language:

| Label | What it establishes | What it does not establish |
|---|---|---|
| Static | Files, paths, syntax, locks, and declared contracts are internally consistent | Code imports or executes on project data |
| Dry run | Runner ordering and targets resolve | Inputs exist or computations succeed |
| Artifact check | A named existing/generated artifact passes schema or numerical checks | Upstream acquisition is reproducible |
| Branch run | One supported branch executed with stated inputs | Other branches or the full pipeline pass |
| Full run | `run_all.sh` completed with named environment and data snapshot | Causal assumptions are valid |
| Interpretation audit | Estimates, support, diagnostics, and manuscript claims were reconciled | The design is causally identified in nature |

## When sources conflict

Do not average conflicting claims. Apply the [authority map](@/architecture/authority-map.md), quote the concrete conflict in the work log, and update the lowest-authority stale source or reclassify it. If executable sources disagree, stop and resolve the contract rather than choosing the favorable result.

## After editing

- Run narrow syntax/tests while iterating.
- Run `./scripts/run_tests.sh` for code, contracts, runners, AGENTS, or agent-doc changes.
- Regenerate retained empirical outputs only from the owning runner.
- Inspect numerical and diagnostic changes, not just exit codes.
- Regenerate grounding when watched inputs change.
- If a registered excerpt changed, review the old/new `accept-snippet-drift` diff before accepting its new digest. Ordinary generation must remain blocked until then.
- Report skipped checks and missing prerequisites explicitly.
