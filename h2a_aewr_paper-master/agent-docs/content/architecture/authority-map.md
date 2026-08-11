+++
title = "Authority map"
description = "Which source wins when code, prose, outputs, and historical notes disagree."
weight = 1
+++

## Ranking

1. Executable code, supported runners, lockfiles, and machine-checked contracts.
2. Generated agent pages after verification passes.
3. Curated agent pages and the nearest code-directory README.
4. Root README and retained output artifacts.
5. Draft prose, `markdowns/`, papers, PDFs, and Word notes.

This is not a blanket claim that code is scientifically correct. It is a rule
for identifying what the repository currently implements. Identification,
measurement, and interpretation remain subject to their substantive audits.

## Status vocabulary

- **Active:** implemented in a supported runner and guarded by current checks.
- **Compatibility:** retained executable code used as a benchmark or migration record.
- **Proposed:** a design or source that has not become a supported runner.
- **Historical:** dated reasoning preserved for provenance but superseded as implementation guidance.
- **Generated:** a deterministic projection; valid only when freshness verification passes.
- **Unverified:** a claim with no current executable check or reconciled source.

## Anti-staleness rule

Dynamic facts belong in `agent-docs/assumptions.toml` or generated pages, not in
root instructions. When a constant, runner step, dependency, or file inventory
changes, verification fails until the projection and assumption statements are
reviewed together.

Executable contracts that benefit from direct inspection belong in
`agent-docs/snippets.toml`. The [rendered excerpts](@/generated/code-grounding.md)
are source-derived, carry source and excerpt hashes, and cannot be refreshed
past changed code without an explicit digest-acceptance step.
