+++
title = "H-2A AEWR coding-agent guide"
description = "A source-ranked, executable, freshness-checked map of the research codebase."
sort_by = "weight"
template = "index.html"
page_template = "page.html"
+++

This guide is an on-demand grounding layer, not a second repository truth. The
small `AGENTS.md` hierarchy tells an agent how to behave; these pages explain
why, connect claims to sources, and expose generated facts. The manifest and
assumption checks make silent drift visible. Implementation excerpts are
generated from stable source selectors; each fence carries source and excerpt
hashes and is blocked on unreviewed excerpt drift. The generated page and index
report the current count rather than copying it into curated prose.

> [!IMPORTANT]
> Run `python scripts/agent_grounding.py verify` before relying on a generated
> page. A failure means the repository changed after the projection was built.

Begin with the [operating protocol](@/operating-protocol.md), then open the
nearest design or stage contract named by
`python scripts/agent_grounding.py snapshot --scope <path>`.
Use [source-linked code grounding](@/generated/code-grounding.md) when the
implemented contract matters more than a prose summary.
