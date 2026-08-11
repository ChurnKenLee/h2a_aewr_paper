+++
title = "Coding-agent workflow"
description = "Scoped discovery, reviewed drift acceptance, and machine-readable context."

[extra]
scopes = ["AGENTS.md", "content", "scripts/agent_docs.py", "static"]
+++

## Start from a scoped snapshot

Run the snapshot before planning a change:

```sh
python scripts/agent_docs.py snapshot --scope code/designs/panel_iv
```

It reports the applicable instruction chain, nearest operational READMEs,
canonical pages selected through their declared scopes or direct grounding
links, high-risk assumptions declared for that scope, and source regions
connected to the target.

{{ grounding(path="scripts/agent_docs.py", anchor="agent-context-snapshot", sha256="a3e844a66bf429ed4b6523c59fe512b3f110a0250151e23bb76fc5375f0a0524") }}

## Verify before trusting generated context

`verify` independently extracts every referenced region and checks its digest.
It also rejects stale `static/grounding-manifest.json` and `static/llms.txt`.
The same gate evaluates the declared assumption registry. `generate` refuses
to rewrite those projections while a source reference or assumption check is
invalid.

{{ grounding(path="scripts/agent_docs.py", anchor="agent-docs-verification", sha256="d39d838d53eff3dc8245bb8bbf8dc01e47702df8b7abecc276cdc054397bd00d") }}

## Review source drift explicitly

`accept-drift` displays a unified diff between the previously reviewed excerpt
and current source. Its `--write` mode updates the page digest only after that
page's normalized prose changed, then regenerates context when all references
and declared assumptions are valid.

{{ grounding(path="scripts/agent_docs.py", anchor="agent-drift-review", sha256="b72791c27ccab838bc10d39c8ab4c45be09c4709d80db8ebb8b53ac993b72293") }}

This is intentionally stricter than copying the new digest from a failed Zola
build. The documentation edit is the review record; the digest merely locks the
reviewed source state.
