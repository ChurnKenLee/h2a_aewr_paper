# Canonical documentation instructions

- These Zola pages are written primarily to ground coding agents, while
  remaining readable to researchers. State authority, assumptions, caveats,
  inputs, outputs, and operational consequences explicitly.
- Put commands and directory-local ownership in READMEs. Put cross-cutting
  contracts and research-design semantics here.
- Place each `grounding` shortcode next to the claim supported by its narrow,
  named source region. Literal examples are illustrative, not authoritative.
- Never accept source drift by changing only a digest. Update the page, inspect
  the old/new excerpt with `agent_docs.py accept-drift`, then use `--write`.
- Keep high-risk, scope-aware invariants in `agent/assumptions.toml`; explain
  their substantive meaning and limitations in the relevant canonical page.
- Do not edit `static/grounding-manifest.json` or `static/llms.txt` directly.
  They are generated projections for LLM consumption.
- Use Zola `@/` links for internal pages. Leave
  `python scripts/agent_docs.py verify`, `python scripts/test_agent_docs.py`,
  `zola check --skip-external-links`, and `zola build` green.
