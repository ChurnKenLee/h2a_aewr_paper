# Coding-agent knowledge base

This is a dependency-free Zola site whose Markdown remains useful without the
rendered site. It keeps automatically discoverable `AGENTS.md` files compact
while providing deep, searchable grounding on demand.

Inside `devenv shell`:

<!-- grounding-fence: illustrative -->
```sh
agent-docs-generate
agent-docs-check
agent-docs-serve
```

Equivalent direct commands:

<!-- grounding-fence: illustrative -->
```sh
python scripts/agent_grounding.py generate
python scripts/agent_grounding.py verify
zola --root agent-docs check --skip-external-links
zola --root agent-docs serve
```

Quick CI skips external HTTP checks so transient provider failures cannot block
code changes. Scheduled CI runs `zola check` without that flag and detects
external-link drift.

The generated manifest hashes the repository inputs that agents rely on.
Changing any watched file makes verification fail until generated grounding is
recomputed and reviewed.

Authoritative code fences are registered in `snippets.toml` and rendered into
`content/generated/code-grounding.md`. Each record uses a symbol or exact-line
boundary selector and stores a reviewed excerpt SHA-256. `verify` independently
re-extracts all snippets and syntax-checks them. `generate` refuses excerpt
drift; review it with `python scripts/agent_grounding.py accept-snippet-drift
--id <id>` and add `--write` only after approving the displayed diff.
