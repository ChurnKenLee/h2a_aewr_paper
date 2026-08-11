# Agent-documentation instructions

- The Zola source is both human-browsable and agent-readable. Keep pages useful as plain Markdown; do not require rendered HTML or client-side JavaScript to recover a contract.
- Curated pages explain authority, intent, caveats, and workflows. Generated pages report inventories, runner facts, locks, assumption checks, and drift. Never copy generated facts into curated prose unless the statement needs human interpretation.
- Do not edit `content/generated/*`, `static/grounding-manifest.json`, or `static/llms.txt` directly. Change their source or generator, run `python scripts/agent_grounding.py generate`, then verify.
- Do not edit `content/generated/code-grounding.md` or `static/grounding-snippets.json`. Authoritative fences come only from `snippets.toml`. A changed excerpt digest blocks both verify and ordinary generation until `accept-snippet-drift --id <id>` displays the old/new diff and a reviewer reruns it with `--write`.
- Every literal fence in curated grounding Markdown needs a language and an immediately preceding `grounding-fence` classification comment. Such fences are non-authoritative; use registered source snippets extensively for implementation contracts.
- Every current factual claim should name a source path, assumption ID, or executable verification. Mark proposed, historical, superseded, and unverified material explicitly.
- Use Zola `@/` internal links so broken links fail the site check. Keep external links primary and stable; quick CI skips external fetches, while scheduled CI checks them.
- A code, runner, lockfile, assumption, AGENTS, or curated-doc change must leave `python scripts/agent_grounding.py verify` and `zola --root agent-docs check --skip-external-links` green.
