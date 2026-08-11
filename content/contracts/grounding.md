+++
title = "Maintaining grounded documentation"
description = "How to add, review, and refresh checked code references."

[extra]
scopes = ["AGENTS.md", "content", "scripts/agent_docs.py", "templates"]
+++

A grounding reference records that a documentation claim was reviewed against
a named region of source code. Zola extracts that region, calculates its
SHA-256 digest, and fails the build when the recorded digest is stale.

## Add a code region

Use a stable, unique name and the repository's comment syntax:

```r
# docs-ground:start example-contract
example_setting <- "implemented value"
# docs-ground:end example-contract
```

Keep regions narrow enough that every included change could affect the nearby
documentation. Do not use line numbers: they change when unrelated lines move.

## Reference it from a page

Place the reference next to the claim it supports:

```text
{{/* grounding(path="code/example.R", anchor="example-contract", sha256="REPLACE_WITH_CURRENT_DIGEST") */}}
```

Run Zola first. For a new reference it fails with the current digest; review
the region and insert that digest into the page. Then generate the
machine-readable context and run all checks:

```sh
python scripts/agent_docs.py generate
python scripts/agent_docs.py verify
zola check --skip-external-links
```

The first run fails and reports the current digest. Review the code and prose,
replace the placeholder with the reported digest, and run the check again.

## Respond to a stale-documentation failure

When an anchored code region changes:

1. Preview the reviewed and current source excerpts:

   ```sh
   python scripts/agent_docs.py accept-drift \
     --document content/path/to/page.md --anchor anchor-name
   ```

2. Update the page's explanation, assumptions, inputs, outputs, or
   interpretation. This review is required even when the existing claim still
   holds; record why it remains accurate.
3. Accept the reviewed excerpt and regenerate agent context:

   ```sh
   python scripts/agent_docs.py accept-drift \
     --document content/path/to/page.md --anchor anchor-name --write
   ```

   The command refuses to update the digest if the page's prose and context are
   unchanged since the previous review.
4. Run `python scripts/agent_docs.py verify`,
   `zola check --skip-external-links`, and `zola build` again.

Never refresh digests automatically in CI. Changing the digest is the explicit
record that the documentation was reviewed. A matching digest proves that the
linked code has not changed since review; it does not independently prove that
the prose is correct.

## Context for coding agents

Before editing, an agent runs:

```sh
python scripts/agent_docs.py snapshot --scope code/or/file/to/change
```

The snapshot names the active `AGENTS.md` chain, nearest operational READMEs,
canonical pages related to that scope, and direct source-to-page dependencies.
The generated `static/grounding-manifest.json` exposes the same reviewed
excerpts and all scoped agent instructions as structured data. `static/llms.txt`
provides a compact discovery index for tools consuming the rendered site.
