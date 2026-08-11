+++
title = "Declared assumptions"
description = "Machine-checked, scope-aware invariants that coding agents must surface before changes."

[extra]
scopes = ["agent/assumptions.toml", "AGENTS.md", "code", "scripts"]
+++

High-risk repository and empirical invariants are declared in
`agent/assumptions.toml`. Each record has an owner, risk, source, review
trigger, affected scopes, and one or more executable text checks.

The checks are deliberately modest. They catch removal or accidental drift of
an explicitly declared condition; they do not prove identification or
statistical validity. The linked canonical design page and executable code
remain the sources needed for substantive review.

A scoped context snapshot includes only the declarations relevant to the
target path. The complete registry and its checks are also copied into
`static/grounding-manifest.json` and summarized in `static/llms.txt` after
successful generation.

Change a declaration when its stated invariant, owner, source, affected scope,
or review trigger changes. Do not weaken a check merely to make verification
pass; decide whether the implementation or the declaration is wrong first.
