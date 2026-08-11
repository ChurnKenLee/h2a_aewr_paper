+++
title = "Authority and evidence"
description = "How an agent resolves conflicting claims and reports validation."

[extra]
scopes = ["AGENTS.md", "content", "draft", "outputs"]
+++

Use sources in this order when they disagree:

1. Executable code, pipeline runners, environment locks, and machine checks.
2. `static/grounding-manifest.json`, only when `agent_docs.py verify` passes.
3. Canonical Zola pages and the nearest directory README.
4. The root README.
5. Historical proposals, manuscript prose, and retained output artifacts.

A source-linked excerpt establishes freshness, not truth. The surrounding page
must still state assumptions, units, samples, inference, limitations, and the
effect of the implementation on downstream consumers.

Validation evidence must distinguish static checks from empirical execution.
A parser, dry run, or successful Zola build does not show that a model ran or
that its numerical results remained stable. Agents should report exactly which
checks ran, which data were available, and which empirical checks were skipped.
