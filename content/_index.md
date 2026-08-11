+++
title = "H-2A Paper Documentation"
template = "index.html"
+++

This site is the canonical technical reference and coding-agent knowledge base
for the H-2A empirical pipeline. It documents contracts and research designs
whose claims are checked against named regions of the implementation during
every Zola build.

- [Pipeline architecture](@/architecture/pipeline.md)
- [Artifact ownership](@/architecture/artifact-ownership.md)
- [Generating the shared county-year panel](@/architecture/shared-panel-generation.md)
- [Maintaining grounded documentation](@/contracts/grounding.md)
- [Geographic code contract](@/contracts/geographic-codes.md)
- [Shared county-year panel](@/contracts/shared-panel.md)
- [Causal outcomes and model inputs](@/contracts/causal-outcomes-and-model-inputs.md)
- [H-2A prediction contract](@/contracts/prediction-model.md)
- [Difference-in-differences](@/designs/difference-in-differences.md)
- [Panel IV](@/designs/panel-iv.md)
- [Mundlak–Chamberlain](@/designs/mundlak-chamberlain.md)
- [Coding-agent workflow](@/reference/agent-workflow.md)
- [Authority and evidence](@/reference/authority.md)
- [Declared assumptions](@/reference/assumptions.md)

Repository READMEs remain the operational entry points for commands, script
order, and produced artifacts.

Coding agents should begin with a scoped context snapshot. Tools consuming the
rendered site can discover its canonical pages through [`/llms.txt`](/llms.txt)
and load reviewed excerpts and instruction files from
[`/grounding-manifest.json`](/grounding-manifest.json).
