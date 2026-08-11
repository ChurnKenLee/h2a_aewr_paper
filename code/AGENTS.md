# Code-wide instructions

- Start with the stage README and the canonical Zola pages selected by
  `python scripts/agent_docs.py snapshot --scope <target-path>`.
- Confirm the owning producer and downstream consumers before changing a
  persistent artifact. Repair schemas at their owner, not at consumers.
- Persistent county identifiers use the canonical string representation.
  Source-specific identifiers may exist only while reading raw inputs.
- R entry points source `code/paths.R`; Python entry points resolve the
  repository root explicitly. Do not introduce working-directory-dependent
  paths.
- Fail on missing required columns, empty outputs, duplicate declared keys,
  incompatible model metadata, or geographic drift.
- Never import design-specific treatments, instruments, samples, or estimators
  into the source, derived, shared-helper, cleaning, or shared-build stages.
- For a changed script, validate syntax, inspect every input and output path,
  run the narrowest artifact check available, and dry-run its owning runner.
