# Code-wide instructions

- Start with the stage README and the generated pipeline contract page. Confirm the owning producer and all downstream consumers before changing a persistent artifact.
- Persistent county identifiers must use the canonical string representation. Source-specific identifiers may exist only while reading raw input.
- R entry points source `code/paths.R`; Python entry points resolve the repository root explicitly. Do not introduce CWD-dependent relative paths.
- Fail on missing required columns, empty outputs, duplicate declared keys, incompatible model metadata, or geographic drift. Do not silently coerce a stale artifact into the new schema.
- Keep top-level scripts executable as runners expect. Marimo applications are flattened by `scripts/pipeline_helpers.sh`; ordinary Python programs are run directly.
- For a changed script, perform syntax validation, inspect every input/output path, run the narrowest available artifact check, and dry-run its owning runner.
- Never import a design-specific treatment, instrument, sample, or estimator into `a01_sources`, `b01_derived`, `c00_shared`, `c01_clean`, or `c02_build`.
