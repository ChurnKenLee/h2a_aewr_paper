# Mundlak–Chamberlain instructions

- Version 4 is the supported Python runner path. The R version-2.3/version-3
  estimator files are retained historical compatibility code unless explicitly
  targeted.
- Preserve the declared linear-history registry, causal-first common full-rank
  basis, pooled rich-projection candidate, county-FE sensitivities, common
  six-outcome sample, exact full-model leverage, named gradients,
  input/code/sample hashes, and resource guards.
- The exhaustive registry is not the default queue. Do not trigger it without
  an explicit resource decision.
- The scalar continuous CCV-HC3 mixture is experimental and not Lean-proved
  exact. Never omit that status or substitute partial-regressor leverage for
  full-model HC3 leverage.
- Keep all named full-history dose coordinates separate; automatic
  collinearity dropping cannot choose the causal basis, and QR may not reduce
  the history dimension.
- Do not reintroduce polynomials, imposed trends, randomization inference, a
  bootstrap, pandas, or pickle into the supported version-4 path.
- Keep support tables, lead placebos, out-of-range diagnostics, model warnings,
  and selected-primary logic visible even when results are unfavorable.
