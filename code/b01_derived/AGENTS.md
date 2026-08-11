# Derived-stage instructions

- Derived artifacts are reusable inputs, not design-specific results. Keep
  outcome construction and predictors independent of identification strategy.
- Preserve producer order within ACS, wage, price-index, and H-2A prediction
  families.
- The supported county price/output index is the NASS/CDL chained-Fisher
  implementation. Older Nielsen/FAF proposal documents are historical.
- H-2A PPML models and scores carry compatible cutoff and model-spec metadata.
  Never combine static and dynamic models or infer metadata from filenames.
- Preserve training/scoring separation: fitting creates cutoff-stamped models;
  scoring emits one static score per county and cutoff.
- Model-code changes require numerical and downstream panel-impact checks;
  successful parsing and serialization are not sufficient evidence.
