# B01 derived-stage instructions

- Derived artifacts are reusable inputs, not design-specific results. Keep outcome construction and predictors independent of the downstream identification strategy.
- Preserve documented producer order within the ACS, wage, price-index, and H-2A prediction families.
- The current county price/output index is the NASS/CDL chained-Fisher implementation in `02_price_index_nass_synthetic_cdl.py`; the Word note proposing Nielsen/FAF gravity recovery is historical.
- H-2A PPML models and scores must carry compatible cutoff and model-spec metadata. Never combine static and dynamic specifications or infer a missing cutoff from filenames alone.
- Preserve training/scoring separation: fitting creates stamped cutoff-specific model artifacts; scoring discovers compatible completed models and emits one static score per county and cutoff.
- A model-code change requires targeted numerical checks, metadata validation, and downstream panel-impact review; syntax and successful serialization are insufficient.
