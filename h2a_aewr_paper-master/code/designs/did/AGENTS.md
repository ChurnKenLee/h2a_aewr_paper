# Difference-in-differences instructions

- This branch consumes the shared panel and owns `did_county_year_panel.parquet` plus its retained tables and coefficient plot.
- Preserve the continuous baseline treatment definition, the post-2011 timing, direct county/year fixed effects, and CZ-by-AEWR-region clustered inference unless the requested change explicitly revises the estimand.
- Regression families retain a four-column layout: full/no-border samples, each without/with controls, with a common sample where documented.
- The event-study reference and 2011 slope handling are substantive. Do not change coefficient indexing or reference periods as a formatting refactor.
- Run the branch runner in dry-run mode for ordering, parse every R file, and compare regenerated estimates/tables when data are available. Do not infer estimate stability from a successful formula parse.
