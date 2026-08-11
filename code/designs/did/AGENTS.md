# Difference-in-differences instructions

- This branch consumes the shared panel and owns its design panel, retained
  tables, and coefficient plot.
- Preserve the continuous baseline treatment, post-2011 timing, direct county
  and year fixed effects, and CZ-by-AEWR-region clustered inference unless the
  requested change explicitly revises the estimand.
- Regression families retain four columns: full and no-border samples, each
  without and with controls, using common samples where documented.
- Event-study reference and 2011-slope handling are substantive. Do not change
  coefficient indexing or reference periods as formatting work.
- Dry-run the branch runner, parse every changed R file, and compare regenerated
  estimates when data are available.
