# Shared descriptives

These figures depend only on source artifacts and the shared county panel:

| Script | Output |
| --- | --- |
| `01_h2a_workers.R` | `outputs/figures/fig_line_ts_h2a_workers_certified.png` |
| `02_aewr_p25_map.R` | `outputs/figures/map_aewr_cz_p25_change_from_trend_2022_2008.png` |

Run:

```sh
./scripts/run_descriptives.sh
```

`panel_iv_intuition.py` is an interactive Marimo notebook that is not part of
the runner and writes nothing. It reads the panel-IV county-year panel, the
wage-reform case file and the AEWR and FLS region tables, and shows the
region-year AEWR variation, the cluster-year first stage, region-year reduced
forms, the concentration of H-2A use, whether the AEWR binds, and regional case
studies. Open it with:

```sh
uv run --no-sync marimo edit code/descriptives/panel_iv_intuition.py
```
