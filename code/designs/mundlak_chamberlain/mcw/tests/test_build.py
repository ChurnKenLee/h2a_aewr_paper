"""Synthetic contract tests for :mod:`mcw.build`."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_PARENT))

from mcw.build import (
    BASELINE_IMPUTATION_LABEL,
    BITE_APPROXIMATION_LABEL,
    CLUSTER_COLUMNS,
    FRACTION_AFFECTED_COLUMNS,
    ONE_LAG_BENCHMARK_COLUMNS,
    OUTCOME_COLUMNS,
    PanelBuildError,
    build_mcw_panel,
    full_history_columns,
    one_lag_columns,
    treatment_coordinate_column,
)


def _synthetic_panel() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    counties = (
        ("01001", "01", "001", "01", 10.0),
        ("02013", "02", "007", "02", 20.0),
    )
    real_growth = {
        2008: 0.90,
        2009: 0.95,
        2010: 1.0,
        2011: 1.1,
        2012: 1.21,
        2013: 1.331,
    }
    for county, state, cz, region, base_aewr in counties:
        county_wage_offset = (base_aewr - 10.0) / 10.0
        for year, growth in real_growth.items():
            wage_shift = 2.0 * float(year - 2008)
            rows.append(
                {
                    "county_fips": county,
                    "state_fips": state,
                    "cz_id": cz,
                    "aewr_region_id": region,
                    "year": year,
                    "aewr": base_aewr * growth,
                    "aewr_ppi": base_aewr * growth,
                    "prevailing_ag_min_wage": 7.0,
                    "wage_p10": 5.0 + wage_shift + county_wage_offset,
                    "wage_p25": 6.0 + wage_shift + county_wage_offset,
                    "wage_p50": 8.0 + wage_shift + county_wage_offset,
                    "wage_p75": 10.0 + wage_shift + county_wage_offset,
                    "wage_p90": 12.0 + wage_shift + county_wage_offset,
                    "nbr_applications_start_year": (
                        0.5 if year == 2013 else base_aewr / 100.0
                    ),
                    "nbr_employers_balanced_start_year": 2.0,
                    "nbr_workers_requested_start_year": 3.0,
                    "nbr_workers_certified_start_year": base_aewr / 5.0,
                    "man_hours_certified_start_year": 5.0,
                    "emp_farm": 100.0 + float(year - 2008),
                    "ln_pop_census": (
                        10.0 + base_aewr / 100.0 + 0.01 * float(year - 2008)
                    ),
                    "farm_emp_share": base_aewr / 100.0,
                    "emp_pop_ratio": base_aewr / 200.0,
                    "share_farm_crop_cashandinc": base_aewr / 50.0,
                    "share_farm_laborexp_prodexp": base_aewr / 80.0,
                    "share_farm_animal_cashandinc": base_aewr / 70.0,
                    "share_farm_prodexp_cashandinc": base_aewr / 60.0,
                    "census_cropland_2007": base_aewr * 1000.0,
                }
            )
    return pl.DataFrame(rows)


class BuildPanelTests(unittest.TestCase):
    def _build(self, frame: pl.DataFrame) -> pl.DataFrame:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.parquet"
            frame.write_parquet(source)
            return build_mcw_panel(
                source,
                history_years=(2011, 2012, 2013),
                analysis_years=(2012, 2013),
            )

    def test_paths_bites_outcomes_and_string_clusters(self) -> None:
        panel = self._build(_synthetic_panel())
        self.assertEqual(panel.height, 4)
        self.assertEqual(
            full_history_columns(2013),
            (
                "mc_aewr_log_level_2011",
                "mc_aewr_log_level_2012",
                "mc_aewr_log_level_2013",
            ),
        )
        self.assertEqual(
            one_lag_columns(2013),
            ("mc_aewr_log_level_2012", "mc_aewr_log_level_2013"),
        )
        self.assertEqual(one_lag_columns(2011), ("mc_aewr_log_level_2011",))

        row = panel.filter(
            (pl.col("county_fips") == "01001") & (pl.col("year") == 2013)
        ).row(0, named=True)
        self.assertAlmostEqual(row["mc_aewr_log_level_2011"], 100.0 * math.log(1.1))
        self.assertAlmostEqual(row["mc_aewr_log_level_2012"], 100.0 * math.log(1.21))
        self.assertAlmostEqual(row["mc_aewr_dollar_level_2013"], 3.31)
        self.assertAlmostEqual(row["mc_aewr_log_change_2013"], 100.0 * math.log(1.1))
        self.assertAlmostEqual(
            row[ONE_LAG_BENCHMARK_COLUMNS[0]], row["mc_aewr_log_level_2013"]
        )
        self.assertAlmostEqual(
            row[ONE_LAG_BENCHMARK_COLUMNS[1]], row["mc_aewr_log_level_2012"]
        )

        self.assertAlmostEqual(row["mc_bite_f0809_2011"], 2.0)
        self.assertEqual(row["mc_baseline_bite_missing"], 0)
        self.assertAlmostEqual(row["mc_baseline_bite"], 1.5)
        self.assertAlmostEqual(row["mc_baseline_bite_imputed"], 1.5)
        self.assertAlmostEqual(row[FRACTION_AFFECTED_COLUMNS["f0809"]], 0.5)
        self.assertAlmostEqual(
            row["mc_exposure_log_f0809_2012"],
            0.5 * row["mc_aewr_log_level_2012"],
        )
        self.assertEqual(row["mc_baseline_year_count_f0809"], 2)
        self.assertEqual(row["mc_baseline_year_count_f0810"], 3)
        self.assertEqual(row["mc_baseline_farm_employment"], 101.0)
        self.assertEqual(row["mc_baseline_farm_employment_year_count"], 3)
        self.assertAlmostEqual(row["mc_baseline_farm_employment_share"], 0.1)
        self.assertTrue(math.isfinite(row["mc_baseline_bite_z"]))
        self.assertEqual(row["mc_bite_approximation_method"], BITE_APPROXIMATION_LABEL)

        self.assertEqual(row[OUTCOME_COLUMNS["applications"]], 0.5)
        self.assertEqual(row[OUTCOME_COLUMNS["employers"]], 2.0)
        self.assertEqual(row[OUTCOME_COLUMNS["any_application"]], 1)
        self.assertEqual(row[CLUSTER_COLUMNS["county"]], "01001")
        self.assertEqual(row[CLUSTER_COLUMNS["state"]], "01")
        self.assertEqual(row[CLUSTER_COLUMNS["region"]], "01")
        self.assertEqual(row[CLUSTER_COLUMNS["cz_region"]], "001::01")
        for column in ("county_fips", "state_fips", "cz_id", "aewr_region_id"):
            self.assertEqual(panel.schema[column], pl.String)
        for column in CLUSTER_COLUMNS.values():
            self.assertEqual(panel.schema[column], pl.String)

    def test_duplicate_county_year_fails_before_construction(self) -> None:
        source = _synthetic_panel()
        duplicate = pl.concat([source, source.head(1)])
        with self.assertRaisesRegex(PanelBuildError, "county-year keys are not unique"):
            self._build(duplicate)

    def test_non_string_geography_is_rejected_without_lossy_cast(self) -> None:
        source = _synthetic_panel().with_columns(pl.col("state_fips").cast(pl.Int32))
        with self.assertRaisesRegex(
            PanelBuildError, "state_fips must be stored as a string"
        ):
            self._build(source)

    def test_baseline_imputation_is_labeled_and_uses_region_median(self) -> None:
        source = _synthetic_panel()
        donor = source.filter(pl.col("county_fips") == "01001").with_columns(
            pl.lit("01003").alias("county_fips"),
            pl.lit(15_000.0).alias("census_cropland_2007"),
        )
        source = pl.concat([source, donor]).with_columns(
            pl.when(pl.col("county_fips") == "01001")
            .then(None)
            .otherwise(pl.col("census_cropland_2007"))
            .alias("census_cropland_2007")
        )
        panel = self._build(source)
        row = panel.filter(
            (pl.col("county_fips") == "01001") & (pl.col("year") == 2013)
        ).row(0, named=True)
        self.assertIsNone(row["mc_baseline_cropland"])
        self.assertEqual(row["mc_baseline_cropland_missing"], 1)
        self.assertEqual(row["mc_baseline_cropland_imputed"], 15_000.0)
        self.assertTrue(math.isfinite(row["mc_baseline_cropland_z"]))
        self.assertEqual(
            row["mc_baseline_imputation_method"], BASELINE_IMPUTATION_LABEL
        )

    def test_inconsistent_region_year_aewr_fails(self) -> None:
        source = _synthetic_panel().with_columns(
            pl.when((pl.col("county_fips") == "01001") & (pl.col("year") == 2012))
            .then(pl.col("aewr_ppi") + 1.0)
            .otherwise(pl.col("aewr_ppi"))
            .alias("aewr_ppi")
        )
        extra = source.filter(
            (pl.col("county_fips") == "01001") & (pl.col("year") == 2012)
        ).with_columns(
            pl.lit("01003").alias("county_fips"),
            (pl.col("aewr_ppi") + 1.0).alias("aewr_ppi"),
        )
        source = pl.concat([source, extra])
        with self.assertRaisesRegex(
            PanelBuildError, "AEWR must be unique within region-year"
        ):
            self._build(source)

    def test_coordinate_helper_rejects_unknown_family(self) -> None:
        with self.assertRaisesRegex(PanelBuildError, "unknown treatment family"):
            treatment_coordinate_column("not_a_family", 2013)


if __name__ == "__main__":
    unittest.main()
