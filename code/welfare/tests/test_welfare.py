"""Synthetic contract and analytic-identity tests for code/welfare/envelope.py."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

WELFARE_DIR = Path(__file__).resolve().parents[1]
if str(WELFARE_DIR) not in sys.path:
    sys.path.insert(0, str(WELFARE_DIR))

from envelope import (
    add_coverage_adjustment,
    evaluate_bounds_grid,
    load_welfare_contract,
    scalar_envelope_metrics,
    sha256_file,
)


def _synthetic_welfare_panel() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for year in (2024, 2025):
        rows.extend(
            [
                {
                    "county_fips": "01001",
                    "state_fips": "01",
                    "aewr_region_id": "10",
                    "year": year,
                    "aewr_ppi": 12.0,
                    "aewr": 16.0,
                    "man_hours_certified_start_year": 1000.0,
                    "nbr_workers_certified_start_year": 10.0,
                    "cert_hours_with_hourly_wage_and_aewr_start_year": 1000.0,
                    "cert_hours_at_aewr_start_year": 900.0,
                    "farm_laborexpense_ppi": 100.0,
                    "farm_cashcrops_ppi": 500.0,
                },
                {
                    "county_fips": "01003",
                    "state_fips": "01",
                    "aewr_region_id": "10",
                    "year": year,
                    "aewr_ppi": 12.0,
                    "aewr": 16.0,
                    "man_hours_certified_start_year": 500.0,
                    "nbr_workers_certified_start_year": 5.0,
                    "cert_hours_with_hourly_wage_and_aewr_start_year": 0.0,
                    "cert_hours_at_aewr_start_year": 0.0,
                    "farm_laborexpense_ppi": 50.0,
                    "farm_cashcrops_ppi": 250.0,
                },
            ]
        )
    return pl.DataFrame(rows)


class WelfareEnvelopeTests(unittest.TestCase):
    def test_eta_zero_exact_first_order_transfer(self) -> None:
        for delta in (1.0, -3.5):
            for form in ("constant_elasticity", "constant_semi_elasticity"):
                res = scalar_envelope_metrics(15.0, 2000.0, delta, 0.0, form)
                self.assertAlmostEqual(
                    res["change_producer_surplus"], -2000.0 * delta, places=10
                )
                self.assertAlmostEqual(
                    res["second_order_behavioral_cushion"], 0.0, places=10
                )
                self.assertAlmostEqual(res["change_exposed_hours"], 0.0, places=10)

    def test_eta_minus_one_log_limit_and_zero_wage_bill_change(self) -> None:
        at_m1 = scalar_envelope_metrics(15.0, 1000.0, 1.0, -1.0, "constant_elasticity")
        lo = scalar_envelope_metrics(
            15.0, 1000.0, 1.0, -1.0 - 1e-6, "constant_elasticity"
        )
        hi = scalar_envelope_metrics(
            15.0, 1000.0, 1.0, -1.0 + 1e-6, "constant_elasticity"
        )
        expected = -1000.0 * 15.0 * math.log(16.0 / 15.0)
        self.assertAlmostEqual(at_m1["change_producer_surplus"], expected, places=9)
        self.assertAlmostEqual(
            at_m1["change_producer_surplus"], lo["change_producer_surplus"], places=3
        )
        self.assertAlmostEqual(
            at_m1["change_producer_surplus"], hi["change_producer_surplus"], places=3
        )
        self.assertAlmostEqual(at_m1["change_h2a_wage_bill"], 0.0, places=9)

    def test_convexity_nonnegative_cushion_for_increase_and_cut(self) -> None:
        for delta in (1.0, -3.0):
            for eta in (-0.3, -0.75, -1.0, -1.2):
                for form in ("constant_elasticity", "constant_semi_elasticity"):
                    res = scalar_envelope_metrics(16.0, 1000.0, delta, eta, form)
                    self.assertGreater(res["second_order_behavioral_cushion"], 0.0)
                    self.assertGreater(res["second_order_pct_of_abs_first_order"], 0.0)

    def test_monotonicity_and_semi_elasticity_closeness(self) -> None:
        etas = (-1.2, -1.0, -0.75, -0.3, 0.0)
        surpluses = [
            scalar_envelope_metrics(15.0, 1000.0, 1.0, eta, "constant_elasticity")[
                "change_producer_surplus"
            ]
            for eta in etas
        ]
        # More negative eta yields smaller producer loss (less negative Delta Pi).
        for i in range(len(surpluses) - 1):
            self.assertGreater(surpluses[i], surpluses[i + 1])

        ce = scalar_envelope_metrics(15.0, 1000.0, 1.0, -1.0, "constant_elasticity")
        se = scalar_envelope_metrics(
            15.0, 1000.0, 1.0, -1.0, "constant_semi_elasticity"
        )
        rel_gap = abs(
            ce["change_producer_surplus"] - se["change_producer_surplus"]
        ) / abs(ce["change_producer_surplus"])
        self.assertLess(rel_gap, 0.001)

    def test_coverage_adjustment_fallback_and_grid_evaluation(self) -> None:
        panel = add_coverage_adjustment(_synthetic_welfare_panel())
        # County 01003 has 0 comparable hours -> falls back to region-year share 0.9.
        c2 = panel.filter(
            (pl.col("county_fips") == "01003") & (pl.col("year") == 2024)
        ).row(0, named=True)
        self.assertAlmostEqual(c2["coverage_adjusted_binding_share"], 0.9, places=10)
        self.assertEqual(c2["coverage_adjustment_source"], "aewr_region_year_fallback")

        oct_panel = panel.filter(pl.col("year") == 2024).with_columns(
            pl.lit(16.0).alias("fls_2025_nominal_aewr"),
            pl.lit(-3.0).alias("oct2025_statutory_cut"),
        )
        contract = {
            "canonical_etas": (0.0, -0.3, -1.0),
            "parametric_exposure_shares": (0.85, 0.94, 1.0),
            "track1_ci95_low": -1.1,
            "track1_point_estimate": -0.2,
            "passthrough_estimate": 0.62,
            "real_aewr_column": "aewr_ppi",
            "nominal_aewr_column": "aewr",
            "counterfactual_delta": 1.0,
            "start_year": 2024,
            "complete_end_year": 2024,
            "provisional_end_year": 2025,
            "snapshot_year": 2024,
            "percent_shifts": (0.10, -0.10),
        }
        grid = evaluate_bounds_grid(panel, oct_panel, contract)
        row_0 = grid.filter(
            (pl.col("window") == "2024_snapshot_real")
            & (pl.col("counterfactual") == "plus_1_dollar")
            & (pl.col("exposure_method") == "coverage_adjusted")
            & (pl.col("passthrough_scenario") == "statutory_full_pi_100")
            & (pl.col("elasticity") == 0.0)
            & (pl.col("functional_form") == "constant_elasticity")
        ).row(0, named=True)
        # Total hours = 1500, coverage-adjusted exposed = 1350 -> Delta Pi(0) = -1350.
        self.assertAlmostEqual(
            row_0["avg_annual_change_producer_surplus_cert_space"], -1350.0, places=8
        )

    def test_contract_loader_fails_closed_on_pending_or_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_panel = root / "panel.parquet"
            p_t1 = root / "q.csv"
            p_w = root / "w.csv"
            p_s = root / "s.parquet"
            for p in (p_panel, p_t1, p_w, p_s):
                p.write_text("dummy\n", encoding="utf-8")
            c_path = root / "contract.toml"
            c_path.write_text(
                f"""
contract_version = 2
status = "scenario_bounds_grid"
[authorization]
authorized_on = "2026-11-26"
note = "test"
[elasticity_grid]
definition = "dlog_certified_hours_dlog_real_aewr"
canonical_values = [0.0, -1.0]
parametric_exposure_shares = [0.85, 1.0]
[track1_source]
design = "t1"
artifact = "q.csv"
artifact_sha256 = "PENDING_PATH2"
row_selector = "r1"
inference_method = "cv3j"
point_estimate = -0.2
standard_error = 0.4
ci95_low = -1.0
ci95_high = 0.6
n_states = 46
[passthrough_source]
design = "w"
artifact = "w.csv"
artifact_sha256 = "{sha256_file(p_w)}"
row_selector = "r"
estimate = 0.62
standard_error = 0.1
[application]
population = "all_shared_panel_county_years"
source_panel = "panel.parquet"
source_panel_sha256 = "{sha256_file(p_panel)}"
real_aewr_column = "aewr_ppi"
real_aewr_unit = "2012 USD"
nominal_aewr_column = "aewr"
nominal_aewr_unit = "nominal USD"
counterfactual_delta = 1.0
start_year = 2011
complete_end_year = 2024
provisional_end_year = 2025
snapshot_year = 2024
[secondary_counterfactuals]
percent_shifts = [0.10, -0.10]
skill_rates_artifact = "s.parquet"
skill_rates_sha256 = "{sha256_file(p_s)}"
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unresolved placeholder"):
                load_welfare_contract(c_path, root)

    def test_contract_loader_cross_checks_quantity_and_wage_csv_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_panel = root / "panel.parquet"
            p_t1 = root / "q.csv"
            p_w = root / "w.csv"
            p_s = root / "s.parquet"
            p_panel.write_text("dummy\n", encoding="utf-8")
            p_s.write_text("dummy\n", encoding="utf-8")
            p_t1.write_text(
                "window,contrast,specification,outcome,estimator,treatment,"
                "elasticity,elasticity_se_jk,elasticity_low_jk,elasticity_high_jk,states\n"
                "full_fy,fy2025_2026,B_primary,hours,PPML,x_us_skill_1,"
                "-0.20,0.40,-1.00,0.60,46\n",
                encoding="utf-8",
            )
            p_w.write_text(
                "track,specification,tau,tier,weights,term,estimate\n"
                "wage,occupation_dw,2026,R7,h0,dose,0.622887\n",
                encoding="utf-8",
            )
            c_path = root / "contract.toml"
            template = (
                "contract_version = 2\n"
                'status = "scenario_bounds_grid"\n'
                "[authorization]\n"
                'authorized_on = "2026-11-26"\n'
                'note = "test"\n'
                "[elasticity_grid]\n"
                'definition = "dlog_certified_hours_dlog_real_aewr"\n'
                "canonical_values = [0.0, -1.0]\n"
                "parametric_exposure_shares = [0.85, 1.0]\n"
                "[track1_source]\n"
                'design = "t1"\n'
                'artifact = "q.csv"\n'
                f'artifact_sha256 = "{sha256_file(p_t1)}"\n'
                'row_selector = "window=full_fy,contrast=fy2025_2026,specification=B_primary,outcome=hours,estimator=PPML,treatment=x_us_skill_1"\n'
                'inference_method = "cv3j"\n'
                "point_estimate = {pt}\n"
                "standard_error = 0.40\n"
                "ci95_low = -1.00\n"
                "ci95_high = 0.60\n"
                "n_states = 46\n"
                "[passthrough_source]\n"
                'design = "w"\n'
                'artifact = "w.csv"\n'
                f'artifact_sha256 = "{sha256_file(p_w)}"\n'
                'row_selector = "r"\n'
                "estimate = 0.622887\n"
                "standard_error = 0.1\n"
                "[application]\n"
                'population = "all_shared_panel_county_years"\n'
                'source_panel = "panel.parquet"\n'
                f'source_panel_sha256 = "{sha256_file(p_panel)}"\n'
                'real_aewr_column = "aewr_ppi"\n'
                'real_aewr_unit = "2012 USD"\n'
                'nominal_aewr_column = "aewr"\n'
                'nominal_aewr_unit = "nominal USD"\n'
                "counterfactual_delta = 1.0\n"
                "start_year = 2011\n"
                "complete_end_year = 2024\n"
                "provisional_end_year = 2025\n"
                "snapshot_year = 2024\n"
                "[secondary_counterfactuals]\n"
                "percent_shifts = [0.10, -0.10]\n"
                'skill_rates_artifact = "s.parquet"\n'
                f'skill_rates_sha256 = "{sha256_file(p_s)}"\n'
            )
            c_path.write_text(template.format(pt="-0.20"), encoding="utf-8")
            loaded = load_welfare_contract(c_path, root)
            self.assertAlmostEqual(loaded["track1_ci95_low"], -1.00, places=6)

            c_path.write_text(template.format(pt="-0.25"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees with q.csv"):
                load_welfare_contract(c_path, root)


if __name__ == "__main__":
    unittest.main()
