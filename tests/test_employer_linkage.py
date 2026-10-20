"""Unit tests for the worksite-business linkage contract."""

from __future__ import annotations

import unittest

import polars as pl

from h2a.employer_linkage import (
    ADDENDUM_SOURCE,
    H2A_SOURCE,
    TIERS,
    _new_partition,
    _union,
    build_review_report,
    classify_application_role,
    link_employer_records,
    normalize_fein,
    normalize_name,
    normalize_phone,
    normalize_state,
    normalize_zip,
    validate_linkage_partitions,
    validate_review_labels,
)


def make_record(
    index: int,
    name: str,
    *,
    state: str = "NC",
    postal_code: str = "27000",
    address: str = "",
    phone: str = "",
    fein: str = "",
    trade_name: str = "",
    first_year: int = 2015,
    last_year: int = 2015,
    alias_only: bool = False,
) -> dict[str, object]:
    return {
        "source_dataset": H2A_SOURCE,
        "source_name_raw": name,
        "source_trade_name_raw": trade_name,
        "source_address_1_raw": address,
        "source_address_2_raw": "",
        "source_city_raw": "CITY",
        "source_state_raw": state,
        "source_postal_code_raw": postal_code,
        "source_phone_raw": phone,
        "source_fein_raw": fein,
        "first_fiscal_year": first_year,
        "last_fiscal_year": last_year,
        "source_row_count": 1,
        "membership_count": 1,
        "included_business_ever": not alias_only,
        "alias_evidence_ever": alias_only,
        "employer_record_roles": (
            "post2020_individual_master_alias_only"
            if alias_only
            else "pre2020_standalone_business"
        ),
        "data_regimes": "pre_2020" if last_year < 2020 else "post_2020",
        "employer_record_id": f"wbr_{index:024x}",
        "normalized_name": normalize_name(name),
        "normalized_trade_name": normalize_name(trade_name),
        "normalized_address_1": normalize_name(address),
        "normalized_address_2": "",
        "normalized_city": "CITY",
        "normalized_state": state,
        "normalized_postal_code": postal_code,
        "normalized_phone": phone,
        "normalized_fein": fein,
        "normalized_full_address": (
            f"{normalize_name(address)}||CITY|{state}|{postal_code}"
            if address
            else ""
        ),
    }


def make_membership(records: list[dict[str, object]]) -> pl.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "case_number": f"CASE-{record['employer_record_id']}",
                "fiscal_year": record["last_fiscal_year"],
                "employer_record_id": record["employer_record_id"],
                "employer_is_worksite_business": record["included_business_ever"],
                "employer_is_alias_evidence_only": record["alias_evidence_ever"],
                "employer_record_role": record["employer_record_roles"],
            }
        )
    return pl.DataFrame(rows)


class NormalizationTests(unittest.TestCase):
    def test_normalization(self) -> None:
        self.assertEqual(normalize_name("  José & Sons, L.L.C. "), "JOSE AND SONS L L C")
        self.assertEqual(normalize_zip("1234"), "01234")
        self.assertEqual(normalize_phone("+1 (919) 555-0100"), "9195550100")
        self.assertEqual(normalize_phone("000-000-0000"), "")
        self.assertEqual(normalize_fein("12-3456789"), "123456789")
        self.assertEqual(normalize_state("North Carolina"), "NC")


class RoleTests(unittest.TestCase):
    def base(self, **updates: object) -> dict[str, object]:
        row: dict[str, object] = {
            "source_dataset": H2A_SOURCE,
            "source_name_raw": "Grower LLC",
            "fiscal_year": 2018,
            "case_entry_count": 2,
            "has_named_addendum": False,
            "source_primary_sub_raw": "SUB",
            "source_organization_flag_raw": "",
            "source_type_of_employer_application_raw": "",
            "source_ag_association_or_agency_raw": "",
            "source_h2a_labor_contractor_raw": "",
        }
        row.update(updates)
        return row

    def test_role_classification_across_2020_break(self) -> None:
        self.assertEqual(
            classify_application_role(self.base(case_entry_count=1)),
            "pre2020_standalone_business",
        )
        self.assertEqual(
            classify_application_role(self.base()),
            "pre2020_sub_employer_business",
        )
        self.assertEqual(
            classify_application_role(
                self.base(
                    source_primary_sub_raw="PRI",
                    source_organization_flag_raw="Individual Employer",
                )
            ),
            "pre2020_substantive_master_business",
        )
        self.assertEqual(
            classify_application_role(
                self.base(
                    source_primary_sub_raw="PRI",
                    source_organization_flag_raw="Association - Filing as Agent",
                )
            ),
            "pre2020_association_agent_contractor_master_excluded",
        )
        self.assertEqual(
            classify_application_role(
                self.base(fiscal_year=2021, has_named_addendum=False)
            ),
            "post2020_master_no_named_addendum_business",
        )
        self.assertEqual(
            classify_application_role(
                self.base(
                    fiscal_year=2021,
                    has_named_addendum=True,
                    source_type_of_employer_application_raw="Individual Employer",
                )
            ),
            "post2020_individual_master_alias_only",
        )
        self.assertEqual(
            classify_application_role(
                self.base(
                    source_dataset=ADDENDUM_SOURCE,
                    fiscal_year=2021,
                )
            ),
            "post2020_addendum_business",
        )


class LinkageRuleTests(unittest.TestCase):
    def link(
        self, records: list[dict[str, object]], membership: pl.DataFrame | None = None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        frame = pl.DataFrame(records)
        linked, audit, _ = link_employer_records(
            frame, membership if membership is not None else make_membership(records)
        )
        validate_linkage_partitions(linked)
        return linked, audit

    def same(self, linked: pl.DataFrame, tier: str, left: int, right: int) -> bool:
        values = linked.sort("employer_record_id")[f"employer_id_{tier}"]
        return values[left] == values[right]

    def test_exact_fein_all_tiers(self) -> None:
        records = [
            make_record(0, "ALPHA HOLDINGS", fein="123456789"),
            make_record(1, "OMEGA PRODUCE", fein="123456789", state="CA"),
        ]
        linked, _ = self.link(records)
        for tier in TIERS:
            self.assertTrue(self.same(linked, tier, 0, 1))

    def test_conservative_exact_name_requires_contact(self) -> None:
        records = [
            make_record(0, "ACME ORCHARDS LLC", address="10 MAIN ST"),
            make_record(1, "ACME ORCHARDS LLC", address="10 MAIN ST"),
            make_record(2, "ACME ORCHARDS LLC", state="CA", postal_code="90210"),
        ]
        linked, _ = self.link(records)
        self.assertTrue(self.same(linked, "conservative", 0, 1))
        self.assertFalse(self.same(linked, "conservative", 0, 2))
        self.assertFalse(self.same(linked, "balanced", 0, 2))

    def test_balanced_rules_and_high_recall_extensions(self) -> None:
        records = [
            make_record(0, "SUNRISE ORCHARDS LLC", address="1 APPLE RD"),
            make_record(1, "SUNRISE ORCHARD LLC", address="1 APPLE RD"),
            make_record(2, "ALEXANDER FAMILY ORCHARDS LLC", postal_code="27100"),
            make_record(3, "ALEXANDER FAMILY ORCHARD LLC", postal_code="27100"),
            make_record(4, "BLUEBERRY HILL FARMS LLC", postal_code="27200"),
            make_record(5, "BLUEBERRY HILLS FARM LLC", postal_code="27300"),
            make_record(6, "CEDAR VALLEY PRODUCE LLC", postal_code="27400"),
            make_record(7, "CEDAR VALLEY PRODUCE LLC", postal_code="27500"),
        ]
        linked, audit = self.link(records)
        self.assertTrue(self.same(linked, "balanced", 0, 1))
        self.assertTrue(self.same(linked, "balanced", 2, 3))
        self.assertFalse(self.same(linked, "balanced", 4, 5))
        self.assertTrue(self.same(linked, "high_recall", 4, 5))
        self.assertFalse(self.same(linked, "balanced", 6, 7))
        self.assertTrue(self.same(linked, "high_recall", 6, 7))
        accepted_rules = set(audit.filter(pl.col("decision") == "accepted")["linkage_rule"])
        self.assertIn("similar_name_exact_address", accepted_rules)
        self.assertIn("similar_name_state_zip_shared_core", accepted_rules)
        self.assertIn("similar_name_same_state_shared_core", accepted_rules)
        self.assertIn("exact_distinctive_name_state_cross_zip", accepted_rules)

    def test_dba_phone_and_exact_state_zip_rules(self) -> None:
        records = [
            make_record(
                0,
                "LEGAL ENTITY ONE LLC",
                trade_name="MOUNTAIN VIEW PRODUCE",
                phone="9195550100",
            ),
            make_record(
                1,
                "LEGAL ENTITY TWO LLC",
                trade_name="MOUNTAIN VIEW PRODUCE",
                phone="9195550100",
            ),
            make_record(2, "SUNRISE ORCHARDS LLC", phone="9195550200"),
            make_record(3, "SUNRISE ORCHARD LLC", phone="9195550200"),
            make_record(4, "DISTINCTIVE CEDAR HOLDINGS LLC", postal_code="27601"),
            make_record(5, "DISTINCTIVE CEDAR HOLDINGS LLC", postal_code="27601"),
        ]
        linked, audit = self.link(records)
        self.assertTrue(self.same(linked, "conservative", 0, 1))
        self.assertFalse(self.same(linked, "conservative", 2, 3))
        self.assertTrue(self.same(linked, "balanced", 2, 3))
        self.assertFalse(self.same(linked, "conservative", 4, 5))
        self.assertTrue(self.same(linked, "balanced", 4, 5))
        accepted_rules = set(audit.filter(pl.col("decision") == "accepted")["linkage_rule"])
        self.assertIn("exact_dba_exact_phone", accepted_rules)
        self.assertIn("similar_name_exact_phone", accepted_rules)
        self.assertIn("exact_distinctive_name_state_zip", accepted_rules)

    def test_same_case_individual_master_bridge(self) -> None:
        records = [
            make_record(
                0,
                "WASHINGTON APPLE GROWERS LLC",
                first_year=2021,
                last_year=2021,
                alias_only=True,
            ),
            make_record(
                1,
                "WASHINGTON APPLE GROWER LLC",
                state="CA",
                postal_code="90210",
                first_year=2021,
                last_year=2021,
            ),
        ]
        membership = make_membership(records).with_columns(
            pl.lit("CASE-SHARED").alias("case_number"),
            pl.when(pl.col("employer_is_alias_evidence_only"))
            .then(pl.lit("post2020_individual_master_alias_only"))
            .otherwise(pl.lit("post2020_addendum_business"))
            .alias("employer_record_role"),
        )
        linked, audit = self.link(records, membership)
        self.assertFalse(self.same(linked, "conservative", 0, 1))
        self.assertTrue(self.same(linked, "balanced", 0, 1))
        self.assertIn(
            "same_case_individual_master_addendum_bridge",
            set(audit.filter(pl.col("decision") == "accepted")["linkage_rule"]),
        )

    def test_conflicting_feins_and_shared_contacts_are_rejected(self) -> None:
        records = [
            make_record(
                0,
                "CONFLICT PRODUCE LLC",
                address="99 MARKET ST",
                fein="123456789",
            ),
            make_record(
                1,
                "CONFLICT PRODUCE LLC",
                address="99 MARKET ST",
                fein="987654321",
            ),
        ]
        records.extend(
            make_record(
                index + 2,
                f"DISTINCTIVE ALPHA {index} HOLDINGS",
                phone="9195550100",
                state="",
                postal_code="",
            )
            for index in range(11)
        )
        linked, audit = self.link(records)
        self.assertFalse(self.same(linked, "conservative", 0, 1))
        reasons = set(audit["rejection_reason"])
        self.assertIn("conflicting_fein", reasons)
        self.assertIn("shared_contact_more_than_ten_names", reasons)

    def test_name_sprawl_and_shuffle_determinism(self) -> None:
        rows = [make_record(i, f"DISTINCTIVE NAME {i}") for i in range(101)]
        aliases = [(row["normalized_name"],) for row in rows]
        partition = _new_partition(rows, aliases, "balanced")
        for index in range(1, 100):
            self.assertEqual(_union(partition, 0, index, "test_fuzzy")[0], "accepted")
        self.assertEqual(
            _union(partition, 0, 100, "test_fuzzy"),
            ("rejected", "tier_name_sprawl"),
        )
        frame = pl.DataFrame(rows[:8])
        membership = make_membership(rows[:8])
        linked, _, _ = link_employer_records(frame, membership, collect_audit=False)
        shuffled, _, _ = link_employer_records(
            frame.sample(fraction=1, shuffle=True, seed=1729),
            membership,
            collect_audit=False,
        )
        columns = ["employer_record_id", *(f"employer_id_{tier}" for tier in TIERS)]
        self.assertTrue(
            linked.select(columns)
            .sort("employer_record_id")
            .equals(shuffled.select(columns).sort("employer_record_id"))
        )


class ReviewLabelTests(unittest.TestCase):
    def test_review_labels_must_be_complete_and_valid(self) -> None:
        sample = pl.DataFrame({"pair_id": ["a", "b"]})
        labels = pl.DataFrame(
            {"pair_id": ["a", "b"], "same_business": ["true", "uncertain"]}
        )
        validated = validate_review_labels(sample, labels)
        self.assertEqual(validated["same_business"].to_list(), ["true", "uncertain"])
        with self.assertRaisesRegex(ValueError, "every sampled pair"):
            validate_review_labels(sample, labels.head(1))
        with self.assertRaisesRegex(ValueError, "true, false, or uncertain"):
            validate_review_labels(
                sample,
                pl.DataFrame(
                    {"pair_id": ["a", "b"], "same_business": ["yes", "false"]}
                ),
            )
        with self.assertRaisesRegex(ValueError, "true, false, or uncertain"):
            validate_review_labels(
                sample,
                pl.DataFrame(
                    {"pair_id": ["a", "b"], "same_business": [None, "false"]}
                ),
            )

    def test_uncertain_counts_as_false_in_review_precision(self) -> None:
        labeled = pl.DataFrame(
            {
                "sample_type": ["accepted_balanced_edge"] * 2,
                "linkage_rule": ["exact_fein"] * 2,
                "data_regime": ["pre_2020"] * 2,
                "sampling_weight": [1.0, 3.0],
                "same_business": ["true", "uncertain"],
            }
        )
        report, report_hash = build_review_report(labeled)
        overall = report.filter(pl.col("stratum") == "all")["estimate"].item()
        self.assertEqual(overall, 0.25)
        self.assertRegex(report_hash, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
