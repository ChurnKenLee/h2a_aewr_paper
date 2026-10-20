"""Deterministic role-aware linkage of H-2A worksite businesses.

The functions in this module construct identity evidence and linkage audits.
They do not use application outcomes, certification decisions, worker counts,
wages, hours, or downstream design variables.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from typing import Final

import polars as pl
import us
from rapidfuzz import fuzz

# docs-ground:start employer-linkage-contract
LINKAGE_UNIT: Final = "worksite_business"
LINKAGE_VERSION: Final = "v2"
INPUT_MIN_YEAR: Final = 2008
INPUT_MAX_YEAR: Final = 2024
AS_OF_YEAR: Final = 2022
REVIEW_SEED: Final = 1729
REVIEW_ACCEPTED_COUNT: Final = 140
REVIEW_REJECTED_COUNT: Final = 40
REVIEW_CLUSTER_COUNT: Final = 20
TIERS: Final = ("conservative", "balanced", "high_recall")
TIER_NAME_SPRAW_LIMITS: Final = {
    "conservative": 25,
    "balanced": 100,
    "high_recall": 500,
}
SHARED_CONTACT_NAME_LIMIT: Final = 10
BALANCED_CONTACT_SIMILARITY: Final = 97
BALANCED_STATE_ZIP_SIMILARITY: Final = 98
BALANCED_SAME_CASE_SIMILARITY: Final = 97
HIGH_RECALL_STATE_SIMILARITY: Final = 94

H2A_SOURCE: Final = "h2a_with_fips"
ADDENDUM_SOURCE: Final = "h2a_addendum_b_with_fips"

RAW_IDENTITY_COLUMNS: Final = (
    "source_dataset",
    "source_name_raw",
    "source_trade_name_raw",
    "source_address_1_raw",
    "source_address_2_raw",
    "source_city_raw",
    "source_state_raw",
    "source_postal_code_raw",
    "source_phone_raw",
    "source_fein_raw",
)

ROLE_RAW_COLUMNS: Final = (
    "source_primary_sub_raw",
    "source_organization_flag_raw",
    "source_type_of_employer_application_raw",
    "source_ag_association_or_agency_raw",
    "source_h2a_labor_contractor_raw",
)

MEMBERSHIP_JOIN_COLUMNS: Final = (
    "source_dataset",
    "case_number",
    "fiscal_year",
    *RAW_IDENTITY_COLUMNS[1:],
    *ROLE_RAW_COLUMNS,
)

VALID_RECORD_ROLES: Final = {
    "pre2020_standalone_business",
    "pre2020_sub_employer_business",
    "pre2020_substantive_master_business",
    "pre2020_association_agent_contractor_master_excluded",
    "post2020_addendum_business",
    "post2020_blank_addendum_excluded",
    "post2020_master_no_named_addendum_business",
    "post2020_individual_master_alias_only",
    "post2020_master_with_addendum_excluded",
}

INCLUDED_RECORD_ROLES: Final = {
    "pre2020_standalone_business",
    "pre2020_sub_employer_business",
    "pre2020_substantive_master_business",
    "post2020_addendum_business",
    "post2020_master_no_named_addendum_business",
}
# docs-ground:end employer-linkage-contract

PLACEHOLDER_ALIASES: Final = {
    "",
    "D B A",
    "DBA",
    "N A",
    "NA",
    "NIL",
    "NO DBA",
    "NO D B A",
    "NONE",
    "NONE USED",
    "NOT APPLICABLE",
    "SAME",
    "TBD",
    "UNKNOWN",
    "VARIOUS",
}

PLACEHOLDER_ADDRESSES: Final = {
    "",
    "N A",
    "NA",
    "NONE",
    "NOT APPLICABLE",
    "SAME",
    "TBD",
    "UNKNOWN",
    "VARIOUS",
}

GENERIC_NAME_TOKENS: Final = {
    "AG",
    "AGRICULTURAL",
    "AGRICULTURE",
    "AND",
    "BUSINESS",
    "CO",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "DBA",
    "ENTERPRISE",
    "ENTERPRISES",
    "FARM",
    "FARMING",
    "FARMS",
    "GROWER",
    "GROWERS",
    "INC",
    "INCORPORATED",
    "LIMITED",
    "LLC",
    "LLP",
    "LP",
    "LTD",
    "NURSERIES",
    "NURSERY",
    "OF",
    "ORCHARD",
    "ORCHARDS",
    "PARTNERSHIP",
    "RANCH",
    "RANCHES",
    "SERVICE",
    "SERVICES",
    "THE",
}


def require_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
    """Fail with a compact schema error when required fields are absent."""
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {', '.join(missing)}"
        )


def normalize_text(value: object) -> str:
    """Normalize free text without removing substantive business tokens."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").upper().strip()
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(value: object) -> str:
    return normalize_text(value)


def normalize_zip(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().removesuffix(".0")
    digits = re.sub(r"\D", "", text)
    if len(digits) < 4:
        return ""
    return digits[:5].zfill(5)


def normalize_phone(value: object) -> str:
    digits = re.sub(r"\D", "", "" if value is None else str(value))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10 or len(set(digits)) == 1:
        return ""
    return digits


def normalize_fein(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().removesuffix(".0")
    digits = re.sub(r"\D", "", text)
    if len(digits) != 9 or len(set(digits)) == 1:
        return ""
    return digits


@cache
def normalize_state(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    if text in {"DC", "DISTRICT OF COLUMBIA"}:
        return "DC"
    state = us.states.lookup(text)
    return "" if state is None else state.abbr


def core_name_tokens(name: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                token
                for token in name.split()
                if len(token) >= 4
                and not token.isdigit()
                and token not in GENERIC_NAME_TOKENS
            }
        )
    )


def is_valid_alias(name: str) -> bool:
    """Return whether an alias is usable as name evidence."""
    if name in PLACEHOLDER_ALIASES:
        return False
    compact = re.sub(r"[^A-Z0-9]", "", name)
    return len(compact) >= 4 and bool(core_name_tokens(name))


def valid_full_address(row: dict[str, object]) -> str:
    address_1 = str(row["normalized_address_1"])
    if (
        address_1 in PLACEHOLDER_ADDRESSES
        or len(re.sub(r"[^A-Z0-9]", "", address_1)) < 5
        or not row["normalized_state"]
        or not row["normalized_postal_code"]
    ):
        return ""
    return "|".join(
        [
            address_1,
            str(row["normalized_address_2"]),
            str(row["normalized_city"]),
            str(row["normalized_state"]),
            str(row["normalized_postal_code"]),
        ]
    )


def _stable_hash(prefix: str, values: list[object] | tuple[object, ...]) -> str:
    serialized = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
    return prefix + digest


def classify_application_role(row: dict[str, object]) -> str:
    """Classify one source membership without consulting application outcomes."""
    source = str(row["source_dataset"])
    fiscal_year = int(row["fiscal_year"])
    named = bool(normalize_name(row.get("source_name_raw")))
    if source == ADDENDUM_SOURCE:
        return (
            "post2020_addendum_business"
            if named
            else "post2020_blank_addendum_excluded"
        )
    if source != H2A_SOURCE:
        raise ValueError(f"unknown H-2A linkage source: {source}")

    if fiscal_year < 2020:
        if int(row.get("case_entry_count", 0)) == 1:
            return "pre2020_standalone_business"
        organization = normalize_text(row.get("source_organization_flag_raw"))
        primary_sub = normalize_text(row.get("source_primary_sub_raw"))
        is_master = (
            fiscal_year < 2016 and bool(organization)
        ) or (2016 <= fiscal_year < 2020 and primary_sub == "PRI")
        if not is_master:
            return "pre2020_sub_employer_business"
        role_text = " ".join(
            normalize_text(row.get(column))
            for column in (
                "source_organization_flag_raw",
                "source_type_of_employer_application_raw",
                "source_ag_association_or_agency_raw",
            )
        )
        contractor = normalize_text(
            row.get("source_h2a_labor_contractor_raw")
        ) == "Y"
        nonbusiness_master = (
            organization == "A"
            or "ASSOCIATION" in role_text
            or "AGENT" in role_text
            or "CONTRACTOR" in role_text
            or contractor
        )
        return (
            "pre2020_association_agent_contractor_master_excluded"
            if nonbusiness_master
            else "pre2020_substantive_master_business"
        )

    if not bool(row.get("has_named_addendum", False)):
        return "post2020_master_no_named_addendum_business"
    role_text = " ".join(
        normalize_text(row.get(column))
        for column in (
            "source_type_of_employer_application_raw",
            "source_ag_association_or_agency_raw",
            "source_organization_flag_raw",
        )
    )
    if "INDIVIDUAL EMPLOYER" in role_text:
        return "post2020_individual_master_alias_only"
    return "post2020_master_with_addendum_excluded"


def _string_column(source: str, target: str | None = None) -> pl.Expr:
    return pl.col(source).cast(pl.String).fill_null("").alias(target or source)


def _empty_role_columns() -> list[pl.Expr]:
    return [pl.lit("").alias(column) for column in ROLE_RAW_COLUMNS]


def build_membership_records(
    h2a: pl.DataFrame,
    addendum: pl.DataFrame,
    *,
    input_max_year: int = INPUT_MAX_YEAR,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Construct the outcome-free source-row role and membership map."""
    h2a_required = {
        "case_number",
        "fiscal_year",
        "employer_name",
        "trade_name_dba",
        "employer_address_1",
        "employer_address_2",
        "employer_city",
        "employer_state",
        "employer_postal_code",
        "employer_phone",
        "employer_fein",
        "primary_sub",
        "organization_flag",
        "type_of_employer_application",
        "ag_association_or_agency",
        "h2a_labor_contractor",
    }
    addendum_required = {
        "case_number",
        "fiscal_year",
        "business_name",
        "worksite_address_1",
        "worksite_address_2",
        "worksite_city",
        "worksite_state",
        "worksite_zip",
    }
    require_columns(h2a, h2a_required, H2A_SOURCE)
    require_columns(addendum, addendum_required, ADDENDUM_SOURCE)

    observed_min = min(h2a["fiscal_year"].min(), addendum["fiscal_year"].min())
    observed_max = max(h2a["fiscal_year"].max(), addendum["fiscal_year"].max())
    if observed_min != INPUT_MIN_YEAR or observed_max != input_max_year:
        raise ValueError(
            "employer linkage requires the pinned FY2008--"
            f"{input_max_year} source vintage; got FY{observed_min}--{observed_max}"
        )
    if (h2a["fiscal_year"] > input_max_year).any() or (
        addendum["fiscal_year"] > input_max_year
    ).any():
        raise ValueError("source rows exceed the declared input vintage")

    h2a_rows = h2a.select(
        pl.lit(H2A_SOURCE).alias("source_dataset"),
        _string_column("case_number"),
        pl.col("fiscal_year").cast(pl.Int32),
        _string_column("employer_name", "source_name_raw"),
        _string_column("trade_name_dba", "source_trade_name_raw"),
        _string_column("employer_address_1", "source_address_1_raw"),
        _string_column("employer_address_2", "source_address_2_raw"),
        _string_column("employer_city", "source_city_raw"),
        _string_column("employer_state", "source_state_raw"),
        _string_column("employer_postal_code", "source_postal_code_raw"),
        _string_column("employer_phone", "source_phone_raw"),
        _string_column("employer_fein", "source_fein_raw"),
        _string_column("primary_sub", "source_primary_sub_raw"),
        _string_column("organization_flag", "source_organization_flag_raw"),
        _string_column(
            "type_of_employer_application",
            "source_type_of_employer_application_raw",
        ),
        _string_column(
            "ag_association_or_agency",
            "source_ag_association_or_agency_raw",
        ),
        _string_column(
            "h2a_labor_contractor", "source_h2a_labor_contractor_raw"
        ),
    )
    addendum_rows = addendum.select(
        pl.lit(ADDENDUM_SOURCE).alias("source_dataset"),
        _string_column("case_number"),
        pl.col("fiscal_year").cast(pl.Int32),
        _string_column("business_name", "source_name_raw"),
        pl.lit("").alias("source_trade_name_raw"),
        _string_column("worksite_address_1", "source_address_1_raw"),
        _string_column("worksite_address_2", "source_address_2_raw"),
        _string_column("worksite_city", "source_city_raw"),
        _string_column("worksite_state", "source_state_raw"),
        _string_column("worksite_zip", "source_postal_code_raw"),
        pl.lit("").alias("source_phone_raw"),
        pl.lit("").alias("source_fein_raw"),
        *_empty_role_columns(),
    )

    h2a_keys = h2a_rows.select(MEMBERSHIP_JOIN_COLUMNS).unique()
    case_counts = h2a_keys.group_by("case_number", "fiscal_year").agg(
        pl.len().cast(pl.Int32).alias("case_entry_count")
    )
    named_addendum_cases = (
        addendum_rows.with_columns(
            pl.col("source_name_raw")
            .str.strip_chars()
            .ne("")
            .alias("named_addendum")
        )
        .group_by("case_number", "fiscal_year")
        .agg(pl.col("named_addendum").any().alias("has_named_addendum"))
    )

    h2a_membership = (
        h2a_rows.group_by(MEMBERSHIP_JOIN_COLUMNS)
        .agg(pl.len().cast(pl.UInt32).alias("source_row_count"))
        .join(
            case_counts,
            on=["case_number", "fiscal_year"],
            how="left",
            validate="m:1",
        )
        .join(
            named_addendum_cases,
            on=["case_number", "fiscal_year"],
            how="left",
            validate="m:1",
        )
        .with_columns(pl.col("has_named_addendum").fill_null(False))
    )
    addendum_membership = (
        addendum_rows.group_by(MEMBERSHIP_JOIN_COLUMNS)
        .agg(pl.len().cast(pl.UInt32).alias("source_row_count"))
        .with_columns(
            pl.lit(1, dtype=pl.Int32).alias("case_entry_count"),
            pl.lit(True).alias("has_named_addendum"),
        )
    )
    membership = pl.concat(
        [h2a_membership, addendum_membership], how="vertical_relaxed"
    ).sort(MEMBERSHIP_JOIN_COLUMNS)
    membership = membership.with_columns(
        pl.struct(
            [
                "source_dataset",
                "fiscal_year",
                "source_name_raw",
                "case_entry_count",
                "has_named_addendum",
                *ROLE_RAW_COLUMNS,
            ]
        )
        .map_elements(classify_application_role, return_dtype=pl.String)
        .alias("employer_record_role"),
        pl.struct(MEMBERSHIP_JOIN_COLUMNS)
        .map_elements(
            lambda row: _stable_hash(
                "wbm_", tuple(row[column] for column in MEMBERSHIP_JOIN_COLUMNS)
            ),
            return_dtype=pl.String,
        )
        .alias("source_membership_id"),
    ).with_columns(
        pl.col("employer_record_role")
        .is_in(INCLUDED_RECORD_ROLES)
        .and_(pl.col("source_name_raw").str.strip_chars().ne(""))
        .alias("employer_is_worksite_business"),
        pl.col("employer_record_role")
        .eq("post2020_individual_master_alias_only")
        .alias("employer_is_alias_evidence_only"),
        pl.when(pl.col("fiscal_year") < 2020)
        .then(pl.lit("pre_2020"))
        .otherwise(pl.lit("post_2020"))
        .alias("data_regime"),
    )
    if not set(membership["employer_record_role"].unique()).issubset(
        VALID_RECORD_ROLES
    ):
        raise ValueError("membership contains an invalid record role")
    if membership["source_membership_id"].n_unique() != membership.height:
        raise ValueError("source membership IDs are not unique")
    if membership.select(MEMBERSHIP_JOIN_COLUMNS).is_duplicated().any():
        raise ValueError("source membership join keys are duplicated")

    source_diagnostics = (
        membership.group_by("source_dataset")
        .agg(
            pl.col("source_row_count").sum().alias("source_rows"),
            pl.len().alias("membership_rows"),
            pl.col("employer_is_worksite_business")
            .sum()
            .alias("included_membership_rows"),
            pl.col("employer_is_alias_evidence_only")
            .sum()
            .alias("alias_only_membership_rows"),
        )
        .sort("source_dataset")
    )
    return membership, source_diagnostics


def build_identity_records(
    membership: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Collapse memberships to stable raw identities and normalize evidence."""
    named = membership.filter(
        pl.col("source_name_raw").str.strip_chars().ne("")
    )
    records = (
        named.group_by(RAW_IDENTITY_COLUMNS)
        .agg(
            pl.col("fiscal_year").min().alias("first_fiscal_year"),
            pl.col("fiscal_year").max().alias("last_fiscal_year"),
            pl.col("source_row_count").sum().alias("source_row_count"),
            pl.len().cast(pl.UInt32).alias("membership_count"),
            pl.col("employer_is_worksite_business")
            .any()
            .alias("included_business_ever"),
            pl.col("employer_is_alias_evidence_only")
            .any()
            .alias("alias_evidence_ever"),
            pl.col("employer_record_role")
            .unique()
            .sort()
            .str.join("|")
            .alias("employer_record_roles"),
            pl.col("data_regime")
            .unique()
            .sort()
            .str.join("|")
            .alias("data_regimes"),
        )
        .with_columns(
            pl.struct(RAW_IDENTITY_COLUMNS)
            .map_elements(
                lambda row: _stable_hash(
                    "wbr_", tuple(row[column] for column in RAW_IDENTITY_COLUMNS)
                ),
                return_dtype=pl.String,
            )
            .alias("employer_record_id"),
            pl.col("source_name_raw")
            .map_elements(normalize_name, return_dtype=pl.String)
            .alias("normalized_name"),
            pl.col("source_trade_name_raw")
            .map_elements(normalize_name, return_dtype=pl.String)
            .alias("normalized_trade_name"),
            pl.col("source_address_1_raw")
            .map_elements(normalize_text, return_dtype=pl.String)
            .alias("normalized_address_1"),
            pl.col("source_address_2_raw")
            .map_elements(normalize_text, return_dtype=pl.String)
            .alias("normalized_address_2"),
            pl.col("source_city_raw")
            .map_elements(normalize_text, return_dtype=pl.String)
            .alias("normalized_city"),
            pl.col("source_state_raw")
            .map_elements(normalize_state, return_dtype=pl.String)
            .alias("normalized_state"),
            pl.col("source_postal_code_raw")
            .map_elements(normalize_zip, return_dtype=pl.String)
            .alias("normalized_postal_code"),
            pl.col("source_phone_raw")
            .map_elements(normalize_phone, return_dtype=pl.String)
            .alias("normalized_phone"),
            pl.col("source_fein_raw")
            .map_elements(normalize_fein, return_dtype=pl.String)
            .alias("normalized_fein"),
        )
        .with_columns(
            pl.struct(
                [
                    "normalized_address_1",
                    "normalized_address_2",
                    "normalized_city",
                    "normalized_state",
                    "normalized_postal_code",
                ]
            )
            .map_elements(valid_full_address, return_dtype=pl.String)
            .alias("normalized_full_address")
        )
        .sort("employer_record_id")
    )
    if records["employer_record_id"].n_unique() != records.height:
        raise ValueError("employer record IDs contain a hash collision")
    if records.select(RAW_IDENTITY_COLUMNS).is_duplicated().any():
        raise ValueError("raw identity keys are duplicated")
    membership_with_ids = membership.join(
        records.select(*RAW_IDENTITY_COLUMNS, "employer_record_id"),
        on=list(RAW_IDENTITY_COLUMNS),
        how="left",
        validate="m:1",
    )
    missing_included_ids = membership_with_ids.filter(
        pl.col("employer_is_worksite_business")
        & pl.col("source_name_raw").str.strip_chars().ne("")
        & pl.col("employer_record_id").is_null()
    )
    if missing_included_ids.height:
        raise ValueError("named included records are missing record IDs")
    return records, membership_with_ids


def _aliases_for_records(rows: list[dict[str, object]]) -> tuple[list[tuple[str, ...]], Counter]:
    trade_to_primary: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        trade = str(row["normalized_trade_name"])
        if trade:
            trade_to_primary[trade].add(str(row["normalized_name"]))
    valid_trade_names = {
        trade
        for trade, primary_names in trade_to_primary.items()
        if is_valid_alias(trade)
        and len(primary_names) <= SHARED_CONTACT_NAME_LIMIT
    }
    aliases: list[tuple[str, ...]] = []
    frequency: Counter = Counter()
    for row in rows:
        values = []
        primary = str(row["normalized_name"])
        trade = str(row["normalized_trade_name"])
        if is_valid_alias(primary):
            values.append(primary)
        if trade in valid_trade_names and trade not in values:
            values.append(trade)
        record_aliases = tuple(values)
        aliases.append(record_aliases)
        frequency.update(record_aliases)
    return aliases, frequency


def _new_partition(
    rows: list[dict[str, object]], aliases: list[tuple[str, ...]], tier: str
) -> dict[str, object]:
    return {
        "parent": list(range(len(rows))),
        "size": [1] * len(rows),
        "anchor": [str(row["employer_record_id"]) for row in rows],
        "feins": [
            {str(row["normalized_fein"])} if row["normalized_fein"] else set()
            for row in rows
        ],
        "names": [set(value) for value in aliases],
        "evidence": [set() for _ in rows],
        "tier": tier,
    }


def _find(partition: dict[str, object], value: int) -> int:
    parent = partition["parent"]
    root = value
    while parent[root] != root:
        root = parent[root]
    while parent[value] != value:
        next_value = parent[value]
        parent[value] = root
        value = next_value
    return root


def _union(
    partition: dict[str, object], left: int, right: int, rule: str
) -> tuple[str, str]:
    left_root = _find(partition, left)
    right_root = _find(partition, right)
    if left_root == right_root:
        return "redundant", "already_linked"
    feins = partition["feins"]
    combined_feins = feins[left_root] | feins[right_root]
    if len(combined_feins) > 1:
        return "rejected", "conflicting_fein"
    names = partition["names"]
    combined_names = names[left_root] | names[right_root]
    tier = str(partition["tier"])
    if rule != "exact_fein" and len(combined_names) > TIER_NAME_SPRAW_LIMITS[tier]:
        return "rejected", "tier_name_sprawl"
    size = partition["size"]
    anchor = partition["anchor"]
    if size[left_root] < size[right_root] or (
        size[left_root] == size[right_root]
        and anchor[left_root] > anchor[right_root]
    ):
        left_root, right_root = right_root, left_root
    partition["parent"][right_root] = left_root
    size[left_root] += size[right_root]
    anchor[left_root] = min(anchor[left_root], anchor[right_root])
    feins[left_root] = combined_feins
    feins[right_root] = set()
    names[left_root] = combined_names
    names[right_root] = set()
    evidence = partition["evidence"]
    evidence[left_root].update(evidence[right_root])
    evidence[left_root].add(rule)
    evidence[right_root] = set()
    return "accepted", ""


def _edge_regime(left: dict[str, object], right: dict[str, object]) -> str:
    regimes = set(str(left["data_regimes"]).split("|")) | set(
        str(right["data_regimes"]).split("|")
    )
    if regimes == {"pre_2020"}:
        return "pre_2020"
    if regimes == {"post_2020"}:
        return "post_2020"
    return "cross_era"


def _audit_row(
    tier: str,
    rule: str,
    left: int,
    right: int,
    rows: list[dict[str, object]],
    alias_frequency: Counter,
    decision: str,
    rejection_reason: str,
    score: float | None,
    threshold: float | None,
) -> dict[str, object]:
    left_row = rows[left]
    right_row = rows[right]
    left_id = str(left_row["employer_record_id"])
    right_id = str(right_row["employer_record_id"])
    pair = tuple(sorted((left_id, right_id)))
    common_frequency = max(
        alias_frequency.get(str(left_row["normalized_name"]), 0),
        alias_frequency.get(str(right_row["normalized_name"]), 0),
    )
    return {
        "edge_id": _stable_hash(
            "wbe_", (tier, rule, *pair, decision, rejection_reason)
        ),
        "linkage_tier": tier,
        "linkage_rule": rule,
        "left_employer_record_id": pair[0],
        "right_employer_record_id": pair[1],
        "decision": decision,
        "rejection_reason": rejection_reason,
        "name_score": score,
        "rule_threshold": threshold,
        "data_regime": _edge_regime(left_row, right_row),
        "cross_state": bool(
            left_row["normalized_state"]
            and right_row["normalized_state"]
            and left_row["normalized_state"] != right_row["normalized_state"]
        ),
        "cross_zip": bool(
            left_row["normalized_postal_code"]
            and right_row["normalized_postal_code"]
            and left_row["normalized_postal_code"]
            != right_row["normalized_postal_code"]
        ),
        "common_name": common_frequency > 25,
        "common_name_frequency": common_frequency,
    }


def link_employer_records(
    records: pl.DataFrame,
    membership: pl.DataFrame,
    *,
    collect_audit: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create the three nested partitions and their edge/cluster audits."""
    active = records.filter(
        pl.col("included_business_ever") | pl.col("alias_evidence_ever")
    ).sort("employer_record_id")
    rows = active.to_dicts()
    if not rows:
        raise ValueError("worksite-business universe is empty")
    record_index = {
        str(row["employer_record_id"]): index for index, row in enumerate(rows)
    }
    aliases, alias_frequency = _aliases_for_records(rows)
    partitions = {
        tier: _new_partition(rows, aliases, tier) for tier in TIERS
    }
    audit: list[dict[str, object]] = []

    def attempt(
        left: int,
        right: int,
        rule: str,
        tiers: tuple[str, ...],
        *,
        score: float | None = None,
        threshold: float | None = None,
    ) -> None:
        if left == right:
            return
        for tier in tiers:
            decision, reason = _union(partitions[tier], left, right, rule)
            if collect_audit and decision != "redundant":
                audit.append(
                    _audit_row(
                        tier,
                        rule,
                        left,
                        right,
                        rows,
                        alias_frequency,
                        decision,
                        reason,
                        score,
                        threshold,
                    )
                )

    def reject_near_threshold(
        left: int,
        right: int,
        rule: str,
        tiers: tuple[str, ...],
        score: float,
        threshold: float,
    ) -> None:
        if not collect_audit or score < threshold - 3:
            return
        for tier in tiers:
            audit.append(
                _audit_row(
                    tier,
                    rule,
                    left,
                    right,
                    rows,
                    alias_frequency,
                    "rejected",
                    "score_below_threshold",
                    score,
                    threshold,
                )
            )

    all_tiers = TIERS
    balanced_tiers = ("balanced", "high_recall")

    fein_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        fein = str(row["normalized_fein"])
        if fein:
            fein_groups[fein].append(index)
    for fein in sorted(fein_groups):
        group = sorted(fein_groups[fein], key=lambda i: rows[i]["employer_record_id"])
        for index in group[1:]:
            attempt(group[0], index, "exact_fein", all_tiers, score=100, threshold=100)

    for contact_column, contact_label in (
        ("normalized_full_address", "address"),
        ("normalized_phone", "phone"),
    ):
        contact_groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            contact = str(row[contact_column])
            if contact:
                contact_groups[contact].append(index)
        for contact in sorted(contact_groups):
            group = sorted(
                contact_groups[contact], key=lambda i: rows[i]["employer_record_id"]
            )
            contact_names = sorted({alias for i in group for alias in aliases[i]})
            if len(contact_names) > SHARED_CONTACT_NAME_LIMIT:
                if collect_audit and len(group) > 1:
                    for tier in all_tiers:
                        audit.append(
                            _audit_row(
                                tier,
                                f"shared_exact_{contact_label}",
                                group[0],
                                group[1],
                                rows,
                                alias_frequency,
                                "rejected",
                                "shared_contact_more_than_ten_names",
                                None,
                                None,
                            )
                        )
                continue

            exact_groups: dict[str, list[int]] = defaultdict(list)
            for index in group:
                for alias in aliases[index]:
                    exact_groups[alias].append(index)
            for alias in sorted(exact_groups):
                exact_group = sorted(
                    set(exact_groups[alias]),
                    key=lambda i: rows[i]["employer_record_id"],
                )
                for index in exact_group[1:]:
                    uses_dba = alias in {
                        rows[exact_group[0]]["normalized_trade_name"],
                        rows[index]["normalized_trade_name"],
                    }
                    rule = (
                        f"exact_dba_exact_{contact_label}"
                        if uses_dba
                        else f"exact_name_exact_{contact_label}"
                    )
                    attempt(
                        exact_group[0], index, rule, all_tiers, score=100, threshold=100
                    )

            representatives: dict[tuple[str, str], int] = {}
            for index in group:
                fein = str(rows[index]["normalized_fein"])
                for alias in aliases[index]:
                    key = (alias, fein)
                    current = representatives.get(key)
                    if current is None or rows[index]["employer_record_id"] < rows[
                        current
                    ]["employer_record_id"]:
                        representatives[key] = index
            entries = sorted(
                ((alias, index) for (alias, _), index in representatives.items()),
                key=lambda item: (item[0], rows[item[1]]["employer_record_id"]),
            )
            for position, (left_alias, left) in enumerate(entries):
                for right_alias, right in entries[position + 1 :]:
                    if left == right or left_alias == right_alias:
                        continue
                    score = float(fuzz.WRatio(left_alias, right_alias))
                    rule = f"similar_name_exact_{contact_label}"
                    if score >= BALANCED_CONTACT_SIMILARITY:
                        attempt(
                            left,
                            right,
                            rule,
                            balanced_tiers,
                            score=score,
                            threshold=BALANCED_CONTACT_SIMILARITY,
                        )
                    else:
                        reject_near_threshold(
                            left,
                            right,
                            rule,
                            balanced_tiers,
                            score,
                            BALANCED_CONTACT_SIMILARITY,
                        )

    location_entries: list[tuple[str, str, str, str, int]] = []
    location_representatives: dict[tuple[str, str, str, str], int] = {}
    for index, row in enumerate(rows):
        state = str(row["normalized_state"])
        postal_code = str(row["normalized_postal_code"])
        fein = str(row["normalized_fein"])
        for alias in aliases[index]:
            key = (alias, state, postal_code, fein)
            current = location_representatives.get(key)
            if current is None or row["employer_record_id"] < rows[current][
                "employer_record_id"
            ]:
                location_representatives[key] = index
    for key, index in sorted(
        location_representatives.items(),
        key=lambda item: (*item[0], rows[item[1]]["employer_record_id"]),
    ):
        location_entries.append((*key, index))

    exact_zip_groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        state = str(row["normalized_state"])
        postal_code = str(row["normalized_postal_code"])
        if not state or not postal_code:
            continue
        for alias in aliases[index]:
            if is_valid_alias(alias):
                exact_zip_groups[(alias, state, postal_code)].append(index)
    for key in sorted(exact_zip_groups):
        group = sorted(
            set(exact_zip_groups[key]), key=lambda i: rows[i]["employer_record_id"]
        )
        for index in group[1:]:
            attempt(
                group[0],
                index,
                "exact_distinctive_name_state_zip",
                balanced_tiers,
                score=100,
                threshold=100,
            )

    zip_blocks: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for entry_index, (alias, state, postal_code, _, _) in enumerate(location_entries):
        if state and postal_code:
            for token in core_name_tokens(alias):
                zip_blocks[(state, postal_code, token)].append(entry_index)
    for entry_index, (alias, state, postal_code, _, left) in enumerate(
        location_entries
    ):
        if not state or not postal_code:
            continue
        candidates: set[int] = set()
        for token in core_name_tokens(alias):
            candidates.update(zip_blocks[(state, postal_code, token)])
        for candidate_index in sorted(i for i in candidates if i > entry_index):
            candidate_alias, _, _, _, right = location_entries[candidate_index]
            if left == right or alias == candidate_alias:
                continue
            if not (set(core_name_tokens(alias)) & set(core_name_tokens(candidate_alias))):
                continue
            score = float(fuzz.WRatio(alias, candidate_alias))
            if score >= BALANCED_STATE_ZIP_SIMILARITY:
                attempt(
                    left,
                    right,
                    "similar_name_state_zip_shared_core",
                    balanced_tiers,
                    score=score,
                    threshold=BALANCED_STATE_ZIP_SIMILARITY,
                )
            else:
                reject_near_threshold(
                    left,
                    right,
                    "similar_name_state_zip_shared_core",
                    balanced_tiers,
                    score,
                    BALANCED_STATE_ZIP_SIMILARITY,
                )

    active_membership = membership.filter(
        pl.col("employer_record_id").is_not_null()
        & (
            pl.col("employer_is_worksite_business")
            | pl.col("employer_is_alias_evidence_only")
        )
    )
    bridge_groups = active_membership.group_by("case_number", "fiscal_year").agg(
        pl.col("employer_record_id")
        .filter(pl.col("employer_is_alias_evidence_only"))
        .unique()
        .alias("alias_ids"),
        pl.col("employer_record_id")
        .filter(
            pl.col("employer_record_role") == "post2020_addendum_business"
        )
        .unique()
        .alias("business_ids"),
    )
    for bridge in bridge_groups.sort("case_number", "fiscal_year").to_dicts():
        for alias_id in sorted(bridge["alias_ids"]):
            left = record_index.get(alias_id)
            if left is None:
                continue
            for business_id in sorted(bridge["business_ids"]):
                right = record_index.get(business_id)
                if right is None or left == right:
                    continue
                candidate_scores = [
                    fuzz.WRatio(left_alias, right_alias)
                    for left_alias in aliases[left]
                    for right_alias in aliases[right]
                ]
                score = float(max(candidate_scores, default=0))
                if score >= BALANCED_SAME_CASE_SIMILARITY:
                    attempt(
                        left,
                        right,
                        "same_case_individual_master_addendum_bridge",
                        balanced_tiers,
                        score=score,
                        threshold=BALANCED_SAME_CASE_SIMILARITY,
                    )
                else:
                    reject_near_threshold(
                        left,
                        right,
                        "same_case_individual_master_addendum_bridge",
                        balanced_tiers,
                        score,
                        BALANCED_SAME_CASE_SIMILARITY,
                    )

    exact_state_groups: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for alias, state, postal_code, _, index in location_entries:
        if not state or not postal_code or not is_valid_alias(alias):
            continue
        key = (alias, state)
        current = exact_state_groups[key].get(postal_code)
        if current is None or rows[index]["employer_record_id"] < rows[current][
            "employer_record_id"
        ]:
            exact_state_groups[key][postal_code] = index
    for key in sorted(exact_state_groups):
        zip_representatives = [
            exact_state_groups[key][postal_code]
            for postal_code in sorted(exact_state_groups[key])
        ]
        for index in zip_representatives[1:]:
            attempt(
                zip_representatives[0],
                index,
                "exact_distinctive_name_state_cross_zip",
                ("high_recall",),
                score=100,
                threshold=100,
            )

    state_representatives: dict[tuple[str, str, str], int] = {}
    for alias, state, _, fein, index in location_entries:
        if not state:
            continue
        key = (alias, state, fein)
        current = state_representatives.get(key)
        if current is None or rows[index]["employer_record_id"] < rows[current][
            "employer_record_id"
        ]:
            state_representatives[key] = index
    state_entries = [
        (*key, index)
        for key, index in sorted(
            state_representatives.items(),
            key=lambda item: (*item[0], rows[item[1]]["employer_record_id"]),
        )
    ]
    state_blocks: dict[tuple[str, str], list[int]] = defaultdict(list)
    for entry_index, (alias, state, _, _) in enumerate(state_entries):
        for token in core_name_tokens(alias):
            state_blocks[(state, token)].append(entry_index)
    for entry_index, (alias, state, _, left) in enumerate(state_entries):
        candidates: set[int] = set()
        for token in core_name_tokens(alias):
            candidates.update(state_blocks[(state, token)])
        for candidate_index in sorted(i for i in candidates if i > entry_index):
            candidate_alias, _, _, right = state_entries[candidate_index]
            if left == right or alias == candidate_alias:
                continue
            score = float(fuzz.WRatio(alias, candidate_alias))
            if score >= HIGH_RECALL_STATE_SIMILARITY:
                attempt(
                    left,
                    right,
                    "similar_name_same_state_shared_core",
                    ("high_recall",),
                    score=score,
                    threshold=HIGH_RECALL_STATE_SIMILARITY,
                )
            else:
                reject_near_threshold(
                    left,
                    right,
                    "similar_name_same_state_shared_core",
                    ("high_recall",),
                    score,
                    HIGH_RECALL_STATE_SIMILARITY,
                )

    output = active
    cluster_rows: list[dict[str, object]] = []
    for tier in TIERS:
        partition = partitions[tier]
        roots = [_find(partition, index) for index in range(len(rows))]
        entity_ids = [
            "emp_wb_v2_"
            + tier
            + "_"
            + hashlib.sha256(
                str(partition["anchor"][root]).encode("utf-8")
            ).hexdigest()[:24]
            for root in roots
        ]
        member_groups: dict[int, list[int]] = defaultdict(list)
        for index, root in enumerate(roots):
            member_groups[root].append(index)
        cluster_sizes = [len(member_groups[root]) for root in roots]
        business_sizes = [
            sum(bool(rows[i]["included_business_ever"]) for i in member_groups[root])
            for root in roots
        ]
        evidence_values = [
            "|".join(sorted(partition["evidence"][root])) or "singleton"
            for root in roots
        ]
        output = output.with_columns(
            pl.Series(f"employer_id_{tier}", entity_ids, dtype=pl.String),
            pl.Series(
                f"employer_cluster_size_{tier}", cluster_sizes, dtype=pl.UInt32
            ),
            pl.Series(
                f"employer_business_count_{tier}", business_sizes, dtype=pl.UInt32
            ),
            pl.Series(
                f"employer_linkage_evidence_{tier}",
                evidence_values,
                dtype=pl.String,
            ),
        )
        for root, members in sorted(
            member_groups.items(), key=lambda item: partition["anchor"][item[0]]
        ):
            member_rows = [rows[index] for index in members]
            names = {alias for index in members for alias in aliases[index]}
            states = {
                str(row["normalized_state"])
                for row in member_rows
                if row["normalized_state"]
            }
            postal_codes = {
                str(row["normalized_postal_code"])
                for row in member_rows
                if row["normalized_postal_code"]
            }
            feins = {
                str(row["normalized_fein"])
                for row in member_rows
                if row["normalized_fein"]
            }
            entity_id = entity_ids[members[0]]
            cluster_rows.append(
                {
                    "linkage_tier": tier,
                    "employer_id": entity_id,
                    "anchor_employer_record_id": partition["anchor"][root],
                    "identity_record_count": len(members),
                    "included_business_record_count": sum(
                        bool(row["included_business_ever"]) for row in member_rows
                    ),
                    "alias_only_record_count": sum(
                        bool(row["alias_evidence_ever"])
                        and not bool(row["included_business_ever"])
                        for row in member_rows
                    ),
                    "distinct_name_count": len(names),
                    "state_count": len(states),
                    "postal_code_count": len(postal_codes),
                    "fein_count": len(feins),
                    "first_fiscal_year": min(
                        int(row["first_fiscal_year"]) for row in member_rows
                    ),
                    "last_fiscal_year": max(
                        int(row["last_fiscal_year"]) for row in member_rows
                    ),
                    "linkage_evidence": evidence_values[members[0]],
                    "large_cluster": len(members) >= 10,
                    "transitive_cluster": len(members) >= 3,
                    "cross_era": min(
                        int(row["first_fiscal_year"]) for row in member_rows
                    )
                    < 2020
                    <= max(int(row["last_fiscal_year"]) for row in member_rows),
                    "multi_state": len(states) > 1,
                }
            )

    edge_schema = {
        "edge_id": pl.String,
        "linkage_tier": pl.String,
        "linkage_rule": pl.String,
        "left_employer_record_id": pl.String,
        "right_employer_record_id": pl.String,
        "decision": pl.String,
        "rejection_reason": pl.String,
        "name_score": pl.Float64,
        "rule_threshold": pl.Float64,
        "data_regime": pl.String,
        "cross_state": pl.Boolean,
        "cross_zip": pl.Boolean,
        "common_name": pl.Boolean,
        "common_name_frequency": pl.Int64,
    }
    edge_audit = pl.DataFrame(audit, schema=edge_schema) if audit else pl.DataFrame(
        schema=edge_schema
    )
    if edge_audit.height:
        audit_key = [
            "linkage_tier",
            "linkage_rule",
            "left_employer_record_id",
            "right_employer_record_id",
            "decision",
            "rejection_reason",
        ]
        edge_audit = (
            edge_audit.unique(audit_key, maintain_order=True)
            .sort(audit_key)
        )
        if edge_audit["edge_id"].is_duplicated().any():
            raise ValueError("edge audit IDs are duplicated")
    clusters = pl.DataFrame(cluster_rows).sort("linkage_tier", "employer_id")
    return output.sort("employer_record_id"), edge_audit, clusters


def validate_linkage_partitions(linked: pl.DataFrame) -> None:
    """Enforce IDs, FEIN integrity, and exact partition nesting."""
    if linked["employer_record_id"].n_unique() != linked.height:
        raise ValueError("linked record keys are duplicated")
    for tier in TIERS:
        entity = f"employer_id_{tier}"
        if linked[entity].null_count():
            raise ValueError(f"active records are missing {entity}")
        conflicts = (
            linked.filter(pl.col("normalized_fein") != "")
            .group_by(entity)
            .agg(pl.col("normalized_fein").n_unique().alias("feins"))
            .filter(pl.col("feins") > 1)
        )
        if conflicts.height:
            raise ValueError(f"{entity} contains conflicting FEINs")
    for finer, coarser in (
        ("conservative", "balanced"),
        ("balanced", "high_recall"),
    ):
        violations = (
            linked.group_by(f"employer_id_{finer}")
            .agg(pl.col(f"employer_id_{coarser}").n_unique().alias("partitions"))
            .filter(pl.col("partitions") != 1)
        )
        if violations.height:
            raise ValueError(
                f"{finer} partition is not nested within {coarser}"
            )


def build_vintage_comparison(
    full_linked: pl.DataFrame, as_of_linked: pl.DataFrame
) -> pl.DataFrame:
    """Compare FY2008--2022 partitions under full and as-of-2022 evidence."""
    rows: list[pl.DataFrame] = []
    for tier in TIERS:
        full_id = f"employer_id_{tier}"
        joined = as_of_linked.select("employer_record_id", full_id).rename(
            {full_id: "as_of_2022_employer_id"}
        ).join(
            full_linked.select("employer_record_id", full_id).rename(
                {full_id: "full_vintage_employer_id"}
            ),
            on="employer_record_id",
            how="left",
            validate="1:1",
        )
        full_parts = joined.group_by("full_vintage_employer_id").agg(
            pl.col("as_of_2022_employer_id")
            .n_unique()
            .alias("as_of_parts_in_full_cluster")
        )
        asof_parts = joined.group_by("as_of_2022_employer_id").agg(
            pl.col("full_vintage_employer_id")
            .n_unique()
            .alias("full_parts_in_as_of_cluster")
        )
        joined = joined.join(
            full_parts, on="full_vintage_employer_id", how="left", validate="m:1"
        ).join(
            asof_parts, on="as_of_2022_employer_id", how="left", validate="m:1"
        ).with_columns(
            pl.lit(tier).alias("linkage_tier"),
            pl.when(
                (pl.col("as_of_parts_in_full_cluster") == 1)
                & (pl.col("full_parts_in_as_of_cluster") == 1)
            )
            .then(pl.lit("stable_partition"))
            .when(
                (pl.col("as_of_parts_in_full_cluster") > 1)
                & (pl.col("full_parts_in_as_of_cluster") == 1)
            )
            .then(pl.lit("full_vintage_merges_as_of_clusters"))
            .when(
                (pl.col("as_of_parts_in_full_cluster") == 1)
                & (pl.col("full_parts_in_as_of_cluster") > 1)
            )
            .then(pl.lit("full_vintage_splits_as_of_cluster"))
            .otherwise(pl.lit("nonmonotone_partition_change"))
            .alias("partition_comparison"),
        )
        rows.append(joined)
    return pl.concat(rows, how="vertical").sort("linkage_tier", "employer_record_id")


def _allocate_stratified_sample(
    candidates: list[dict[str, object]],
    count: int,
    sample_type: str,
) -> list[dict[str, object]]:
    if len(candidates) < count:
        raise ValueError(
            f"review sample requires {count} {sample_type} candidates; "
            f"only {len(candidates)} are available"
        )
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        priority = any(
            bool(candidate.get(flag, False))
            for flag in (
                "cross_era",
                "cross_state",
                "multi_state",
                "common_name",
                "large_cluster",
            )
        )
        stratum = "|".join(
            [
                str(candidate.get("linkage_rule", "cluster_check")),
                str(candidate.get("data_regime", "mixed")),
                "priority" if priority else "standard",
            ]
        )
        candidate["review_stratum"] = stratum
        groups[stratum].append(candidate)
    for stratum, group in groups.items():
        group.sort(
            key=lambda row: hashlib.sha256(
                f"{REVIEW_SEED}|{sample_type}|{stratum}|{row['candidate_key']}".encode()
            ).hexdigest()
        )

    allocation = {stratum: 0 for stratum in groups}
    ordered_strata = sorted(groups)
    for stratum in ordered_strata:
        if sum(allocation.values()) == count:
            break
        allocation[stratum] = 1
    while sum(allocation.values()) < count:
        available = [
            stratum
            for stratum in ordered_strata
            if allocation[stratum] < len(groups[stratum])
        ]
        if not available:
            break
        selected_stratum = max(
            available,
            key=lambda stratum: (
                math.sqrt(len(groups[stratum])) / (allocation[stratum] + 1),
                stratum,
            ),
        )
        allocation[selected_stratum] += 1

    selected: list[dict[str, object]] = []
    for stratum in ordered_strata:
        take = allocation[stratum]
        population = len(groups[stratum])
        for candidate in groups[stratum][:take]:
            candidate = dict(candidate)
            candidate["sample_probability"] = take / population
            candidate["sampling_weight"] = population / take
            selected.append(candidate)
    if len(selected) != count:
        raise ValueError(f"failed to allocate the {sample_type} sample")
    return selected


def build_review_sample(
    linked: pl.DataFrame,
    edge_audit: pl.DataFrame,
) -> pl.DataFrame:
    """Construct the seeded 140/40/20 manual-review sample."""
    record_lookup = {
        str(row["employer_record_id"]): row for row in linked.to_dicts()
    }
    cluster_state_counts = dict(
        linked.group_by("employer_id_balanced")
        .agg(
            pl.col("normalized_state")
            .filter(pl.col("normalized_state") != "")
            .n_unique()
            .alias("state_count")
        )
        .iter_rows()
    )

    def enrich_edge(row: dict[str, object], sample_type: str) -> dict[str, object]:
        left = record_lookup[str(row["left_employer_record_id"])]
        right = record_lookup[str(row["right_employer_record_id"])]
        cluster_size = max(
            int(left["employer_cluster_size_balanced"]),
            int(right["employer_cluster_size_balanced"]),
        )
        return {
            **row,
            "sample_type": sample_type,
            "candidate_key": str(row["edge_id"]),
            "cluster_id": str(left["employer_id_balanced"]),
            "cluster_size": cluster_size,
            "large_cluster": cluster_size >= 10,
            "cross_era": row["data_regime"] == "cross_era",
            "multi_state": max(
                cluster_state_counts[str(left["employer_id_balanced"])],
                cluster_state_counts[str(right["employer_id_balanced"])],
            )
            > 1,
        }

    accepted = [
        enrich_edge(row, "accepted_balanced_edge")
        for row in edge_audit.filter(
            (pl.col("linkage_tier") == "balanced")
            & (pl.col("decision") == "accepted")
        ).to_dicts()
    ]
    rejected = [
        enrich_edge(row, "near_threshold_rejected")
        for row in edge_audit.filter(
            (pl.col("linkage_tier") == "balanced")
            & (pl.col("decision") == "rejected")
            & (pl.col("rejection_reason") == "score_below_threshold")
        ).to_dicts()
    ]

    cluster_candidates: list[dict[str, object]] = []
    groups = linked.group_by("employer_id_balanced").agg(
        pl.col("employer_record_id").sort().alias("member_ids"),
        pl.len().alias("cluster_size"),
        pl.col("first_fiscal_year").min().alias("first_fiscal_year"),
        pl.col("last_fiscal_year").max().alias("last_fiscal_year"),
        pl.col("normalized_state").filter(pl.col("normalized_state") != "").n_unique().alias("state_count"),
    ).filter(pl.col("cluster_size") >= 3)
    for group in groups.sort("employer_id_balanced").to_dicts():
        anchor_id = group["member_ids"][0]
        remote_id = group["member_ids"][-1]
        anchor = record_lookup[anchor_id]
        remote = record_lookup[remote_id]
        if anchor_id == remote_id:
            continue
        cluster_candidates.append(
            {
                "edge_id": _stable_hash(
                    "wbc_", ("balanced", anchor_id, remote_id)
                ),
                "linkage_tier": "balanced",
                "linkage_rule": "anchor_remote_cluster_check",
                "left_employer_record_id": anchor_id,
                "right_employer_record_id": remote_id,
                "decision": "accepted_cluster_membership",
                "rejection_reason": "",
                "name_score": float(
                    fuzz.WRatio(anchor["normalized_name"], remote["normalized_name"])
                ),
                "rule_threshold": None,
                "data_regime": _edge_regime(anchor, remote),
                "cross_state": bool(
                    anchor["normalized_state"]
                    and remote["normalized_state"]
                    and anchor["normalized_state"] != remote["normalized_state"]
                ),
                "cross_zip": bool(
                    anchor["normalized_postal_code"]
                    and remote["normalized_postal_code"]
                    and anchor["normalized_postal_code"]
                    != remote["normalized_postal_code"]
                ),
                "common_name": False,
                "common_name_frequency": 0,
                "sample_type": "anchor_remote_cluster_check",
                "candidate_key": _stable_hash("wbcand_", (anchor_id, remote_id)),
                "cluster_id": group["employer_id_balanced"],
                "cluster_size": int(group["cluster_size"]),
                "large_cluster": int(group["cluster_size"]) >= 10,
                "cross_era": int(group["first_fiscal_year"]) < 2020 <= int(group["last_fiscal_year"]),
                "multi_state": int(group["state_count"]) > 1,
            }
        )

    selected = [
        *_allocate_stratified_sample(
            accepted, REVIEW_ACCEPTED_COUNT, "accepted_balanced_edge"
        ),
        *_allocate_stratified_sample(
            rejected, REVIEW_REJECTED_COUNT, "near_threshold_rejected"
        ),
        *_allocate_stratified_sample(
            cluster_candidates,
            REVIEW_CLUSTER_COUNT,
            "anchor_remote_cluster_check",
        ),
    ]
    review_rows: list[dict[str, object]] = []
    for candidate in selected:
        left = record_lookup[str(candidate["left_employer_record_id"])]
        right = record_lookup[str(candidate["right_employer_record_id"])]
        pair_id = _stable_hash(
            "wbrev_",
            (
                REVIEW_SEED,
                candidate["sample_type"],
                candidate["candidate_key"],
            ),
        )
        review_rows.append(
            {
                "pair_id": pair_id,
                "review_seed": REVIEW_SEED,
                "sample_type": candidate["sample_type"],
                "review_stratum": candidate["review_stratum"],
                "sampling_weight": candidate["sampling_weight"],
                "sample_probability": candidate["sample_probability"],
                "linkage_rule": candidate["linkage_rule"],
                "edge_decision": candidate["decision"],
                "edge_rejection_reason": candidate["rejection_reason"],
                "name_score": candidate["name_score"],
                "rule_threshold": candidate["rule_threshold"],
                "data_regime": candidate["data_regime"],
                "cross_era": candidate["data_regime"] == "cross_era",
                "cross_state": candidate["cross_state"],
                "cross_zip": candidate["cross_zip"],
                "common_name": candidate["common_name"],
                "large_cluster": candidate["large_cluster"],
                "multi_state": candidate.get("multi_state", False),
                "cluster_id": candidate["cluster_id"],
                "cluster_size": candidate["cluster_size"],
                "left_employer_record_id": candidate["left_employer_record_id"],
                "left_name": left["source_name_raw"],
                "left_trade_name": left["source_trade_name_raw"],
                "left_address": left["source_address_1_raw"],
                "left_city": left["source_city_raw"],
                "left_state": left["source_state_raw"],
                "left_postal_code": left["source_postal_code_raw"],
                "left_phone": left["source_phone_raw"],
                "left_fein": left["source_fein_raw"],
                "left_first_fiscal_year": left["first_fiscal_year"],
                "left_last_fiscal_year": left["last_fiscal_year"],
                "right_employer_record_id": candidate["right_employer_record_id"],
                "right_name": right["source_name_raw"],
                "right_trade_name": right["source_trade_name_raw"],
                "right_address": right["source_address_1_raw"],
                "right_city": right["source_city_raw"],
                "right_state": right["source_state_raw"],
                "right_postal_code": right["source_postal_code_raw"],
                "right_phone": right["source_phone_raw"],
                "right_fein": right["source_fein_raw"],
                "right_first_fiscal_year": right["first_fiscal_year"],
                "right_last_fiscal_year": right["last_fiscal_year"],
            }
        )
    result = pl.DataFrame(review_rows).sort("sample_type", "review_stratum", "pair_id")
    expected_counts = {
        "accepted_balanced_edge": REVIEW_ACCEPTED_COUNT,
        "near_threshold_rejected": REVIEW_REJECTED_COUNT,
        "anchor_remote_cluster_check": REVIEW_CLUSTER_COUNT,
    }
    observed_counts = dict(
        result.group_by("sample_type").len().iter_rows()
    )
    if observed_counts != expected_counts or result["pair_id"].n_unique() != 200:
        raise ValueError("manual-review sample does not satisfy 140/40/20")
    return result


def validate_review_labels(
    review_sample: pl.DataFrame, labels: pl.DataFrame
) -> pl.DataFrame:
    """Validate and attach complete true/false/uncertain manual labels."""
    require_columns(labels, {"pair_id", "same_business"}, "review labels")
    labels = labels.select(
        pl.col("pair_id").cast(pl.String).str.strip_chars().fill_null(""),
        pl.col("same_business")
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
        .fill_null(""),
    )
    if labels["pair_id"].is_duplicated().any():
        raise ValueError("review labels contain duplicate pair IDs")
    valid_values = {"true", "false", "uncertain"}
    invalid = labels.filter(~pl.col("same_business").is_in(valid_values))
    if invalid.height:
        raise ValueError(
            "same_business labels must be true, false, or uncertain"
        )
    expected = set(review_sample["pair_id"].to_list())
    observed = set(labels["pair_id"].to_list())
    if expected != observed:
        raise ValueError(
            "review labels must contain every sampled pair exactly once; "
            f"missing={len(expected - observed)}, unexpected={len(observed - expected)}"
        )
    return review_sample.join(labels, on="pair_id", how="left", validate="1:1")


def build_review_report(labeled_sample: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    """Report weighted review metrics without applying an empirical gate."""
    labeled = labeled_sample.with_columns(
        (pl.col("same_business") == "true").alias("label_positive"),
        (pl.col("same_business") != "true").alias("label_negative_or_uncertain"),
    )
    rows: list[dict[str, object]] = []

    def add_metric(
        frame: pl.DataFrame, group: str, metric: str, stratum: str, value: pl.Expr
    ) -> None:
        if frame.is_empty():
            return
        summary = frame.select(
            pl.len().alias("sample_count"),
            pl.col("sampling_weight").sum().alias("weight_sum"),
            (
                (value.cast(pl.Float64) * pl.col("sampling_weight")).sum()
                / pl.col("sampling_weight").sum()
            ).alias("estimate"),
        ).row(0, named=True)
        rows.append(
            {
                "metric_group": group,
                "metric": metric,
                "stratum": stratum,
                **summary,
            }
        )

    accepted = labeled.filter(pl.col("sample_type") == "accepted_balanced_edge")
    add_metric(
        accepted,
        "accepted_edges",
        "weighted_precision_uncertain_is_false",
        "all",
        pl.col("label_positive"),
    )
    for column in ("linkage_rule", "data_regime"):
        for value in sorted(accepted[column].unique().to_list()):
            add_metric(
                accepted.filter(pl.col(column) == value),
                "accepted_edges",
                "weighted_precision_uncertain_is_false",
                f"{column}={value}",
                pl.col("label_positive"),
            )
    near_miss = labeled.filter(pl.col("sample_type") == "near_threshold_rejected")
    add_metric(
        near_miss,
        "rejected_candidates",
        "weighted_near_miss_yield",
        "all",
        pl.col("label_positive"),
    )
    cluster_checks = labeled.filter(
        pl.col("sample_type") == "anchor_remote_cluster_check"
    )
    add_metric(
        cluster_checks,
        "cluster_checks",
        "weighted_cluster_check_failure_rate",
        "all",
        pl.col("label_negative_or_uncertain"),
    )
    report = pl.DataFrame(rows).sort("metric_group", "metric", "stratum")
    report_hash = hashlib.sha256(report.write_csv().encode("utf-8")).hexdigest()
    return report, report_hash


def _add_provenance(
    frame: pl.DataFrame, review_report_hash: str
) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(LINKAGE_UNIT).alias("employer_linkage_unit"),
        pl.lit(LINKAGE_VERSION).alias("employer_linkage_version"),
        pl.lit(INPUT_MAX_YEAR, dtype=pl.Int32).alias(
            "employer_linkage_input_max_year"
        ),
        pl.lit("complete").alias("employer_linkage_review_status"),
        pl.lit(review_report_hash).alias("employer_linkage_review_report_hash"),
    )


def finalize_linkage_artifacts(
    artifacts: dict[str, pl.DataFrame], labels: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    """Validate labels, create the report, and stamp promotable artifacts."""
    labeled_sample = validate_review_labels(artifacts["review_sample"], labels)
    report, report_hash = build_review_report(labeled_sample)
    report_key = ["metric_group", "metric", "stratum"]
    if report.is_empty() or report.select(report_key).is_duplicated().any():
        raise ValueError("manual-review report has invalid keys")
    finalized = dict(artifacts)
    finalized["review_sample"] = labeled_sample
    finalized["review_report"] = report
    for name in (
        "crosswalk",
        "membership",
        "edge_audit",
        "cluster_diagnostics",
        "vintage_comparison",
    ):
        finalized[name] = _add_provenance(finalized[name], report_hash)
    return finalized


def build_linkage_artifacts(
    h2a: pl.DataFrame,
    addendum: pl.DataFrame,
    *,
    verify_determinism: bool = True,
) -> dict[str, pl.DataFrame]:
    """Build all candidate v2 artifacts except the manual review report."""
    membership, source_diagnostics = build_membership_records(h2a, addendum)
    records, membership = build_identity_records(membership)
    linked, edge_audit, cluster_diagnostics = link_employer_records(
        records, membership
    )
    validate_linkage_partitions(linked)
    if verify_determinism:
        shuffled = records.sample(fraction=1, shuffle=True, seed=REVIEW_SEED)
        relinked, _, _ = link_employer_records(
            shuffled, membership, collect_audit=False
        )
        columns = ["employer_record_id", *(f"employer_id_{tier}" for tier in TIERS)]
        if not linked.select(columns).sort("employer_record_id").equals(
            relinked.select(columns).sort("employer_record_id")
        ):
            raise ValueError("linkage IDs are not shuffle deterministic")

    id_columns = [
        "employer_record_id",
        *(
            column
            for tier in TIERS
            for column in (
                f"employer_id_{tier}",
                f"employer_cluster_size_{tier}",
                f"employer_business_count_{tier}",
                f"employer_linkage_evidence_{tier}",
            )
        ),
    ]
    records_with_ids = records.join(
        linked.select(id_columns),
        on="employer_record_id",
        how="left",
        validate="1:1",
    )
    membership_with_ids = membership.join(
        linked.select(id_columns),
        on="employer_record_id",
        how="left",
        validate="m:1",
    )
    missing_named_included = membership_with_ids.filter(
        pl.col("employer_is_worksite_business")
        & pl.col("source_name_raw").str.strip_chars().ne("")
        & pl.col("employer_id_balanced").is_null()
    )
    if missing_named_included.height:
        raise ValueError("named included businesses are missing linkage IDs")

    asof_membership = membership.filter(pl.col("fiscal_year") <= AS_OF_YEAR)
    asof_records, asof_membership = build_identity_records(asof_membership)
    asof_linked, _, _ = link_employer_records(
        asof_records, asof_membership, collect_audit=False
    )
    validate_linkage_partitions(asof_linked)
    vintage_comparison = build_vintage_comparison(linked, asof_linked)
    review_sample = build_review_sample(linked, edge_audit)
    artifacts = {
        "crosswalk": records_with_ids.sort("employer_record_id"),
        "membership": membership_with_ids.sort("source_membership_id"),
        "edge_audit": edge_audit,
        "cluster_diagnostics": cluster_diagnostics,
        "vintage_comparison": vintage_comparison,
        "review_sample": review_sample,
        "source_diagnostics": source_diagnostics,
    }
    artifact_keys = {
        "crosswalk": ["employer_record_id"],
        "membership": ["source_membership_id"],
        "edge_audit": ["edge_id"],
        "cluster_diagnostics": ["linkage_tier", "employer_id"],
        "vintage_comparison": ["linkage_tier", "employer_record_id"],
        "review_sample": ["pair_id"],
        "source_diagnostics": ["source_dataset"],
    }
    for name, keys in artifact_keys.items():
        frame = artifacts[name]
        if frame.is_empty():
            raise ValueError(f"{name} artifact is empty")
        if frame.select(keys).is_duplicated().any():
            raise ValueError(f"{name} artifact keys are duplicated")
        if frame.select(
            pl.any_horizontal(pl.col(key).is_null() for key in keys).any()
        ).item():
            raise ValueError(f"{name} artifact keys contain nulls")
    return artifacts


def read_review_labels(path: Path) -> pl.DataFrame:
    """Read labels or raise the promotion-blocking missing-label error."""
    if not path.exists():
        raise FileNotFoundError(
            "manual review is incomplete: label every pair in "
            "data/intermediate/h2a_employer_linkage_review_sample.csv and save "
            "pair_id,same_business to "
            "data/raw/h2a/employer_linkage_review_labels.csv"
        )
    return pl.read_csv(path, infer_schema_length=1000)
