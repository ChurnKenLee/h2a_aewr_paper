# Purpose: Link H-2A worksite businesses into alternative employer entities.
# Inputs: h2a_with_fips.parquet and h2a_addendum_b_with_fips.parquet.
# Output: data/intermediate/h2a_employer_crosswalk.parquet.

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import hashlib
    import json
    import re
    import unicodedata
    from collections import Counter, defaultdict
    from functools import lru_cache

    import marimo as mo
    import polars as pl
    import us
    from rapidfuzz import fuzz

    from h2a.paths import INTERMEDIATE

    return (
        Counter,
        INTERMEDIATE,
        defaultdict,
        fuzz,
        hashlib,
        json,
        lru_cache,
        mo,
        pl,
        re,
        unicodedata,
        us,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Match H-2A employers

    This notebook constructs three nested, national longitudinal employer
    identifiers. It never rewrites the two input Parquet files; their exact raw
    identity fields are retained in the crosswalk as many-to-one join keys.
    """)


@app.cell
def _():
    H2A_SOURCE = "h2a_with_fips"
    ADDENDUM_SOURCE = "h2a_addendum_b_with_fips"

    RAW_IDENTITY_COLUMNS = [
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
    ]

    GENERIC_NAME_TOKENS = {
        "AG",
        "AGRICULTURAL",
        "AGRICULTURE",
        "AND",
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

    PLACEHOLDER_ADDRESSES = {
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

    return (
        ADDENDUM_SOURCE,
        GENERIC_NAME_TOKENS,
        H2A_SOURCE,
        PLACEHOLDER_ADDRESSES,
        RAW_IDENTITY_COLUMNS,
    )


@app.cell
def _(
    GENERIC_NAME_TOKENS,
    PLACEHOLDER_ADDRESSES,
    RAW_IDENTITY_COLUMNS,
    hashlib,
    json,
    lru_cache,
    re,
    unicodedata,
    us,
):
    def normalize_text(value):
        if value is None:
            return ""
        text = unicodedata.normalize("NFKD", str(value))
        text = text.encode("ascii", "ignore").decode("ascii").upper().strip()
        text = text.replace("&", " AND ")
        text = re.sub(r"[^A-Z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def normalize_name(value):
        return normalize_text(value)

    def normalize_zip(value):
        if value is None:
            return ""
        text = str(value).strip().removesuffix(".0")
        digits = re.sub(r"\D", "", text)
        if len(digits) < 4:
            return ""
        return digits[:5].zfill(5)

    def normalize_phone(value):
        digits = re.sub(r"\D", "", "" if value is None else str(value))
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) != 10 or len(set(digits)) == 1:
            return ""
        return digits

    def normalize_fein(value):
        if value is None:
            return ""
        text = str(value).strip().removesuffix(".0")
        digits = re.sub(r"\D", "", text)
        if len(digits) != 9 or len(set(digits)) == 1:
            return ""
        return digits

    @lru_cache(maxsize=None)
    def normalize_state(value):
        text = normalize_text(value)
        if not text:
            return ""
        if text in {"DC", "DISTRICT OF COLUMBIA"}:
            return "DC"
        state = us.states.lookup(text)
        return "" if state is None else state.abbr

    def valid_full_address(row):
        address_1 = row["normalized_address_1"]
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
                row["normalized_address_2"],
                row["normalized_city"],
                row["normalized_state"],
                row["normalized_postal_code"],
            ]
        )

    def core_name_tokens(name):
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

    def blocking_ngrams(name):
        core = "".join(core_name_tokens(name))
        compact = core or re.sub(r"[^A-Z0-9]", "", name)
        if len(compact) < 4:
            return ()
        return tuple(
            sorted({compact[index : index + 3] for index in range(len(compact) - 2)})
        )

    def raw_identity_fingerprint(row):
        payload = [row[column] for column in RAW_IDENTITY_COLUMNS]
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return "er_" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]

    return (
        blocking_ngrams,
        core_name_tokens,
        normalize_fein,
        normalize_name,
        normalize_phone,
        normalize_state,
        normalize_text,
        normalize_zip,
        raw_identity_fingerprint,
        valid_full_address,
    )


@app.cell
def _(
    ADDENDUM_SOURCE,
    H2A_SOURCE,
    RAW_IDENTITY_COLUMNS,
    normalize_fein,
    normalize_name,
    normalize_phone,
    normalize_state,
    normalize_text,
    normalize_zip,
    pl,
    raw_identity_fingerprint,
    valid_full_address,
):
    def require_columns(frame, required, label):
        missing = sorted(set(required).difference(frame.columns))
        if missing:
            raise ValueError(
                f"{label} is missing required columns: {', '.join(missing)}"
            )

    def string_column(source, target=None):
        return pl.col(source).cast(pl.String).fill_null("").alias(target or source)

    def build_identity_records(h2a, addendum):
        h2a_required = {
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
        }
        addendum_required = {
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

        h2a_rows = h2a.select(
            pl.lit(H2A_SOURCE).alias("source_dataset"),
            pl.col("fiscal_year").cast(pl.Int32),
            string_column("employer_name", "source_name_raw"),
            string_column("trade_name_dba", "source_trade_name_raw"),
            string_column("employer_address_1", "source_address_1_raw"),
            string_column("employer_address_2", "source_address_2_raw"),
            string_column("employer_city", "source_city_raw"),
            string_column("employer_state", "source_state_raw"),
            string_column("employer_postal_code", "source_postal_code_raw"),
            string_column("employer_phone", "source_phone_raw"),
            string_column("employer_fein", "source_fein_raw"),
        )
        addendum_rows = addendum.select(
            pl.lit(ADDENDUM_SOURCE).alias("source_dataset"),
            pl.col("fiscal_year").cast(pl.Int32),
            string_column("business_name", "source_name_raw"),
            pl.lit("").alias("source_trade_name_raw"),
            string_column("worksite_address_1", "source_address_1_raw"),
            string_column("worksite_address_2", "source_address_2_raw"),
            string_column("worksite_city", "source_city_raw"),
            string_column("worksite_state", "source_state_raw"),
            string_column("worksite_zip", "source_postal_code_raw"),
            pl.lit("").alias("source_phone_raw"),
            pl.lit("").alias("source_fein_raw"),
        )
        raw_rows = pl.concat([h2a_rows, addendum_rows], how="vertical")
        raw_rows = raw_rows.with_columns(
            pl.col("source_name_raw").str.strip_chars().ne("").alias("has_name")
        )

        source_diagnostics = (
            raw_rows.group_by("source_dataset")
            .agg(
                pl.len().alias("source_rows"),
                pl.col("has_name").sum().alias("rows_with_name"),
                (~pl.col("has_name")).sum().alias("blank_name_rows"),
            )
            .sort("source_dataset")
        )

        records = (
            raw_rows.filter("has_name")
            .drop("has_name")
            .group_by(RAW_IDENTITY_COLUMNS)
            .agg(
                pl.col("fiscal_year").min().alias("first_fiscal_year"),
                pl.col("fiscal_year").max().alias("last_fiscal_year"),
                pl.len().alias("source_row_count"),
            )
            .with_columns(
                pl.struct(RAW_IDENTITY_COLUMNS)
                .map_elements(raw_identity_fingerprint, return_dtype=pl.String)
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

        if records.get_column("employer_record_id").n_unique() != records.height:
            raise AssertionError("employer_record_id contains a hash collision")
        if records.select(RAW_IDENTITY_COLUMNS).is_duplicated().any():
            raise AssertionError(
                "Raw employer join keys must be unique in the crosswalk"
            )
        if (
            records.get_column("source_row_count").sum()
            != raw_rows.filter("has_name").height
        ):
            raise AssertionError(
                "Employer identity collapse changed source-row coverage"
            )

        source_diagnostics = source_diagnostics.join(
            records.group_by("source_dataset").agg(pl.len().alias("identity_records")),
            on="source_dataset",
            how="left",
            validate="1:1",
        )
        return records, source_diagnostics

    return (build_identity_records,)


@app.cell
def _(
    Counter,
    blocking_ngrams,
    core_name_tokens,
    defaultdict,
    fuzz,
    pl,
):
    class DisjointSet:
        def __init__(self, record_ids, feins, aliases, max_fuzzy_names):
            self.parent = list(range(len(record_ids)))
            self.size = [1] * len(record_ids)
            self.anchor = list(record_ids)
            self.fein = [value or "" for value in feins]
            self.names = [set(value) for value in aliases]
            self.max_fuzzy_names = max_fuzzy_names
            self.accepted = Counter()
            self.rejected_fein_conflicts = Counter()
            self.rejected_name_sprawl = Counter()
            self.evidence = {}

        def find(self, value):
            root = value
            while self.parent[root] != root:
                root = self.parent[root]
            while self.parent[value] != value:
                parent = self.parent[value]
                self.parent[value] = root
                value = parent
            return root

        def union(self, left, right, evidence):
            left_root = self.find(left)
            right_root = self.find(right)
            if left_root == right_root:
                return False

            left_fein = self.fein[left_root]
            right_fein = self.fein[right_root]
            if left_fein and right_fein and left_fein != right_fein:
                self.rejected_fein_conflicts[evidence] += 1
                return False

            combined_names = self.names[left_root] | self.names[right_root]
            if (evidence.startswith("fuzzy_") or evidence == "exact_dba") and len(
                combined_names
            ) > self.max_fuzzy_names:
                self.rejected_name_sprawl[evidence] += 1
                return False

            if self.size[left_root] < self.size[right_root] or (
                self.size[left_root] == self.size[right_root]
                and self.anchor[left_root] > self.anchor[right_root]
            ):
                left_root, right_root = right_root, left_root

            self.parent[right_root] = left_root
            self.size[left_root] += self.size[right_root]
            self.anchor[left_root] = min(
                self.anchor[left_root], self.anchor[right_root]
            )
            self.fein[left_root] = left_fein or right_fein
            self.names[left_root] = combined_names
            self.names[right_root] = set()
            combined_evidence = self.evidence.pop(left_root, set())
            combined_evidence.update(self.evidence.pop(right_root, set()))
            combined_evidence.add(evidence)
            self.evidence[left_root] = combined_evidence
            self.accepted[evidence] += 1
            return True

        def output_columns(self, method):
            roots = [self.find(index) for index in range(len(self.parent))]
            entity_ids = [
                f"emp_{method}_{self.anchor[root].removeprefix('er_')}"
                for root in roots
            ]
            cluster_sizes = [self.size[root] for root in roots]
            evidence = [
                "|".join(sorted(self.evidence.get(root, {"singleton"})))
                for root in roots
            ]
            return entity_ids, cluster_sizes, evidence

    def link_employer_records(records):
        rows = records.select(
            "employer_record_id",
            "normalized_name",
            "normalized_trade_name",
            "normalized_full_address",
            "normalized_phone",
            "normalized_fein",
            "normalized_state",
            "normalized_postal_code",
        ).to_dicts()
        record_ids = [row["employer_record_id"] for row in rows]
        feins = [row["normalized_fein"] for row in rows]
        invalid_trade_names = {
            "N A",
            "NA",
            "NIL",
            "NO DBA",
            "NO D B A",
            "NONE",
            "NONE USED",
            "NOT APPLICABLE",
            "SAME",
            "UNKNOWN",
        }
        trade_name_primary_names = defaultdict(set)
        for row in rows:
            trade_name = row["normalized_trade_name"]
            if trade_name:
                trade_name_primary_names[trade_name].add(row["normalized_name"])
        valid_trade_names = {
            trade_name
            for trade_name, primary_names in trade_name_primary_names.items()
            if trade_name not in invalid_trade_names and len(primary_names) <= 10
        }
        aliases = [
            tuple(
                dict.fromkeys(
                    [
                        row["normalized_name"],
                        *(
                            [row["normalized_trade_name"]]
                            if row["normalized_trade_name"] in valid_trade_names
                            else []
                        ),
                    ]
                )
            )
            for row in rows
        ]
        methods = {
            "conservative": DisjointSet(record_ids, feins, aliases, 25),
            "balanced": DisjointSet(record_ids, feins, aliases, 100),
            "high_recall": DisjointSet(record_ids, feins, aliases, 500),
        }

        def union_methods(left, right, evidence, selected_methods):
            for method in selected_methods:
                methods[method].union(left, right, evidence)

        all_methods = ("conservative", "balanced", "high_recall")
        balanced_methods = ("balanced", "high_recall")
        skipped_shared_contacts = Counter()

        fein_groups = defaultdict(list)
        for index, fein in enumerate(feins):
            if fein:
                fein_groups[fein].append(index)
        for fein in sorted(fein_groups):
            group = fein_groups[fein]
            for index in group[1:]:
                union_methods(group[0], index, "exact_fein", all_methods)

        primary_name_groups = defaultdict(list)
        trade_name_groups = defaultdict(list)
        for index, row in enumerate(rows):
            primary_name_groups[row["normalized_name"]].append(index)
            if row["normalized_trade_name"] in valid_trade_names:
                trade_name_groups[row["normalized_trade_name"]].append(index)

        def union_exact_group(group, evidence):
            known_feins = {feins[index] for index in group if feins[index]}
            if len(known_feins) <= 1:
                for index in group[1:]:
                    union_methods(group[0], index, evidence, all_methods)
                return len(known_feins) > 1

            partitions = defaultdict(list)
            for index in group:
                partitions[feins[index]].append(index)
            for partition in sorted(partitions):
                partition_group = partitions[partition]
                for index in partition_group[1:]:
                    union_methods(partition_group[0], index, evidence, all_methods)
            return True

        ambiguous_exact_names = 0
        for primary_name in sorted(primary_name_groups):
            group = sorted(set(primary_name_groups[primary_name]))
            ambiguous_exact_names += union_exact_group(group, "exact_name")

        for trade_name in sorted(trade_name_groups):
            group = sorted(
                set(trade_name_groups[trade_name] + primary_name_groups[trade_name])
            )
            union_exact_group(group, "exact_dba")

        def corroborated_groups(column):
            groups = defaultdict(list)
            for index, row in enumerate(rows):
                value = row[column]
                if value:
                    groups[value].append(index)
            return groups

        def alias_representatives(group):
            representatives = {}
            for index in group:
                for alias in aliases[index]:
                    key = (alias, feins[index])
                    current = representatives.get(key)
                    if current is None or record_ids[index] < record_ids[current]:
                        representatives[key] = index
            return sorted(
                ((alias, index) for (alias, _), index in representatives.items()),
                key=lambda item: (item[0], record_ids[item[1]]),
            )

        for column, evidence_name in [
            ("normalized_phone", "fuzzy_name_exact_phone"),
            ("normalized_full_address", "fuzzy_name_exact_address"),
        ]:
            groups = corroborated_groups(column)
            for group_key in sorted(groups):
                representatives = alias_representatives(groups[group_key])
                # Filing agents and shared administrative worksites can supply
                # one phone/address for hundreds of unrelated growers. Such
                # identifiers are not employer evidence. Moderately reused
                # contacts are allowed only in the less conservative tiers.
                if len(representatives) > 25:
                    skipped_shared_contacts[evidence_name] += 1
                    continue
                for left_position, (left_name, left_index) in enumerate(
                    representatives
                ):
                    for right_name, right_index in representatives[left_position + 1 :]:
                        score = fuzz.WRatio(left_name, right_name)
                        if score >= 95 and len(representatives) <= 10:
                            union_methods(
                                left_index, right_index, evidence_name, all_methods
                            )
                        elif score >= 92:
                            union_methods(
                                left_index,
                                right_index,
                                evidence_name,
                                balanced_methods,
                            )

        location_entries = {}
        for index, row in enumerate(rows):
            for alias in aliases[index]:
                key = (
                    alias,
                    row["normalized_state"],
                    row["normalized_postal_code"],
                    feins[index],
                )
                current = location_entries.get(key)
                if current is None or record_ids[index] < record_ids[current]:
                    location_entries[key] = index
        location_entries = [
            (*key, index)
            for key, index in sorted(
                location_entries.items(),
                key=lambda item: (*item[0], record_ids[item[1]]),
            )
        ]

        zip_blocks = defaultdict(list)
        for entry_index, (alias, state, postal_code, _, _) in enumerate(
            location_entries
        ):
            if not state or not postal_code:
                continue
            for ngram in blocking_ngrams(alias):
                zip_blocks[(state, postal_code, ngram)].append(entry_index)

        for entry_index, (alias, state, postal_code, _, record_index) in enumerate(
            location_entries
        ):
            if not state or not postal_code:
                continue
            candidates = set()
            for ngram in blocking_ngrams(alias):
                candidates.update(zip_blocks[(state, postal_code, ngram)])
            for candidate_index in sorted(
                value for value in candidates if value > entry_index
            ):
                candidate_alias, _, _, _, candidate_record = location_entries[
                    candidate_index
                ]
                if fuzz.WRatio(alias, candidate_alias) >= 96:
                    union_methods(
                        record_index,
                        candidate_record,
                        "fuzzy_name_same_state_zip",
                        balanced_methods,
                    )

        state_entries = {}
        for alias, state, _, fein, record_index in location_entries:
            if not state:
                continue
            key = (alias, state, fein)
            current = state_entries.get(key)
            if current is None or record_ids[record_index] < record_ids[current]:
                state_entries[key] = record_index
        state_entries = [
            (*key, index)
            for key, index in sorted(
                state_entries.items(),
                key=lambda item: (*item[0], record_ids[item[1]]),
            )
        ]

        state_token_blocks = defaultdict(list)
        for entry_index, (alias, state, _, _) in enumerate(state_entries):
            for token in core_name_tokens(alias):
                state_token_blocks[(state, token)].append(entry_index)

        for entry_index, (alias, state, _, record_index) in enumerate(state_entries):
            candidates = set()
            for token in core_name_tokens(alias):
                candidates.update(state_token_blocks[(state, token)])
            for candidate_index in sorted(
                value for value in candidates if value > entry_index
            ):
                candidate_alias, _, _, candidate_record = state_entries[candidate_index]
                if fuzz.WRatio(alias, candidate_alias) >= 94:
                    union_methods(
                        record_index,
                        candidate_record,
                        "fuzzy_name_same_state",
                        ("high_recall",),
                    )

        output = records
        cluster_counts = {}
        accepted_edges = {}
        rejected_conflicts = {}
        rejected_name_sprawl = {}
        for method, disjoint_set in methods.items():
            entity_ids, cluster_sizes, evidence = disjoint_set.output_columns(method)
            output = output.with_columns(
                pl.Series(f"employer_id_{method}", entity_ids, dtype=pl.String),
                pl.Series(
                    f"employer_cluster_size_{method}",
                    cluster_sizes,
                    dtype=pl.UInt32,
                ),
                pl.Series(
                    f"employer_linkage_evidence_{method}",
                    evidence,
                    dtype=pl.String,
                ),
            )
            cluster_counts[method] = len(set(entity_ids))
            accepted_edges[method] = dict(sorted(disjoint_set.accepted.items()))
            rejected_conflicts[method] = dict(
                sorted(disjoint_set.rejected_fein_conflicts.items())
            )
            rejected_name_sprawl[method] = dict(
                sorted(disjoint_set.rejected_name_sprawl.items())
            )

        diagnostics = {
            "identity_records": records.height,
            "ambiguous_exact_names": ambiguous_exact_names,
            "excluded_shared_or_placeholder_trade_names": len(
                trade_name_primary_names.keys() - valid_trade_names
            ),
            "cluster_counts": cluster_counts,
            "accepted_edges": accepted_edges,
            "rejected_fein_conflicts": rejected_conflicts,
            "rejected_name_sprawl": rejected_name_sprawl,
            "skipped_shared_contacts": dict(sorted(skipped_shared_contacts.items())),
        }
        return output, diagnostics

    def run_synthetic_checks():
        synthetic = pl.DataFrame(
            {
                "employer_record_id": [f"er_{index:024x}" for index in range(14)],
                "normalized_name": [
                    "ACME ORCHARDS LLC",
                    "ACME ORCHARD LLC",
                    "SMITH FARMS LLC",
                    "SMITH FARM LLC",
                    "BLUEBERRY HILL FARMS LLC",
                    "BLUEBERRY HILLS FARM LLC",
                    "ALPHA FARM LLC",
                    "OMEGA FARM LLC",
                    "SAME NAME LLC",
                    "SAME NAME LLC",
                    "FIRST LEGAL NAME LLC",
                    "SECOND LEGAL NAME LLC",
                    "FEIN LEGAL NAME ONE LLC",
                    "FEIN LEGAL NAME TWO LLC",
                ],
                "normalized_trade_name": [
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "SHARED DBA",
                    "SHARED DBA",
                    "",
                    "",
                ],
                "normalized_full_address": [
                    "10 MAIN ST||RALEIGH|NC|27601",
                    "10 MAIN ST||RALEIGH|NC|27601",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
                "normalized_phone": [
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "9105550100",
                    "9105550100",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
                "normalized_fein": [
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "111111112",
                    "222222223",
                    "",
                    "",
                    "333333334",
                    "333333334",
                ],
                "normalized_state": ["NC"] * 14,
                "normalized_postal_code": [
                    "27601",
                    "27601",
                    "27511",
                    "27511",
                    "28001",
                    "28002",
                    "28301",
                    "28301",
                    "28401",
                    "28401",
                    "28501",
                    "28502",
                    "28601",
                    "28602",
                ],
            }
        )
        linked, diagnostics = link_employer_records(synthetic)

        def same(method, left, right):
            column = f"employer_id_{method}"
            return linked[left, column] == linked[right, column]

        assert same("conservative", 0, 1)
        assert not same("conservative", 2, 3)
        assert same("balanced", 2, 3)
        assert not same("balanced", 4, 5)
        assert same("high_recall", 4, 5)
        assert not same("high_recall", 6, 7)
        assert not same("high_recall", 8, 9)
        assert same("conservative", 10, 11)
        assert same("conservative", 12, 13)

        shuffled = synthetic.sample(fraction=1, shuffle=True, seed=1729)
        relinked, _ = link_employer_records(shuffled)
        id_columns = [
            "employer_record_id",
            "employer_id_conservative",
            "employer_id_balanced",
            "employer_id_high_recall",
        ]
        assert (
            linked.select(id_columns)
            .sort("employer_record_id")
            .equals(relinked.select(id_columns).sort("employer_record_id"))
        )
        return pl.DataFrame(
            {
                "check": ["synthetic_linkage_contract"],
                "status": ["passed"],
                "records": [linked.height],
                "rejected_fein_conflicts": [
                    sum(
                        sum(value.values())
                        for value in diagnostics["rejected_fein_conflicts"].values()
                    )
                ],
            }
        )

    return link_employer_records, run_synthetic_checks


@app.cell
def _(INTERMEDIATE, pl):
    h2a_with_fips = pl.read_parquet(INTERMEDIATE / "h2a_with_fips.parquet")
    h2a_addendum_b_with_fips = pl.read_parquet(
        INTERMEDIATE / "h2a_addendum_b_with_fips.parquet"
    )
    return h2a_addendum_b_with_fips, h2a_with_fips


@app.cell
def _(
    build_identity_records,
    h2a_addendum_b_with_fips,
    h2a_with_fips,
):
    employer_identity_records, employer_source_diagnostics = build_identity_records(
        h2a_with_fips,
        h2a_addendum_b_with_fips,
    )
    return employer_identity_records, employer_source_diagnostics


@app.cell
def _(run_synthetic_checks):
    synthetic_check_results = run_synthetic_checks()
    return (synthetic_check_results,)


@app.cell
def _(employer_identity_records, link_employer_records):
    h2a_employer_crosswalk, employer_linkage_diagnostics = link_employer_records(
        employer_identity_records
    )
    return employer_linkage_diagnostics, h2a_employer_crosswalk


@app.cell
def _(
    INTERMEDIATE,
    RAW_IDENTITY_COLUMNS,
    h2a_employer_crosswalk,
    pl,
):
    entity_columns = [
        "employer_id_conservative",
        "employer_id_balanced",
        "employer_id_high_recall",
    ]
    if h2a_employer_crosswalk.get_column("employer_record_id").null_count() > 0:
        raise AssertionError("Crosswalk contains missing employer_record_id values")
    if h2a_employer_crosswalk.select(RAW_IDENTITY_COLUMNS).is_duplicated().any():
        raise AssertionError("Crosswalk raw join keys are not many-to-one")
    for column in entity_columns:
        if h2a_employer_crosswalk.get_column(column).null_count() > 0:
            raise AssertionError(f"Crosswalk contains missing {column} values")
        conflicting_feins = (
            h2a_employer_crosswalk.filter(pl.col("normalized_fein") != "")
            .group_by(column)
            .agg(pl.col("normalized_fein").n_unique().alias("n_feins"))
            .filter(pl.col("n_feins") > 1)
        )
        if conflicting_feins.height:
            raise AssertionError(f"{column} contains clusters with conflicting FEINs")

    cluster_counts = [
        h2a_employer_crosswalk.get_column(column).n_unique()
        for column in entity_columns
    ]
    if not cluster_counts[2] <= cluster_counts[1] <= cluster_counts[0]:
        raise AssertionError("Employer linkage tiers are not properly nested")

    crosswalk_path = INTERMEDIATE / "h2a_employer_crosswalk.parquet"
    h2a_employer_crosswalk.write_parquet(crosswalk_path)
    return cluster_counts, crosswalk_path


@app.cell
def _(
    cluster_counts,
    employer_linkage_diagnostics,
    employer_source_diagnostics,
    h2a_employer_crosswalk,
    mo,
    pl,
    synthetic_check_results,
):
    cluster_summary = pl.DataFrame(
        {
            "linkage_tier": ["conservative", "balanced", "high_recall"],
            "employer_entities": cluster_counts,
        }
    )
    largest_clusters = (
        h2a_employer_crosswalk.select(
            "employer_id_high_recall",
            "employer_cluster_size_high_recall",
            "normalized_name",
            "employer_linkage_evidence_high_recall",
        )
        .sort("employer_cluster_size_high_recall", descending=True)
        .unique("employer_id_high_recall", keep="first", maintain_order=True)
        .head(25)
    )
    diagnostics_text = mo.md(
        f"""
        **Ambiguous exact names with multiple FEINs:**
        {employer_linkage_diagnostics["ambiguous_exact_names"]:,}

        **Placeholder or broadly shared DBA values excluded:**
        {employer_linkage_diagnostics["excluded_shared_or_placeholder_trade_names"]:,}

        **Accepted links by tier:**
        `{employer_linkage_diagnostics["accepted_edges"]}`

        **Rejected FEIN conflicts by tier:**
        `{employer_linkage_diagnostics["rejected_fein_conflicts"]}`

        **Rejected fuzzy links exceeding tier name-sprawl limits:**
        `{employer_linkage_diagnostics["rejected_name_sprawl"]}`

        **Shared contact groups excluded from matching:**
        `{employer_linkage_diagnostics["skipped_shared_contacts"]}`
        """
    )
    mo.vstack(
        [
            mo.md("## Validation"),
            mo.ui.table(synthetic_check_results),
            mo.md("## Source coverage"),
            mo.ui.table(employer_source_diagnostics),
            mo.md("## Entity counts"),
            mo.ui.table(cluster_summary),
            diagnostics_text,
            mo.md("## Largest high-recall clusters"),
            mo.ui.table(largest_clusters),
        ]
    )


if __name__ == "__main__":
    app.run()
