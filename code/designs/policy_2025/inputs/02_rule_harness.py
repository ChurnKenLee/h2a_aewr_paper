# Purpose: Score candidate filer-linkage rules that use only fields available
#   in every year from FY2008 (name, mailing address, agent) against two answer
#   keys the rules never see: the FEIN (FY2024 on) and the phone (FY2015 on).
# Inputs (scratch): filer_cases.parquet from 01_filer_records.py.
# Outputs (scratch): harness_single_rules.csv, harness_rule_sets.csv,
#   harness_return_rates.csv, harness_return_errors.csv, harness_fuzzy_bins.csv,
#   filer_ids.parquet, filer_units.parquet.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from collections import defaultdict
    from pathlib import Path

    import marimo as mo
    import polars as pl
    from rapidfuzz import fuzz

    from h2a.employer_linkage import GENERIC_NAME_TOKENS, is_valid_alias
    from h2a.paths import ROOT

    return GENERIC_NAME_TOKENS, Path, ROOT, defaultdict, fuzz, is_valid_alias, mo, os, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rule harness for a filer identifier usable from FY2008

    Every rule reads only the employer name, its mailing address and the agent
    or attorney name. FEIN and phone are answer keys only.

    Scores are per case (B-cubed), so that a filer with hundreds of numbered
    job orders does not dominate:

    - **precision:** for a case, the share of cases grouped with it that have
      its answer key, averaged over cases;
    - **recall:** for a case, the share of cases with its answer key that are
      grouped with it, averaged over cases.

    Rule sets are closed transitively over all years before scoring, as a
    full-vintage linkage would be.
    """)


@app.cell
def _(Path, ROOT, os, pl):
    scratch = Path(os.environ.get("EMPLOYER_PANEL_SCRATCH", "")).expanduser().resolve()
    if not os.environ.get("EMPLOYER_PANEL_SCRATCH") or scratch == ROOT or ROOT in scratch.parents:
        raise ValueError("Set EMPLOYER_PANEL_SCRATCH to a directory outside the repository")
    cases = pl.read_parquet(scratch / "filer_cases.parquet")
    if cases.height == 0:
        raise ValueError("filer_cases.parquet is empty")
    return cases, scratch


@app.cell
def _(GENERIC_NAME_TOKENS, cases, is_valid_alias, pl):
    UNIT_COLUMNS = ["k_exact", "k_legal", "k_core", "k_sorted", "address_key", "city", "state", "zip", "agent_key"]
    SINGULAR_GENERIC = {"FARM", "ORCHARD", "RANCH", "NURSERY", "GROWER", "ENTERPRISE", "SERVICE", "HARVESTING", "PRODUCE"}

    def key_distinct(core_name):
        # Core name without generic farm words; blank when nothing else is left.
        tokens = [
            token
            for token in core_name.split()
            if token not in GENERIC_NAME_TOKENS and token not in SINGULAR_GENERIC
        ]
        return " ".join(sorted(set(tokens))) if len("".join(tokens)) >= 4 else ""

    NAME_KEYS = ["k_exact", "k_legal", "k_core", "k_sorted", "k_distinct"]
    GEOGRAPHIES = {
        "address": ["address_key"],
        "state_zip": ["state", "zip"],
        "state_city": ["state", "city"],
        "state_agent": ["state", "agent_key"],
        "state": ["state"],
        "national": [],
    }
    # Wide geographies need a distinctive name (the v2 test); tight ones do not.
    GUARDED = {"state_agent", "state", "national"}

    # A unit is one distinct combination of the fields the rules can read.
    units = (
        cases.select(UNIT_COLUMNS)
        .unique()
        .sort(UNIT_COLUMNS)
        .with_row_index("unit")
        .with_columns(
            pl.col("k_exact").map_elements(is_valid_alias, return_dtype=pl.Boolean).alias("distinctive"),
            pl.col("k_core").map_elements(key_distinct, return_dtype=pl.String).alias("k_distinct"),
        )
    )
    cases_u = cases.join(units.select("unit", *UNIT_COLUMNS), on=UNIT_COLUMNS, how="left", validate="m:1")
    if cases_u["unit"].null_count():
        raise ValueError("a case has no unit")

    def rule_key(name_key, geography, guard):
        parts = [name_key, *GEOGRAPHIES[geography]]
        usable = pl.all_horizontal([pl.col(part) != "" for part in parts])
        if guard:
            usable = usable & pl.col("distinctive")
        return pl.when(usable).then(pl.concat_str(parts, separator="|")).otherwise(None)

    RULES = {
        f"{name_key[2:]}+{geography}": (name_key, geography, geography in GUARDED)
        for name_key in NAME_KEYS
        for geography in GEOGRAPHIES
    }
    # The same two exact rules with the v2 distinctiveness test, as a baseline.
    RULES["exact+address (v2 test)"] = ("k_exact", "address", True)
    RULES["exact+state_zip (v2 test)"] = ("k_exact", "state_zip", True)

    # An address used by at most three core names across all years.
    address_names = (
        units.filter(pl.col("address_key") != "")
        .group_by("address_key")
        .agg(pl.col("k_core").n_unique().alias("address_names"))
    )
    unit_keys = (
        units.join(address_names, on="address_key", how="left")
        .with_columns(
            [rule_key(name_key, geography, guard).alias(rule) for rule, (name_key, geography, guard) in RULES.items()]
            + [
                pl.when((pl.col("address_key") != "") & (pl.col("address_names") <= 3))
                .then(pl.col("address_key"))
                .otherwise(None)
                .alias("address_only")
            ]
        )
        .sort("unit")
    )
    ALL_RULES = [*RULES, "address_only"]
    return ALL_RULES, cases_u, unit_keys, units


@app.cell
def _(defaultdict, fuzz, unit_keys):
    # Similar-name links: plain edit-distance ratio on core names inside a block.
    def similar_pairs(block_columns, threshold, max_block=150):
        blocks = defaultdict(list)
        for _record in unit_keys.select("unit", "k_core", *block_columns).to_dicts():
            if _record["k_core"] and all(_record[column] for column in block_columns):
                blocks[tuple(_record[column] for column in block_columns)].append(_record)
        pairs = []
        skipped = 0
        for members in blocks.values():
            names = {}
            for member in members:
                names.setdefault(member["k_core"], member["unit"])
            if len(names) > max_block:
                skipped += 1
                continue
            keys = list(names)
            for left in range(len(keys)):
                for right in range(left + 1, len(keys)):
                    if fuzz.ratio(keys[left], keys[right]) >= threshold:
                        pairs.append((names[keys[left]], names[keys[right]]))
        return pairs, skipped

    similar_address, skipped_address = similar_pairs(["address_key"], 85)
    similar_state_zip, skipped_state_zip = similar_pairs(["state", "zip"], 94)
    PAIR_RULES = {"~similar name, same address (85)": similar_address, "~similar name, same state and ZIP (94)": similar_state_zip}
    print("similar-name links:", len(similar_address), len(similar_state_zip), "| blocks skipped:", skipped_address, skipped_state_zip)
    return (PAIR_RULES,)


@app.cell
def _(PAIR_RULES, defaultdict, pl, unit_keys):
    def components(rules):
        # Union units that share a key under any rule, or are paired by a "~" rule.
        count = unit_keys.height
        parent = list(range(count))

        def find(value):
            root = value
            while parent[root] != root:
                root = parent[root]
            while parent[value] != value:
                parent[value], value = root, parent[value]
            return root

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        unit_list = unit_keys["unit"].to_list()
        for rule in rules:
            if rule.startswith("~"):
                for left, right in PAIR_RULES[rule]:
                    union(left, right)
                continue
            groups = defaultdict(list)
            for unit, key in zip(unit_list, unit_keys[rule].to_list()):
                if key is not None:
                    groups[key].append(unit)
            for members in groups.values():
                for member in members[1:]:
                    union(members[0], member)
        return pl.DataFrame(
            {"unit": unit_keys["unit"], "cluster": [find(value) for value in range(count)]},
            schema={"unit": pl.UInt32, "cluster": pl.Int64},
        )

    # Nested ladder: each set contains the one before it.
    STEPS = [
        ("R0 exact name at address or state+ZIP, v2 name test", ["exact+address (v2 test)", "exact+state_zip (v2 test)"]),
        ("R1 same, short names allowed", ["exact+address", "exact+state_zip"]),
        ("R2 + core name at address or state+ZIP", ["legal+address", "legal+state_zip", "core+address", "core+state_zip", "sorted+address", "sorted+state_zip"]),
        ("R3 + core name in state and city", ["exact+state_city", "legal+state_city", "core+state_city", "sorted+state_city"]),
        ("R4 + distinctive words at address, ZIP or city", ["distinct+address", "distinct+state_zip", "distinct+state_city"]),
        ("R5 + similar name at address or state+ZIP", list(PAIR_RULES)),
        ("R6 + core name, state and agent", ["exact+state_agent", "legal+state_agent", "core+state_agent", "sorted+state_agent"]),
        ("R7 + core name in state", ["exact+state", "legal+state", "core+state", "sorted+state"]),
        ("R8 + core name, any state", ["exact+national", "legal+national", "core+national", "sorted+national"]),
        ("R9 + same address, any name", ["address_only"]),
    ]
    LADDER = {}
    _running = []
    for _name, _rules in STEPS:
        _running = [*_running, *_rules]
        LADDER[_name] = list(_running)
    clusters = {name: components(rules) for name, rules in LADDER.items()}
    return LADDER, clusters


@app.cell
def _(cases_u, pl):
    # Answer keys by window. A phone used by more than ten core names in the
    # window is a shared contact and gives no answer.
    def truth_cases(first_year, last_year, key):
        window = cases_u.filter(pl.col("fiscal_year").is_between(first_year, last_year) & (pl.col(key) != ""))
        if key == "phone":
            shared = (
                window.group_by("phone")
                .agg(pl.col("k_core").n_unique().alias("names"))
                .filter(pl.col("names") > 10)
            )
            window = window.join(shared.select("phone"), on="phone", how="anti")
        return window.select("fiscal_year", "case_number", "unit", pl.col(key).alias("truth"))

    WINDOWS = {
        "FEIN FY2024-26": truth_cases(2024, 2026, "fein"),
        "phone FY2024-26": truth_cases(2024, 2026, "phone"),
        "phone FY2020-23": truth_cases(2020, 2023, "phone"),
        "phone FY2015-19": truth_cases(2015, 2019, "phone"),
    }

    def bcubed(frame, link):
        # frame: cases with `truth` and a link column (never null).
        scored = frame.with_columns(
            pl.len().over(link).alias("n_link"),
            pl.len().over("truth").alias("n_truth"),
            pl.len().over([link, "truth"]).alias("n_both"),
        )
        by_truth = frame.group_by("truth").agg(pl.len().alias("n"), pl.col(link).n_unique().alias("links"))
        by_link = frame.group_by(link).agg(pl.col("truth").n_unique().alias("truths"))
        return {
            "cases": frame.height,
            "precision": round(float((scored["n_both"] / scored["n_link"]).mean()), 4),
            "recall": round(float((scored["n_both"] / scored["n_truth"]).mean()), 4),
            "answer_keys_split": round(float((by_truth["links"] > 1).mean()), 4),
            "groups_with_two_answer_keys": round(float((by_link["truths"] > 1).mean()), 4),
        }

    return WINDOWS, bcubed


@app.cell
def _(ALL_RULES, LADDER, WINDOWS, bcubed, clusters, pl, unit_keys):
    single_rows = []
    for _window_name in ("FEIN FY2024-26", "phone FY2015-19"):
        for _rule in ALL_RULES:
            # A case the rule does not cover is grouped only with its own unit.
            _frame = WINDOWS[_window_name].join(unit_keys.select("unit", _rule), on="unit", how="left").with_columns(
                pl.coalesce(pl.col(_rule), pl.concat_str([pl.lit("unit:"), pl.col("unit").cast(pl.String)])).alias("link")
            )
            single_rows.append({"window": _window_name, "rule": _rule, **bcubed(_frame, "link")})
    single_rules = pl.DataFrame(single_rows)

    set_rows = []
    for _set_name in LADDER:
        for _window_name, _truth in WINDOWS.items():
            _frame = _truth.join(clusters[_set_name], on="unit", how="left")
            set_rows.append({"rule_set": _set_name, "window": _window_name, **bcubed(_frame, "cluster")})
    rule_sets = pl.DataFrame(set_rows)
    return rule_sets, single_rules


@app.cell
def _(LADDER, cases_u, clusters, pl):
    # Share of filers in FY t that file again in FY t+1, by identifier, and the
    # errors a rule set makes against the FEIN for FY2024 to FY2025.
    def return_rates(frame, identifier, label):
        present = frame.filter(pl.col(identifier).is_not_null()).select("fiscal_year", identifier).unique()
        following = present.with_columns((pl.col("fiscal_year") - 1).alias("fiscal_year"), pl.lit(True).alias("returns"))
        joined = present.join(following, on=["fiscal_year", identifier], how="left")
        return (
            joined.group_by("fiscal_year")
            .agg(pl.col("returns").fill_null(False).mean().round(4).alias("return_rate"))
            .with_columns(pl.lit(label).alias("identifier"))
        )

    rate_frames = [
        return_rates(cases_u.with_columns(pl.when(pl.col("fein") != "").then(pl.col("fein")).alias("fein_id")), "fein_id", "FEIN"),
        return_rates(cases_u.with_columns(pl.when(pl.col("phone") != "").then(pl.col("phone")).alias("phone_id")), "phone_id", "phone"),
    ]
    filer_ids = cases_u.select("fiscal_year", "case_number", "unit")
    error_rows = []
    _fein_2025 = cases_u.filter(pl.col("fiscal_year") == 2025).select("fein").unique()
    for _name in LADDER:
        _short = _name.split(" ")[0]
        _with = cases_u.join(clusters[_name], on="unit", how="left", validate="m:1")
        rate_frames.append(return_rates(_with, "cluster", _short))
        filer_ids = filer_ids.join(clusters[_name].rename({"cluster": f"filer_{_short}"}), on="unit", how="left", validate="m:1")
        _base = _with.filter(pl.col("fiscal_year") == 2024).with_columns(
            pl.col("fein").is_in(_fein_2025["fein"].implode()).alias("fein_returns"),
            pl.col("cluster").is_in(_with.filter(pl.col("fiscal_year") == 2025)["cluster"].unique().implode()).alias("rule_returns"),
        )
        error_rows.append(
            {
                "rule_set": _name,
                "cases_fy2024": _base.height,
                "returning_filer_missed": round(float((~_base.filter(pl.col("fein_returns"))["rule_returns"]).mean()), 4),
                "departed_filer_found": round(float(_base.filter(~pl.col("fein_returns"))["rule_returns"].mean()), 4),
                "filers_fy2025_by_rule": _with.filter(pl.col("fiscal_year") == 2025)["cluster"].n_unique(),
                "filers_fy2025_by_fein": _fein_2025.height,
            }
        )
    return_errors = pl.DataFrame(error_rows)
    return_rate_table = (
        pl.concat(rate_frames)
        .filter(pl.col("fiscal_year") <= 2024)
        .pivot(on="identifier", index="fiscal_year", values="return_rate")
        .sort("fiscal_year")
    )
    return filer_ids, return_errors, return_rate_table


@app.cell
def _(WINDOWS, defaultdict, fuzz, pl, unit_keys):
    # How often do two different core names in the same place share a FEIN, by
    # edit-distance ratio? One pair per pair of names; blocks above 150 names skipped.
    _fein_unit = WINDOWS["FEIN FY2024-26"].group_by("unit").agg(pl.col("truth").n_unique().alias("n"), pl.col("truth").first()).filter(pl.col("n") == 1)
    _fein_units = unit_keys.join(_fein_unit.select("unit", "truth"), on="unit", how="inner")
    BINS = [(0, 70), (70, 80), (80, 85), (85, 90), (90, 94), (94, 97), (97, 101)]
    fuzzy_rows = []
    for _block_name, _block_columns in {"same address": ["address_key"], "same state and ZIP": ["state", "zip"], "same state and city": ["state", "city"]}.items():
        _blocks = defaultdict(dict)
        for _record in _fein_units.select("k_core", "truth", *_block_columns).to_dicts():
            if _record["k_core"] and all(_record[_column] for _column in _block_columns):
                _blocks[tuple(_record[_column] for _column in _block_columns)].setdefault((_record["k_core"], _record["truth"]), True)
        _tally = defaultdict(lambda: [0, 0])
        for _members in _blocks.values():
            _items = list(_members)
            if len(_items) > 150:
                continue
            for _left in range(len(_items)):
                for _right in range(_left + 1, len(_items)):
                    if _items[_left][0] == _items[_right][0]:
                        continue
                    _value = fuzz.ratio(_items[_left][0], _items[_right][0])
                    for _low, _high in BINS:
                        if _low <= _value < _high:
                            _tally[(_low, _high)][0] += 1
                            _tally[(_low, _high)][1] += int(_items[_left][1] == _items[_right][1])
        for _low, _high in BINS:
            _pairs, _same = _tally[(_low, _high)]
            fuzzy_rows.append(
                {
                    "block": _block_name,
                    "ratio_from": _low,
                    "ratio_below": _high,
                    "name_pairs": _pairs,
                    "share_same_fein": round(_same / _pairs, 4) if _pairs else None,
                }
            )
    fuzzy_bins = pl.DataFrame(fuzzy_rows)
    return (fuzzy_bins,)


@app.cell
def _(
    filer_ids,
    fuzzy_bins,
    return_errors,
    return_rate_table,
    rule_sets,
    scratch,
    single_rules,
    unit_keys,
):
    single_rules.write_csv(scratch / "harness_single_rules.csv")
    rule_sets.write_csv(scratch / "harness_rule_sets.csv")
    return_rate_table.write_csv(scratch / "harness_return_rates.csv")
    return_errors.write_csv(scratch / "harness_return_errors.csv")
    fuzzy_bins.write_csv(scratch / "harness_fuzzy_bins.csv")
    filer_ids.write_parquet(scratch / "filer_ids.parquet")
    unit_keys.select("unit", "k_exact", "k_legal", "k_core", "k_sorted", "k_distinct", "address_key", "city", "state", "zip", "agent_key", "distinctive").write_parquet(scratch / "filer_units.parquet")


@app.cell
def _(fuzzy_bins, mo, pl, return_errors, return_rate_table, rule_sets, single_rules, units):
    with pl.Config(tbl_rows=200, tbl_cols=30, tbl_width_chars=260, fmt_str_lengths=70, tbl_hide_dataframe_shape=True):
        print("units:", units.height, "| distinctive:", int(units["distinctive"].sum()))
        print(single_rules.pivot(on="window", index="rule", values=["precision", "recall"]).sort("rule"))
        print(rule_sets)
        print(return_errors)
        print(return_rate_table)
        print(fuzzy_bins)
    mo.vstack([mo.ui.table(single_rules), mo.ui.table(rule_sets), mo.ui.table(return_errors), mo.ui.table(return_rate_table), mo.ui.table(fuzzy_bins)])


if __name__ == "__main__":
    app.run()
