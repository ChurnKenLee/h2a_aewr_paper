# Purpose: Build scratch tables for the employer-panel exploration: one row per
#   filer case with candidate name keys, the worksite-business rows with the
#   same keys, and an inventory of identity fields in the raw disclosure files.
# Inputs: h2a_employer_record_membership.parquet, h2a_allocated_worksites.parquet,
#   h2a_disclosure_records.parquet, raw H-2A disclosure workbooks.
# Outputs (scratch only, EMPLOYER_PANEL_SCRATCH): filer_cases.parquet,
#   business_rows.parquet, raw_field_inventory.csv.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import re
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from h2a.employer_linkage import (
        normalize_fein,
        normalize_name,
        normalize_phone,
        normalize_state,
        normalize_text,
        normalize_zip,
    )
    from h2a.paths import INTERMEDIATE, RAW, ROOT

    return (
        INTERMEDIATE,
        Path,
        RAW,
        ROOT,
        mo,
        normalize_fein,
        normalize_name,
        normalize_phone,
        normalize_state,
        normalize_text,
        normalize_zip,
        os,
        pl,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Filer records for the employer-panel exploration

    The filer is the employer named on the case: the entity that files the job
    order and sets the offer. It is not the v2 linkage unit, which is the
    worksite business. This notebook writes scratch tables only and changes no
    retained artifact.

    Name keys use fields that exist in every year from FY2008: the employer
    name, its mailing address and the agent or attorney name. Phone (FY2015 on)
    and FEIN (FY2024 on) are carried only as answer keys for scoring.
    """)


@app.cell
def _(Path, ROOT, os):
    scratch_value = os.environ.get("EMPLOYER_PANEL_SCRATCH", "")
    if not scratch_value:
        raise ValueError("Set EMPLOYER_PANEL_SCRATCH to a directory outside the repository")
    scratch = Path(scratch_value).expanduser().resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        raise ValueError("EMPLOYER_PANEL_SCRATCH must be outside the repository")
    scratch.mkdir(parents=True, exist_ok=True)
    return (scratch,)


@app.cell
def _(normalize_name, re):
    LEGAL_FORM_TOKENS = {
        "LLC", "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD",
        "LIMITED", "LP", "LLP", "LLLP", "PLLC", "PC", "PA", "PARTNERSHIP", "THE", "DBA",
    }
    PLURAL_TOKENS = {
        "FARMS": "FARM", "ORCHARDS": "ORCHARD", "RANCHES": "RANCH", "NURSERIES": "NURSERY",
        "GROWERS": "GROWER", "ENTERPRISES": "ENTERPRISE", "BROTHERS": "BROS", "SONS": "SON",
        "VINEYARDS": "VINEYARD", "GREENHOUSES": "GREENHOUSE", "DAIRIES": "DAIRY",
        "SERVICES": "SERVICE", "PRODUCTS": "PRODUCT", "ACRES": "ACRE", "GARDENS": "GARDEN",
        "HARVESTERS": "HARVESTER", "FARMING": "FARM",
    }
    SITE_TOKENS = {"SITE", "WORKSITE", "FIELD", "BLOCK", "LOCATION", "NO", "NUMBER", "TEMP"}

    def collapse_initials(tokens):
        # "L L C" -> "LLC", "J W SELF" -> "JW SELF"
        merged = []
        run = ""
        for token in tokens:
            if len(token) == 1 and token.isalpha():
                run += token
                continue
            if run:
                merged.append(run)
                run = ""
            merged.append(token)
        if run:
            merged.append(run)
        return merged

    def key_exact(value):
        return normalize_name(value)

    def key_legal(value):
        tokens = collapse_initials(normalize_name(value).split())
        kept = [token for token in tokens if token not in LEGAL_FORM_TOKENS]
        return " ".join(kept) if kept else " ".join(tokens)

    def core_tokens(value):
        text = "" if value is None else str(value)
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.split(r"\b(?:D\s*/?\s*B\s*/?\s*A)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
        tokens = collapse_initials(normalize_name(text).split())
        kept = []
        for token in tokens:
            if token in LEGAL_FORM_TOKENS or token in SITE_TOKENS or token == "AND":
                continue
            if token.isdigit():
                continue
            kept.append(PLURAL_TOKENS.get(token, token))
        return kept if kept else tokens

    def key_core(value):
        return " ".join(core_tokens(value))

    def key_sorted(value):
        return " ".join(sorted(set(core_tokens(value))))

    def key_person(value):
        # Order-free key for an agent or attorney name; single letters dropped.
        tokens = [token for token in normalize_name(value).split() if len(token) > 1]
        return " ".join(sorted(set(tokens)))

    return key_core, key_exact, key_legal, key_person, key_sorted


@app.cell
def _(RAW, pl, re):
    # Raw disclosure workbooks: agent fields per case, and an inventory of
    # identity fields. Columns are read as text.
    FIELD_FAMILIES = {
        "fein": r"^EMPLOYER_FEIN$",
        "phone": r"^EMPLOYER_PHONE$",
        "trade_name": r"^TRADE_NAME_DBA$",
        "previous_trade_name": r"^PREV_TRADE_NAME_DBA_1$",
        "contact_email": r"^EMPLOYER_POC_EMAIL$",
        "contact_phone": r"^EMPLOYER_POC_PHONE$",
        "email_to_apply": r"^EMAIL_TO_APPLY$",
        "phone_to_apply": r"^PHONE_TO_APPLY$",
        "agent_name": r"^(AGENT_ATTORNEY_NAME|ATTORNEY_AGENT_LAST_NAME)$",
        "law_firm": r"^(LAWFIRM_NAME|LAWFIRM_NAME_BUSINESS_NAME)$",
        "housing_address": r"^HOUSING_ADDRESS_LOCATION$",
        "worksite_address": r"^WORKSITE_ADDRESS$",
        "worksite_zip": r"^(WORKSITE_POSTAL_CODE|WORKSITE_ZIP)$",
        "naics": r"^NAICS_CODE$",
    }

    def fiscal_year_of(path):
        match = re.search(r"FY(\d{2,4})", path.name)
        value = int(match.group(1))
        return value if value > 100 else 2000 + value

    raw_paths = sorted(
        path
        for path in (RAW / "h2a").glob("H*2A*.xls*")
        if "Addendum" not in path.name and "Worksite County" not in path.name
    )
    inventory_rows = []
    agent_frames = []
    for raw_path in raw_paths:
        fiscal_year = fiscal_year_of(raw_path)
        sheet = pl.read_excel(raw_path, engine="calamine", infer_schema_length=0)
        sheet = sheet.rename({column: column.strip().upper() for column in sheet.columns})
        case_column = "CASE_NUMBER" if "CASE_NUMBER" in sheet.columns else "CASE_NO"
        row = {"fiscal_year": fiscal_year, "file": raw_path.name, "rows": sheet.height, "columns": sheet.width}
        for family, pattern in FIELD_FAMILIES.items():
            matches = [column for column in sheet.columns if re.search(pattern, column)]
            if matches:
                values = sheet[matches[0]].cast(pl.String).str.strip_chars()
                row[family] = round(float((values.is_not_null() & (values != "")).mean()), 3)
            else:
                row[family] = None
        inventory_rows.append(row)
        if "AGENT_ATTORNEY_NAME" in sheet.columns:
            agent = sheet.select(
                pl.col(case_column).alias("case_number"),
                pl.col("AGENT_ATTORNEY_NAME").alias("agent_name_raw"),
            )
        else:
            agent = sheet.select(
                pl.col(case_column).alias("case_number"),
                pl.concat_str(
                    [pl.col("ATTORNEY_AGENT_FIRST_NAME"), pl.col("ATTORNEY_AGENT_LAST_NAME")],
                    separator=" ",
                    ignore_nulls=True,
                ).alias("agent_name_raw"),
            )
        law_columns = [c for c in ("LAWFIRM_NAME", "LAWFIRM_NAME_BUSINESS_NAME") if c in sheet.columns]
        agent = agent.with_columns(
            (sheet[law_columns[0]] if law_columns else pl.lit(None, dtype=pl.String)).alias("law_firm_raw"),
            pl.lit(fiscal_year, dtype=pl.Int32).alias("fiscal_year"),
        )
        agent_frames.append(
            agent.with_columns(pl.col("case_number").cast(pl.String).str.strip_chars())
            .group_by("fiscal_year", "case_number")
            .agg(
                pl.col("agent_name_raw").drop_nulls().first(),
                pl.col("law_firm_raw").drop_nulls().first(),
            )
        )
    raw_field_inventory = pl.DataFrame(inventory_rows).sort("fiscal_year")
    agent_by_case = pl.concat(agent_frames)
    if agent_by_case.select("fiscal_year", "case_number").is_duplicated().any():
        raise ValueError("agent table is not unique by fiscal year and case")
    return agent_by_case, raw_field_inventory


@app.cell
def _(INTERMEDIATE, pl):
    membership = pl.read_parquet(
        INTERMEDIATE / "h2a_employer_record_membership.parquet",
        columns=[
            "source_dataset", "case_number", "fiscal_year", "source_name_raw", "source_trade_name_raw",
            "source_address_1_raw", "source_address_2_raw", "source_city_raw", "source_state_raw",
            "source_postal_code_raw", "source_phone_raw", "source_fein_raw",
            "source_h2a_labor_contractor_raw", "employer_record_role", "employer_is_worksite_business",
            "employer_id_balanced", "employer_id_high_recall",
        ],
    )
    case_workers = (
        pl.scan_parquet(INTERMEDIATE / "h2a_allocated_worksites.parquet")
        .group_by("fiscal_year", "case_number")
        .agg(pl.col("adjusted_nbr_workers_certified").sum().alias("certified_workers"))
        .collect()
    )
    return case_workers, membership


@app.cell
def _(
    agent_by_case,
    case_workers,
    key_core,
    key_exact,
    key_legal,
    key_person,
    key_sorted,
    membership,
    normalize_fein,
    normalize_phone,
    normalize_state,
    normalize_text,
    normalize_zip,
    pl,
):
    def add_name_keys(frame, column):
        names = frame.select(column).unique()
        keyed = names.with_columns(
            pl.col(column).map_elements(key_exact, return_dtype=pl.String).alias("k_exact"),
            pl.col(column).map_elements(key_legal, return_dtype=pl.String).alias("k_legal"),
            pl.col(column).map_elements(key_core, return_dtype=pl.String).alias("k_core"),
            pl.col(column).map_elements(key_sorted, return_dtype=pl.String).alias("k_sorted"),
        )
        return frame.join(keyed, on=column, how="left", validate="m:1")

    def add_address_keys(frame):
        return frame.with_columns(
            pl.col("source_address_1_raw").map_elements(normalize_text, return_dtype=pl.String).alias("address_1"),
            pl.col("source_city_raw").map_elements(normalize_text, return_dtype=pl.String).alias("city"),
            pl.col("source_state_raw").map_elements(normalize_state, return_dtype=pl.String).alias("state"),
            pl.col("source_postal_code_raw").map_elements(normalize_zip, return_dtype=pl.String).alias("zip"),
        ).with_columns(
            pl.when((pl.col("address_1").str.len_chars() >= 5) & (pl.col("state") != "") & (pl.col("zip") != ""))
            .then(pl.concat_str(["address_1", "city", "state", "zip"], separator="|"))
            .otherwise(pl.lit(""))
            .alias("address_key")
        )

    # Filer rows: every case-level row that is not a pre-2020 sub-employer.
    filer_rows = membership.filter(
        (pl.col("source_dataset") == "h2a_with_fips")
        & ~pl.col("employer_record_role").str.contains("sub_employer")
        & (pl.col("source_name_raw").str.strip_chars() != "")
    )
    filer_cases = (
        filer_rows.group_by("fiscal_year", "case_number")
        .agg(
            pl.col("source_name_raw").n_unique().alias("filer_names"),
            pl.all().exclude("source_name_raw", "fiscal_year", "case_number").first(),
            pl.col("source_name_raw").first(),
        )
    )
    if (filer_cases["filer_names"] > 1).any():
        raise ValueError("a case has more than one filer name")
    filer_cases = add_address_keys(add_name_keys(filer_cases.drop("filer_names"), "source_name_raw"))
    filer_cases = (
        filer_cases.with_columns(
            pl.col("source_phone_raw").map_elements(normalize_phone, return_dtype=pl.String).alias("phone"),
            pl.col("source_fein_raw").map_elements(normalize_fein, return_dtype=pl.String).alias("fein"),
            (pl.col("source_h2a_labor_contractor_raw").str.to_uppercase() == "Y").fill_null(False).alias("labor_contractor"),
        )
        .join(agent_by_case, on=["fiscal_year", "case_number"], how="left", validate="1:1")
        .with_columns(
            pl.col("agent_name_raw").fill_null("").map_elements(key_person, return_dtype=pl.String).alias("agent_key"),
            pl.col("law_firm_raw").fill_null("").map_elements(key_legal, return_dtype=pl.String).alias("law_firm_key"),
        )
        .join(case_workers, on=["fiscal_year", "case_number"], how="left", validate="1:1")
        .with_columns(pl.col("certified_workers").fill_null(0.0))
        .select(
            "fiscal_year", "case_number", "employer_record_role", "labor_contractor", "certified_workers",
            "source_name_raw", "source_trade_name_raw", "k_exact", "k_legal", "k_core", "k_sorted",
            "address_1", "city", "state", "zip", "address_key", "phone", "fein",
            "agent_name_raw", "agent_key", "law_firm_key",
        )
        .sort("fiscal_year", "case_number")
    )
    if filer_cases.select("fiscal_year", "case_number").is_duplicated().any() or filer_cases.height == 0:
        raise ValueError("filer cases are empty or not unique by fiscal year and case")

    # Worksite-business rows (the v2 unit) with the same name keys.
    business_rows = add_address_keys(
        add_name_keys(
            membership.filter(
                pl.col("employer_is_worksite_business") & (pl.col("source_name_raw").str.strip_chars() != "")
            ),
            "source_name_raw",
        )
    ).select(
        "source_dataset", "fiscal_year", "case_number", "employer_record_role", "employer_id_balanced",
        "employer_id_high_recall", "source_name_raw", "k_exact", "k_legal", "k_core", "k_sorted",
        "address_1", "city", "state", "zip", "address_key",
    )
    return business_rows, filer_cases


@app.cell
def _(business_rows, filer_cases, raw_field_inventory, scratch):
    filer_cases.write_parquet(scratch / "filer_cases.parquet")
    business_rows.write_parquet(scratch / "business_rows.parquet")
    raw_field_inventory.write_csv(scratch / "raw_field_inventory.csv")


@app.cell
def _(business_rows, filer_cases, mo, pl, raw_field_inventory):
    coverage = (
        filer_cases.group_by("fiscal_year")
        .agg(
            pl.len().alias("cases"),
            (pl.col("address_key") != "").mean().round(3).alias("address"),
            (pl.col("agent_key") != "").mean().round(3).alias("agent"),
            (pl.col("law_firm_key") != "").mean().round(3).alias("law_firm"),
            (pl.col("phone") != "").mean().round(3).alias("phone"),
            (pl.col("fein") != "").mean().round(3).alias("fein"),
        )
        .sort("fiscal_year")
    )
    with pl.Config(tbl_rows=40, tbl_cols=30, tbl_width_chars=240):
        print(raw_field_inventory.drop("file"))
        print(coverage)
        print("filer cases:", filer_cases.height, "| worksite-business rows:", business_rows.height)
    mo.vstack([mo.ui.table(raw_field_inventory), mo.ui.table(coverage)])


if __name__ == "__main__":
    app.run()
