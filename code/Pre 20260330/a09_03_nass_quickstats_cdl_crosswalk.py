import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import pyprojroot
    import dotenv, os
    import ibis
    import ibis.selectors as s
    from ibis import _
    import polars as pl
    import pdfplumber
    import dspy
    from pydantic import BaseModel, Field
    from typing import List, Literal
    import json
    import tqdm
    from itertools import islice
    import time
    import tempfile
    import duckdb

    return (
        BaseModel,
        Field,
        List,
        Path,
        dspy,
        duckdb,
        islice,
        json,
        os,
        pdfplumber,
        pl,
        pyprojroot,
        tempfile,
        tqdm,
    )


@app.cell
def _(dspy, os, pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path / 'binaries'
    json_path = root_path / 'code' / 'json'
    cdl_path = root_path / 'data' / 'croplandcros_cdl'
    # Configure Gemini
    gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    gemini = dspy.LM('gemini/gemini-3-flash-preview', api_key=gemini_api_key)
    dspy.configure(lm=gemini)
    return cdl_path, json_path


@app.cell
def _(Path, cdl_path, json, json_path, pl):
    # NASS crop definitions
    with open(json_path / 'nass_quickstats_survey_obs_selection.json', 'r') as _f:
        nass_obs_selection = json.load(_f)

    # Load CroplandCROS CDL code table
    cdl_names = pl.read_excel(Path(cdl_path / 'CDL_codes_names_colors.xlsx'), read_options={"header_row": 3})
    cdl_names = (cdl_names
        .rename(str.lower)
        .select([
            pl.col('codes'),
            pl.col('class_names')
        ])
        .rename(mapping={
         'codes':'cdl_code','class_names':'cdl_name'
        })
    )
    return cdl_names, nass_obs_selection


@app.cell
def _(Path, cdl_fsa_code_dict, cdl_path, pl):
    # We have additional information from CDL-to-FSA crosswalk, which we can incorporate
    # CDL to FSA crosswalk
    cdl_fsa_xwalk = pl.read_excel(Path(cdl_path / 'FSA-to-CDL_Crosswalk.xlsx'))
    cdl_fsa_xwalk = cdl_fsa_xwalk.rename(str.lower)

    # CDL codes map to multiple FSA crops; aggregate to CDL codes
    # Split comma-separated elements within strings, explode, then implode back into list
    cdl_fsa_xwalk = (
        cdl_fsa_xwalk
            .group_by("cdl_code")
            .agg([
                pl.col("fsa_crop_name").str.split(",").list.explode().str.strip_chars().unique().implode().alias("fsa_crops"),
                pl.col("fsa_type_name").str.split(",").list.explode().str.strip_chars().unique().implode().alias("fsa_types"),
                pl.col("fsa_intended_use").str.split(",").list.explode().str.strip_chars().unique().implode().alias("fsa_intended_use_code")
        ])
    )

    # Add column for translation of FSA Intended Use Code
    cdl_fsa_xwalk = cdl_fsa_xwalk.with_columns(
        pl.col('fsa_intended_use_code')
            .list.eval(
                pl.element().replace_strict(cdl_fsa_code_dict, default=None)
                .drop_nulls()
            )
        .alias('fsa_intended_use')
    ).select([
        'cdl_code',
        'fsa_crops',
        'fsa_types',
        'fsa_intended_use'
    ])
    return (cdl_fsa_xwalk,)


@app.cell
def _(cdl_path, pdfplumber, pl):
    # Translation for FSA Intended Use codes
    mappings = []
    cdl_fsa = cdl_path / 'intended_use_codes.pdf'
    cdl_fsa_pdf = pdfplumber.open(cdl_fsa)
    for page in cdl_fsa_pdf.pages:
        table = page.extract_table({
            "vertical_strategy": "lines",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
        })

        if not table:
            continue

        for row in table:
            # Filter out headers and empty rows
            # Typical columns: [Code, Intended Use, FSA-578 Printout, Definition]
            # This filters out the header row (row[0] = 'Code')
            # Also fiters out rows added for long definition (row[0] = row[1] = row[2] = '')
            if row and row[0] and len(row[0]) <= 3 and row[0].strip().lower() != 'code':
                code = row[0].strip().upper().replace(".", "")
                if row[1]:
                    name = row[1].strip()
                else:
                    name = ""
                # Definitions often span the last column
                if row[-1]:
                    definition = row[-1].strip()
                else:
                    ""

                mappings.append({
                    "fsa_use_code": code,
                    "fsa_use_name": name,
                    "fsa_definition": definition
                })

    cdl_fsa_mapping = pl.DataFrame(mappings)
    cdl_fsa_code_dict = dict(cdl_fsa_mapping.select("fsa_use_code", "fsa_use_name").iter_rows())
    return (cdl_fsa_code_dict,)


@app.cell
def _(cdl_fsa_xwalk, cdl_names, pl):
    # Add FSA semantic info to CDL codes
    cdl_fsa_context = cdl_names.join(cdl_fsa_xwalk, on=['cdl_code'], how='full')

    # Pivot semantic info from 3 columns (wide) to 1 (long)
    df_long = (
        cdl_fsa_context
        # Transform columns into a 'variable' and 'value' pair
        # We keep cdl_code and cdl_name as our index anchors
        .unpivot(
            index=["cdl_code", "cdl_name"],
            on=["fsa_crops", "fsa_types", "fsa_intended_use"],
            variable_name="fsa_attribute_type",
            value_name="fsa_attribute_value"
        )
        # 'Explode' the lists in the value column
        # This turns [Corn, Popcorn] into two separate rows
        .explode("fsa_attribute_value")
        # Clean up
        .filter(pl.col("fsa_attribute_value").is_not_null())
        .with_columns(
            # Standardize naming for easier SQL queries (e.g., 'fsa_crops' -> 'crop')
            pl.col("fsa_attribute_type")
        )
    )
    # Non-crop definitions are not relevant
    df_long = df_long.filter(pl.col('cdl_name')!='Background')
    return (df_long,)


@app.cell
def _(BaseModel, Field, List):
    # Define pydantic output structure
    class CDLMatchOutput(BaseModel):
        # This ensures a clean list of integers
        codes: List[int] = Field(description="List of integer CDL codes (e.g., [1, 26, 225]). Include all relevant primary and double-crop codes.")

        # This captures the 'Why' for auditing
        reasoning: str = Field(description="Brief agronomic justification for the match.")

    return (CDLMatchOutput,)


@app.function
# Define NASS JSON input structure to optimize LLM processing
def format_nass_definition_for_dspy(nass_entry: dict) -> str:
    """
    Takes a NASS JSON crop definition entry and creates a clean, 
    hierarchical YAML-like block for the LLM.
    """
    # Extract core info
    group = nass_entry.get("group", "UNKNOWN")
    commodity = nass_entry.get("commodity", "UNKNOWN")

    # We focus on the chosen 'definitions' inside the output
    definitions = nass_entry.get("output", {}).get("definitions", [])

    blocks = []
    for i, d in enumerate(definitions):
        block = f"""
        --- NASS DEFINITION {i+1} ---
        GROUP: {group}
        COMMODITY: {commodity}
        CLASS: {d.get('class_desc')}
        PRODUCTION PRACTICE: {d.get('prodn_practice_desc')}
        UTILIZATION PRACTICE: {d.get('util_practice_desc')}
        SEMANTIC HINT: {d.get('semantic_label')}
        """
        blocks.append(block)

    return "\n".join(blocks)


@app.cell
def _(CDLMatchOutput, dspy):
    class NASSCDLMatcherSignature(dspy.Signature):
        """
        Match a NASS crop definition to the correct CDL code using expert heuristic logic.

        DOMAIN HEURISTICS (Rules for resolving metadata gaps):
        1. THE CATCH-ALL PRINCIPLE: Primary CDL codes (e.g., Code 1 for Corn, Code 24 for Winter Wheat, Code 176 for Grass/Pasture) are 'catch-alls'. Even if a specific NASS 'Utilization Practice' like 'Silage' or 'Grazing' is missing from the FSA metadata for a primary code, the primary code is almost always the correct spatial match for that crop.

        2. THE SATELLITE LOGIC: CDL is based on satellite imagery. Satellites see botanical identity (Corn vs. Sorghum), not human intent (Grain vs. Silage). If NASS specifies a practice that is visually identical to the main crop, choose the main crop's CDL code.

        3. COMMODITY > CLASS > PRACTICE: 
           - Priority 1: Commodity (e.g., WHEAT).
           - Priority 2: Class (e.g., WINTER). 
           - Priority 3: Practice (e.g., IRRIGATED). 
           If a Practice is not found in metadata, do not abandon a code that matches the Commodity and Class.

        4. DOUBLE CROP INCLUSION: If NASS indicates a crop that is part of a double-crop CDL name (e.g., 'Dbl Crop WinWht/Soybeans'), you MUST include those double-crop codes in your final answer to ensure acreage is fully captured.
        """

        nass_canonical_definition = dspy.InputField(desc="The hierarchical NASS crop definition. From most general to most specific categorization: GROUP, then COMMODITY, then CLASS. PRODUCTION PRACTICE and UTILIZATION PRACTICE provide additional detail. Semantic hint is a guess at the appropriate common crop name.")

        cdl_table = dspy.InputField(desc="The full list of CDL codes and names, with added FSA metadata")

        answer: CDLMatchOutput = dspy.OutputField(desc="The final structured mapping result.")

    return (NASSCDLMatcherSignature,)


@app.cell
def _(df_long, pl):
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_hide_dataframe_shape=True):
        cdl_fsa_markdown = str(df_long)
    return (cdl_fsa_markdown,)


@app.cell
def _(nass_obs_selection):
    test_nass_entry = nass_obs_selection['FRUIT & TREE NUTS||COFFEE']
    test_nass_input = format_nass_definition_for_dspy(nass_entry = test_nass_entry)
    return


@app.cell
def _(NASSCDLMatcherSignature, dspy, query_fsa_metadata):
    # Initialize the ReAct agent with our SQL tool
    matcher_agent = dspy.ReAct(NASSCDLMatcherSignature, tools=[query_fsa_metadata], max_iters = 20)
    return (matcher_agent,)


@app.cell
def _(Path, duckdb, json_path, tempfile):
    # Define tool that queries SQL database
    # Put CDL codes + FSA semantic hint table into duckdb database
    tmp_dir_handle = tempfile.TemporaryDirectory()
    # Extract the path string
    db_path = Path(tmp_dir_handle.name) / "my_data.duckdb"

    con_writer = duckdb.connect(db_path)
    con_writer.execute("CREATE TABLE cdl_table AS SELECT * FROM df_long")
    con_writer.close() # Close writer so the file is ready for others

    # Now open your READ-ONLY connection
    con_ro = duckdb.connect(db_path, read_only=True)

    # Settings
    output_file = json_path / "nass_to_cdl_mappings.jsonl"

    def query_fsa_metadata(sql_query: str) -> str:
        """
        Query the FSA-CDL crosswalk using DuckDB SQL.
        Table Name: cdl_table
        Columns: 
          - cdl_code: Integer
          - cdl_name: String
          - fsa_attribute_type: (One of 'fsa_crops', 'fsa_types', 'fsa_intended_use')
          - fsa_attribute_value: The actual name/category (e.g., 'Corn', 'Silage', 'HRW')
        Example: SELECT DISTINCT cdl_code FROM cdl_table WHERE fsa_attribute_value LIKE '%Silage%'
        """
        try:
            # We use DuckDB to query the registered Polars dataframe
            result = con_ro.execute(sql_query).df()
            if result.empty:
                return "No matches found in the FSA-CDL crosswalk for that query."
            return result.to_string(index=False)
        except Exception as e:
            return f"SQL Error: {str(e)}"

    return con_ro, output_file, query_fsa_metadata, tmp_dir_handle


@app.cell
def _(
    Path,
    cdl_fsa_markdown,
    islice,
    json,
    matcher_agent,
    nass_obs_selection,
    output_file,
    tqdm,
):
    # Load existing results to avoid duplicate costs
    processed_ids = set()
    if Path(output_file).exists():
        with open(output_file, "r") as _f:
            for line in _f:
                record = json.loads(line)
                processed_ids.add(record["nass_id"])

    print(f"Resuming: {len(processed_ids)} already processed. Total to process: {len(nass_obs_selection)}")

    # Iterate and Predict
    # Use tqdm for a progress bar
    for nass_id, nass_entry in tqdm.tqdm(islice(nass_obs_selection.items(), 300)):

        if nass_id in processed_ids:
            print(f'{nass_id} already processed')
            continue

        try:
            print(f'Processing {nass_id}')
            # Prepare Input
            nass_yaml = format_nass_definition_for_dspy(nass_entry = nass_entry)

            # Execute Agent
            # Note: we pass the summary table to save context window tokens
            # Execute the agent
            prediction = matcher_agent(
                nass_canonical_definition=nass_yaml, 
                cdl_table=cdl_fsa_markdown
            )

            # 3. Store Result
            result = {
                "nass_id": nass_id,
                "mapping": prediction.answer.model_dump(), # Clean nested JSON
                "trajectory": prediction.trajectory # Useful to check the SQL queries later
            }

            # Append to file immediately (Checkpointing)
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")

        except Exception as e:
            print(f"Error processing {nass_id}: {e}")
            # Optional: Store error log
            continue

    print(f"Batch complete. Results saved to {output_file}")
    return


@app.cell
def _(con_ro, tmp_dir_handle):
    con_ro.close()
    # This deletes the temporary file and the folder
    tmp_dir_handle.cleanup()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
