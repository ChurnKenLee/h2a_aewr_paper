import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


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
    import dspy
    from pydantic import BaseModel, Field
    from typing import List
    import json
    import tqdm
    import time

    return (
        BaseModel,
        Field,
        List,
        dspy,
        ibis,
        json,
        mo,
        os,
        pl,
        pyprojroot,
        time,
        tqdm,
    )


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path / 'binaries'
    return binary_path, root_path


@app.cell
def _(dspy, os):
    # Configure Gemini
    gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    gemini = dspy.LM('gemini/gemini-3-flash-preview', api_key=gemini_api_key)
    dspy.configure(lm=gemini)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Preprocess data for DSpy querying
    """)
    return


@app.cell
def _(binary_path, ibis):
    con = ibis.polars.connect()
    survey_crops = con.read_parquet(binary_path / 'qs_survey_crops.parquet')
    census_crops = con.read_parquet(binary_path / 'qs_census_crops.parquet')

    # We only care about years >= 2007
    # Filter to national, state, and county level of aggregation
    # Do not want counts of operations by stat range
    # Convert values to numeric
    census_crops = (
        census_crops
            .filter(
                _.year >= 2007,
                _.agg_level_desc.isin(['NATIONAL', 'STATE', 'COUNTY']),
                _.unit_desc != 'OPERATIONS',
                _.freq_desc == 'ANNUAL'
            ).mutate(
                numeric_value = _.value.re_replace(r',', '').try_cast('float64')
            )
    )

    survey_crops = (
        survey_crops
            .filter(
                _.year >= 2007,
                _.agg_level_desc.isin(['NATIONAL', 'STATE', 'COUNTY']),
                _.unit_desc != 'OPERATIONS',
                _.freq_desc == 'ANNUAL'
            ).mutate(
                numeric_value = _.value.re_replace(r',', '').try_cast('float64')
            )
    )

    census_crops = (
        census_crops.drop(
            _.value
        ).rename(
            {'value':'numeric_value'}
        )
    )

    survey_crops = (
        survey_crops.drop(
            _.value
        ).rename(
            {'value':'numeric_value'}
        )
    )

    # Define the categorization logic for Census Crops
    census_crops = census_crops.mutate(
        observation_type=ibis.cases(
            ((_.unit_desc == "ACRES") & (_.statisticcat_desc != "AREA NON-BEARING"), "area"),
            (_.statisticcat_desc == "PRODUCTION", "production"),
            (_.statisticcat_desc == "YIELD", "yield"),
            else_=None
        )
    )

    # Categorization logic for Survey Crops
    survey_crops = survey_crops.mutate(
        observation_type=ibis.cases(
            (_.unit_desc.isin(["ACRES", "ACRES WHERE TAPS SET"]), "area"),
            (_.statisticcat_desc == "PRICE RECEIVED", "price"),
            (_.statisticcat_desc == "PRODUCTION", "production"),
            (_.statisticcat_desc == "YIELD", "yield"),
            else_=None
        )
    )
    return census_crops, survey_crops


@app.cell
def _(census_crops, survey_crops):
    # Preprocessing: It's helpful to have a view of unique combinations and their counts to speed up the LLM's "browsing"
    unique_census_crops_summary = census_crops.group_by([
        "group_desc", 
        "commodity_desc", 
        "class_desc", 
        "prodn_practice_desc", 
        "util_practice_desc", 
        "observation_type", 
        "agg_level_desc"
    ]).aggregate(
        count=_.count(),
        min_year=_.year.min(),
        max_year=_.year.max(),
        unique_counties=_.county_name.nunique()
    )

    unique_survey_crops_summary = survey_crops.group_by([
        "group_desc", 
        "commodity_desc", 
        "class_desc", 
        "prodn_practice_desc", 
        "util_practice_desc", 
        "observation_type", 
        "agg_level_desc"
    ]).aggregate(
        count=_.count(),
        min_year=_.year.min(),
        max_year=_.year.max(),
        unique_counties=_.county_name.nunique()
    )

    t = survey_crops
    summary_t = unique_survey_crops_summary
    return (summary_t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Summary table is sufficiently small for each group-commodity that we do not need to use tools to query the table, we can just feed the entire table in the signature
    """)
    return


@app.cell
def _(BaseModel, Field, List):
    # Structured output defintion
    # This is a crop for any given group-commodity
    class CropDefinition(BaseModel):
        class_desc: str = Field(description="The class name from USDA QuickStats, e.g., class WINTER for commodity WHEAT")
        prodn_practice_desc: str = Field(description="The production practice, e.g., ALL PRODUCTION PRACTICES")
        util_practice_desc: str = Field(description="The utilization practice, e.g., GRAIN")
        reasoning: str = Field(description="Briefly explain why this is a canonical crop based on coverage.")
        semantic_label: str = Field(description="A simplified, plain-English name for this crop (e.g., 'Winter Wheat' or 'Strawberries'). This will be used for future cross-dataset mapping.")

    class CanonicalCropOutput(BaseModel):
        definitions: List[CropDefinition] = Field(description="A list of the identified canonical crop definitions.")

    return (CanonicalCropOutput,)


@app.cell
def _(CanonicalCropOutput, dspy):
    # Signature for DSpy; refine instruction prompt here
    class DefineCanonicalCrops(dspy.Signature):
        """
        TASK: Identify the 'canonical' definitions for a commodity to maximize data coverage for a multi-dataset merge (USDA Census, CroplandCROS Cropland Data Layer (CDL)).

        CONTEXT: Included table is USDA QuickStats survey data aggregated to provide summary counts for multiple crop definitions within a group-commodity. A crop is defined as a unique combination of group (group_desc), commodity (commodity_desc), class (class_desc), production practice (prodn_practice_desc), and utilization practice (util_practice_desc). Summary counts are provided for observation types, i.e., observation_type == price, production, area, yield, for each crop definition, across different levels of aggregation, agg_level_desc == NATIONAL, STATE, COUNTY. This is USDA survey data, so data is collected and available every year.

        PROBLEM: Choice of specific values of class_desc, prodn_practice_desc, util_practice_desc for defining a crop is not straightforward enough to be determined algorithmically. Appropriate choice requires looking at all the values available for any given commodity, e.g., may choose GRAPES-ALL CLASSES as the appropriate representative 'canonical' definition for grapes and ignoring other grape classes, while PEPPERS does not have an appropriate ALL CLASSES to pick, and have to pick both classes PEPPERS-BELL and PEPPERS-CHILE for bell peppers and chile peppers as different crops. Evaluating coverage with different definitions further complicates matters. Add on top the desire for merging with CDL, which defines crops at a level that does not match cleanly with commodity_desc or class_desc alone consistently in NASS QuickStats data.

        HEURISTICS:
        1. PRICE PRIORITY: Price is the 'anchor' variable. A definition with Price data (usually STATE level, ~50 records/year) is almost always preferred. Price coverage is prioritized.
        2. GEOGRAPHIC AGGREGATION SCALING: Recognize that 50 records at the STATE level represents HIGHER coverage than 50 records at the COUNTY level.
        3. TEMPORAL AGGREGATION SCALING: USDA Survey data is collected and published annually, so for example, 3000 records at the COUNTY level for 1 year (as shown by the first and last observed years) is much better than 6000 records at the COUNTY level for 20 years.
        4. TEMPORAL COVERAGE: Cropland Data Layer provides annual data going to 2008, so consistent and wide temporal coverage is preferable.
        5. SINGLE CROP IN COMMODITY: If a commodity contains multiple subclasses that refer to the same crop, e.g., FRUIT & TREE NUTS-GRAPES has classes WINE TYPE, TABLE TYPE, RAISIN TYPE, JUICE TYPE, prefer the universal class ALL CLASSES if available and does not sacrifice coverage too much.
        6. MULTIPLE CROPS IN COMMODITY: If a commodity contains distinct crops that have high-quality coverage, e.g., group-commodity VEGETABLES-PEPPERS has classes CHILE and BELL that are normally considered different crops, identify them as separate definition of a crop.
        7. CDL COMPATIBILITY: The Cropland Data Layer (CDL) usually maps to (commodity + class), but not always. Sometimes maps to commodity alone, so need to choose representative class. Avoid overly specific class, production practice, and utilization practice unless they are the only ones with data.
        8. THE 'ALL CLASSES' TRAP: If 'ALL CLASSES' has Price data but specific subclasses do not, favor 'ALL CLASSES' to ensure the Price variable is available for the merge.
        """
        group = dspy.InputField(desc="The broad USDA sector, e.g., FIELD CROPS vs VEGETABLES.")
        commodity = dspy.InputField(desc="The specific commodity being analyzed. The highest level classification of crops in any USDA group sector.")
        # Pass the data (agg summary table) as a formatted string (Markdown table)
        data_table = dspy.InputField(desc="A Markdown table containing summary counts.")

        # Enforce output type hint via pydantic model
        output: CanonicalCropOutput = dspy.OutputField()

    return (DefineCanonicalCrops,)


@app.cell
def _(DefineCanonicalCrops, dspy):
    # TypedPredictor handles the validation and conversion back to Pydantic objects
    # This enforces output structure
    analyzer = dspy.ChainOfThought(DefineCanonicalCrops)
    return (analyzer,)


@app.cell
def _():
    # A list of test commodities with known issues
    test_cases = [
        {"group": "FIELD CROPS", "commodity": "WHEAT"}, # Multiple Classes (Winter/Spring)
        {"group": "VEGETABLES", "commodity": "TOMATOES"}, # Utilization (Fresh vs Proc)
        {"group": "FIELD CROPS", "commodity": "SOYBEANS"}, # Simple Class, complex practices
        {"group": "VEGETABLES", "commodity": "BEANS"}, # Overlapping Class names
        {"group": "VEGETABLES", "commodity": "MELONS"} # Class identifies distinct crops
    ]

    # group_val = test_cases[3]['group']
    # comm_val = test_cases[3]['commodity']
    # table_str = markdown_summary_table(group_val, comm_val)
    # prediction = analyzer(
    #     group=group_val,
    #     commodity=comm_val,
    #     data_table=table_str
    # )

    # # Access the data programmatically
    # # Because of the type hint, prediction.output is already a CanonicalCropOutput object
    # for definition in prediction.output.definitions:
    #     print(f"Class: {definition.class_desc}")
    #     print(f"Production practice: {definition.prodn_practice_desc}")
    #     print(f"Utilization practice: {definition.util_practice_desc}")
    #     print(f"Logic: {definition.reasoning}")
    #     print("-" * 20)
    return


@app.cell
def _(summary_t):
    # Create markdown table (using pandas + tabulate)
    # Get summary stats from Ibis/Polars table
    def markdown_summary_table(group_val, comm_val):
        summary_df = (summary_t
            .filter([
                _.group_desc == group_val,
                _.commodity_desc == comm_val,
                _.observation_type.notnull() # Only include categorized obs to save on tokens
            ]).to_polars())

        if summary_df.is_empty():
            return None

        # Convert to Markdown Table (good for LLMs)
        table_str = summary_df.to_pandas().to_markdown(index=False)
        return table_str

    return (markdown_summary_table,)


@app.cell
def _(summary_t):
    # Extract all unique tasks from summary table
    all_tasks = (summary_t
        .select("group_desc", "commodity_desc")
        .distinct()
        .to_polars()
        .to_dicts())
    return (all_tasks,)


@app.cell
def _(json, root_path):
    # Store LM response results in json
    json_path = root_path / 'code' / 'json'
    mapping_file = json_path / 'nass_quickstats_survey_obs_selection.json'

    # Load existing results to resume a partial run
    if mapping_file.exists():
        with open(mapping_file, 'r') as _f:
            master_results = json.load(_f)
    else:
        master_results = {}
    return mapping_file, master_results


@app.function
# We can call .model_dump() on the output pydantic v2 object to serialize to JSON
def capture_full_prediction(prediction, task_key, group, commodity):
    # 1. Start with the 'metadata' you need for lookup
    record = {
        "task_key": task_key,
        "group": group,
        "commodity": commodity,
    }

    # 2. Extract all keys from the Prediction object
    # This captures 'rationale' (from COT) and any other fields
    for key in prediction.keys():
        val = getattr(prediction, key)

        # Check if the value is a Pydantic V2 model
        if hasattr(val, "model_dump"):
            record[key] = val.model_dump() # output goes here
        else:
            record[key] = val # reasoning goes here

    return record


@app.cell
def _(
    all_tasks,
    analyzer,
    json,
    mapping_file,
    markdown_summary_table,
    master_results,
    time,
    tqdm,
):
    for task in tqdm.tqdm(all_tasks, desc="Processing Commodities"):
        g_val = task['group_desc']
        c_val = task['commodity_desc']

        # Create a unique key for the dictionary
        task_key = f"{g_val}||{c_val}"
        print(task_key)

        # Skip if already processed
        if task_key in master_results:
            print(f'{task_key} already processed')
            continue

        # Generate the markdown table that provides summary obs counts to LM
        table_str = markdown_summary_table(g_val, c_val)

        # No statistic available for this particular group-commodity
        if table_str is None:
            print(f'{task_key} has no available observation types')
            continue

        try:
            # Run the LLM
            prediction = analyzer(
                group=g_val,
                commodity=c_val,
                data_table=table_str
            )

            # Store as a dict that is straight JSON
            master_results[task_key] = capture_full_prediction(prediction, task_key, g_val, c_val)

            # Autosave every 10 iterations to prevent data loss
            if len(master_results) % 10 == 0:
                with open(mapping_file, 'w') as _f:
                    json.dump(master_results, _f, indent=2)

        except Exception as e:
            print(f"Error processing {task_key}: {e}")
            # Optional: Sleep to handle rate limits
            time.sleep(5)

    # Save the responses
    with open(mapping_file, 'w') as f:
        json.dump(master_results, f, indent=4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now extract the rows we want and export binaries
    """)
    return


@app.cell
def _(json, mapping_file, pl):
    # NASS crop definitions
    with open(mapping_file, 'r') as _f:
        nass_obs_selection_json = json.load(_f)
    nass_obs_selection = pl.from_dicts(list(nass_obs_selection_json.values())) # Unpack JSON in pl df
    nass_obs_selection = (nass_obs_selection
        .drop('reasoning') # shares same name with column inside definitions
        .unnest('output') # output only has 1 item inside: definitions of crops selected
        .explode('definitions') # definitions has multiple entries, each definition is a selected crop
        .unnest('definitions') # each crop has multiple elements within each definition (class, prodn, util)
        .rename(mapping={
            'task_key':'nass_id',
            'group':'group_desc',
            'commodity':'commodity_desc',
            'reasoning':'nass_reasoning',
            'semantic_label':'nass_semantic_label'
        }))
    return (nass_obs_selection,)


@app.cell
def _(census_crops, nass_obs_selection, survey_crops):
    census_selected_crops = census_crops.to_polars()
    census_selected_crops = census_selected_crops.join(
        nass_obs_selection,
        how='inner',
        on=['group_desc',
            'commodity_desc',
            'class_desc',
            'prodn_practice_desc',
            'util_practice_desc']
    )

    survey_selected_crops = survey_crops.to_polars()
    survey_selected_crops = survey_selected_crops.join(
        nass_obs_selection,
        how='inner',
        on=['group_desc',
            'commodity_desc',
            'class_desc',
            'prodn_practice_desc',
            'util_practice_desc']
    )
    return census_selected_crops, survey_selected_crops


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Need to add cropland asset data for legacy pipeline reasons for now
    """)
    return


@app.cell
def _(binary_path, census_selected_crops, pl, survey_selected_crops):
    census_economics = pl.read_parquet(binary_path / "qs_census_economics.parquet")
    census_economics = (census_economics
        .filter(
            (pl.col("freq_desc") == "ANNUAL") & 
            (pl.col("agg_level_desc").is_in(["NATIONAL", "STATE", "COUNTY"]))
        )
        .with_columns(
            pl.when(pl.col("value") == "(Z)") # "(Z)" means less than 0.5
                .then(pl.lit(0.0))
                .otherwise(
                    pl.col("value")
                        .str.replace_all(",", "")
                        .cast(pl.Float64, strict=False)
                )
                .alias("value")
        )
        .filter(pl.col("value").is_not_null())
    )

    # 3. Create Cropland Subset
    census_cropland = (census_economics
        .filter(
            (pl.col("commodity_desc").is_in(["FARM OPERATIONS", "AG LAND"])) &
            (pl.col("short_desc") == "AG LAND, CROPLAND - ACRES") &
            (pl.col("domain_desc") == "TOTAL")
        )
        .with_columns(
            pl.lit('cropland_asset').alias('observation_type')
        ))

    # Convert from categorical to string for concatenating
    census_cropland = census_cropland.with_columns(
        pl.col(pl.Categorical).cast(pl.String)
    )

    # Combine and export
    # Note that polars' handling of categoricals/enums are non-standard and R cannot parse it
    # So we keep all categoricals as strings for exporting
    # Since the parquet file is massively reduced in size, should not blow up memory usage in R
    qs_census_selected_obs = pl.concat([census_selected_crops, census_cropland], how="diagonal")
    qs_survey_selected_obs = survey_selected_crops

    qs_census_selected_obs.write_parquet(binary_path / "qs_census_selected_obs.parquet")
    qs_survey_selected_obs.write_parquet(binary_path / "qs_survey_selected_obs.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
