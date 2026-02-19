import marimo

__generated_with = "0.19.11"
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
    import dspy
    from typing import List, Dict
    import json

    return dspy, ibis, json, mo, os, pyprojroot


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path / 'binaries'
    return (binary_path,)


@app.cell
def _(dspy, os):
    # Configure Gemini
    gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    gemini = dspy.LM('google/gemini-3-flash-preview', api_key='YOUR_API_KEY')
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

    # Filter to national, state, and county level of aggregation
    # Do not want counts of operations by stat range
    # Convert values to numeric
    census_crops = (
        census_crops
            .filter(
                _.agg_level_desc.isin(['NATIONAL', 'STATE', 'COUNTY']),
                _.unit_desc != 'OPERATIONS',
                _.freq_desc == 'ANNUAL'
            ).mutate(
                numeric_value = _.value.re_replace(r',', '').cast('float64')
            )
    )

    survey_crops = (
        survey_crops
            .filter(
                _.agg_level_desc.isin(['NATIONAL', 'STATE', 'COUNTY']),
                _.unit_desc != 'OPERATIONS',
                _.freq_desc == 'ANNUAL'
            ).mutate(
                numeric_value = _.value.re_replace(r',', '').cast('float64')
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
def _(ibis, t):
    def create_summary_view(table):
        # Group by the hierarchy and observation_type to get counts
        summary = table.group_by([
            "commodity_desc", 
            "class_desc", 
            "prodn_practice_desc", 
            "util_practice_desc", 
            "observation_type"
        ]).aggregate(
            count=ibis._.count(),
            unique_years=ibis._.year.nunique(),
            unique_counties=ibis._.location_desc.nunique()
        )
        return summary

    summary_t = create_summary_view(t)
    return (summary_t,)


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

    summary_t = unique_survey_crops_summary
    return (summary_t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define set of tools we want DSpy to have access to
    """)
    return


@app.cell
def _(json, summary_t, t):
    def get_hierarchy_exploration(commodity: str) -> str:
        """
        Returns all unique combinations of class, production practice, and utilization practice 
        for a commodity. Use this first to see what sub-categories exist.
        """
        try:
            commodity = commodity.upper()
            res = summary_t.filter(summary_t.commodity_desc == commodity).select(
                "class_desc", "prodn_practice_desc", "util_practice_desc"
            ).distinct().to_polars()
        
            if res.is_empty():
                return f"No hierarchy found for {commodity}."
            
            return json.dumps(res.to_dicts())
        
        except Exception as e:
            return f"Error exploring hierarchy: {str(e)}"

    def find_price_anchors(commodity: str) -> str:
        """
        CRITICAL TOOL: Searches specifically for PRICE data across all classes.
        Use this to identify which classes have price data, as price is the highest priority.
        Returns counts and aggregation levels (usually STATE).
        """
        try:
            commodity = commodity.upper()
            res = summary_t.filter([
                summary_t.commodity_desc == commodity,
                summary_t.observation_type == 'price'
            ]).select(
                "class_desc", "prodn_practice_desc", "util_practice_desc", 
                "agg_level_desc", "count", "unique_years"
            ).to_polars()
        
            if res.is_empty():
                return "No PRICE data found for any class in this commodity."
            
            return json.dumps(res.to_dicts())
        
        except Exception as e:
            return f"Error finding price: {str(e)}"

    def get_coverage_by_agg_level(commodity: str, class_val: str) -> str:
        """
        Returns detailed coverage counts grouped by observation_type AND agg_level_desc.
        Use this to compare the richness of data (e.g., County Yields vs State Prices).
        Helps determine if a class is a 'major' crop with full geography or a 'minor' one.
        """
        try:
            commodity = commodity.upper()
            class_val = class_val.upper()
        
            res = summary_t.filter([
                summary_t.commodity_desc == commodity,
                summary_t.class_desc == class_val
            ]).select(
                "observation_type", "agg_level_desc", "prodn_practice_desc", 
                "util_practice_desc", "count", "unique_years", "unique_locations"
            ).to_polars()
        
            if res.is_empty():
                return f"No coverage data found for class {class_val}."
            
            return json.dumps(res.to_dicts())
        
        except Exception as e:
            return f"Error getting coverage: {str(e)}"

    def get_semantic_samples(commodity: str, class_val: str) -> str:
        """
        Returns the 'short_desc' (the actual data label) for a specific class. 
        Use this when you are unsure if two classes (e.g., 'DRY BEANS' and 'DRY BEANS, EXCL CHICKPEAS') 
        are separate crops or overlapping. The short_desc usually reveals the true definition.
        """
        try:
            # We query the raw table (t) for this, limited to a few rows
            commodity = commodity.upper()
            class_val = class_val.upper()
        
            res = t.filter([
                t.commodity_desc == commodity,
                t.class_desc == class_val
            ]).select("short_desc", "unit_desc").distinct().limit(10).to_polars()
        
            return json.dumps(res.to_dicts())
        
        except Exception as e:
            return f"Error getting samples: {str(e)}"

    return find_price_anchors, get_hierarchy_exploration


@app.cell
def _(dspy):
    class IdentifyCanonicalCrops(dspy.Signature):
        """
        Identify canonical crop definitions for a specific commodity within a sector group.
    
        Context: You are looking at {commodity} in the {group} sector. 
        Be aware that the same commodity in a different group (e.g., Horticulture vs Vegetables) 
        is a different agricultural product with different units and practices.
    
        PRIORITY: PRICE coverage is the most important metric.
        """
        group = dspy.InputField(desc="The group_desc (e.g., VEGETABLES, FIELD CROPS)")
        commodity = dspy.InputField(desc="The commodity_desc (e.g., TOMATOES, CORN)")
    
        canonical_definitions = dspy.OutputField(desc="JSON list of valid (class, prod, util) definitions.")
        reasoning = dspy.OutputField(desc="Why these were chosen based on Price/Yield coverage.")

    return


@app.cell
def _(find_price_anchors, get_hierarchy_exploration):
    get_hierarchy_exploration(commodity = "BEANS")
    find_price_anchors(commodity = "BEANS")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Select crop definitions we want for NASS census and survey data
    """)
    return


if __name__ == "__main__":
    app.run()
