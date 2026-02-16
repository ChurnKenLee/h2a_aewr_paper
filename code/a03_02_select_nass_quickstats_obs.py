import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import pyprojroot
    import ibis
    import ibis.selectors as s
    from ibis import _
    import dspy

    return ibis, mo, pyprojroot


@app.cell
def _(pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    binary_path = root_path / 'binaries'
    return (binary_path,)


@app.cell
def _(binary_path, ibis):
    con = ibis.polars.connect()
    survey_crops_table = con.read_parquet(binary_path / 'qs_survey_crops.parquet')
    return (survey_crops_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define set of tools we want DSpy to have access to
    """)
    return


@app.cell
def _(expr):
    class QuickStatsTools:
        def __init__(self, table):
            self.table = table # Ibis table object
        
        def get_commodity_structure(self, commodity: str) -> str:
            """
            Returns unique class_desc, prodn_practice_desc, and util_practice_desc for a commodity.
            """
            # Filter and get unique combinations
            res = (
                self.table
                    .filter(_.commodity_desc == commodity)
                    .select("class_desc", "prodn_practice_desc", "util_practice_desc")
                    .distinct()
                  )
            res_dict = (res.to_polars().to_dicts())
            return res_dict

            def get_coverage_stats(self, commodity: str, statistic: str, class_val: str = None) -> str:
                """
                Returns row counts and non-missing value counts for a specific statistic.
                """
                com_tab = self.table.filter(self.table.commodity_desc == commodity)
                stats_tab = com_tab.filter(self.table.statisticcat_desc == statistic)
    
                if class_val:
                    expr = expr.filter(self.table.CLASS_DESC == class_val)
    
                # Count total rows and non-null/non-(D) values
                # In QuickStats, (D) is a string, so we check for numeric-like values
                stats = (expr.aggregate(
                    total_rows=expr.count(),
                    valid_values=expr.VALUE.cast("float64").count() # Ibis handles the cast/count
                ).execute())
    
                return str(stats.to_dicts())

    return (QuickStatsTools,)


@app.cell
def _(QuickStatsTools, survey_crops_table):
    tools_instance = QuickStatsTools(survey_crops_table)
    test_out = tools_instance.get_commodity_structure(commodity='CORN')
    test_out
    return


@app.cell
def _():
    # class QuickStatsTools:
    #     def __init__(self, table):
    #         self.table = table # Ibis table object

    #     @dspy.tool
    #     def get_commodity_structure(self, commodity: str) -> str:
    #         """Returns unique Class, Production, and Util practices for a commodity."""
    #         # Filter and get unique combinations
    #         res = (self.table
    #                .filter(self.table.COMMODITY_DESC == commodity)
    #                .select("CLASS_DESC", "PRODN_PRACTICE_DESC", "UTIL_PRACTICE_DESC")
    #                .distinct()
    #                .execute())
    #         return str(res.to_dicts())

    #     @dspy.tool
    #     def get_coverage_stats(self, commodity: str, statistic: str, class_val: str = None) -> str:
    #         """Returns row counts and non-missing value counts for a specific statistic."""
    #         expr = self.table.filter(self.table.COMMODITY_DESC == commodity)
    #         expr = expr.filter(self.table.STATISTICCAT_DESC == statistic)

    #         if class_val:
    #             expr = expr.filter(self.table.CLASS_DESC == class_val)

    #         # Count total rows and non-null/non-(D) values
    #         # In QuickStats, (D) is a string, so we check for numeric-like values
    #         stats = (expr.aggregate(
    #             total_rows=expr.count(),
    #             valid_values=expr.VALUE.cast("float64").count() # Ibis handles the cast/count
    #         ).execute())

    #         return str(stats.to_dicts())

    #     @dspy.tool
    #     def peek_short_desc(self, commodity: str, limit: int = 10) -> str:
    #         """Shows actual NASS row descriptions to help identify colloquial names."""
    #         res = (self.table
    #                .filter(self.table.COMMODITY_DESC == commodity)
    #                .select("SHORT_DESC")
    #                .distinct()
    #                .limit(limit)
    #                .execute())
    #         return str(res["SHORT_DESC"].to_list())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Select crop definitions we want for NASS census and survey data
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
