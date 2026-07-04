import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from h2a.paths import RAW, INTERMEDIATE
    import polars as pl

    return INTERMEDIATE, Path, RAW, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CDL crop type crosswalk

    Build a complete CDL code to crop type crosswalk using NASS crop groups when
    the CDL/NASS crosswalk is unambiguous. Manually resolved rows are limited to
    CDL-only codes, non-crop land-cover classes, double-crop classes, and CDL
    labels that NASS maps to multiple groups.
    """)
    return


@app.cell
def _(INTERMEDIATE, Path, RAW, pl):
    cdl_path = RAW / "croplandcros_cdl"

    cdl_codebook = (
        pl.read_excel(
            Path(cdl_path / "CDL_codes_names_colors.xlsx"),
            read_options={"header_row": 3},
        )
        .rename(str.lower)
        .select(
            pl.col("codes").cast(pl.Int64).alias("cdl_code"),
            pl.col("class_names").cast(pl.String).str.strip_chars().alias("cdl_name"),
        )
        .with_columns(
            pl.when(pl.col("cdl_name") == "")
            .then(None)
            .otherwise(pl.col("cdl_name"))
            .alias("cdl_name")
        )
    )

    cdl_nass_xwalk = pl.read_parquet(INTERMEDIATE / "cdl_nass_xwalk.parquet")
    return cdl_codebook, cdl_nass_xwalk


@app.cell
def _(cdl_nass_xwalk, pl):
    NASS_CROP_GROUPS = [
        "FIELD CROPS",
        "FRUIT & TREE NUTS",
        "HORTICULTURE",
        "VEGETABLES",
    ]

    nass_groups_by_cdl = (
        cdl_nass_xwalk.filter(
            pl.col("cdl_code").is_not_null(),
            pl.col("group_desc").is_in(NASS_CROP_GROUPS),
        )
        .group_by("cdl_code")
        .agg(pl.col("group_desc").unique().sort().alias("xwalk_nass_crop_groups"))
        .with_columns(
            pl.when(pl.col("xwalk_nass_crop_groups").list.len() == 1)
            .then(pl.col("xwalk_nass_crop_groups").list.first())
            .otherwise(None)
            .alias("xwalk_crop_type")
        )
    )
    return (nass_groups_by_cdl,)


@app.cell
def _(pl):
    def manual_row(
        cdl_code,
        crop_type,
        is_crop,
        nass_crop_groups,
        source,
        note,
    ):
        return {
            "cdl_code": cdl_code,
            "manual_crop_type": crop_type,
            "manual_is_crop": is_crop,
            "manual_nass_crop_groups": nass_crop_groups,
            "manual_source": source,
            "manual_note": note,
        }


    manual_crop_type = pl.DataFrame(
        [
            # NASS crosswalk ambiguities resolved to the concrete CDL class.
            manual_row(
                12,
                "VEGETABLES",
                True,
                ["VEGETABLES"],
                "manual_nass_resolution",
                "Sweet corn is a NASS vegetable; field crop matches are totals.",
            ),
            manual_row(
                41,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_nass_resolution",
                "CDL sugarbeets correspond to NASS sugarbeets.",
            ),
            manual_row(
                42,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_nass_resolution",
                "CDL dry beans correspond to NASS dry edible beans.",
            ),
            manual_row(
                43,
                "VEGETABLES",
                True,
                ["VEGETABLES"],
                "manual_nass_resolution",
                "Potatoes are treated as vegetables for crop type summaries.",
            ),
            manual_row(
                44,
                "MIXED CROPS",
                True,
                ["FIELD CROPS", "FRUIT & TREE NUTS", "HORTICULTURE", "VEGETABLES"],
                "manual_mixed_cdl_class",
                "Other Crops spans multiple NASS crop groups.",
            ),
            manual_row(
                46,
                "VEGETABLES",
                True,
                ["VEGETABLES"],
                "manual_nass_resolution",
                "Sweet potatoes are treated as vegetables for crop type summaries.",
            ),
            manual_row(
                47,
                "MIXED FRUIT & TREE NUTS/VEGETABLES",
                True,
                ["FRUIT & TREE NUTS", "VEGETABLES"],
                "manual_mixed_cdl_class",
                "Misc Vegs & Fruits spans fruit and vegetable NASS groups.",
            ),
            manual_row(
                48,
                "VEGETABLES",
                True,
                ["VEGETABLES"],
                "manual_nass_resolution",
                "Watermelons are grouped with melons in NASS vegetables.",
            ),
            manual_row(
                53,
                "MIXED FIELD CROPS/VEGETABLES",
                True,
                ["FIELD CROPS", "VEGETABLES"],
                "manual_mixed_cdl_class",
                "CDL peas can include dry edible peas and green peas.",
            ),
            manual_row(
                57,
                "MIXED FIELD CROPS/HORTICULTURE/VEGETABLES",
                True,
                ["FIELD CROPS", "HORTICULTURE", "VEGETABLES"],
                "manual_mixed_cdl_class",
                "NASS places herbs across field crops, horticulture, and vegetables.",
            ),
            manual_row(
                58,
                "MIXED FIELD CROPS/HORTICULTURE",
                True,
                ["FIELD CROPS", "HORTICULTURE"],
                "manual_mixed_cdl_class",
                "Clover is a NASS field crop; wildflowers are horticulture.",
            ),
            manual_row(
                59,
                "MIXED FIELD CROPS/HORTICULTURE",
                True,
                ["FIELD CROPS", "HORTICULTURE"],
                "manual_mixed_cdl_class",
                "Sod is horticulture; grass seed is closest to NASS grass field crops.",
            ),
            manual_row(
                60,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_from_nass_metadata",
                "Switchgrass appears in NASS field crops.",
            ),
            manual_row(
                70,
                "HORTICULTURE",
                True,
                ["HORTICULTURE"],
                "manual_from_nass_metadata",
                "Cut Christmas trees are a NASS horticulture commodity.",
            ),
            manual_row(
                71,
                "FRUIT & TREE NUTS",
                True,
                ["FRUIT & TREE NUTS"],
                "manual_nass_resolution",
                "Other Tree Crops are overwhelmingly fruit and tree nut crops.",
            ),
            manual_row(
                92,
                "AQUACULTURE",
                False,
                ["AQUACULTURE"],
                "manual_from_nass_metadata",
                "Aquaculture is a NASS animals/products group, not a crop group.",
            ),
            manual_row(
                209,
                "VEGETABLES",
                True,
                ["VEGETABLES"],
                "manual_nass_resolution",
                "Cantaloupes are grouped with melons in NASS vegetables.",
            ),
            manual_row(
                210,
                "FRUIT & TREE NUTS",
                True,
                ["FRUIT & TREE NUTS"],
                "manual_from_nass_metadata",
                "Prunes are a NASS fruit crop.",
            ),
            manual_row(
                213,
                "VEGETABLES",
                True,
                ["VEGETABLES"],
                "manual_nass_resolution",
                "Honeydew melons are grouped with melons in NASS vegetables.",
            ),
            manual_row(
                221,
                "FRUIT & TREE NUTS",
                True,
                ["FRUIT & TREE NUTS"],
                "manual_nass_resolution",
                "Strawberries are a NASS fruit crop.",
            ),
            manual_row(
                224,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_from_nass_metadata",
                "Vetch appears under NASS legumes in field crops.",
            ),
            manual_row(
                247,
                "VEGETABLES",
                True,
                ["VEGETABLES"],
                "manual_from_nass_metadata",
                "Turnips are a NASS vegetable.",
            ),
            # Double-crop classes.
            manual_row(
                26,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_double_crop",
                "Winter wheat and soybeans are both field crops.",
            ),
            manual_row(
                225,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_double_crop",
                "Winter wheat and corn are both field crops.",
            ),
            manual_row(
                226,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_double_crop",
                "Oats and corn are both field crops.",
            ),
            manual_row(
                228,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_double_crop",
                "Triticale and corn are both field crops.",
            ),
            manual_row(
                230,
                "MIXED FIELD CROPS/VEGETABLES",
                True,
                ["FIELD CROPS", "VEGETABLES"],
                "manual_double_crop",
                "Lettuce is a vegetable; durum wheat is a field crop.",
            ),
            manual_row(
                231,
                "VEGETABLES",
                True,
                ["VEGETABLES"],
                "manual_double_crop",
                "Lettuce and cantaloupe are both vegetables.",
            ),
            manual_row(
                232,
                "MIXED FIELD CROPS/VEGETABLES",
                True,
                ["FIELD CROPS", "VEGETABLES"],
                "manual_double_crop",
                "Lettuce is a vegetable; cotton is a field crop.",
            ),
            manual_row(
                233,
                "MIXED FIELD CROPS/VEGETABLES",
                True,
                ["FIELD CROPS", "VEGETABLES"],
                "manual_double_crop",
                "Lettuce is a vegetable; barley is a field crop.",
            ),
            manual_row(
                234,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_double_crop",
                "Durum wheat and sorghum are both field crops.",
            ),
            manual_row(
                235,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_double_crop",
                "Barley and sorghum are both field crops.",
            ),
            manual_row(
                236,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_double_crop",
                "Winter wheat and sorghum are both field crops.",
            ),
            manual_row(
                237,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_double_crop",
                "Barley and corn are both field crops.",
            ),
            manual_row(
                238,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_double_crop",
                "Winter wheat and cotton are both field crops.",
            ),
            manual_row(
                239,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_double_crop",
                "Soybeans and cotton are both field crops.",
            ),
            manual_row(
                240,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_double_crop",
                "Soybeans and oats are both field crops.",
            ),
            manual_row(
                241,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_double_crop",
                "Corn and soybeans are both field crops.",
            ),
            manual_row(
                254,
                "FIELD CROPS",
                True,
                ["FIELD CROPS"],
                "manual_double_crop",
                "Barley and soybeans are both field crops.",
            ),
            # CDL land-cover, no-data, and non-crop classes.
            *[
                manual_row(
                    _cdl_code,
                    "NON-CROP",
                    False,
                    [],
                    "manual_non_crop_cdl_class",
                    "CDL land-cover, no-data, fallow, pasture, or non-crop class.",
                )
                for _cdl_code in [
                    0,
                    61,
                    63,
                    64,
                    65,
                    81,
                    82,
                    83,
                    87,
                    88,
                    111,
                    112,
                    121,
                    122,
                    123,
                    124,
                    131,
                    141,
                    142,
                    143,
                    152,
                    176,
                    190,
                    195,
                ]
            ],
        ],
        schema_overrides={"manual_nass_crop_groups": pl.List(pl.Utf8)},
    )
    return (manual_crop_type,)


@app.cell
def _(cdl_codebook, manual_crop_type, nass_groups_by_cdl, pl):
    empty_groups = pl.lit([]).cast(pl.List(pl.Utf8))

    cdl_crop_type = (
        cdl_codebook.join(nass_groups_by_cdl, on="cdl_code", how="left", validate="1:1")
        .join(manual_crop_type, on="cdl_code", how="left", validate="1:1")
        .with_columns(
            (pl.col("cdl_name").is_null()).alias("is_reserved_cdl_code"),
            pl.when(pl.col("manual_nass_crop_groups").is_not_null())
            .then(pl.col("manual_nass_crop_groups"))
            .when(pl.col("xwalk_nass_crop_groups").is_not_null())
            .then(pl.col("xwalk_nass_crop_groups"))
            .otherwise(empty_groups)
            .alias("nass_crop_groups"),
            pl.when(pl.col("cdl_name").is_null())
            .then(pl.lit("RESERVED/UNUSED"))
            .otherwise(pl.coalesce("manual_crop_type", "xwalk_crop_type"))
            .alias("crop_type"),
            pl.when(pl.col("cdl_name").is_null())
            .then(False)
            .otherwise(
                pl.coalesce(
                    "manual_is_crop",
                    pl.col("xwalk_crop_type").is_not_null(),
                    pl.lit(False),
                )
            )
            .alias("is_crop"),
            pl.when(pl.col("cdl_name").is_null())
            .then(pl.lit("cdl_codebook_reserved"))
            .when(pl.col("manual_crop_type").is_not_null())
            .then(pl.col("manual_source"))
            .when(pl.col("xwalk_crop_type").is_not_null())
            .then(pl.lit("nass_cdl_xwalk"))
            .otherwise(None)
            .alias("crop_type_source"),
            pl.when(pl.col("manual_note").is_not_null())
            .then(pl.col("manual_note"))
            .otherwise(None)
            .alias("crop_type_note"),
        )
        .with_columns(
            pl.col("crop_type").str.to_lowercase().alias("crop_type_label"),
            pl.col("cdl_code").cast(pl.String),
        )
        .select(
            [
                "cdl_code",
                "cdl_name",
                "crop_type",
                "crop_type_label",
                "is_crop",
                "is_reserved_cdl_code",
                "nass_crop_groups",
                "crop_type_source",
                "crop_type_note",
            ]
        )
        .sort("cdl_code")
    )
    return (cdl_crop_type,)


@app.cell
def _(INTERMEDIATE, cdl_crop_type):
    cdl_crop_type.write_parquet(INTERMEDIATE / "cdl_crop_type.parquet")
    cdl_crop_type
    return


if __name__ == "__main__":
    app.run()
