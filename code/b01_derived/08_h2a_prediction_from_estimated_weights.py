# Purpose: Score county H-2A usage from saved cutoff-specific PPML models.
# Inputs: Cutoff model parameters, climate normals, soil cells, and 2011 BEA employment.
# Output: h2a_prediction_using_elastic_net_by_cutoff.parquet.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl

    from h2a.geography import assert_geo_columns
    from h2a.paths import INTERMEDIATE

    return INTERMEDIATE, assert_geo_columns, mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Predict H-2A usage from saved PPML models

    This notebook discovers every completed cutoff-specific model, reconstructs
    its training-time transformations and categorical mappings, and applies the
    saved PPML equation to county climate normals and soil cells. It does not
    recalculate any preprocessing statistic from the scoring sample.
    """)


@app.cell
def _(INTERMEDIATE, pl):
    model_spec = "climate_norm_static_v1"

    def _cutoff_from_path(path):
        try:
            return int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Cannot parse cutoff year from {path.name!r}.") from exc

    model_paths = sorted(
        INTERMEDIATE.glob("h2a_prediction_elastic_net_model_cutoff_*.parquet"),
        key=_cutoff_from_path,
    )
    if not model_paths:
        raise FileNotFoundError(
            f"No cutoff-specific H-2A model Parquets were found in {INTERMEDIATE}."
        )

    model_cutoff_years = [_cutoff_from_path(path) for path in model_paths]
    if len(model_cutoff_years) != len(set(model_cutoff_years)):
        raise ValueError("More than one model artifact exists for a cutoff year.")
    for _path, _cutoff in zip(model_paths, model_cutoff_years, strict=True):
        if "model_spec" not in pl.read_parquet_schema(_path):
            raise ValueError(
                f"Stale or incompatible model {_path.name}: it has no "
                f"model_spec={model_spec!r} marker."
            )
        _model_specs = (
            pl.scan_parquet(_path)
            .select("model_spec")
            .unique()
            .collect()["model_spec"]
            .to_list()
        )
        if _model_specs != [model_spec]:
            raise ValueError(
                f"Stale or incompatible model {_path.name}: expected "
                f"model_spec={model_spec!r}, observed {_model_specs!r}."
            )

    print(
        f"Discovered {len(model_paths)} cutoff models spanning "
        f"{model_cutoff_years[0]}-{model_cutoff_years[-1]}."
    )
    return model_cutoff_years, model_paths, model_spec


@app.cell
def _(model_paths, model_spec, pl):
    expected_model_schema = {
        "cutoff_year": pl.Int32,
        "model_spec": pl.String,
        "record_type": pl.String,
        "interaction_order": pl.Int32,
        "feature_id": pl.Int32,
        "feature_name": pl.String,
        "categorical_columns": pl.List(pl.String),
        "categorical_values": pl.List(pl.String),
        "covariate_scope": pl.String,
        "covariate": pl.String,
        "coefficient": pl.Float32,
        "imputation_value": pl.Float32,
        "center": pl.Float32,
        "scale": pl.Float32,
    }
    _reference_path = model_paths[0]
    if pl.read_parquet_schema(_reference_path) != expected_model_schema:
        raise ValueError(f"Unexpected model schema in {_reference_path.name}.")
    if pl.scan_parquet(_reference_path).select("model_spec").unique().collect()[
        "model_spec"
    ].to_list() != [model_spec]:
        raise ValueError("The reference model has an incompatible model_spec.")

    reference_transforms = (
        pl.scan_parquet(_reference_path)
        .filter(pl.col("record_type") == "continuous_transform")
        .select(
            "covariate_scope",
            "covariate",
            "imputation_value",
            "center",
            "scale",
        )
        .collect()
    )
    if reference_transforms.height == 0:
        raise ValueError("The reference model has no continuous transforms.")
    _transform_keys = reference_transforms.select("covariate_scope", "covariate")
    if _transform_keys.unique().height != _transform_keys.height:
        raise ValueError("The reference model contains duplicate transforms.")

    county_covariates = reference_transforms.filter(
        pl.col("covariate_scope") == "county"
    )["covariate"].to_list()
    patch_covariates = reference_transforms.filter(
        pl.col("covariate_scope") == "patch"
    )["covariate"].to_list()
    if not county_covariates or not patch_covariates:
        raise ValueError("Both county and patch transforms are required.")
    _normal_covariates = [
        column for column in county_covariates if column.startswith("normal_cb_")
    ]
    if not _normal_covariates or county_covariates != _normal_covariates:
        raise ValueError(
            "Saved model must contain normal_cb_* county covariates only."
        )

    reference_categorical_metadata = (
        pl.scan_parquet(_reference_path)
        .filter((pl.col("record_type") == "weight") & pl.col("covariate").is_null())
        .select(
            "interaction_order",
            "feature_id",
            "feature_name",
            "categorical_columns",
            "categorical_values",
        )
        .sort("interaction_order", "feature_id")
        .collect()
    )
    if reference_categorical_metadata.height == 0:
        raise ValueError("The reference model has no categorical feature metadata.")

    categorical_covariates = []
    for _columns in reference_categorical_metadata["categorical_columns"].to_list():
        for _column in _columns:
            if _column not in categorical_covariates:
                categorical_covariates.append(_column)

    print(
        f"Model design: {len(county_covariates)} county covariates, "
        f"{len(patch_covariates)} patch covariates, and "
        f"{len(categorical_covariates)} categorical covariates."
    )
    return (
        categorical_covariates,
        county_covariates,
        expected_model_schema,
        patch_covariates,
        reference_categorical_metadata,
        reference_transforms,
    )


@app.cell
def _(
    INTERMEDIATE,
    assert_geo_columns,
    categorical_covariates,
    county_covariates,
    np,
    patch_covariates,
    pl,
):
    climate_source = pl.read_parquet(
        INTERMEDIATE / "county_h2a_prediction_climate_basis_annual.parquet",
        columns=["county_fips"] + county_covariates,
    )
    assert_geo_columns(climate_source, ["county_fips"])
    climate = climate_source.unique().sort("county_fips")
    if climate.height == 0 or climate["county_fips"].n_unique() != climate.height:
        raise ValueError("Climate normals must have exactly one row per county.")
    if not np.isfinite(climate.select(county_covariates).to_numpy()).all():
        raise ValueError("Scoring climate normals must be finite.")

    bea = (
        pl.read_parquet(
            INTERMEDIATE / "bea_farm_nonfarm_emp.parquet",
            columns=["year", "county_fips", "bea_farm_emp"],
        )
        .filter(pl.col("year") == 2011)
        .select(
            "county_fips",
            pl.col("bea_farm_emp").alias("bea_farm_emp_2011"),
        )
        .filter(
            pl.col("bea_farm_emp_2011").is_not_null(),
            pl.col("bea_farm_emp_2011") > 0,
        )
    )
    assert_geo_columns(bea, ["county_fips"])

    county_lookup = (
        climate.select("county_fips")
        .unique()
        .join(bea, on="county_fips", how="inner")
        .sort("county_fips")
        .with_row_index("_county_index")
    )
    county_scoring = (
        climate.join(county_lookup, on="county_fips", how="inner", validate="1:1")
        .sort("_county_index")
        .with_row_index("_group_index")
    )
    if county_scoring.height == 0:
        raise ValueError("No counties remain after joining climate and BEA inputs.")

    _soil_columns = (
        ["county_fips", "soil_cell_id", "total_acres"]
        + patch_covariates
        + categorical_covariates
    )
    soil = pl.read_parquet(
        INTERMEDIATE / "county_h2a_prediction_gnatsgo_soil_cells.parquet",
        columns=_soil_columns,
    )
    assert_geo_columns(soil, ["county_fips"])
    soil_scoring = (
        soil.join(
            county_lookup,
            on="county_fips",
            how="inner",
        )
        .with_columns(
            [pl.col(column).fill_null("MISSING") for column in categorical_covariates]
        )
        .sort("_county_index", "soil_cell_id")
        .with_columns(
            (
                pl.col("total_acres")
                / pl.col("total_acres").sum().over("_county_index")
            ).alias("_acreage_frac")
        )
        .with_columns(
            (pl.col("_acreage_frac") * pl.col("bea_farm_emp_2011"))
            .cast(pl.Float32)
            .alias("_patch_exposure")
        )
    )
    _patch_exposure = soil_scoring["_patch_exposure"].to_numpy()
    if not np.isfinite(_patch_exposure).all() or (_patch_exposure < 0).any():
        raise ValueError("Patch exposures must be finite and nonnegative.")

    _exposure_check = soil_scoring.group_by("_county_index").agg(
        pl.col("_patch_exposure").sum().alias("patch_exposure_sum"),
        pl.col("bea_farm_emp_2011").first(),
    )
    if not np.allclose(
        _exposure_check["patch_exposure_sum"].to_numpy(),
        _exposure_check["bea_farm_emp_2011"].to_numpy(),
        rtol=2e-6,
        atol=2e-4,
    ):
        raise ValueError("County patch exposures do not sum to 2011 BEA employment.")

    print(
        f"Scoring design contains {county_scoring.height:,} counties and "
        f"{soil_scoring.height:,} soil patches with fixed 2011 exposure."
    )
    return county_scoring, soil_scoring


@app.cell
def _(np, pl):
    def build_feature_ids(soil_scoring, categorical_metadata):
        feature_ids = {}
        orders = sorted(categorical_metadata["interaction_order"].unique())

        for order in orders:
            order_metadata = categorical_metadata.filter(
                pl.col("interaction_order") == order
            )
            combo_records = {}
            for row in order_metadata.iter_rows(named=True):
                columns = tuple(row["categorical_columns"])
                combo_records.setdefault(columns, []).append(row)

            order_id_columns = []
            for columns, records in sorted(
                combo_records.items(),
                key=lambda item: min(row["feature_id"] for row in item[1]),
            ):
                lookup_rows = []
                for row in records:
                    values = row["categorical_values"]
                    lookup_row = dict(zip(columns, values, strict=True))
                    lookup_row["_feature_id"] = row["feature_id"]
                    lookup_rows.append(lookup_row)
                lookup = pl.DataFrame(lookup_rows)
                if lookup.select(list(columns)).unique().height != lookup.height:
                    raise ValueError(
                        f"Duplicate category mapping for order {order}, "
                        f"columns {columns}."
                    )

                mapped = (
                    soil_scoring.select(list(columns))
                    .with_row_index("_row_index")
                    .join(lookup, on=list(columns), how="left", validate="m:1")
                    .sort("_row_index")
                )
                if (
                    mapped.height != soil_scoring.height
                    or mapped["_feature_id"].null_count()
                ):
                    raise ValueError(
                        f"Scoring data contain an unseen category for order "
                        f"{order}, columns {columns}."
                    )
                order_id_columns.append(
                    mapped["_feature_id"].to_numpy().astype(np.int32)
                )

            feature_ids[int(order)] = np.column_stack(order_id_columns)

        return feature_ids

    return (build_feature_ids,)


@app.cell
def _(np, pl):
    def load_saved_model(
        model_path,
        cutoff_year,
        expected_model_spec,
        expected_model_schema,
        reference_categorical_metadata,
        reference_transforms,
        county_covariates,
        patch_covariates,
    ):
        if pl.read_parquet_schema(model_path) != expected_model_schema:
            raise ValueError(f"Unexpected model schema in {model_path.name}.")

        model_specs = (
            pl.scan_parquet(model_path)
            .select("model_spec")
            .unique()
            .collect()["model_spec"]
            .to_list()
        )
        if model_specs != [expected_model_spec]:
            raise ValueError(
                f"{model_path.name} contains model specs {model_specs}, "
                f"expected only {expected_model_spec!r}."
            )

        model_cutoffs = (
            pl.scan_parquet(model_path)
            .select("cutoff_year")
            .unique()
            .collect()["cutoff_year"]
            .to_list()
        )
        if model_cutoffs != [cutoff_year]:
            raise ValueError(
                f"{model_path.name} contains cutoff values {model_cutoffs}, "
                f"expected only {cutoff_year}."
            )

        transforms = (
            pl.scan_parquet(model_path)
            .filter(pl.col("record_type") == "continuous_transform")
            .select(
                "covariate_scope",
                "covariate",
                "imputation_value",
                "center",
                "scale",
            )
            .collect()
        )
        if transforms.select("covariate_scope", "covariate").rows() != (
            reference_transforms.select("covariate_scope", "covariate").rows()
        ):
            raise ValueError(
                f"Continuous covariate layout changed in {model_path.name}."
            )
        if not transforms.select(
            pl.all_horizontal(
                pl.col("imputation_value", "center", "scale").is_finite()
            ).all()
        ).item():
            raise ValueError(f"Non-finite transform in {model_path.name}.")
        if not (transforms["scale"] > 0).all():
            raise ValueError(f"Nonpositive transform scale in {model_path.name}.")

        categorical_metadata = (
            pl.scan_parquet(model_path)
            .filter((pl.col("record_type") == "weight") & pl.col("covariate").is_null())
            .select(
                "interaction_order",
                "feature_id",
                "feature_name",
                "categorical_columns",
                "categorical_values",
            )
            .sort("interaction_order", "feature_id")
            .collect()
        )
        if not categorical_metadata.equals(reference_categorical_metadata):
            raise ValueError(
                f"Categorical feature mapping changed in {model_path.name}."
            )

        bias_rows = (
            pl.scan_parquet(model_path)
            .filter(pl.col("record_type") == "global_bias")
            .select("coefficient")
            .collect()
        )
        if bias_rows.height != 1 or not np.isfinite(bias_rows["coefficient"][0]):
            raise ValueError(f"Invalid global bias in {model_path.name}.")
        bias = np.float32(bias_rows["coefficient"][0])

        weights = (
            pl.scan_parquet(model_path)
            .filter(pl.col("record_type") == "weight")
            .select(
                "interaction_order",
                "feature_id",
                "covariate_scope",
                "covariate",
                "coefficient",
            )
            .collect()
        )
        if not weights["coefficient"].is_finite().all():
            raise ValueError(f"Non-finite coefficient in {model_path.name}.")

        covariate_keys = [("county", covariate) for covariate in county_covariates] + [
            ("patch", covariate) for covariate in patch_covariates
        ]
        coefficient_map = pl.DataFrame(
            {
                "covariate_scope": [key[0] for key in covariate_keys],
                "covariate": [key[1] for key in covariate_keys],
                "_coefficient_index": np.arange(
                    1, len(covariate_keys) + 1, dtype=np.int32
                ),
            }
        )
        mapped_weights = weights.join(
            coefficient_map,
            on=["covariate_scope", "covariate"],
            how="left",
            validate="m:1",
        ).with_columns(
            pl.when(pl.col("covariate").is_null())
            .then(pl.lit(0, dtype=pl.Int32))
            .otherwise(pl.col("_coefficient_index"))
            .alias("_coefficient_index")
        )
        if mapped_weights.filter(pl.col("_coefficient_index").is_null()).height:
            raise ValueError(f"Unknown covariate coefficient in {model_path.name}.")

        coefficient_width = 1 + len(covariate_keys)
        weight_matrices = {}
        for order in sorted(mapped_weights["interaction_order"].unique()):
            order_weights = mapped_weights.filter(pl.col("interaction_order") == order)
            if (
                order_weights.select("feature_id", "_coefficient_index").unique().height
                != order_weights.height
            ):
                raise ValueError(
                    f"Duplicate order-{order} coefficient in {model_path.name}."
                )
            feature_count = int(order_weights["feature_id"].max()) + 1
            matrix = np.full(
                (feature_count, coefficient_width), np.nan, dtype=np.float32
            )
            matrix[
                order_weights["feature_id"].to_numpy(),
                order_weights["_coefficient_index"].to_numpy(),
            ] = order_weights["coefficient"].to_numpy()
            if not np.isfinite(matrix).all():
                raise ValueError(
                    f"Incomplete order-{order} weights in {model_path.name}."
                )
            weight_matrices[int(order)] = matrix

        return bias, weight_matrices, transforms

    return (load_saved_model,)


@app.cell
def _(np, pl):
    def apply_saved_transforms(frame, scope, covariates, transform_rows):
        transform_lookup = {
            (row["covariate_scope"], row["covariate"]): row
            for row in transform_rows.iter_rows(named=True)
        }
        values = frame.select(covariates).to_numpy().astype(np.float32)

        for column_index, covariate in enumerate(covariates):
            transform = transform_lookup[(scope, covariate)]
            column = values[:, column_index]
            missing = np.isnan(column)
            if missing.any():
                column[missing] = np.float32(transform["imputation_value"])
            if not np.isfinite(column).all():
                raise ValueError(
                    f"Non-finite scoring value for {scope} covariate {covariate!r}."
                )
            values[:, column_index] = (
                column - np.float32(transform["center"])
            ) / np.float32(transform["scale"])

        return values

    def score_saved_model(
        cutoff_year,
        model_spec,
        bias,
        weight_matrices,
        transforms,
        feature_ids,
        county_scoring,
        soil_scoring,
        county_covariates,
        patch_covariates,
        chunk_size=50_000,
    ):
        X_county = apply_saved_transforms(
            county_scoring, "county", county_covariates, transforms
        )
        X_patch = apply_saved_transforms(
            soil_scoring, "patch", patch_covariates, transforms
        )
        county_index = soil_scoring["_county_index"].to_numpy().astype(np.int32)
        patch_exposure = soil_scoring["_patch_exposure"].to_numpy().astype(np.float32)
        predicted_count = np.zeros(county_scoring.height, dtype=np.float32)
        county_width = len(county_covariates)

        for chunk_start in range(0, soil_scoring.height, chunk_size):
            chunk_end = min(chunk_start + chunk_size, soil_scoring.height)
            chunk = slice(chunk_start, chunk_end)
            chunk_count = chunk_end - chunk_start
            patch_log_rate = np.full(chunk_count, bias, dtype=np.float32)
            county_slopes = np.zeros((chunk_count, county_width), dtype=np.float32)

            for order in sorted(weight_matrices):
                matrix = weight_matrices[order]
                ids_for_order = feature_ids[order][chunk]
                for combo_index in range(ids_for_order.shape[1]):
                    gathered = matrix[ids_for_order[:, combo_index]]
                    patch_log_rate += gathered[:, 0]
                    county_slopes += gathered[:, 1 : 1 + county_width]
                    patch_log_rate += np.sum(
                        gathered[:, 1 + county_width :] * X_patch[chunk],
                        axis=1,
                        dtype=np.float32,
                    )

            chunk_county_index = county_index[chunk]
            log_rate = patch_log_rate + np.sum(
                county_slopes * X_county[chunk_county_index],
                axis=1,
                dtype=np.float32,
            )
            np.minimum(log_rate, np.float32(15.0), out=log_rate)
            predicted_patch_count = patch_exposure[chunk] * np.exp(log_rate)
            np.add.at(predicted_count, chunk_county_index, predicted_patch_count)

        if not np.isfinite(predicted_count).all() or (predicted_count < 0).any():
            raise ValueError(
                f"Cutoff-{cutoff_year} predictions must be finite and nonnegative."
            )

        result = (
            county_scoring.select("county_fips", "bea_farm_emp_2011")
            .with_columns(
                pl.lit(cutoff_year, dtype=pl.Int32).alias("cutoff_year"),
                pl.lit(model_spec).alias("model_spec"),
                pl.Series("predicted_h2a_count", predicted_count),
            )
            .with_columns(
                (pl.col("predicted_h2a_count") / pl.col("bea_farm_emp_2011"))
                .cast(pl.Float32)
                .alias("predicted_h2a_share_2011")
            )
            .select(
                "cutoff_year",
                "model_spec",
                "county_fips",
                "predicted_h2a_count",
                "bea_farm_emp_2011",
                "predicted_h2a_share_2011",
            )
        )
        if not np.allclose(
            result["predicted_h2a_share_2011"].to_numpy(),
            result["predicted_h2a_count"].to_numpy()
            / result["bea_farm_emp_2011"].to_numpy(),
            rtol=2e-6,
            atol=2e-7,
        ):
            raise ValueError("Predicted H-2A share normalization is inconsistent.")
        return result

    return (score_saved_model,)


@app.cell
def _(
    INTERMEDIATE,
    assert_geo_columns,
    build_feature_ids,
    county_covariates,
    county_scoring,
    expected_model_schema,
    load_saved_model,
    model_cutoff_years,
    model_paths,
    model_spec,
    patch_covariates,
    pl,
    reference_categorical_metadata,
    reference_transforms,
    score_saved_model,
    soil_scoring,
):
    feature_ids = build_feature_ids(soil_scoring, reference_categorical_metadata)
    prediction_frames = []

    for cutoff_year, model_path in zip(model_cutoff_years, model_paths, strict=True):
        bias, weight_matrices, transforms = load_saved_model(
            model_path,
            cutoff_year,
            model_spec,
            expected_model_schema,
            reference_categorical_metadata,
            reference_transforms,
            county_covariates,
            patch_covariates,
        )
        cutoff_predictions = score_saved_model(
            cutoff_year,
            model_spec,
            bias,
            weight_matrices,
            transforms,
            feature_ids,
            county_scoring,
            soil_scoring,
            county_covariates,
            patch_covariates,
        )
        prediction_frames.append(cutoff_predictions)
        print(
            f"Scored cutoff {cutoff_year}: "
            f"{cutoff_predictions.height:,} counties, predicted count range "
            f"{cutoff_predictions['predicted_h2a_count'].min():.4g}-"
            f"{cutoff_predictions['predicted_h2a_count'].max():.4g}."
        )

    predictions_by_cutoff = pl.concat(prediction_frames).sort(
        "cutoff_year", "county_fips"
    )
    assert_geo_columns(predictions_by_cutoff, ["county_fips"])
    _expected_rows = len(model_paths) * county_scoring.height
    if predictions_by_cutoff.height != _expected_rows:
        raise ValueError(
            f"Expected {_expected_rows:,} predictions, observed "
            f"{predictions_by_cutoff.height:,}."
        )
    if (
        predictions_by_cutoff.select("cutoff_year", "county_fips")
        .unique()
        .height
        != predictions_by_cutoff.height
    ):
        raise ValueError("Predictions are not unique by cutoff year and county.")

    prediction_output_path = (
        INTERMEDIATE / "h2a_prediction_using_elastic_net_by_cutoff.parquet"
    )
    predictions_by_cutoff.write_parquet(prediction_output_path)
    print(f"Saved cutoff-specific county predictions to {prediction_output_path}")
    return (predictions_by_cutoff,)


@app.cell
def _(pl, predictions_by_cutoff):
    prediction_diagnostics = (
        predictions_by_cutoff.group_by("cutoff_year")
        .agg(
            pl.len().alias("n_counties"),
            pl.col("predicted_h2a_count").min().alias("min_count"),
            pl.col("predicted_h2a_count").median().alias("median_count"),
            pl.col("predicted_h2a_count").max().alias("max_count"),
            pl.col("predicted_h2a_share_2011").max().alias("max_share_2011"),
        )
        .sort("cutoff_year")
    )
    return (prediction_diagnostics,)


if __name__ == "__main__":
    app.run()
