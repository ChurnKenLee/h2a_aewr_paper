import marimo

__generated_with = "0.22.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import pyprojroot
    import dotenv, os
    import json
    import polars as pl
    from google import genai
    from google.genai import types
    from google.genai._transformers import process_schema
    from google.maps import places_v1
    from google.protobuf.json_format import MessageToJson, MessageToDict
    from typing import Optional, List, get_args, get_origin
    from pydantic import BaseModel, Field
    import time
    import os
    import us
    from ratelimit import limits, sleep_and_retry
    import pickle

    return (
        MessageToDict,
        dotenv,
        json,
        limits,
        mo,
        os,
        pl,
        places_v1,
        pyprojroot,
        sleep_and_retry,
        time,
        us,
    )


@app.cell
def _(dotenv, os, pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    dotenv.load_dotenv()
    api_key = os.getenv('GOOGLE_PLACES_API_KEY')

    binary_path = root_path / 'binaries'
    json_path = root_path / 'code' / 'json'
    return api_key, binary_path, json_path


@app.cell
def _(us):
    def state_abbr_to_name(val):
        if val is None or val == '':
            return ''

        val_str = str(val).strip()

        if val_str.upper() == 'DC':
            return 'District of Columbia'

        state_obj = us.states.lookup(val_str)
        if state_obj:
            return state_obj.name
        else:
            return ''

    return (state_abbr_to_name,)


@app.cell
def _(pl, state_abbr_to_name):
    # This mimics: ', '.join([x for x in columns if x != ''])
    def concat_location(city_col, county_col, state_col):
        concat_df = pl.concat_list(
            [city_col, county_col, state_col]
        ).list.eval(
            pl.element().filter(
                (pl.element() != "") & (pl.element().is_not_null())
            )
        ).list.join(", ")

        return concat_df

    # Function applies clean name and concat to df
    def process_df(df):
        clean_names_df = df.with_columns([
            # Apply the state lookup
            pl.col("state").map_elements(state_abbr_to_name, return_dtype=pl.String).alias("state_name"),
            pl.col("state_suggested").map_elements(state_abbr_to_name, return_dtype=pl.String).alias("state_name_suggested"),
        ]).with_columns([
            # Concatenate columns
            concat_location(pl.col("city"), pl.col("county"), pl.col("state_name"))
                .alias("original_location_name"),
            concat_location(pl.col("city_suggested"), pl.col("county_suggested"), pl.col("state_name_suggested"))
                .alias("suggested_location_name"),
        ])

        return clean_names_df

    return (process_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Get Place ID
    """)
    return


@app.cell
def _(binary_path, client, json, json_path, pl, places_v1, process_df, time):
    # Load cleaned unmatched locations, clean state names, and concat place names to get location name
    h2a_df = pl.read_csv(json_path / "unmatched_h2a_with_suggestions.csv", infer_schema=False).fill_null('')
    add_b_df = pl.read_csv(json_path / "unmatched_add_b_with_suggestions.csv", infer_schema=False).fill_null('')
    h2a_df = process_df(h2a_df)
    add_b_df = process_df(add_b_df)

    search_cache_file = json_path / "placeid_search_queries_cache.json"
    if search_cache_file.exists():
        with open(search_cache_file, "r") as _fp:
            search_query_cache = json.load(_fp)
    else:
        search_query_cache = {}

    def rate_limited_queries(queries, qpm, save_frequency=50):
        response_list =[]
        # Calculate delay in seconds
        delay = (60 / qpm) + 0.01

        new_calls_since_last_save = 0

        for query in queries:
            # Polars might pass None for null values, so check for truthiness
            if not query or ',' not in query:
                response_list.append([])
                continue

            # Check if query was previously executed ---
            if query in search_query_cache:
                response_list.append(search_query_cache[query])
                continue

            # Create request
            request = places_v1.SearchTextRequest(text_query=query)

            # Call API
            response = client.search_text(
                request=request, 
                metadata=[("x-goog-fieldmask", "places.id")]
            )

            placeid_candidates = [place.id for place in response.places]

            # Add to cache
            search_query_cache[query] = placeid_candidates
            response_list.append(placeid_candidates)

            new_calls_since_last_save += 1

            # Save to disk periodically
            if new_calls_since_last_save >= save_frequency:
                with open(search_cache_file, "w") as _fp:
                    json.dump(search_query_cache, _fp)
                print(f"Checkpoint: Saved {len(search_query_cache)} queries to cache.")
                new_calls_since_last_save = 0

            # Actually sleep to respect the rate limit
            time.sleep(delay)

        # =Final save to catch any remaining queries
        if new_calls_since_last_save > 0:
            with open(search_cache_file, "w") as _fp:
                json.dump(search_query_cache, _fp)
        if new_calls_since_last_save == 0:
            print("No new locations so no new PlaceIDs")

        return response_list

    # --- Process h2a_df ---
    h2a_original_locations_list = h2a_df["original_location_name"].to_list()
    h2a_original_locations_candidate_list = rate_limited_queries(h2a_original_locations_list, 600)
    h2a_suggested_locations_list = h2a_df["suggested_location_name"].to_list()
    h2a_suggested_locations_candidate_list = rate_limited_queries(h2a_suggested_locations_list, 600)
    h2a_df = h2a_df.with_columns([
        pl.Series(
            name="original_location_placeid", 
            values=h2a_original_locations_candidate_list
        ),
        pl.Series(
            name="suggested_location_placeid", 
            values=h2a_suggested_locations_candidate_list
        )
    ])

    # --- Process add_b_df ---
    add_b_original_locations_list = add_b_df["original_location_name"].to_list()
    add_b_original_locations_candidate_list = rate_limited_queries(add_b_original_locations_list, 600)
    add_b_suggested_locations_list = add_b_df["suggested_location_name"].to_list()
    add_b_suggested_locations_candidate_list = rate_limited_queries(add_b_suggested_locations_list, 600)
    add_b_df = add_b_df.with_columns([
        pl.Series(
            name="original_location_placeid", 
            values=add_b_original_locations_candidate_list
        ),
        pl.Series(
            name="suggested_location_placeid", 
            values=add_b_suggested_locations_candidate_list
        )
    ])

    # # Save to Parquet using Polars write method
    h2a_df.write_parquet(binary_path / "h2a_location_placeids.parquet")
    add_b_df.write_parquet(binary_path / "add_b_location_placeids.parquet")

    # Define the operation as an expression to reuse it
    def add_common_id_cols(df: pl.DataFrame) -> pl.DataFrame:
        df_with_common_id = df.with_columns(
            common_placeid = pl.col("original_location_placeid").list.set_intersection("suggested_location_placeid")
        ).with_columns(
            has_common_placeid = pl.col("common_placeid").list.len() > 0
        )
        return df_with_common_id

    # Apply to your DataFrames
    h2a_df = add_common_id_cols(h2a_df)
    add_b_df = add_common_id_cols(add_b_df)
    return add_b_df, h2a_df


@app.cell
def _(add_b_df, h2a_df, pl):
    # Create a list of the Series we want to extract and flatten
    series_to_combine = [
        h2a_df["original_location_placeid"].explode(),
        h2a_df["suggested_location_placeid"].explode(),
        add_b_df["original_location_placeid"].explode(),
        add_b_df["suggested_location_placeid"].explode()
    ]

    # Concatenate them, remove nulls/None, and get unique values
    placeid_list = (
        pl.concat(series_to_combine)
        .drop_nulls()
        .unique()
        .to_list()
    )
    return (placeid_list,)


@app.cell
def _(api_key, limits, places_v1, sleep_and_retry):
    client = places_v1.PlacesClient(client_options={"api_key": api_key})

    @sleep_and_retry
    @limits(calls=9, period=1)
    def get_address_components_from_placeid(placeid):
        # Place ID format for Places API
        name = "places/" + placeid

        # We only request 'addressComponents' to minimize latency/cost
        # This corresponds to the X-Goog-FieldMask header
        field_mask = [("x-goog-fieldmask", "addressComponents")]

        # Construct request
        request = places_v1.GetPlaceRequest(name=name)

        # Send request
        # We pass the field_mask via the metadata argument in the new SDK
        response = client.get_place(
            request=request,
            metadata=field_mask
        )

        return response

    return client, get_address_components_from_placeid


@app.cell
def _(
    MessageToDict,
    get_address_components_from_placeid,
    json,
    json_path,
    placeid_list,
):
    # 1. Load existing JSON responses if available
    json_file_path = json_path / "placeid_address_components_mapping.json"

    if json_file_path.exists():
        with open(json_file_path, "r") as _fp:
            placeid_address_components_dict = json.load(_fp)
        print(f"Loaded {len(placeid_address_components_dict)} pre-existing Place API responses from JSON.")
    else:
        placeid_address_components_dict = {}

    new_calls = 0

    for _placeid in placeid_list:
        # 2. Only fetch if we haven't seen this PlaceID before
        if _placeid not in placeid_address_components_dict:
            _response = get_address_components_from_placeid(_placeid)

            # Convert the Protobuf response immediately to a standard Python dictionary
            placeid_address_components_dict[_placeid] = MessageToDict(_response._pb)

            new_calls += 1

            # 3. Periodically backup to JSON so you don't lose progress
            if new_calls % 50 == 0:
                with open(json_file_path, "w") as _fp:
                    json.dump(placeid_address_components_dict, _fp, indent=2)
                print(f"Checkpoint: Saved {len(placeid_address_components_dict)} responses to JSON.")

    print(f"Made {new_calls} new API calls for Place Details.")

    # 4. Final save of the JSON mapping
    with open(json_file_path, "w") as _fp:
        json.dump(placeid_address_components_dict, _fp, indent=2)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
