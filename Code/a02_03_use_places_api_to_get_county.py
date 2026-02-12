import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import json
    import pandas as pd
    from google import genai
    from google.genai import types
    from google.genai._transformers import process_schema
    from google.maps import places_v1
    from google.protobuf.json_format import MessageToJson
    from typing import Optional, List, get_args, get_origin
    from pydantic import BaseModel, Field
    import time
    import os
    import us
    from ratelimit import limits, sleep_and_retry
    import pickle
    return (
        MessageToJson,
        json,
        limits,
        pd,
        pickle,
        places_v1,
        sleep_and_retry,
        us,
    )


@app.cell
def _():
    with open("../tools/google_places_api_key.txt") as _f:
        _line = _f.readlines()
    api_key = _line[0]
    return (api_key,)


@app.cell
def _(us):
    # Function converts state abbreviations to names
    def state_abbr_to_name(input):
        if input == '':
            return ''

        # Include DC
        if str(input).upper() == 'DC':
            return 'District of Columbia'

        # US package lookup can handle names, abbreviations, FIPs
        state_obj = us.states.lookup(str(input).strip())

        return state_obj.name if state_obj else ''
    return (state_abbr_to_name,)


@app.cell
def _(api_key, pd, places_v1, state_abbr_to_name):
    # Load cleaned unmatched locations
    h2a_df = pd.read_csv("json/unmatched_h2a_with_suggestions.csv", dtype = str).fillna('')
    add_b_df = pd.read_csv("json/unmatched_add_b_with_suggestions.csv", dtype = str).fillna('')

    # Define the names we want to use to query
    # Add full names for states
    h2a_df['state_name'] = h2a_df['state'].apply(lambda _x: state_abbr_to_name(_x))
    h2a_df['state_name_suggested'] = h2a_df['state_suggested'].apply(lambda _x: state_abbr_to_name(_x))
    add_b_df['state_name'] = add_b_df['state'].apply(lambda _x: state_abbr_to_name(_x))
    add_b_df['state_name_suggested'] = add_b_df['state_suggested'].apply(lambda _x: state_abbr_to_name(_x))

    # Contatenate location name elements
    h2a_df['original_location_name'] = h2a_df[['city', 'county', 'state_name']].apply(lambda _x: ', '.join(_x[_x!='']), axis=1)
    h2a_df['suggested_location_name'] = h2a_df[['city_suggested', 'county_suggested', 'state_name_suggested']].apply(lambda _x: ', '.join(_x[_x!='']), axis=1)

    add_b_df['original_location_name'] = add_b_df[['city', 'county', 'state_name']].apply(lambda _x: ', '.join(_x[_x!='']), axis=1)
    add_b_df['suggested_location_name'] = add_b_df[['city_suggested', 'county_suggested', 'state_name_suggested']].apply(lambda _x: ', '.join(_x[_x!='']), axis=1)

    # Rate limit our queries manually
    client = places_v1.PlacesClient(client_options={"api_key": api_key})
    def rate_limited_queries(queries, qpm):
        response_list = []
        delay = 60/qpm + 0.01

        # The field mask has to be passed in the call
        for query in queries:

            # Skip invalid requests
            if query == '':
                response_list.append([])
                continue
            if ',' not in query:
                response_list.append([])
                continue

            # Create request
            request = places_v1.SearchTextRequest(text_query=query)

            # The field mask has to be passed in the call
            response = client.search_text(
                    request = request, 
                    metadata = [("x-goog-fieldmask", "places.id")]
                )
            placeid_candidates = [place.id for place in response.places]
            response_list.append(placeid_candidates)

        return response_list

    h2a_original_locations_list = list(h2a_df['original_location_name'])
    h2a_original_locations_candidate_list = rate_limited_queries(h2a_original_locations_list, 600)
    h2a_df['original_location_placeid'] = h2a_original_locations_candidate_list

    h2a_suggested_locations_list = list(h2a_df['suggested_location_name'])
    h2a_suggested_locations_candidate_list = rate_limited_queries(h2a_suggested_locations_list, 600)
    h2a_df['suggested_location_placeid'] = h2a_suggested_locations_candidate_list

    add_b_original_locations_list = list(add_b_df['original_location_name'])
    add_b_original_locations_candidate_list = rate_limited_queries(add_b_original_locations_list, 600)
    add_b_df['original_location_placeid'] = add_b_original_locations_candidate_list

    add_b_suggested_locations_list = list(add_b_df['suggested_location_name'])
    add_b_suggested_locations_candidate_list = rate_limited_queries(add_b_suggested_locations_list, 600)
    add_b_df['suggested_location_placeid'] = add_b_suggested_locations_candidate_list

    # Save Place ID suggestions
    h2a_df.to_parquet("../binaries/h2a_location_placeids.parquet")
    add_b_df.to_parquet("../binaries/add_b_location_placeids.parquet")
    return add_b_df, h2a_df


@app.cell
def _(pd):
    # Read Place ID suggestions
    h2a_df = pd.read_parquet("../binaries/h2a_location_placeids.parquet")
    add_b_df = pd.read_parquet("../binaries/add_b_location_placeids.parquet")
    return add_b_df, h2a_df


@app.function
def get_common_elements(list1, list2):
    # Convert lists to sets and find the intersection
    common = set(list1) & set(list2)
    # Convert back to a list
    return list(common)


@app.cell
def _(add_b_df, h2a_df):
    h2a_df['common_placeid'] = h2a_df.apply(lambda _x: get_common_elements(_x['original_location_placeid'], _x['suggested_location_placeid']), axis=1)
    add_b_df['common_placeid'] = add_b_df.apply(lambda _x: get_common_elements(_x['original_location_placeid'], _x['suggested_location_placeid']), axis=1)
    return


@app.cell
def _(add_b_df, h2a_df):
    h2a_df['has_common_placeid'] = h2a_df['common_placeid'].astype('bool')
    add_b_df['has_common_placeid'] = add_b_df['common_placeid'].astype('bool')
    return


@app.cell
def _(add_b_df, h2a_df):
    # Get all Place IDs we need to match to address components
    def put_placeid_in_list(df_col):
        list_of_placeids = []
        for placeid_list in df_col:
            if placeid_list.any():
                for placeid in placeid_list:
                    list_of_placeids.append(placeid)
            else:
                continue
        return list_of_placeids

    h2a_original_placeid_list = put_placeid_in_list(h2a_df['original_location_placeid'])
    h2a_suggested_placeid_list = put_placeid_in_list(h2a_df['suggested_location_placeid'])
    add_b_original_placeid_list = put_placeid_in_list(add_b_df['original_location_placeid'])
    add_b_suggested_placeid_list = put_placeid_in_list(add_b_df['suggested_location_placeid'])
    placeid_list = h2a_original_placeid_list + h2a_suggested_placeid_list + add_b_original_placeid_list + add_b_suggested_placeid_list
    placeid_list = list(set(placeid_list)) # We want only unique Place IDs; no sense making duplicate calls
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
    return (get_address_components_from_placeid,)


@app.cell
def _(get_address_components_from_placeid, pickle, placeid_list):
    placeid_address_components_dict = {}

    for _placeid in placeid_list:
        _response = get_address_components_from_placeid(_placeid)
        placeid_address_components_dict[_placeid] = _response

    # Pickle results
    with open("json/placeid_address_components_mapping_pickle.pickle", "wb") as _fp:
        pickle.dump(placeid_address_components_dict, _fp)
    return


@app.cell
def _(MessageToJson, json, pickle):
    # Store mappings as JSON as well for stability
    with open("json/placeid_address_components_mapping_pickle.pickle", "rb") as _fp:
        placeid_address_components_dict_loaded = pickle.load(_fp)

    placeid_address_components_dict_json = {}
    for _placeid, _response in placeid_address_components_dict_loaded.items():
        placeid_address_components_dict_json[_placeid] = MessageToJson(_response._pb) # Convert protocol response to JSON for writing to JSON file

    with open("json/placeid_address_components_mapping_json.json", 'w') as _fp:
        json.dump(placeid_address_components_dict_json, _fp)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
