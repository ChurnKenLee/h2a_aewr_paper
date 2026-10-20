# Purpose: Resolve H-2A location pieces that local matching left unmatched,
#   using Gemini name cleanup and Google Places county lookups.
# Inputs: h2a_location_pieces.parquet (from 02_01) and the Gemini/Places caches
#   in data/intermediate/cache; GOOGLE_GEMINI_API_KEY and GOOGLE_PLACES_API_KEY
#   only when H2A_RUN_LOCATION_APIS=1.
# Outputs: h2a_location_api_counties.parquet (one row per queued piece) and,
#   with networking enabled, appended Gemini/Places caches.
#
# Default runs make no network calls: every answer comes from the caches and
# the script reports how many requests a live run would need. A county is
# accepted only when Places places the original and the Gemini-suggested name
# in a common county.

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import hashlib
    import json
    import math
    import os
    import time

    import addfips
    import dotenv
    import polars as pl
    import us

    from h2a.paths import CACHE, INTERMEDIATE

    dotenv.load_dotenv()
    RUN_APIS = os.getenv("H2A_RUN_LOCATION_APIS") == "1"
    GEMINI_MODEL = "gemini-3-flash-preview"
    GEMINI_ROWS_PER_REQUEST = 20
    GEMINI_MAX_WAIT_SECONDS = int(os.getenv("H2A_GEMINI_MAX_WAIT_SECONDS", "3600"))
    PLACES_SEARCH_DELAY_SECONDS = 60 / 600 + 0.01  # at most 600 searches per minute
    PLACES_DETAILS_DELAY_SECONDS = 1 / 9  # at most 9 details per second

    # Cache files (kept from the earlier two-script workflow)
    GEMINI_CACHE = {
        "disclosure": CACHE / "cleaned_h2a_locations_response_string.txt",
        "addendum_b": CACHE / "cleaned_add_b_locations_response_string.txt",
    }
    GEMINI_JOB_FILE = {
        "disclosure": CACHE / "h2a_active_job_id.txt",
        "addendum_b": CACHE / "add_b_active_job_id.txt",
    }
    GEMINI_REQUEST_FILE = {
        "disclosure": CACHE / "clean_h2a_locations_request.jsonl",
        "addendum_b": CACHE / "clean_add_b_locations_request.jsonl",
    }
    SEARCH_CACHE = CACHE / "placeid_search_queries_cache.json"
    DETAILS_CACHE = CACHE / "placeid_address_components_mapping.json"
    return (
        DETAILS_CACHE,
        GEMINI_CACHE,
        GEMINI_JOB_FILE,
        GEMINI_MAX_WAIT_SECONDS,
        GEMINI_MODEL,
        GEMINI_REQUEST_FILE,
        GEMINI_ROWS_PER_REQUEST,
        INTERMEDIATE,
        PLACES_DETAILS_DELAY_SECONDS,
        PLACES_SEARCH_DELAY_SECONDS,
        RUN_APIS,
        SEARCH_CACHE,
        addfips,
        hashlib,
        json,
        math,
        os,
        pl,
        time,
        us,
    )


@app.cell
def _(INTERMEDIATE, hashlib, pl):
    # Queue: unique unmatched pieces that carry a city or county name.
    # State-only pieces cannot be placed in a county and are not sent.
    PIECE_KEYS = ["xcity", "county_list", "xstate", "xzip"]
    pieces = pl.read_parquet(INTERMEDIATE / "h2a_location_pieces.parquet")
    queue = (
        pieces.filter(
            (pl.col("matched_county_fips_list") == "")
            & ((pl.col("xcity").str.strip_chars() != "") | (pl.col("county_list").str.strip_chars() != ""))
        )
        .select(["source", *PIECE_KEYS])
        .unique()
        .sort(["source", *PIECE_KEYS])
    )

    def location_hash(city, county, state, zip_code):
        raw = f"{city or ''}|{county or ''}|{state or ''}|{zip_code or ''}".lower()
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def location_hashes(city, county, state, zip_code):
        # The earlier workflow round-tripped ZIPs through CSV type inference,
        # which could drop leading zeros; accept either form as a cache hit.
        hashes = [location_hash(city, county, state, zip_code)]
        if zip_code and zip_code.isdigit():
            hashes.append(location_hash(city, county, state, str(int(zip_code)) if int(zip_code) else ""))
        return hashes

    print(f"Queued pieces: {queue.height} ({dict(queue.group_by('source').len().sort('source').rows())})")
    return PIECE_KEYS, location_hash, location_hashes, queue


@app.cell
def _(json):
    # Gemini structured-output schema and batch-response parser
    GEMINI_SCHEMA = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "location_hash": {"type": "STRING", "description": "The unique hash ID of the row from the input data."},
                "city": {"type": "STRING", "description": "Corrected city name."},
                "county": {"type": "STRING", "description": "County name if present or easily inferable."},
                "state": {"type": "STRING", "description": "Two-letter state abbreviation (e.g., NY, CA)."},
                "zipcode": {"type": "STRING", "description": "5-digit zipcode as a string."},
                "confidence": {"type": "STRING", "description": "High/Medium/Low confidence in the correction."},
            },
            "required": ["location_hash", "confidence"],
        },
    }

    def gemini_prompt(rows_json):
        return f"""
            You are an expert data cleaning assistant.
            I have a list of US locations that contain errors such as typos, swapped columns, blank entries, or formatting issues.
            Your goal is to clean these locations so they can be successfully queried against the Google Places API to find the County and FIPS code.

            Instruction:
            1. Fix typos in City and County names.
            2. Move data to the correct field if swapped.
            3. Ensure Zipcode is a 5-digit string.
            4. If a field is missing, leave it null.
            5. Return a list of CleanedLocation objects.
            6. EXTREMELY IMPORTANT: Preserve the 'location_hash' for each row exactly.

            Input Data (JSON): {rows_json}
            """

    def parse_gemini_cache(path):
        # location_hash -> suggestion dict; later responses replace earlier ones
        suggestions = {}
        if not path.exists():
            return suggestions
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                candidates = item["response"].get("candidates", [])
                rows = json.loads(candidates[0]["content"]["parts"][0]["text"]) if candidates else []
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
            for row in rows:
                if isinstance(row, dict) and row.get("location_hash"):
                    suggestions[row["location_hash"]] = {
                        key: ("" if row.get(key) in (None, "null") else str(row.get(key)))
                        for key in ["city", "county", "state", "zipcode", "confidence"]
                    }
        return suggestions

    return GEMINI_SCHEMA, gemini_prompt, parse_gemini_cache


@app.cell
def _(
    GEMINI_CACHE,
    GEMINI_JOB_FILE,
    GEMINI_MAX_WAIT_SECONDS,
    GEMINI_MODEL,
    GEMINI_REQUEST_FILE,
    GEMINI_ROWS_PER_REQUEST,
    GEMINI_SCHEMA,
    RUN_APIS,
    gemini_prompt,
    json,
    location_hash,
    location_hashes,
    math,
    os,
    parse_gemini_cache,
    pl,
    queue,
    time,
):
    # 1. Gemini suggestions, one batch job per source; resumes a pending job
    def run_gemini_batch(source, rows):
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])
        job_file = GEMINI_JOB_FILE[source]
        if job_file.exists():
            job_name = job_file.read_text().strip()
            print(f"{source}: resuming Gemini job {job_name}")
        else:
            with GEMINI_REQUEST_FILE[source].open("w", encoding="utf-8") as handle:
                for start in range(0, len(rows), GEMINI_ROWS_PER_REQUEST):
                    chunk = json.dumps(rows[start : start + GEMINI_ROWS_PER_REQUEST])
                    handle.write(json.dumps({
                        "key": f"request_chunk_{start}",
                        "request": {
                            "contents": [{"parts": [{"text": gemini_prompt(chunk)}], "role": "user"}],
                            "generation_config": {
                                "response_mime_type": "application/json",
                                "response_schema": GEMINI_SCHEMA,
                                "temperature": 0.1,
                            },
                        },
                    }) + "\n")
            upload = client.files.upload(
                file=GEMINI_REQUEST_FILE[source],
                config=types.UploadFileConfig(display_name=f"{source}_locations", mime_type="jsonl"),
            )
            job_name = client.batches.create(
                model=GEMINI_MODEL, src=upload.name, config={"display_name": f"clean_{source}_locations"}
            ).name
            job_file.write_text(job_name)
            print(f"{source}: submitted Gemini job {job_name}")
        done = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
        deadline = time.time() + GEMINI_MAX_WAIT_SECONDS
        job = client.batches.get(name=job_name)
        while job.state.name not in done and time.time() < deadline:
            time.sleep(30)
            job = client.batches.get(name=job_name)
        if job.state.name not in done:
            print(f"{source}: Gemini job still {job.state.name}; rerun to resume")
            return
        if job.state.name == "JOB_STATE_SUCCEEDED" and job.dest and job.dest.file_name:
            content = client.files.download(file=job.dest.file_name).decode("utf-8")
            with GEMINI_CACHE[source].open("a", encoding="utf-8") as handle:
                handle.write("\n" + content.strip())
            print(f"{source}: appended Gemini results")
        else:
            print(f"{source}: Gemini job ended {job.state.name}: {job.error}")
        job_file.unlink()

    suggestion_rows = []
    for _source in ["disclosure", "addendum_b"]:
        _cached = parse_gemini_cache(GEMINI_CACHE[_source])
        _rows = queue.filter(pl.col("source") == _source).to_dicts()
        _missing = [
            r for r in _rows
            if not any(h in _cached for h in location_hashes(r["xcity"], r["county_list"], r["xstate"], r["xzip"]))
        ]
        print(
            f"{_source}: {len(_rows)} queued, {len(_missing)} without a Gemini answer "
            f"-> {math.ceil(len(_missing) / GEMINI_ROWS_PER_REQUEST)} Gemini requests"
        )
        if _missing and RUN_APIS:
            run_gemini_batch(
                _source,
                [
                    {
                        "location_hash": location_hash(r["xcity"], r["county_list"], r["xstate"], r["xzip"]),
                        "city": r["xcity"], "county": r["county_list"], "state": r["xstate"], "zip": r["xzip"],
                    }
                    for r in _missing
                ],
            )
            _cached = parse_gemini_cache(GEMINI_CACHE[_source])
        for _r in _rows:
            _hit = next(
                (h for h in location_hashes(_r["xcity"], _r["county_list"], _r["xstate"], _r["xzip"]) if h in _cached),
                None,
            )
            _s = _cached.get(_hit, {})
            suggestion_rows.append({
                **_r,
                "has_gemini_suggestion": _hit is not None,
                "city_suggested": _s.get("city", ""),
                "county_suggested": _s.get("county", ""),
                "state_suggested": _s.get("state", ""),
                "suggestion_confidence": _s.get("confidence", ""),
            })
    suggestions = pl.DataFrame(
        suggestion_rows,
        schema={
            "source": pl.String, "xcity": pl.String, "county_list": pl.String, "xstate": pl.String,
            "xzip": pl.String, "has_gemini_suggestion": pl.Boolean, "city_suggested": pl.String,
            "county_suggested": pl.String, "state_suggested": pl.String, "suggestion_confidence": pl.String,
        },
    )
    return (suggestions,)


@app.cell
def _(
    DETAILS_CACHE,
    PLACES_DETAILS_DELAY_SECONDS,
    PLACES_SEARCH_DELAY_SECONDS,
    RUN_APIS,
    SEARCH_CACHE,
    json,
    os,
    pl,
    suggestions,
    time,
    us,
):
    # 2. Places text search for the original and suggested names, then place
    #    details (address components) for every candidate place ID
    def state_name(value):
        value = str(value or "").strip()
        if value.upper() in ("DC", "DISTRICT OF COLUMBIA"):
            return "District of Columbia"
        state = us.states.lookup(value) if value else None
        return state.name if state else ""

    def query_text(city, county, state):
        return ", ".join(part for part in [city, county, state_name(state)] if part)

    places = suggestions.with_columns(
        state_name=pl.col("xstate").map_elements(state_name, return_dtype=pl.String),
        original_query=pl.struct("xcity", "county_list", "xstate").map_elements(
            lambda r: query_text(r["xcity"], r["county_list"], r["xstate"]), return_dtype=pl.String
        ),
        suggested_query=pl.struct("city_suggested", "county_suggested", "state_suggested").map_elements(
            lambda r: query_text(r["city_suggested"], r["county_suggested"], r["state_suggested"]),
            return_dtype=pl.String,
        ),
    )
    search_cache = json.loads(SEARCH_CACHE.read_text()) if SEARCH_CACHE.exists() else {}
    details_cache = json.loads(DETAILS_CACHE.read_text()) if DETAILS_CACHE.exists() else {}

    # A query needs at least two components ("city, state"); state-only text is never sent
    queries = sorted({
        q for q in places["original_query"].to_list() + places["suggested_query"].to_list() if q and "," in q
    })
    new_queries = [q for q in queries if q not in search_cache]
    print(f"Places text searches: {len(queries)} needed, {len(new_queries)} not cached")
    if new_queries and RUN_APIS:
        from google.maps import places_v1

        _client = places_v1.PlacesClient(client_options={"api_key": os.environ["GOOGLE_PLACES_API_KEY"]})
        for _i, _query in enumerate(new_queries, start=1):
            _response = _client.search_text(
                request=places_v1.SearchTextRequest(text_query=_query),
                metadata=[("x-goog-fieldmask", "places.id")],
            )
            search_cache[_query] = [place.id for place in _response.places]
            if _i % 50 == 0 or _i == len(new_queries):
                SEARCH_CACHE.write_text(json.dumps(search_cache))
            time.sleep(PLACES_SEARCH_DELAY_SECONDS)

    place_ids = sorted({pid for q in queries for pid in search_cache.get(q, [])})
    new_place_ids = [pid for pid in place_ids if pid not in details_cache]
    print(
        f"Places details: {len(place_ids)} needed, {len(new_place_ids)} not cached"
        + ("" if RUN_APIS else " (searches not yet run add more)" if new_queries else "")
    )
    if new_place_ids and RUN_APIS:
        from google.maps import places_v1
        from google.protobuf.json_format import MessageToDict

        _client = places_v1.PlacesClient(client_options={"api_key": os.environ["GOOGLE_PLACES_API_KEY"]})
        for _i, _pid in enumerate(new_place_ids, start=1):
            _response = _client.get_place(
                request=places_v1.GetPlaceRequest(name=f"places/{_pid}"),
                metadata=[("x-goog-fieldmask", "addressComponents")],
            )
            details_cache[_pid] = MessageToDict(_response._pb)
            if _i % 50 == 0 or _i == len(new_place_ids):
                DETAILS_CACHE.write_text(json.dumps(details_cache, indent=2))
            time.sleep(PLACES_DETAILS_DELAY_SECONDS)
    return details_cache, places, search_cache


@app.cell
def _(INTERMEDIATE, addfips, details_cache, pl, places, search_cache):
    # 3. Counties: Places address components (administrative_area_level_2)
    #    for each query; keep counties common to the original and suggested
    #    names and convert them to 2010 FIPS
    MODERN_TO_2010_COUNTY = {
        "Oglala Lakota County": "Shannon County",
        "Kusilvak Census Area": "Wade Hampton Census Area",
        "Chugach Census Area": "Valdez-Cordova Census Area",
        "Copper River Census Area": "Valdez-Cordova Census Area",
        "Bedford City": "Bedford County",
        # Connecticut planning regions (2022) to majority-landmass 2010 counties
        "Capitol Planning Region": "Hartford County",
        "Greater Bridgeport Planning Region": "Fairfield County",
        "Lower Connecticut River Valley Planning Region": "Middlesex County",
        "Naugatuck Valley Planning Region": "New Haven County",
        "Northeastern Connecticut Planning Region": "Windham County",
        "Northwest Hills Planning Region": "Litchfield County",
        "South Central Connecticut Planning Region": "New Haven County",
        "Southeastern Connecticut Planning Region": "New London County",
        "Western Connecticut Planning Region": "Fairfield County",
    }
    county_af = addfips.AddFIPS(vintage=2010)

    def query_counties(query):
        names = set()
        for pid in search_cache.get(query, []) if query else []:
            for component in details_cache.get(pid, {}).get("addressComponents", []):
                if "administrative_area_level_2" in component.get("types", []):
                    names.add(component["longText"])
        return names

    def common_county_fips(original_query, suggested_query, state):
        common = query_counties(original_query) & query_counties(suggested_query)
        fips = {county_af.get_county_fips(MODERN_TO_2010_COUNTY.get(n, n), state) for n in common} - {None}
        return ",".join(sorted(fips))

    api_counties = places.with_columns(
        places_county_fips_list=pl.struct("original_query", "suggested_query", "state_name").map_elements(
            lambda r: common_county_fips(r["original_query"], r["suggested_query"], r["state_name"]),
            return_dtype=pl.String,
        ),
        # A query that is never sent (empty or state-only) counts as answered
        original_query_cached=~pl.col("original_query").str.contains(",")
        | pl.col("original_query").is_in(list(search_cache)),
        suggested_query_cached=~pl.col("suggested_query").str.contains(",")
        | pl.col("suggested_query").is_in(list(search_cache)),
    ).with_columns(
        api_status=pl.when(pl.col("places_county_fips_list") != "")
        .then(pl.lit("resolved"))
        .when(~pl.col("has_gemini_suggestion") | ~pl.col("original_query_cached") | ~pl.col("suggested_query_cached"))
        .then(pl.lit("not_queried"))
        .otherwise(pl.lit("no_common_county"))
    )
    api_counties.write_parquet(INTERMEDIATE / "h2a_location_api_counties.parquet")
    print(api_counties.group_by("source", "api_status").len().sort("source", "api_status"))
    return


if __name__ == "__main__":
    app.run()
