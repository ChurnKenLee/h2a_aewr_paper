---
title: A02 02 Clean Unmatched Locations Using Gemini
marimo-version: 0.22.4
width: full
---

```python {.marimo}
from pathlib import Path
import pyprojroot
import dotenv, os
import json
import polars as pl
from google import genai
from google.genai import types
from google.genai._transformers import process_schema
from typing import Optional, List, get_args, get_origin
from pydantic import BaseModel, Field, ValidationError
import hashlib
import time
```

```python {.marimo}
root_path = pyprojroot.find_root(criterion='pyproject.toml')
json_path = root_path / 'code' / 'json'
dotenv.load_dotenv()
gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
```

```python {.marimo}
# Define a stable hash function
def get_location_hash(city, county, state, zip_code):
    # Ensure all parts are strings and handle nulls
    city = str(city) if city else ""
    county = str(county) if county else ""
    state = str(state) if state else ""
    zip_code = str(zip_code) if zip_code else ""

    raw = f"{city}|{county}|{state}|{zip_code}".lower()
    return hashlib.md5(raw.encode('utf-8')).hexdigest()
```

```python {.marimo}
# Define the Structured Output schema we will use for Gemini
# This ensures Gemini returns valid JSON matching our DataFrame structure
class CleanedLocation(BaseModel):
    location_hash: str = Field(..., description="The unique hash ID of the row from the input data.")
    city: Optional[str] = Field(None, description="Corrected city name.")
    county: Optional[str] = Field(None, description="County name if present or easily inferable.")
    state: Optional[str] = Field(None, description="Two-letter state abbreviation (e.g., NY, CA).")
    zipcode: Optional[str] = Field(None, description="5-digit zipcode as a string.")
    confidence: str = Field(..., description="High/Medium/Low confidence in the correction.")

try:
    from types import UnionType
except ImportError:
    UnionType = object()

def to_gemini_schema(model_class):
    properties = {}
    required_fields = []

    for name, field in model_class.model_fields.items():
        # Get the type annotation
        field_type = field.annotation
        is_optional = False

        # Handle Optional[T] / Union[T, None]
        if get_origin(field_type) is UnionType or get_origin(field_type) is list or str(field_type).startswith("typing.Optional"):
             # Simple check for Optional/Union types 
             # (In Python <3.10 Optional is Union[T, NoneType])
            args = get_args(field_type)
            if type(None) in args:
                is_optional = True
                # Grab the non-None type
                non_none_args = [arg for arg in args if arg is not type(None)]
                if non_none_args:
                    field_type = non_none_args[0]

        # Map Python types to Gemini types
        gemini_type = "STRING" # Default
        if field_type == int:
            gemini_type = "INTEGER"
        elif field_type == float:
            gemini_type = "NUMBER"
        elif field_type == bool:
            gemini_type = "BOOLEAN"

        # Build property definition
        properties[name] = {
            "type": gemini_type,
            "description": field.description
        }

        # Add to required list ONLY if not optional
        if field.is_required() and not is_optional:
            required_fields.append(name)

    # 3. Construct the final structure
    # Since you want a LIST of locations, we wrap the object in an ARRAY
    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": properties,
            "required": required_fields
        }
    }

genai_schema = to_gemini_schema(CleanedLocation)
```

```python {.marimo}
# This function generates the batch request in the form of a JSONL file
# chunk_size = # of rows in each request
def write_batch_request_jsonl(df: pd.DataFrame, df_original_id: str, response_schema: dict, jsonl_file_path: str, chunk_size: int):

    if df_original_id not in df.columns:
        raise Exception("Missing ID in input dataframe for matching rows with output")

    total_chunks = (len(df) // chunk_size) + 1
    print(f"Writing {total_chunks} requests (chunks) to {jsonl_file_path}")

    jsonl_entries_list = []
    for i in range(0, len(df), chunk_size):
        #chunk = df.slice(i, chunk_size) # This gets added to our prompt as json
        #chunk_in_json = chunk.to_json(orient='records')

        chunk = df.slice(i, chunk_size) # This gets added to our prompt as json
        chunk_in_json = json.dumps(chunk.to_dicts())

        # Define prompt here
        prompt_string = f"""
        You are an expert data cleaning assistant. 
        I have a list of US locations that contain errors such as typos, swapped columns, blank entries, or formatting issues.
        Your goal is to clean these locations so they can be successfully queried against the Google Places API to find the County and FIPS code.

        Instruction:
        1. Fix typos in City and County names.
        2. Move data to the correct field if swapped.
        3. Ensure Zipcode is a 5-digit string.
        4. If a field is missing, leave it null.
        5. Return a list of CleanedLocation objects.
        6. EXTREMELY IMPORTANT: Preserve the '{df_original_id}' for each row exactly.

        Input Data (JSON): {chunk_in_json}
        """

        # Construct a batch request in the JSONL file
        # Each line should be a GenerateContentRequest object, though documented examples don't specify model
        # REMEMBER TO USE DOUBLE QUOTES BCOS JSON
        request_entry = {
            "key": f"request_chunk_{i}",
            "request": {
                "contents": [{
                    "parts":[{"text": prompt_string}],
                    "role": "user"
                }],
                "generation_config": {
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                    "temperature": 0.1
                }
            }
        }

        jsonl_entries_list.append(request_entry)

    # Write list of requests to JSONL file
    with open(jsonl_file_path, "w") as f:
        for req in jsonl_entries_list:
            f.write(json.dumps(req) + "\n")
```

```python {.marimo}
# Define function that uploads file using Files API, returns the uploaded file object that has name to be called for processing
def upload_jsonl_using_files_api(jsonl_path, upload_display_name):
    # Upload file to Google Gen AI
    uploaded_file = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(
            display_name=upload_display_name,
            mime_type='jsonl'
        ),
    )

    print(f"Uploaded file: {uploaded_file.name}")
    return uploaded_file
```

```python {.marimo}
# Function calls Google GenAI BatchGenerateContent on uploaded file, returns batch job object with job details
def initiate_genai_batch(uploaded_file_name, model_id, job_display_name):
    file_batch_job = client.batches.create(
        model=model_id,
        src=uploaded_file_name,
        config={
            'display_name': job_display_name
        },
    )

    print(f"Initiated batch job: {file_batch_job.name}")
    return file_batch_job
```

```python {.marimo}
# Define function that polls for completion and downloads completed result, returns completed job obj
def poll_genai_job(job_name, sleep_duration, output_file_name):
    # Poll for job completion
    completed_states = set([
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    ])

    print(f"Polling status for job: {job_name}")

    batch_job = client.batches.get(name=job_name) # Initial poll
    while batch_job.state.name not in completed_states:
        print(f'Job current state: {batch_job.state.name}, waiting {sleep_duration} seconds')
        time.sleep(sleep_duration) # Wait before polling again
        batch_job = client.batches.get(name=job_name)


    print(f"Job finished with state: {batch_job.state.name}")
    if batch_job.state.name == 'JOB_STATE_FAILED':
        print(f"Error: {batch_job.error}")

    # Retrieve results
    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':

        # If batch job was created with a file
        if batch_job.dest and batch_job.dest.file_name:
            # Results are in a JSONL file
            result_file_name = batch_job.dest.file_name
            print(f"Downloading results file: {result_file_name}")
            file_content = client.files.download(file=result_file_name).decode('utf-8')

    else:
        print(f"Job did not succeed. Final state: {batch_job.state.name}")
        if batch_job.error:
            print(f"Error: {batch_job.error}")

    # Dump response into text file
    with open(output_file_name, "w") as _f:
        _f.write(file_content)
    print(f"Results string written to {output_file_name}")

    return batch_job
```

```python {.marimo}
def process_gemini_batch_with_cache(
    df_to_clean: pl.DataFrame,
    hash_col: str,
    cache_file: Path,
    request_jsonl_path: Path,
    upload_display_name: str,
    job_display_name: str,
    model_id: str = "gemini-3-flash-preview",
    batch_rows: int = 20,
    sleep_duration: int = 10
):
    # 1. Parse existing cache
    existing_results_df = parse_results(cache_file)
    if len(existing_results_df) > 0:
        already_processed_hashes = existing_results_df[hash_col].to_list()
    else:
        already_processed_hashes = []

    # 2. Filter the dataframe to ONLY hashes we haven't processed yet
    new_df = df_to_clean.filter(~pl.col(hash_col).is_in(already_processed_hashes))

    # If nothing is new, return the cached results immediately!
    if len(new_df) == 0:
        print(f"All locations already cached for {job_display_name}! Skipping API calls.")
        return existing_results_df

    print(f"Found {len(new_df)} new locations to process. Initiating Gemini Batch Job...")

    # 3. Create request file for NEW rows
    write_batch_request_jsonl(new_df, hash_col, genai_schema, request_jsonl_path, batch_rows)

    # 4. Upload & Initiate Job
    upload = upload_jsonl_using_files_api(request_jsonl_path, upload_display_name)
    batch_job = initiate_genai_batch(upload.name, model_id, job_display_name)

    # 5. Poll & Download to a TEMPORARY file
    temp_output_file = request_jsonl_path.with_name(f"temp_output_{job_display_name}.txt")
    finished_job = poll_genai_job(batch_job.name, sleep_duration, temp_output_file)

    # 6. Append new results to the master cache file
    if finished_job.state.name == 'JOB_STATE_SUCCEEDED' and temp_output_file.exists():
        with open(temp_output_file, 'r') as temp_f:
            new_content = temp_f.read()

        # Append to cache
        with open(cache_file, 'a') as cache_f:
            # Ensure there's a newline before appending in case the file didn't end with one
            cache_f.write("\n" + new_content.strip())

        print(f"Successfully appended new results to {cache_file.name}")
        temp_output_file.unlink() # Delete the temp file to keep workspace clean
    else:
        print("Job did not succeed or no output file was created.")

    # 7. Re-parse the updated cache file and return the combined results
    return parse_results(cache_file)
```

```python {.marimo}
# Define a function to merge results together into a coherent dataframe
def parse_results(response_string_file):
    # If the cache file doesn't exist yet, return an empty DataFrame with the expected schema
    if not Path(response_string_file).exists():
        return pl.DataFrame(schema={
            "location_hash": pl.String, "city": pl.String, "county": pl.String, 
            "state": pl.String, "zipcode": pl.String, "confidence": pl.String
        })

    all_cleaned_locations = []

    # Load results string from saved local file
    with open(response_string_file, 'r') as f:
        lines = f.readlines()

    for l in lines:
        # Strip all leading/trailing whitespace and newlines
        clean_line = l.strip()

        # Skip completely empty lines
        if not clean_line:
            continue

        try:
            batch_item = json.loads(clean_line)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line. Error: {e}")
            continue

        chunk_key = batch_item.get("key", "Unknown Key")

        # Check if batch item request was successful
        if 'response' not in batch_item:
            print(f'Skipping {chunk_key}: No response found (Status: {batch_item.get('status')})')
            continue

        # Extract the model's text output
        # Standard Gemini response structure: response -> candidates -> content -> parts -> text
        candidates = batch_item['response'].get('candidates', [])
        if not candidates:
            print(f"Skipping {chunk_key}: No candidates returned.")
            continue
        model_text = candidates[0]["content"]["parts"][0]["text"]

        # Parse the Inner JSON (array of locations)
        # Because schema was an ARRAY, model_text is '[{...}, {...}]'
        locations_data = json.loads(model_text)

         # Validate and convert to Pydantic models
        for loc_dict in locations_data:
            try:
                validated_locations_data = CleanedLocation(**loc_dict).model_dump()
                all_cleaned_locations.append(validated_locations_data)
            except ValidationError as e:
                print(f"Validation error for ID {loc_dict.get('original_id')}: {e}")

    if not all_cleaned_locations:
        return pl.DataFrame(schema={
            "location_hash": pl.String, "city": pl.String, "county": pl.String, 
            "state": pl.String, "zipcode": pl.String, "confidence": pl.String
        })

    # Change null to ''
    combined_results = pl.DataFrame(all_cleaned_locations).with_columns(
        pl.selectors.string().str.replace_all("null", "")
)
    return combined_results
```

```python {.marimo}
# Initiate GenAI client here
client = genai.Client(api_key = gemini_api_key)
```

```python {.marimo}
def save_job_id(tracking_file: Path, job_id: str):
    with open(tracking_file, 'w') as f:
        f.write(job_id)

def load_job_id(tracking_file: Path) -> str:
    if tracking_file.exists():
        with open(tracking_file, 'r') as f:
            return f.read().strip()
    return None

def clear_job_id(tracking_file: Path):
    if tracking_file.exists():
        tracking_file.unlink()
```

```python {.marimo}
def submit_gemini_batch_async(
    df_to_clean: pl.DataFrame,
    hash_col: str,
    cache_file: Path,
    request_jsonl_path: Path,
    tracking_file: Path,
    upload_display_name: str,
    job_display_name: str,
    model_id: str = "gemini-3-flash-preview",
    batch_rows: int = 20
):
    # 1. Check if there is already a pending job tracked
    existing_job_id = load_job_id(tracking_file)
    if existing_job_id:
        print(f"A job is already pending: {existing_job_id}. Please check its status first.")
        return existing_job_id

    # 2. Parse cache and filter for new locations
    existing_results_df = parse_results(cache_file)
    if len(existing_results_df) > 0:
        already_processed_hashes = existing_results_df[hash_col].to_list()
    else:
        already_processed_hashes =[]

    new_df = df_to_clean.filter(~pl.col(hash_col).is_in(already_processed_hashes))

    if len(new_df) == 0:
        print(f"All locations already cached for {job_display_name}! No job needed.")
        return None

    # 3. Submit the job
    print(f"Found {len(new_df)} new locations. Submitting Gemini Batch Job...")
    write_batch_request_jsonl(new_df, hash_col, genai_schema, request_jsonl_path, batch_rows)

    upload = upload_jsonl_using_files_api(request_jsonl_path, upload_display_name)
    batch_job = initiate_genai_batch(upload.name, model_id, job_display_name)

    # 4. Save Job ID to tracking file so we can check it later
    save_job_id(tracking_file, batch_job.name)
    print(f"Job successfully submitted! Job ID: {batch_job.name}")
    print("You can now safely interrupt the notebook and retrieve results later.")

    return batch_job.name
```

```python {.marimo}
def check_and_retrieve_gemini_batch(
    cache_file: Path,
    tracking_file: Path
) -> pl.DataFrame:

    job_name = load_job_id(tracking_file)

    if not job_name:
        print("No pending jobs tracked. Returning currently cached results.")
        return parse_results(cache_file)

    # Ask Google for the current status of the job
    batch_job = client.batches.get(name=job_name)
    completed_states =['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED']

    if batch_job.state.name not in completed_states:
        print(f"Job {job_name} is currently: {batch_job.state.name}. Check back later.")
        return parse_results(cache_file)

    print(f"Job {job_name} finished with state: {batch_job.state.name}")

    # If successful, download and append
    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
        if batch_job.dest and batch_job.dest.file_name:
            result_file_name = batch_job.dest.file_name
            print(f"Downloading results file: {result_file_name}")
            file_content = client.files.download(file=result_file_name).decode('utf-8')

            # Append to master cache file
            with open(cache_file, 'a') as cache_f:
                cache_f.write("\n" + file_content.strip())
            print(f"Successfully appended new results to {cache_file.name}")

        # Clear the tracking file so we are ready for future jobs
        clear_job_id(tracking_file)

    else:
        print(f"Job did not succeed. Error: {batch_job.error}")
        # Clear the tracking file so the user can re-submit the failed delta
        clear_job_id(tracking_file)

    # Return the newly updated cache
    return parse_results(cache_file)
```

```python {.marimo}
h2a_df = pl.read_csv(json_path / "unmatched_h2a_locations.csv").fill_null('')
h2a_df_copy = h2a_df.with_columns(
    pl.struct(['city', 'county', 'state', 'zip']).map_elements(
        lambda x: get_location_hash(x['city'], x['county'], x['state'], x['zip']),
        return_dtype=pl.String
    ).alias('location_hash')
)

# File Paths
h2a_cache_file = Path(json_path / "cleaned_h2a_locations_response_string.txt")
h2a_tracking_file = Path(json_path / "h2a_active_job_id.txt")
h2a_jsonl_request = Path(json_path / "clean_h2a_locations_request.jsonl")
```

```python {.marimo}
# Run this cell to submit (it will skip instantly if everything is cached or a job is pending)
submit_gemini_batch_async(
    df_to_clean=h2a_df_copy,
    hash_col='location_hash',
    cache_file=h2a_cache_file,
    request_jsonl_path=h2a_jsonl_request,
    tracking_file=h2a_tracking_file,
    upload_display_name="h2a_batch_upload",
    job_display_name="clean_h2a_batch",
    model_id="gemini-3-flash-preview",
    batch_rows=20
)
```

```python {.marimo}
# Run this cell to check the status. 
# If it's done, it automatically downloads, appends, and merges!
h2a_results_df = check_and_retrieve_gemini_batch(
    cache_file=h2a_cache_file,
    tracking_file=h2a_tracking_file
)

# Merge back with the unmatched locations 
# (If the job is still running, this just merges whatever we already had)
h2a_with_suggestions = h2a_df_copy.join(
    h2a_results_df,
    on='location_hash',
    how='left',
    suffix='_suggested'
).rename({
    'zipcode': 'zip_suggested',
    'confidence': 'suggestion_confidence'
}).drop('location_hash')

h2a_with_suggestions.write_csv(Path(json_path / 'unmatched_h2a_with_suggestions.csv'))
```

```python {.marimo}
add_b_df = pl.read_csv(json_path / "unmatched_add_b_locations.csv").fill_null('')
add_b_df_copy = add_b_df.with_columns(
    pl.struct(['city', 'county', 'state', 'zip']).map_elements(
        lambda x: get_location_hash(x['city'], x['county'], x['state'], x['zip']),
        return_dtype=pl.String
    ).alias('location_hash')
)

# File Paths
add_b_cache_file = Path(json_path / "cleaned_add_b_locations_response_string.txt")
add_b_tracking_file = Path(json_path / "add_b_active_job_id.txt")
add_b_jsonl_request = Path(json_path / "clean_add_b_locations_request.jsonl")
```

```python {.marimo}
# Run this cell to submit (it will skip instantly if everything is cached or a job is pending)
submit_gemini_batch_async(
    df_to_clean=add_b_df_copy,
    hash_col='location_hash',
    cache_file=add_b_cache_file,
    request_jsonl_path=add_b_jsonl_request,
    tracking_file=add_b_tracking_file,
    upload_display_name="add_b_batch_upload",
    job_display_name="clean_add_b_batch",
    model_id="gemini-3-flash-preview",
    batch_rows=20
)
```

```python {.marimo}
# Run this cell to check the status. 
# If it's done, it automatically downloads, appends, and merges!
add_b_results_df = check_and_retrieve_gemini_batch(
    cache_file=add_b_cache_file,
    tracking_file=add_b_tracking_file
)

# Merge back with the unmatched locations 
# (If the job is still running, this just merges whatever we already had)
add_b_with_suggestions = add_b_df_copy.join(
    add_b_results_df,
    on='location_hash',
    how='left',
    suffix='_suggested'
).rename({
    'zipcode': 'zip_suggested',
    'confidence': 'suggestion_confidence'
}).drop('location_hash')

add_b_with_suggestions.write_csv(Path(json_path / 'unmatched_add_b_with_suggestions.csv'))
```

```python {.marimo}
client.close()
```

```python {.marimo}

```