import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import pyprojroot
    import dotenv, os
    import json
    import polars as pl
    from google import genai
    from google.genai import types
    from google.genai._transformers import process_schema
    from typing import Optional, List, get_args, get_origin
    from pydantic import BaseModel, Field
    import time

    return (
        BaseModel,
        Field,
        Optional,
        Path,
        dotenv,
        genai,
        get_args,
        get_origin,
        json,
        os,
        pl,
        pyprojroot,
        time,
        types,
    )


@app.cell
def _(dotenv, os, pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    dotenv.load_dotenv()
    gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    return gemini_api_key, root_path


@app.cell
def _(BaseModel, Field, Optional, get_args, get_origin):
    # Define the Structured Output schema we will use for Gemini
    # This ensures Gemini returns valid JSON matching our DataFrame structure
    class CleanedLocation(BaseModel):
        original_id: int = Field(..., description="The index of the row from the input data.")
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
    return (CleanedLocation,)


@app.cell
def _(json, pd):
    # This function generates the batch request in the form of a JSONL file
    # chunk_size = # of rows in each request
    def write_batch_request_jsonl(df: pd.DataFrame, df_original_id: str, response_schema: dict, jsonl_file_path: str, chunk_size: int):

        if df_original_id not in df.columns:
            raise Exception("Missing ID in input dataframe for matching rows with output")

        total_chunks = (len(df) // chunk_size) + 1
        print(f"Writing {total_chunks} requests (chunks) to {jsonl_file_path}")

        jsonl_entries_list = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size] # This gets added to our prompt as json
            chunk_in_json = chunk.to_json(orient='records')

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

    return


@app.cell
def _(gemini_api_key, genai):
    # Initiate GenAI client here
    client = genai.Client(api_key = gemini_api_key)
    return (client,)


@app.cell
def _(client, types):
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

    return


@app.cell
def _(client):
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

    return


@app.cell
def _(client, time):
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

    return


@app.cell
def _(Path, pl, root_path):
    # Chunk up input dataframe
    # Must add ID to track original rows for merging with cleaned output
    json_path = root_path / 'code' / 'json'
    # Start with H-2A data
    h2a_df = pl.read_csv(json_path / "unmatched_h2a_locations.csv")
    h2a_df_copy = h2a_df.clone()
    h2a_df_copy = h2a_df_copy .with_row_index(name="original_id")

    # Batch specs
    h2a_jsonl_file_path = Path(json_path / "clean_h2a_locations_request.jsonl")
    batch_rows = 20

    # Upload specs
    h2a_upload_display_name = "h2a_batch_jsonl_file_upload"

    # Processing specs
    model_id = "gemini-3-flash-preview"
    h2a_job_display_name = "clean_h2a_batch_job"

    # Retrieval specs
    sleep_duration = 10
    h2a_output_file_name = Path(json_path / "cleaned_h2a_locations_response_string.txt")
    return h2a_df_copy, json_path


@app.cell
def _():
    # write_batch_request_jsonl(h2a_df_copy, 'original_id', genai_schema, h2a_jsonl_file_path, batch_rows)
    # h2a_upload = upload_jsonl_using_files_api(h2a_jsonl_file_path, h2a_upload_display_name)
    # h2a_batch_job = initiate_genai_batch(h2a_upload.name, model_id, h2a_job_display_name)
    # finished_h2a_batch_job = poll_genai_job(h2a_batch_job.name, sleep_duration, h2a_output_file_name)
    return


@app.cell
def _(json_path, pl):
    # Repeat with Addendum B data
    add_b_df = pl.read_csv(json_path / "unmatched_add_b_locations.csv")
    add_b_df_copy = add_b_df.clone()
    add_b_df_copy = add_b_df_copy.with_row_index(name='original_id')

    # Batch specs
    add_b_jsonl_file_path = json_path / "clean_add_b_locations_request.jsonl"
    # batch_rows = 20

    # Upload specs
    add_b_upload_display_name = "add_b_batch_jsonl_file_upload"

    # Processing specs
    # model_id = "gemini-3-flash-preview"
    add_b_job_display_name = "clean_add_b_batch_job"

    # Retrieval specs
    # sleep_duration = 10
    add_b_output_file_name = json_path / "cleaned_add_b_locations_response_string.txt"
    return (add_b_df_copy,)


@app.cell
def _():
    # write_batch_request_jsonl(add_b_df_copy, 'original_id', genai_schema, add_b_jsonl_file_path, batch_rows)
    # add_b_upload = upload_jsonl_using_files_api(add_b_jsonl_file_path, add_b_upload_display_name)
    # add_b_batch_job = initiate_genai_batch(add_b_upload.name, model_id, add_b_job_display_name)
    # finished_add_b_batch_job = poll_genai_job(add_b_batch_job.name, sleep_duration, add_b_output_file_name)
    return


@app.cell
def _(CleanedLocation, ValidationError, json, pl):
    # Define a function to merge results together into a coherent dataframe
    def parse_results(response_string_file):
        all_cleaned_locations = []

        # Load results string from saved local file
        with open(response_string_file) as f:
            lines = f.readlines()

        # Parse response file content
        # Each batch item response appears as a line in the file, as a string containint a JSON object
        for l in lines:
            line = l.splitlines() # Break open string
            if line: # Check for presence of response
                batch_item = json.loads(line[0])
                chunk_key = batch_item.get("key")

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

        # Change null to ''
        combined_results = pl.DataFrame(all_cleaned_locations).with_columns(
            pl.selectors.string().str.replace_all("null", "")
    )
        return combined_results

    return (parse_results,)


@app.cell
def _(Path, json_path, parse_results):
    h2a_results_df = parse_results(Path(json_path / "cleaned_h2a_locations_response_string.txt"))
    add_b_results_df = parse_results(Path(json_path / "cleaned_add_b_locations_response_string.txt"))
    return add_b_results_df, h2a_results_df


@app.cell
def _(Path, h2a_df_copy, h2a_results_df, json_path):
    h2a_with_suggestions = h2a_df_copy.join(
        h2a_results_df,
        on='original_id',
        how='left',
        suffix='_suggested'
    ).rename(
        mapping={
            'zipcode': 'zip_suggested',
            'confidence':'suggestion_confidence'}
    ).drop('original_id')
    h2a_with_suggestions.write_csv(Path(json_path / 'unmatched_h2a_with_suggestions.csv'))
    return


@app.cell
def _(Path, add_b_df_copy, add_b_results_df, json_path):
    add_b_with_suggestions = add_b_df_copy.join(
        add_b_results_df,
        on='original_id',
        how='left',
        suffix='_suggested'
    ).rename(
        mapping={
            'zipcode': 'zip_suggested',
            'confidence':'suggestion_confidence'}
    ).drop('original_id')
    add_b_with_suggestions.write_csv(Path(json_path / 'unmatched_add_b_with_suggestions.csv'))
    return


@app.cell
def _(client):
    client.close()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
