import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import re
    import time
    from pathlib import Path
    import pyprojroot
    import polars as pl
    import pandas as pd
    import requests
    import httpx
    import us
    import seaborn as sns
    import matplotlib.pyplot as plt
    import altair as alt
    import pdfplumber
    # These are for running docling with CUDA 13.0
    # import torch
    # from docling.document_converter import DocumentConverter, PdfFormatOption
    # from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    # from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    # from docling.datamodel.base_models import InputFormat
    import fitz
    from bs4 import BeautifulSoup

    return BeautifulSoup, Path, alt, httpx, mo, pl, pyprojroot, re, us


@app.cell
def _(Path, pyprojroot):
    root_path = pyprojroot.find_root(criterion='pyproject.toml')
    aewr_path = root_path / 'data' / 'aewr'

    # Place Federal Register PDFs matching search criterion here
    Path(aewr_path / 'pdf').mkdir(parents=True, exist_ok=True)
    pdf_path = aewr_path / 'pdf'

    Path(aewr_path / 'txt').mkdir(parents=True, exist_ok=True)
    txt_path = aewr_path / 'txt'

    Path(aewr_path / 'xml').mkdir(parents=True, exist_ok=True)
    xml_path = aewr_path / 'xml'
    return aewr_path, root_path, xml_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For txt and pdf, we can directly construct the correct URL
    """)
    return


@app.cell
def _(aewr_path, pl):
    # List of CFR documents matching search criterion
    cfr_csv = (
        pl.read_csv(aewr_path / 'documents_matching_labor_certification_process_adverse_effect_wage_rates_from_labor_department_and_of_type_notice.csv'
        )
    )

    # New rule went into effect in 2009 that switched calculation of AEWR from using FLS to OES, which was immediately reverted by the incoming Obama administration. However that reversion was stopped by an injunction due to a lawsuit by the NCGA, so for the whole of 2019 (and the first 2 months of 2010), the OES methodology remained in effect. The OES numbers are available in a Final Rule in February 2010 (75 FR 6884, Doc. No. 2010-2731), when the Obama DOL reverted the change via proper notice and comment.
    aewr_2009_csv = (
        pl.read_csv(aewr_path / 'documents_matching_2010_2731.csv')
    )

    # Want to get URL for txt
    # Transform https://.../documents/2005/03/15/05-5026/slug 
    # To: https://.../documents/full_text/text/2005/03/15/05-5026.txt

    cfr_csv = cfr_csv.with_columns(
        txt_url = (
            pl.col("html_url")
            .str.replace("/documents/", "/documents/full_text/text/")
            # This regex removes the slug after the Doc ID and appends .txt
            # It looks for the date pattern, the ID, and then anything following it
            .str.replace(r"(\d{4}/\d{2}/\d{2}/[^/]+)/.*$", r"$1.txt")
        )
    )

    aewr_2009_csv = aewr_2009_csv.with_columns(
        txt_url = (
            pl.col("html_url")
            .str.replace("/documents/", "/documents/full_text/text/")
            # This regex removes the slug after the Doc ID and appends .txt
            # It looks for the date pattern, the ID, and then anything following it
            .str.replace(r"(\d{4}/\d{2}/\d{2}/[^/]+)/.*$", r"$1.txt")
        )
    )

    docs_csv = pl.concat([cfr_csv, aewr_2009_csv])
    return (docs_csv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For XML, we have to get URL from JSON metadata
    """)
    return


@app.cell
def _(re):
    def get_document_metadata(client, doc_num):
        """
        Fetch JSON metadata to find the correct XML URL and Policy Year.
        """
        url = f"https://www.federalregister.gov/api/v1/documents/{doc_num}.json"
        try:
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                # Extract Policy Year from title (e.g., "2024 Adverse Effect...")
                year_match = re.search(r"20\d{2}", data.get("title", ""))
                policy_year = year_match.group(0) if year_match else data.get("publication_date")[:4]

                return {
                    "xml_url": data.get("full_text_xml_url"),
                    "policy_year": policy_year,
                    "doc_num": doc_num
                }
        except Exception as e:
            print(f"Error fetching metadata for {doc_num}: {e}")
        return None

    return (get_document_metadata,)


@app.cell
def _(xml_path):
    def download_xml(client, meta):
        """
        Download the XML file to disk.
        """
        if not meta or not meta["xml_url"]:
            return None

        file_path = xml_path / f"{meta['doc_num']}.xml"
        # Skip if already downloaded
        if file_path.exists():
            return file_path

        try:
            resp = client.get(meta["xml_url"])
            if resp.status_code == 200:
                file_path.write_bytes(resp.content)
                return file_path
        except Exception as e:
            print(f"Download failed for {meta['doc_num']}: {e}")
        return None

    return (download_xml,)


@app.cell(hide_code=True)
def _():
    # # Make pdf and txt download requests to CFR archive
    # txt_files = {}
    # pdf_files = {}
    # for row in docs_csv.iter_rows(named=True):
    #     document_number = row['document_number'].strip()
    #     pdf_url_string = row['pdf_url']
    #     txt_url_string = row['txt_url']
    #     print(f'Getting doc {document_number}')

    #     pdf_files[document_number] = f'{document_number}.pdf'
    #     txt_files[document_number] = f'{document_number}.txt'

    #     # Make pdf request
    #     pdf_file = Path(pdf_path / f'{document_number}.pdf')
    #     if pdf_file.exists():
    #         print('pdf already downloaded')
    #     else:
    #         pdf_response = requests.get(pdf_url_string)
    #         if pdf_response.status_code == 200:
    #             print('pdf request successful')
    #             pdf_file.write_bytes(pdf_response.content)
    #         else:
    #             print('pdf request NOT successful')
    #         time.sleep(1)

    #     # Make txt request
    #     txt_file = Path(txt_path / f'{document_number}.txt')
    #     if txt_file.exists():
    #         print('txt already downloaded')
    #     else:
    #         txt_response = requests.get(txt_url_string)
    #         if txt_response.status_code == 200:
    #             print('txt request successful')
    #             txt_file.write_text(txt_response.text, encoding="utf-8")
    #         else:
    #             print('txt request NOT successful')
    #         time.sleep(1)
    return


@app.cell
def _(us):
    # 50 states minus Alaska
    state_names = [_s.name for _s in us.STATES if _s.name != 'Alaska']
    state_names.append('Idah') # There is a typo in 2005: Idaho is spelled as Idah
    return


@app.cell
def _(BeautifulSoup):
    # We can use GPOTABLE to extract the table from XML
    def parse_aewr_table(xml_path, policy_year, doc_num):
        """
        Parse the GPOTABLE specifically looking for AEWR data.
        """
        with open(xml_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "xml")

        extracted_rows = []

        # Federal Register tables use the GPOTABLE tag
        for table in soup.find_all("GPOTABLE"):
            # Check if this is the 'relevant' table (contains 'State' and 'AEWR' or 'Rate')
            header_text = table.get_text().upper()
            if "STATE" not in header_text or ("AEWR" not in header_text and "WAGE" not in header_text):
                continue

            rows = table.find_all("ROW")
            for row in rows:
                ents = row.find_all("ENT")
                entries = [ent.get_text(strip=True) for ent in ents]
                # AEWR tables usually have 2 columns: [State, Rate], so 2 ENTs
                if len(entries) >= 2:
                    # Clean up rate (remove $, commas, and non-numeric chars)
                    state = entries[0].replace(".", "").strip()
                    rate_raw = entries[1].replace("$", "").replace(",", "").strip()

                    try:
                        rate = float(rate_raw)
                        extracted_rows.append({
                            "state": state,
                            "aewr": rate,
                            "policy_year": int(policy_year),
                            "document_number": doc_num
                        })
                    except ValueError:
                        continue # Skip sub-headers or text rows which will fail rate float conversion

        return extracted_rows

    return (parse_aewr_table,)


@app.cell
def _(docs_csv, download_xml, get_document_metadata, httpx, xml_path):
    # Load document numbers from CSV
    df_input = docs_csv
    # Ensure column name matches CSV ('document_number' is standard FR export)
    doc_numbers = df_input["document_number"].str.strip_chars().to_list()

    doc_num_policy_year_dict = {}
    with httpx.Client(timeout=30.0) as client:
        # 2. Fetch metadata & Download
        print(f"Processing {len(doc_numbers)} documents...")
        for doc_num in doc_numbers:
            meta = get_document_metadata(client, doc_num)
            doc_num_policy_year_dict[doc_num] = meta['policy_year']
            if meta['xml_url']:
                check_path = xml_path / f"{doc_num}.xml"
                if check_path.exists():
                    print(f"{doc_num}.xml already downloaded")
                    continue
                path = download_xml(client, meta)
                print(f"Downloaded {doc_num} ({meta['policy_year']}) XML")
            elif not meta['xml_url']:
                print(f'XML for {doc_num} does not exist')
    return doc_num_policy_year_dict, doc_numbers


@app.cell
def _(doc_num_policy_year_dict, doc_numbers, parse_aewr_table, xml_path):
    # Extract from XML
    all_data = []
    for _f in doc_numbers:
        _file_name = _f + '.xml'
        _xml = xml_path / _file_name
        _policy_year = doc_num_policy_year_dict[_f]
        if _xml.exists():
            rows = parse_aewr_table(_xml, _policy_year, _f)
            all_data.extend(rows)
    return (all_data,)


@app.cell
def _(BeautifulSoup, pl, us, xml_path):
    # There is a special table for 2009 due to the shift to using OES for just that year
    extracted_rows = []
    aewr_2009_xml = xml_path / '2010-2731.xml'
    with open(aewr_2009_xml, "r", encoding="utf-8") as _f:
        soup = BeautifulSoup(_f, "xml")

    # Federal Register tables use the GPOTABLE tag
    for _table in soup.find_all("GPOTABLE"):
        # Check if this is the 'relevant' table (contains 'State' and 'AEWR' or 'Rate')
        _header_text = _table.get_text().upper()
        if "STATE" not in _header_text or ("AEWR" not in _header_text and "WAGE" not in _header_text):
            continue

        _rows = _table.find_all("ROW")
        for _row in _rows:
            _ents = _row.find_all("ENT")
            _entries = [_ent.get_text(strip=True) for _ent in _ents]
            # Special table has 6 columns
            if len(_entries) == 6:
                # Clean up rate (remove $, commas, and non-numeric chars, convert - sign)
                _state = _entries[0].replace(".", "").strip()
                _rate_raw = _entries[3].replace("$", "").replace(",", "").strip()

                extracted_rows.append({
                    "state": _state,
                    "aewr": _rate_raw,
                    "policy_year": 2009,
                    "document_number": '2010-2731'
                })

    aewr_2009 = pl.DataFrame(extracted_rows)
    abbrev_name_dict = us.states.mapping('abbr', 'name')
    aewr_2009 = aewr_2009.with_columns(
        pl.col('aewr').cast(pl.Float64, strict=False).alias('aewr'),
        pl.col('state').replace(abbrev_name_dict).alias('state')
    )
    return (aewr_2009,)


@app.cell
def _(aewr_2009, all_data, pl, us):
    aewr_df = pl.DataFrame(all_data).filter(pl.col('policy_year')!=2009)
    aewr_df = pl.concat([aewr_df, aewr_2009])
    # Fix a few typos
    # Idaho is spelled as Idah in 2005
    # Hawaii AEWR in 2003 is 9.42, not 9.29
    aewr_df = aewr_df.with_columns(
        pl.col('state').replace({'Idah':'Idaho'}),
        pl.when(
            (pl.col('state')=='Hawaii') & 
            (pl.col('policy_year')==2003)
        ).then(
            pl.lit(9.42)
        ).otherwise(
            pl.col('aewr')
        ).alias('aewr')
    )

    # Doc 2024-29549 is actually for 2025 but does not have year in the table name for some reason
    aewr_df = aewr_df.with_columns(
        pl.when(
            pl.col('document_number') == '2024-29549'
        ).then(
            pl.lit(2025)
        ).otherwise(
            pl.col('policy_year')
        ).alias('policy_year')
    )

    # Add state FIPS code and clean state name
    name_fips_dict = us.states.mapping('name', 'fips')
    aewr_df = aewr_df.with_columns(
        pl.col('state').replace(name_fips_dict).alias('state_fips_code')
    ).filter(
        pl.col('state') != 'Nationwide'
    )

    # Harmonize name for exporting
    aewr_df = aewr_df.rename({
        'state':'state_name',
        'policy_year':'year'
    })
    return (aewr_df,)


@app.cell
def _(aewr_df, aewr_path, pl):
    # Validate with USDA data
    usda_aewr = pl.read_excel(aewr_path / 'AEWR-2025.xlsx')
    usda_aewr = usda_aewr.unpivot(
        index=['State', 'Region'],
        variable_name='year',
        value_name='usda_aewr'
    ).rename({
        'State':'state_name',
        'Region':'aewr_region',
    }).with_columns(
        pl.col('year').cast(pl.Int64).alias('year'),
        pl.col('usda_aewr').cast(pl.Float64).alias('usda_aewr')
    )

    check_df = aewr_df.join(
        usda_aewr,
        on=['state_name', 'year'],
        how='right'
    ).with_columns(
        error = abs(pl.col('aewr') - pl.col('usda_aewr'))
    ).filter(pl.col('year')>=2000
    ).sort(pl.col('error'))

    check_df
    return (check_df,)


@app.cell
def _(aewr_path, check_df, pl, root_path):
    # Errors are almost all 0, except for some typos in the USDA dataset I think
    # Export
    export_df = check_df.select([
        'aewr', 'state_fips_code', 'state_name', 'year'
    ]).with_columns(
        pl.col('aewr')
    )
    export_df.write_csv(aewr_path / 'state_year_aewr.csv')
    export_df.write_parquet(root_path / 'Data Int' / 'aewr.parquet')
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    # # Extract table from text files
    # state_regex = "|".join(state_names)

    # def parse_local_file(path):
    #     if path is None: return None

    #     try:
    #         content = Path(path).read_text(encoding="utf-8")

    #         # This regex looks for:
    #         # 1. Start of line -> State Name
    #         # 2. Any number of dots/spaces (Leadership dots)
    #         # 3. Optional $ sign -> The AEWR float
    #         pattern = rf"^({state_regex})[\.\s]+?\$?\s?(\d+\.\d{{2}})"

    #         matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)

    #         # Return as "State:Rate|State:Rate"
    #         return "|".join([f"{m[0]}:{m[1]}" for m in matches])
    #     except:
    #         return None
    return


@app.cell(hide_code=True)
def _():
    # # Need to extract text from pdf by column
    # x0 = 0.05  # Distance of left side of column 1 from left side of page.
    # x1 = 0.35  # Distance of right side of column 1 from left side of page
    # x2 = 0.65  # Distance of right side of column 2 from left side of page
    # x3 = 0.95  # Distance of right side of column 3 from left side of page

    # y0 = 0.05  # Distance of top from top of page.
    # y1 = 0.95  # Distance of bottom from top of page.

    # rows_list = []  # List of rows (as dictionaries) to be converted to dataframe

    # for file in pdf_path.iterdir():
    #     # Read all pdf files in directory
    #     file_type = file.suffix
    #     if file_type != ".pdf":
    #         continue

    #     file_name = file.stem

    #     if file_name == "03-6559":  # correction issued for 2003 AEWR for Hawaii
    #         continue
    #     elif (
    #         file_name == "2013-00115"
    #     ):  # separate AEWR for open range livestock occupation newly added
    #         continue
    #     elif (
    #         file_name == "2023-12896"
    #     ):  # update to AEWR used for states and territories not covered by the Farm Labor Survey
    #         continue

    #     pdf_content = "\"

    #     with pdfplumber.open(file) as pdf:
    #         for i, page in enumerate(pdf.pages):
    #             width = page.width
    #             height = page.height

    #             # Crop pages
    #             left_bbox = (
    #                 x0 * float(width),
    #                 y0 * float(height),
    #                 x1 * float(width),
    #                 y1 * float(height),
    #             )
    #             middle_bbox = (
    #                 x1 * float(width),
    #                 y0 * float(height),
    #                 x2 * float(width),
    #                 y1 * float(height),
    #             )
    #             right_bbox = (
    #                 x2 * float(width),
    #                 y0 * float(height),
    #                 x3 * float(width),
    #                 y1 * float(height),
    #             )

    #             page_crop = page.crop(bbox=left_bbox)
    #             left_text = page_crop.extract_text()
    #             page_crop = page.crop(bbox=middle_bbox)
    #             middle_text = page_crop.extract_text()
    #             page_crop = page.crop(bbox=right_bbox)
    #             right_text = page_crop.extract_text()

    #             page_content = "\n".join([left_text, middle_text, right_text])
    #             pdf_content = pdf_content + page_content

    #         full_string = pdf_content.replace("\n", "\")

    #         # Year AEWR notice is published for
    #         try:
    #             # Skips files for ranching occupations, which we don't care about
    #             # DOL also decided to change the table titling format for 2025 for whatever reason, but we don't care about 2025 for now
    #             notice_year_string = re.search(r"\d\d\d\d [Aa][Dd][Vv][Ee][Rr][Ss][Ee] [Ee][Ff][Ff][Ee][Cc][Tt]", full_string).group(0)
    #             notice_year = notice_year_string[0:4]
    #         except:
    #             continue

    #         # Parse each PDF
    #         print(f"Parsing file {file_name}, for year {notice_year}.")

    #         # Parse state names and corresponding AEWRs within each PDF using regex
    #         for state in state_names:
    #             if (
    #                 notice_year == "2005" and state == "Idaho"
    #             ):  # typo in the 2005 AEWR table for Idaho
    #                 state = "Idah"

    #             # This regex captures everything between the state name and the first .dd after, which are the last 2 digits of the AEWR for that state
    #             state_wage = re.search(rf"{state}[ .$\d]*?\.\d\d", full_string)

    #             # Parse wage within each 'state...wage' string using regex
    #             state_wage_string = state_wage.group(0)
    #             wage = re.search(r"\d+?\.\d\d", state_wage_string)

    #             # Put results into dataframe
    #             row_dict = {}  # define row in dictionary format, column names as keys
    #             aewr = wage.group(0)

    #             if (
    #                 notice_year == "2005" and state == "Idah"
    #             ):  # fix typo in the 2005 AEWR table for Idaho
    #                 state = "Idaho"
    #             if (
    #                 notice_year == "2003" and state == "Hawaii"
    #             ):  # correction issued for 2003 AEWR for Hawaii
    #                 aewr = "9.42"

    #             row_dict.update(
    #                 {"state_name": state, "year": notice_year, "aewr": aewr}
    #             )
    #             rows_list.append(row_dict)

    # # Concatenate all rows into a dataframe
    # aewr_df = pl.from_dicts(rows_list)

    # # Dict for mapping from state names to FIPS
    # name_fips_dict = us.states.mapping('name', 'fips')

    # # Add state FIPS code and export
    # aewr_df = (
    #     aewr_df
    #         .with_columns(
    #             pl.col('state_name')
    #                 .replace(name_fips_dict)
    #                 .alias('state_fips_code')
    #         )
    # )
    # aewr_df.write_csv(aewr_path / 'state_year_aewr.csv')
    # aewr_df.write_parquet(root_path / 'Data Int' / 'aewr.parquet')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What does the evolution of the AEWR look like?
    """)
    return


@app.cell
def _(aewr_df, alt, pl):
    # Clean and Aggregate in Polars
    # Assuming 'year' and 'income' are strings
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    agg_exprs = [
        pl.col("aewr").quantile(q).alias(f"q{int(q*100)}") 
        for q in quantiles
    ]

    # Add the unweighted mean to the list of calculations
    agg_exprs.append(pl.col("aewr").mean().alias("mean_val"))

    stats_df = (
        aewr_df.with_columns([
            pl.col("year").cast(pl.Int32, strict=False),
            pl.col("aewr").cast(pl.Float64, strict=False)
        ])
        .drop_nulls() # Remove rows that weren't valid numbers
        .group_by("year")
        .agg(agg_exprs)
        .sort("year")
    )

    # Build the Altair Chart
    base = alt.Chart(stats_df).encode(
        x=alt.X("year:O", title="Year")
    )

    # Layer 1: Outer Ribbon (5th - 95th)
    ribbon_outer = base.mark_area(opacity=0.15, color="#2c7bb6").encode(
        y=alt.Y("q5:Q", title="Value"),
        y2="q95:Q"
    )

    # Layer 2: Inner Ribbon (25th - 75th)
    ribbon_inner = base.mark_area(opacity=0.4, color="#2c7bb6").encode(
        y="q25:Q",
        y2="q75:Q"
    )

    # Layer 3: Median Line (50th Percentile)
    median_line = base.mark_line(color="white", size=2.5).encode(
        y="q50:Q"
    )

    # Layer 4: Unweighted Mean Line (Dashed)
    mean_line = base.mark_line(
        color="#d7191c", 
        strokeDash=[5, 5], 
        size=2
    ).encode(
        y="mean_val:Q"
    )

    # Layer 5: Invisible points for tooltips (Interactive)
    tooltips = base.mark_point(opacity=0).encode(
        y="mean_val:Q",
        tooltip=[
            alt.Tooltip("year:O"),
            alt.Tooltip("mean_val:Q", title="Mean", format=".2f"),
            alt.Tooltip("q50:Q", title="Median", format=".2f"),
            alt.Tooltip("q25:Q", title="25th Pctl", format=".2f"),
            alt.Tooltip("q75:Q", title="75th Pctl", format=".2f")
        ]
    )

    # Combine everything
    chart = (
        ribbon_outer + 
        ribbon_inner + 
        median_line + 
        mean_line + 
        tooltips
    ).properties(
        width=700,
        height=450,
        title="AEWR: Mean vs Median and Quantiles"
    ).interactive()

    chart.show()
    chart.save('aewr_nominal_quantiles.png')
    return


@app.cell
def _(aewr_df, pl):
    # 1. Clean data and calculate annual changes per ID
    processed_df = (
        aewr_df.with_columns([
            pl.col("state_fips_code").cast(pl.Utf8),
            pl.col("year").cast(pl.Int32, strict=False),
            pl.col("aewr").cast(pl.Float64, strict=False)
        ])
        .drop_nulls()
        .sort(["state_fips_code", "year"])
        .with_columns(
            # Change = Current Year - Previous Year for each ID
            aewr_change = pl.col("aewr").diff().over("state_fips_code")
        )
        .filter(pl.col("aewr_change").is_not_null())
    )

    # 2. Calculate ECDF logic per year
    ecdf_df = (
        processed_df
        .sort(["year", "aewr_change"])
        .with_columns(
            # (Rank of change in that year) / (Total observations in that year)
            cum_prob = (pl.int_range(1, pl.len() + 1).over("year") / pl.len().over("year"))
        )
    )
    return (ecdf_df,)


@app.cell
def _(alt, ecdf_df):
    # Define the base ECDF line
    ecdf_line = alt.Chart().mark_line(interpolate='step-after', color='#2c7bb6').encode(
        x=alt.X("aewr_change:Q", title="aewr Change"),
        y=alt.Y("cum_prob:Q", title="Cumulative Prob"),
        tooltip=["aewr_change", "cum_prob"]
    )

    # Optional: Add a vertical reference line at 0 (shows % who lost vs gained)
    rule = alt.Chart().mark_rule(color='orange', strokeDash=[4,4]).encode(
        x=alt.value(0) # or alt.X(datum=0)
    )

    # Layer the line and the rule, then FACET by year
    faceted_chart = alt.layer(ecdf_line, rule, data=ecdf_df).facet(
        facet=alt.Facet("year:O", title="Annual Change Distribution"),
        columns=3  # Adjust this to 4 or 5 depending on how many years you have
    ).properties(
        title="Yearly Empirical CDF of aewr Changes"
    ).resolve_scale(
        x='shared' # Keeps the X-axis range the same across all facets for comparison
    )

    faceted_chart.show()
    return


if __name__ == "__main__":
    app.run()
