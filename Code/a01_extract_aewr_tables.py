import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import re
    import time
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import pdfplumber
    import requests
    import us
    import marimo as mo
    return Path, pd, pdfplumber, re, us


@app.cell
def _(pd):
    # List of CFR documents matching search criterion
    cfr_csv = pd.read_csv(
        "../Data/aewr/documents_matching_labor_certification_process_adverse_effect_wage_rates_from_labor_department_and_of_type_notice.csv"
    )
    return


@app.cell
def _():
    # # Save all Federal Register PDFs matching search criterion
    # Path("../Data/aewr/pdf/").mkdir(parents=True, exist_ok=True)

    # for index, row in cfr_csv.iterrows():
    #     document_number = row['document_number'].strip()
    #     url_string = row['pdf_url']
    #     response = requests.get(url_string) # URL of the CFR document in pdf format
    #     pdf_file = Path(f"../Data/aewr/pdf/{document_number}.pdf")
    #     pdf_file.write_bytes(response.content)
    #     time.sleep(1) # Pause for 1 second between each pdf download
    return


@app.cell
def _():
    # Need to extract text from pdf by column
    x0 = 0.05  # Distance of left side of column 1 from left side of page.
    x1 = 0.35  # Distance of right side of column 1 from left side of page
    x2 = 0.65  # Distance of right side of column 2 from left side of page
    x3 = 0.95  # Distance of right side of column 3 from left side of page

    y0 = 0.05  # Distance of top from top of page.
    y1 = 0.95  # Distance of bottom from top of page.
    return x0, x1, x2, x3, y0, y1


@app.cell
def _():
    # 50 states minus Alaska
    state_names = [
        "Alabama",
        "Arkansas",
        "Arizona",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Florida",
        "Georgia",
        "Hawaii",
        "Iowa",
        "Idaho",
        "Illinois",
        "Indiana",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Massachusetts",
        "Maryland",
        "Maine",
        "Michigan",
        "Minnesota",
        "Missouri",
        "Mississippi",
        "Montana",
        "North Carolina",
        "North Dakota",
        "Nebraska",
        "New Hampshire",
        "New Jersey",
        "New Mexico",
        "Nevada",
        "New York",
        "Ohio",
        "Oklahoma",
        "Oregon",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Virginia",
        "Vermont",
        "Washington",
        "Wisconsin",
        "West Virginia",
        "Wyoming",
    ]
    return (state_names,)


@app.cell
def _(Path, pdfplumber, re, state_names, x0, x1, x2, x3, y0, y1):
    rows_list = []  # List of rows (as dictionaries) to be converted to dataframe
    for file in Path("../Data/aewr/pdf/").iterdir():
        # Read all pdf files in directory
        file_type = file.suffix
        if file_type != ".pdf":
            continue

        file_name = file.stem

        if file_name == "03-6559":  # correction issued for 2003 AEWR for Hawaii
            continue
        elif (
            file_name == "2013-00115"
        ):  # separate AEWR for open range livestock occupation newly added
            continue
        elif (
            file_name == "2023-12896"
        ):  # update to AEWR used for states and territories not covered by the Farm Labor Survey
            continue

        pdf_content = ""

        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                width = page.width
                height = page.height

                # Crop pages
                left_bbox = (
                    x0 * float(width),
                    y0 * float(height),
                    x1 * float(width),
                    y1 * float(height),
                )
                middle_bbox = (
                    x1 * float(width),
                    y0 * float(height),
                    x2 * float(width),
                    y1 * float(height),
                )
                right_bbox = (
                    x2 * float(width),
                    y0 * float(height),
                    x3 * float(width),
                    y1 * float(height),
                )

                page_crop = page.crop(bbox=left_bbox)
                left_text = page_crop.extract_text()
                page_crop = page.crop(bbox=middle_bbox)
                middle_text = page_crop.extract_text()
                page_crop = page.crop(bbox=right_bbox)
                right_text = page_crop.extract_text()

                page_content = "\n".join([left_text, middle_text, right_text])
                pdf_content = pdf_content + page_content

            full_string = pdf_content.replace("\n", "")

            # Year AEWR notice is published for
            try:
                # Skips files for ranching occupations, which we don't care about
                # DOL also decided to change the table titling format for 2025 for whatever reason, but we don't care about 2025 for now
                notice_year_string = re.search(r"\d\d\d\d [Aa][Dd][Vv][Ee][Rr][Ss][Ee] [Ee][Ff][Ff][Ee][Cc][Tt]", full_string).group(0)
                notice_year = notice_year_string[0:4]
            except:
                continue

            # Parse each PDF
            print(f"Parsing file {file_name}, for year {notice_year}.")

            # Parse state names and corresponding AEWRs within each PDF using regex
            for state in state_names:
                if (
                    notice_year == "2005" and state == "Idaho"
                ):  # typo in the 2005 AEWR table for Idaho
                    state = "Idah"

                # This regex captures everything between the state name and the first .dd after, which are the last 2 digits of the AEWR for that state
                state_wage = re.search(rf"{state}[ .$\d]*?\.\d\d", full_string)

                # Parse wage within each 'state...wage' string using regex
                state_wage_string = state_wage.group(0)
                wage = re.search(r"\d+?\.\d\d", state_wage_string)

                # Put results into dataframe
                row_dict = {}  # define row in dictionary format, column names as keys
                aewr = wage.group(0)

                if (
                    notice_year == "2005" and state == "Idah"
                ):  # fix typo in the 2005 AEWR table for Idaho
                    state = "Idaho"
                if (
                    notice_year == "2003" and state == "Hawaii"
                ):  # correction issued for 2003 AEWR for Hawaii
                    aewr = "9.42"

                row_dict.update(
                    {"state_name": state, "year": notice_year, "aewr": aewr}
                )
                rows_list.append(row_dict)
    return (rows_list,)


@app.cell
def _(pd, rows_list, us):
    # Concatenate all rows into a dataframe
    df = pd.DataFrame(rows_list)

    # Add state FIPS code and export
    df["state_fips"] = df["state_name"].map(us.states.mapping("name", "fips"))
    df = df.rename(columns={"state_fips": "state_fips_code"})
    df = df.drop(columns=["state_name"])
    df.to_csv("../Data/aewr/state_year_aewr.csv", index=False)
    df.to_parquet("../files_for_phil/aewr.parquet", index=False)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
