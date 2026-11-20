# Dataset Creation

## System Details for repoduction

R version 4.2.1 (2022-06-23 ucrt)

Platform: x86_64-w64-mingw32/x64 (64-bit)

Running under: Windows 10 x64 (build 22631)

**R Packages**

- data.table_1.14.2
- readxl_1.4.3   
- dplyr_1.1.4    
- here_1.0.1
- tidyverse_2.0.0
    +  lubridate_1.9.3
    +  forcats_1.0.0    
    +  stringr_1.5.1
    +  readr_2.1.4
    +  tidyr_1.3.0
    +  tibble_3.2.1
    +  purrr_1.0.2
    +  ggplot2_3.4.4

This `h2a_combined.csv` dataset was created using the files in the `Raw Data` directory. 

To reproduce our work please follow these steps:

- Step 1. Download the Repo Directory and unzip it to a location on your computer.

- Step 2. Open the R script `00_Setup_and_Make.R` in an R compatible code editor.
 
- Step 3. On Line 19, set the variable `your_path_before_Repo` to the location where the Repo Directory was unzipped.
 
- Step 4. Run the R script `00_Setup_and_Make.R` in R to create the 'h2a_combined.csv' dataset.

## Accessing Raw Data 

The raw H-2A Disclosure data for this project come from the US Department of Labor's Foreign Labor Certification Performance Data Archive. The data files for fiscal years 2008-2024 were retrieved from: [https://www.dol.gov/agencies/eta/foreign-labor/performance][https://www.dol.gov/agencies/eta/foreign-labor/performance] and the data for fiscal years 2006 and 2007 were obtained from a previous (and no longer maintained) version of this website: [https://www.flcdatacenter.com/CaseH2A.aspx][https://www.flcdatacenter.com/CaseH2A.aspx].

## Dataset Description

The final dataset is best used for analyzing trends and geographic differences in H-2A positions certified. We have taken care in constructing this dataset to harmonize variables that have changed across time and to include only the most useful information for researchers. We have also constructed some variables that we think will be useful, including an indicator for whether the request is for crop and animal employment, whether it is for a contractor or an employer, and the number of weeks requested. The final dataset contains the following features:

- Feature 1: As in the original data, each observation is a single request for H-2A positions. Each request includes features 2-6, in addition to many others.
- Feature 2: Zipcode of employer and worksite.
- Feature 3: Number of positions requested and certified (granted). This can be used to determine whether a request was approved or denied and positions certified reflect the total number of visas issued, a common measure of the number of H-2A workers.
- Feature 4: Approved contract start and end dates. This can be used to identify employment duration.
- Feature 5: Requested weekly hours. 
- Feature 6: Promised pay rate, and the unit of pay (most requests use hourly pay rates).

Please refer to the data dictionary `h2a Data Dictionary.xlsx` in the Repo for more detailed information about each variable in the `h2a_combined.csv` dataset.

## Data Dictionary and Variable Crosswalk

**h2a Data Dictionary.xlsx**

We provide a data dictionary in the `h2a Data Dictionary.xlsx` file that describes the variables in the `h2a_combined.csv` dataset. The data dictionary includes the full set of variables, their descriptions, and the source of the data. We also provide matrix that maps the original variable names to the harmonized variable names in the `h2a_combined.csv` dataset. The raw data shift their variable schemas throughout the years, so to be as clear as possible, we provide this crosswalk to help users understand what variable we used to create this data.

If you have any questions about the data dictionary or variable crosswalk, please contact _alihill@berkeley.edu_.
If you find any errors in the data dictionary or variable crosswalk, please submit an issue or pull request and contact us by email with a subject _H-2A Data Repo_.


**h2a_var_crosswalk.csv**

The `h2a_var_crosswalk.csv` file contains a list of the variables used in each year of the raw data. Due to the shifting variables and variable names across the years, we created this crosswalk to track what variables we need and to create the harmonized variable names. This file is used in the `01_combine_h2a.R` script to subset the raw files to the necessary variables and to create the harmonized variable names.


## Adding New H2A Data to this Repo

To add new H2A data to this repo, follow these steps:

*  Step 1: Download additional H2A data from the source and save to `Raw Data/h2a_program_data/` directory.
*  Step 2: Create a new set of coloumns to match the schema in the `h2a_var_crosswalk.csv` for the new year of data. 
*  Step 3: By hand identify the variables needed and create the harmonized variable names.
*  Step 4: Add the new year to the `dates` variable list on line 46 in the `01_combine_h2a.R` file to ensure it is used in the `combiner` function.
*  Step 5: Run the R script `01_combine_h2a.R` in R to create the `h2a_combined.csv` dataset.

## License

_[Specify the license under which the dataset is released, if applicable.]_

## Citation

If you use this dataset in your research or project, please cite it as:
_
[Provide the citation details here.]_
