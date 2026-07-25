# Purpose: Publish Deming's three-digit SOC decision-intensity crosswalk.
# Input: Tables A1A-A1C of Deming (2021), supplied with the analysis request.
# Output: data/raw/geographic_crosswalks/deming_decision_intensity_3digit_soc.csv.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(tidyverse)

source_url <- paste0(
  "https://static1.squarespace.com/static/60832ecef615231cedd30911/",
  "t/608ac82baec65e6bb38ac16b/1619707947502/",
  "Deming_Decisions_Appendix.pdf"
)

decision_intensity <- tribble(
  ~soc_3digit, ~occupation_category, ~decision_intensity, ~employment_share, ~share_with_ba, ~wage_salary_income,
  "111", "Top Executives and Managers", 9.69, 0.015, 0.585, 133957,
  "112", "Advertising, PR, Sales Managers", 8.20, 0.007, 0.718, 103020,
  "113", "Operations Specialties Managers", 7.39, 0.021, 0.598, 98987,
  "119", "Other Managers", 6.52, 0.062, 0.525, 77320,
  "131", "Business Operations Specialists", 4.92, 0.034, 0.644, 75335,
  "132", "Financial Specialists", 5.30, 0.023, 0.771, 86446,
  "151", "Computer Occupations", 5.70, 0.032, 0.679, 88444,
  "152", "Mathematical Science Occupations", 6.26, 0.002, 0.820, 93271,
  "171", "Architects and Surveyors", 5.58, 0.002, 0.865, 80826,
  "172", "Engineers", 7.15, 0.014, 0.814, 96256,
  "173", "Drafters and Engineering Technicians", 3.82, 0.004, 0.221, 56914,
  "191", "Life Scientists", 6.55, 0.002, 0.988, 83003,
  "192", "Physical Scientists", 6.09, 0.003, 0.983, 84971,
  "193", "Social Scientists and Related", 6.03, 0.002, 0.977, 73488,
  "194", "Life/Physical/Social Science Technicians", 3.05, 0.002, 0.394, 48223,
  "211", "Counselors and Social Workers", 5.59, 0.014, 0.761, 46043,
  "212", "Religious Workers", 5.75, 0.003, 0.718, 44684,
  "231", "Lawyers and Judges", 6.17, 0.007, 0.981, 149559,
  "232", "Legal Support Workers", 4.18, 0.004, 0.464, 54432,
  "251", "Postsecondary Teachers", 5.47, 0.008, 0.924, 68043,
  "252", "K-12 Teachers", 5.42, 0.035, 0.884, 48829,
  "253", "Other Teachers and Instructors", 7.01, 0.005, 0.529, 32632,
  "254", "Librarians and Archivists", 5.17, 0.002, 0.754, 44739,
  "259", "Other Education Occupations", 5.40, 0.008, 0.326, 25682,
  "271", "Art and Design Workers", 3.58, 0.007, 0.606, 50042,
  "272", "Entertainers and Performers", 4.29, 0.004, 0.555, 44240,
  "273", "Media and Communications Workers", 3.66, 0.006, 0.743, 58369,
  "274", "Media/Communications Equipment Workers", 3.74, 0.002, 0.503, 38081,
  "291", "Healthcare Practitioners", 5.57, 0.043, 0.771, 95494,
  "292", "Health Technologists", 3.14, 0.020, 0.213, 45288,
  "299", "Other Healthcare Occupations", 4.00, 0.001, 0.700, 55569,
  "311", "Home Health and Personal Care Aides", 2.36, 0.021, 0.104, 24605,
  "312", "Occupational and Physical Therapy Aides", 3.88, 0.001, 0.267, 36798,
  "319", "Other Healthcare Aides", 1.86, 0.010, 0.159, 30597,
  "331", "Supervisors, Protective Services", 5.31, 0.002, 0.378, 76478,
  "332", "Firefighting and Prevention Workers", 3.26, 0.002, 0.222, 69683,
  "333", "Law Enforcement Workers", 3.51, 0.009, 0.336, 64710,
  "339", "Other Protective Service Workers", 2.76, 0.008, 0.196, 35942,
  "351", "Supervisors, Food Preparation Workers", 3.52, 0.007, 0.136, 32298,
  "352", "Cooks and Food Preparation Workers", 1.83, 0.020, 0.061, 19846,
  "353", "Food and Beverage Serving Workers", 1.16, 0.020, 0.130, 20917,
  "359", "Other Food Preparation and Service Jobs", 1.97, 0.005, 0.068, 15727,
  "371", "Supervisors, Grounds Cleaning/Maintenance", 3.70, 0.003, 0.155, 37785,
  "372", "Building Cleaning and Pest Control", 1.66, 0.026, 0.060, 23467,
  "373", "Grounds Maintenance Workers", 1.69, 0.008, 0.070, 22721,
  "391", "Supervisors, Personal Care and Services", 4.80, 0.001, 0.230, 33722,
  "392", "Animal Care and Service Workers", 3.64, 0.002, 0.222, 20159,
  "393", "Entertainment Attendants", 1.63, 0.002, 0.198, 24897,
  "394", "Funeral Service Workers", 1.37, 0.000, 0.305, 46834,
  "395", "Personal Appearance Workers", 2.01, 0.009, 0.082, 19252,
  "396", "Baggage Porters and Bellhops", 2.38, 0.001, 0.164, 32666,
  "399", "Other Personal Care and Service Workers", 4.46, 0.012, 0.246, 17549,
  "411", "Supervisors, Sales Workers", 5.05, 0.030, 0.309, 58888,
  "412", "Retail Sales Workers", 1.76, 0.038, 0.154, 26265,
  "413", "Sales Representatives, Services", 4.90, 0.011, 0.542, 87290,
  "414", "Sales Representatives, Wholesale and Manufacturing", 3.14, 0.009, 0.493, 81564,
  "419", "Other Sales Workers", 3.67, 0.009, 0.462, 56331,
  "431", "Supervisors, Office and Administrative Support", 5.41, 0.009, 0.377, 59518,
  "432", "Communications Equipment Operators", 1.05, 0.000, 0.214, 35297,
  "433", "Financial Clerks", 2.31, 0.017, 0.241, 41745,
  "434", "Information and Records Clerks", 2.15, 0.036, 0.242, 34138,
  "435", "Scheduling and Dispatching Workers", 1.80, 0.014, 0.165, 42404,
  "436", "Secretaries and Administrative Assistants", 2.75, 0.018, 0.253, 37871,
  "439", "Other Office and Administrative Support Workers", 1.83, 0.018, 0.273, 36159,
  "451", "Farming, Fishing, and Forestry Workers", 4.05, 0.000, 0.177, 45489,
  "452", "Agricultural Workers", 2.74, 0.005, 0.075, 26141,
  "453", "Fishing and Hunting Workers", 2.65, 0.000, 0.108, 22900,
  "454", "Forestry and Logging Workers", 1.89, 0.000, 0.075, 29201,
  "471", "Supervisors, Construction and Extraction", 4.19, 0.006, 0.114, 62880,
  "472", "Construction Trade Workers", 1.43, 0.043, 0.054, 37799,
  "473", "Helpers, Construction Trades", 1.13, 0.000, 0.053, 27383,
  "474", "Other Construction Workers", 1.81, 0.002, 0.120, 47356,
  "475", "Extraction Workers", 1.58, 0.001, 0.062, 60422,
  "491", "Supervisors, Installation and Repair", 5.05, 0.002, 0.154, 67715,
  "492", "Electrical and Electronic Equipment Repair", 2.12, 0.003, 0.161, 48493,
  "493", "Vehicle and Mobile Equipment Repair", 1.99, 0.013, 0.047, 43277,
  "499", "Other Installation, Maintenance, and Repair Workers", 1.89, 0.014, 0.079, 49873,
  "511", "Supervisors, Production", 4.43, 0.006, 0.178, 62469,
  "512", "Assemblers and Fabricators", 1.94, 0.008, 0.061, 35538,
  "513", "Food Processing Workers", 1.33, 0.005, 0.073, 30278,
  "514", "Metal and Plastics Workers", 1.56, 0.011, 0.039, 43941,
  "515", "Printing Workers", 1.54, 0.001, 0.116, 37448,
  "516", "Textile Workers", 1.32, 0.003, 0.073, 24752,
  "517", "Woodworkers", 1.95, 0.001, 0.083, 31483,
  "518", "Plant and System Operators", 2.19, 0.002, 0.174, 67048,
  "519", "Other Production Occupations", 1.48, 0.021, 0.112, 40494,
  "531", "Supervisors, Transportation and Material Moving", 4.25, 0.002, 0.162, 52711,
  "532", "Air Transportation Workers", 4.83, 0.002, 0.625, 103952,
  "533", "Motor Vehicle Operators", 1.65, 0.031, 0.082, 38411,
  "534", "Rail Transportation Workers", 1.73, 0.001, 0.140, 73035,
  "535", "Water Transportation Workers", 2.68, 0.000, 0.176, 65857,
  "536", "Other Transportation Workers", 2.59, 0.002, 0.101, 35193,
  "537", "Material Moving Workers", 1.06, 0.037, 0.065, 29349
) %>%
  mutate(
    source_year = 2018L,
    source_table = case_when(
      row_number() <= 31 ~ "A1A",
      row_number() <= 64 ~ "A1B",
      TRUE ~ "A1C"
    ),
    source_url = source_url,
    source_note = paste0(
      "Deming (2021), decision intensity normalized to a 0-10 ",
      "employment-weighted percentile scale"
    )
  )

stopifnot(
  nrow(decision_intensity) == 93,
  n_distinct(decision_intensity$soc_3digit) == nrow(decision_intensity),
  all(decision_intensity$decision_intensity >= 0),
  all(decision_intensity$decision_intensity <= 10)
)

output_path <- path_raw(
  "geographic_crosswalks",
  "deming_decision_intensity_3digit_soc.csv"
)
write_csv(decision_intensity, output_path)

cat(
  "Wrote",
  nrow(decision_intensity),
  "three-digit SOC decision-intensity rows to",
  output_path,
  "\n"
)
