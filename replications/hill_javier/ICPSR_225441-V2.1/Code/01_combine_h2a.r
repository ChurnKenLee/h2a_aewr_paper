


# Set the directory path
directory <- here(your_path_before_Repo,"Raw Data", "h2a_program_data")

# Get the list of xlsx files in the directory
files <- list.files(directory, pattern = "\\.xlsx$", full.names = TRUE) 

files = files[!grepl("crosswalk", files)]
files = files[!grepl("~", files)]

#read in crosswalk 

h2a_crosswalk = read_excel(here(your_path_before_Repo,"Raw Data",  "h2a_var_crosswalk.xlsx"))

# Combine Function


combiner <- function(year) {
print(year)

files_to_load = files[grepl(year, files)]
  # Read the file
  data <- read_excel(files_to_load)
  
  # pull variables of interest for that year
h2a_var_names = h2a_crosswalk %>% select(contains(year)) 

    # Rename the variables
names(h2a_var_names) = c("flag", "var",'fixed_name')
# Select down to only the variables of interest
h2a_var_names = h2a_var_names %>% filter(flag == 1) %>% select(-flag)

  # Use variables of interest string list to select the columns from the h2a data
  data <- data %>% select_(.dots = h2a_var_names$var)
  # Harmonize names of the variables
  names(data) = h2a_var_names$fixed_name
  # create a file year variable for record keeping
  data = data %>% mutate(file_year = year)
  
  # Return the data
  return(data)
}



# Create a list of years  as strings to loop through
dates = c(2006:2024) %>% lapply(as.character)


# Read and combine all the xlsx files, This returns a list of dataframes
combined_data <- lapply(dates, combiner)

# Combine the list of dataframes into one dataframe
h2a_combined = rbindlist(combined_data, use.names = TRUE, fill = TRUE)


# save the combined h2a data
fwrite(h2a_combined, here(your_path_before_Repo,"Output", "h2a_combined.csv"))


warning("h2a_combined.csv has been saved to the Output folder")

