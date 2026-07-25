# Canonical geographic identifiers used by the supported pipeline.
#
# Identifiers are character strings so leading zeroes survive CSV, Parquet,
# joins, and round trips through Python. County identifiers use the project's
# 2010 geography throughout the analysis pipeline.

clean_geo_code <- function(x) {
  x <- trimws(as.character(x))
  x <- gsub('"', "", x, fixed = TRUE)
  x <- sub("\\.0+$", "", x)
  x[x == ""] <- NA_character_
  x
}

normalize_fixed_width_code <- function(x, width, label) {
  x <- clean_geo_code(x)
  invalid <- !is.na(x) & (!grepl("^[0-9]+$", x) | nchar(x) > width)

  if (any(invalid)) {
    examples <- paste(utils::head(unique(x[invalid]), 5L), collapse = ", ")
    stop(
      label,
      " must contain at most ",
      width,
      " digits; invalid values: ",
      examples,
      call. = FALSE
    )
  }

  padding <- pmax(width - nchar(x), 0L)
  padded <- paste0(
    vapply(padding, function(n) strrep("0", n), character(1)),
    x
  )
  padded[is.na(x)] <- NA_character_
  padded
}

normalize_unpadded_code <- function(x, label) {
  x <- clean_geo_code(x)
  invalid <- !is.na(x) & !grepl("^[0-9]+$", x)

  if (any(invalid)) {
    examples <- paste(utils::head(unique(x[invalid]), 5L), collapse = ", ")
    stop(label, " must contain digits only; invalid values: ", examples,
      call. = FALSE
    )
  }

  x <- sub("^0+(?=[0-9])", "", x, perl = TRUE)
  x
}

state_fips <- function(x) {
  normalize_fixed_width_code(x, 2L, "state_fips")
}

county_code <- function(x) {
  normalize_fixed_width_code(x, 3L, "county_code")
}

county_fips <- function(x) {
  normalize_fixed_width_code(x, 5L, "county_fips")
}

combine_county_fips <- function(state, county) {
  paste0(state_fips(state), county_code(county))
}

state_from_county_fips <- function(county) {
  substr(harmonize_county_fips_2010(county), 1L, 2L)
}

county_code_from_county_fips <- function(county) {
  substr(harmonize_county_fips_2010(county), 3L, 5L)
}

harmonize_county_fips_2010 <- function(x) {
  x <- county_fips(x)
  x[x == "46102"] <- "46113"
  x
}

cz_id <- function(x) {
  normalize_unpadded_code(x, "cz_id")
}

aewr_region_id <- function(x) {
  x <- normalize_unpadded_code(x, "aewr_region_id")
  invalid <- !is.na(x) & !x %in% as.character(seq_len(17L))

  if (any(invalid)) {
    examples <- paste(utils::head(unique(x[invalid]), 5L), collapse = ", ")
    stop(
      "aewr_region_id must be between 1 and 17; invalid values: ",
      examples,
      call. = FALSE
    )
  }

  x
}

oews_area_code <- function(x) {
  x <- clean_geo_code(x)
  invalid <- !is.na(x) & !grepl("^[0-9]+$", x)

  if (any(invalid)) {
    examples <- paste(utils::head(unique(x[invalid]), 5L), collapse = ", ")
    stop(
      "oews_area_code must contain digits only; invalid values: ",
      examples,
      call. = FALSE
    )
  }

  x
}

geo_code_is_valid <- function(x, name) {
  if (!is.character(x)) {
    return(rep(FALSE, length(x)))
  }

  switch(
    name,
    state_fips = is.na(x) | grepl("^[0-9]{2}$", x),
    county_code = is.na(x) | grepl("^[0-9]{3}$", x),
    county_fips = is.na(x) | (grepl("^[0-9]{5}$", x) & x != "46102"),
    neighbor_county_fips = is.na(x) |
      (grepl("^[0-9]{5}$", x) & x != "46102"),
    cz_id = is.na(x) | grepl("^(0|[1-9][0-9]*)$", x),
    aewr_region_id = is.na(x) | x %in% as.character(seq_len(17L)),
    oews_area_code = is.na(x) | grepl("^[0-9]+$", x),
    stop("Unknown geographic identifier: ", name, call. = FALSE)
  )
}

assert_geo_columns <- function(data, required, allow_na = character()) {
  missing_columns <- setdiff(required, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      "Missing required geographic columns: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  for (name in required) {
    value <- data[[name]]

    if (!is.character(value)) {
      stop(name, " must be a character vector.", call. = FALSE)
    }

    invalid <- !geo_code_is_valid(value, name)
    if (any(invalid)) {
      examples <- paste(
        utils::head(unique(value[invalid]), 5L),
        collapse = ", "
      )
      stop(name, " contains malformed values: ", examples, call. = FALSE)
    }

    if (!name %in% allow_na && anyNA(value)) {
      stop(name, " contains missing values.", call. = FALSE)
    }
  }

  invisible(data)
}
