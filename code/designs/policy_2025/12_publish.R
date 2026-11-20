source("code/designs/policy_2025/helpers.R")
out <- id_output_dir(); cd_verify()
validation <- jsonlite::read_json(file.path(out,"validation.json"))
stopifnot(validation$status=="passed")
destinations <- character()
for (name in list.files(out,pattern="\\.csv$")) destinations[name] <- path_tables("policy_2025",name)
for (name in c("wage_filer_state_occupation.parquet","wage_certified_participation.parquet","earnings_pre.parquet","earnings_interim.parquet")) destinations[name] <- path_processed("policy_2025",name)
for (name in c("inputs.json","earnings_pre_inputs.json","earnings_interim_inputs.json","validation.json")) destinations[name] <- path_outputs("reports","policy_2025",name)
for (name in c("policy_2025.md","policy_2025_specifications.md")) destinations[name] <- path_outputs("reports",name)
for (name in c("filer_cases.parquet","filer_ids.parquet")) destinations[file.path("filer",name)] <- path_int("policy_2025","filer",name)
records <- list()
for (name in names(destinations)) {
  source <- file.path(out,name); target <- destinations[[name]]
  if (!file.exists(source)) stop("Required artifact absent: ",name)
  dir.create(dirname(target),recursive=TRUE,showWarnings=FALSE)
  if (!file.copy(source,target,overwrite=TRUE)) stop("Could not publish: ",target)
  hash <- digest::digest(file=source,algo="sha256")
  stopifnot(digest::digest(file=target,algo="sha256")==hash)
  records[[name]] <- list(path=substring(target,nchar(path_root())+2L),sha256=hash,bytes=file.info(target)$size)
}
sources <- c(list.files(CD,recursive=TRUE,full.names=TRUE),path_root("scripts/run_policy_2025.sh"),
  path_root("scripts/pipeline_helpers.sh"),path_code("paths.R"),path_code("c00_shared/geography.R"),
  path_code("a01_sources/04_02_qcew_quarterly_employment.py"),
  path_root("src/h2a/paths.py"),path_root("src/h2a/geography.py"),path_root("src/h2a/employer_linkage.py"),
  path_root("pyproject.toml"),path_root("uv.lock"),path_root("renv.lock"))
sources <- sources[!grepl("/__marimo__/|/__pycache__/|/\\.",sources) & !file.info(sources)$isdir]
hashes <- setNames(lapply(sources,function(f)digest::digest(file=f,algo="sha256")),substring(sources,nchar(path_root())+2L))
jsonlite::write_json(list(design="policy_2025_v1",generated_at=as.character(Sys.time()),
  specification_changed=FALSE,outcomes_previously_inspected=TRUE,
  inputs=jsonlite::read_json(file.path(out,"inputs.json")),source_hashes=hashes,
  outputs=records,validation=validation),path_outputs("reports","policy_2025","provenance.json"),pretty=TRUE,auto_unbox=TRUE)
cat("Published",length(records),"verified policy_2025 artifacts\n")
