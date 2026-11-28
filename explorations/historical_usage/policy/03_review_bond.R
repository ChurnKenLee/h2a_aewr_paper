# Exploratory composition checks for the exact-one-below size contrast.
# Run from root via Rscript --vanilla after configure_r_library.
source("code/paths.R")
library(data.table)
setDTthreads(2L)
out <- commandArgs(trailingOnly = TRUE)[1L]
if (is.na(out)) out <- "/tmp/h2a-history-bond-review"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
stopifnot(startsWith(normalizePath(out), "/tmp/"))

read_columns <- function(name, cols) {
  as.data.table(arrow::read_parquet(path_int(name), col_select = all_of(cols)))
}
# Match bond_thresholds.R case deduplication and windows.
raw <- read_columns("h2a_disclosure_records.parquet", c("fiscal_year","case_number","case_received_date",
                    "decision_date","type_of_employer_application","h2a_labor_contractor",
                    "nbr_workers_requested","worksite_state"))[fiscal_year %between% c(2020L,2025L)]
raw[, received := suppressWarnings(as.IDate(substr(case_received_date,1,10)))]
raw[, decision := suppressWarnings(as.IDate(substr(decision_date,1,10)))]
raw[, workers := suppressWarnings(as.numeric(nbr_workers_requested))]
raw <- unique(raw[,.(fiscal_year,case_number,received,decision,type_of_employer_application,
                     h2a_labor_contractor,workers,worksite_state)])
raw <- raw[nzchar(case_number) & !duplicated(case_number) & !duplicated(case_number, fromLast=TRUE)]
raw[, era := fcase(received >= "2020-04-01" & received < "2022-04-01","early",
                   received >= "2023-04-01" & received < "2025-04-01","late")]
x <- raw[!is.na(era) & as.integer(decision-received) %between% c(0L,180L) &
         type_of_employer_application=="Individual Employer" & h2a_labor_contractor %chin% c("Y","N") &
         workers > 0 & workers == floor(workers)]
ids <- read_columns("policy_2025/filer/filer_ids.parquet",c("fiscal_year","case_number","filer_R7"))
agents <- read_columns("policy_2025/filer/filer_cases.parquet",c("fiscal_year","case_number","agent_key","law_firm_key"))
keys <- c("fiscal_year", "case_number")
stopifnot(!anyDuplicated(ids,by=keys), !anyDuplicated(agents,by=keys))
x <- merge(x,ids,by=keys,all.x=TRUE)
x <- merge(x,agents,by=keys,all.x=TRUE)
stopifnot(!anyNA(x$filer_R7),!anyDuplicated(x$case_number))
x[, contractor := h2a_labor_contractor == "Y"]
x[, k := 50L*floor((workers+5L)/50L)]
x[, exact := as.integer(workers==k-1L)]
x[, stable_all := uniqueN(era)==2L & uniqueN(contractor)==1L, by=filer_R7]
local <- x[k %in% c(150L,200L,250L,300L) & workers>=k-5L & workers<k+5L]
target <- local[,.(n=.N, at_minus_one=sum(exact)),by=.(era,contractor)][order(era,contractor)]
stopifnot(identical(target$n,c(68L,126L,80L,213L)), identical(target$at_minus_one,c(3L,7L,8L,91L)))
local[, recurring_local := uniqueN(era)==2L,by=filer_R7]
# Independent readers can reproduce all normalized agent keys from the filer
# producer. Do not export agent, firm or filer names.
did <- function(y) {
  g <- y[,.(p=mean(exact)),by=.(contractor,era)]
  if (nrow(g)!=4L) return(NA_real_)
  sum(g$p * ifelse(g$contractor,1,-1) * ifelse(g$era=="late",1,-1))
}
cells <- list()
contrasts <- list()
record_sample <- function(y,label) {
  s <- y[,.(cases=.N, filers=uniqueN(filer_R7), hits=sum(exact), share=mean(exact)),by=.(era,contractor)]
  list(cells=s[,sample:=label], result=data.table(sample=label,cases=nrow(y),
          filers=uniqueN(y$filer_R7),did=did(y)))
}
samples <- list(full=local, recurrent_fixed_role_any_size=local[stable_all==TRUE],
                recurrent_fixed_role_in_local_bands=local[stable_all==TRUE & recurring_local==TRUE])
for (label in names(samples)) {
  result <- record_sample(samples[[label]],label)
  cells[[label]] <- result$cells
  contrasts[[label]] <- result$result
}
# Equal-filer within-organization changes, requiring both eras and the same role
# across all observed eligible applications. A local share is undefined if
# the filer has no eligible near-threshold application in a given era.
rec <- samples[["recurrent_fixed_role_in_local_bands"]]
filer <- dcast(rec[,.(share=mean(exact)),by=.(contractor,filer_R7,era)],
               contractor+filer_R7~era,value.var="share")
stopifnot(!anyNA(filer))
g <- filer[,.(filers=.N,early=mean(early),late=mean(late),change=mean(late-early),
              variance=var(late-early)/.N),by=contractor]
se <- sqrt(sum(g$variance))
df <- sum(g$variance)^2 / sum(g$variance^2/(g$filers-1))
eff <- g[contractor==TRUE,change]-g[contractor==FALSE,change]
fwrite(g,file.path(out,"equal_filer_cells.csv"))
fwrite(data.table(did=eff,lo=eff-qt(.975,df)*se,hi=eff+qt(.975,df)*se,df=df),
       file.path(out,"equal_filer_did.csv"))
coverage <- list()
common_did <- list()
exclusions <- list()
for (key in c("agent_key","law_firm_key")) {
  y <- copy(local)
  y[, representative := trimws(get(key))]
  y[, known := !is.na(representative) & nzchar(representative)]
  coverage[[key]] <- y[,.(cases=.N,known=sum(known),known_hits=sum(exact[known]),all_hits=sum(exact),
                          distinct_known=uniqueN(representative[known])),by=.(era,contractor)][,representative_type:=key]
  # Matched four-cell representatives identify within-representative changes.
  # A representative's contractor and comparison clients may differ across eras.
  u <- y[known==TRUE]
  u[, common_all_four := uniqueN(paste(era,contractor))==4L,by=representative]
  a <- u[common_all_four==TRUE]
  for (label in c("known","unknown","common_all_four")) {
    group <- switch(label, known=u, unknown=y[known==FALSE], common_all_four=a)
    entry <- record_sample(group,paste(key,label,sep="_"))
    cells[[paste(key,label)]] <- entry$cells
    contrasts[[paste(key,label)]] <- entry$result
  }
  rep_effect <- a[,.(did=did(.SD),cases=.N),by=representative]
  common_did[[key]] <- data.table(representative_type=key,agents=nrow(rep_effect),cases=nrow(a),
                                mean_did=mean(rep_effect$did),median_did=median(rep_effect$did))
  # Rank using early-window local volume among all employers (predetermined)
  # and late contractor exact hits (retrospective influence only). Ties follow
  # alphabetical order of normalized representative key.
  for (basis in c("early_all_local_cases","late_exact_contractor_cases")) {
    rank <- if (basis=="early_all_local_cases") u[era=="early",.(weight=.N),by=representative] else
      u[era=="late" & contractor==TRUE & exact==1L,.(weight=.N),by=representative]
    setorder(rank,-weight,representative)
    for (top in c(1L,3L,5L)) {
      removed <- head(rank$representative,top)
      retained <- y[!(representative %in% removed)] # preserve cases with unknown representative
      l <- y[era=="late" & contractor==TRUE & representative %in% removed]
      exclusions[[length(exclusions)+1L]] <- data.table(representative_type=key,basis=basis,top=top,
          cases_remaining=nrow(retained),filers_remaining=uniqueN(retained$filer_R7),
          removed_late_contractor_cases=nrow(l),removed_late_contractor_hits=sum(l$exact),
          late_contractor_hits=sum(y[era=="late" & contractor==TRUE,exact]),did=did(retained),
          known_cases_remaining=retained[known==TRUE,.N],known_only_did=did(retained[known==TRUE]))
    }
  }
}
fwrite(rbindlist(cells),file.path(out,"cell_shares.csv"))
fwrite(rbindlist(contrasts),file.path(out,"sample_contrasts.csv"))
fwrite(rbindlist(coverage),file.path(out,"representative_coverage.csv"))
fwrite(rbindlist(common_did),file.path(out,"equal_representative_change.csv"))
fwrite(rbindlist(exclusions),file.path(out,"representative_exclusions.csv"))
