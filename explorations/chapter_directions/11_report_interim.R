source("explorations/chapter_directions/interim_helpers.R")
out<-id_output_dir();cd_verify(file.path(CD,"interim_amendment.sha256"))
r<-read_csv(file.path(out,"earnings_interim_estimates.csv"),show_col_types=FALSE)
table<-function(d) {
  d<-as.data.frame(d)
  for(n in names(d)) {
    if(is.numeric(d[[n]]))d[[n]]<-ifelse(is.na(d[[n]]),"—",formatC(d[[n]],digits=4,format="fg"))
    d[[n]]<-gsub("|","\\|",replace_na(as.character(d[[n]]),"—"),fixed=TRUE)
  }
  if(!nrow(d))return("None.")
  paste(c(paste0("| ",paste(names(d),collapse=" | ")," |"),paste0("| ",paste(rep("---",ncol(d)),collapse=" | ")," |"),apply(d,1,function(x)paste0("| ",paste(x,collapse=" | ")," |"))),collapse="\n")
}
lines<-character();put<-function(...)lines<<-c(lines,paste0(...),"")
put("# Interim agricultural earnings: authorized partial-window analysis")
put("Generated ",as.character(Sys.time()),". This dated addendum supersedes the earlier report's deferred-earnings status for the available quarters. The user authorized partial-window estimates before these outcomes were opened. The original three-quarter target remains a later update; it is no longer wholly uninspected.")
put("## Files and available window")
put("No upload or new download was needed. The existing `data/raw/qcew_episode_2025_2026/2025_qtrly_singlefile.zip` supplies 2025Q1/Q2 for the prior-window lead and 2025Q4 for the first post-policy quarter. The adjacent `2026_qtrly_singlefile.zip` supplies 2026Q1. Prior-year comparisons use the already available 2023/2024 history. Both new archive hashes match their acquisition manifest; retrieved 2026-11-14 UTC, source Last-Modified 2026-08-21.")
put("The primary interim outcome is the mean of YoY log average-weekly-earnings changes in **2025Q4 and 2026Q1**, requiring all four current/prior observations for every county. It retains private crop production (NAICS 111), positive disclosed earnings, employment at least ten, the frozen H-2A exposure and statutory dose, intensity control, state effects, equal county weights and state deletion jackknife t intervals. This changes the time window and potentially the sample from the original three-quarter target.")
put("## Main results")
put("A positive coefficient means earnings fell more where H-2A exposure and the statutory cut were greater. The percent translation uses exposure S=0.30 and a −$4 statutory cut: 100 × [exp(−1.2β)−1]. All intervals are 95% state jackknife intervals. These are covered-job earnings, which combine hourly pay, hours and workforce composition.")
main<-r %>% filter(specification=="crop_primary",tau>=2025) %>% select(window,n,states,estimate,low,high,percent_at_S03_cut_minus4,percent_low,percent_high)
put(table(main))
p<-r %>% filter(window=="interim_2025Q4_2026Q1",specification=="crop_primary")
put("At the stated exposure and cut, the two-quarter estimate is ",sprintf("%.2f%%",p$percent_at_S03_cut_minus4),", with a 95% interval of [",sprintf("%.2f%%",p$percent_low),", ",sprintf("%.2f%%",p$percent_high),"]. ",if(p$low<=0 && p$high>=0)"The interval includes zero; it does not establish no economically meaningful response." else "The interval excludes zero in this specification; the identifying assumptions still require scrutiny.")
put("Quarter-specific rows use each quarter's eligible counties. The rows ending `common_sample` use exactly the two-quarter counties, so their difference is not driven by changes in county eligibility. The prior-window lead measures 2024Q4–2025Q2, before the reform; it is a falsification diagnostic, not a reform effect.")
put("## Sensitivities and comparison industries")
put(table(r %>% filter(window=="interim_2025Q4_2026Q1") %>% select(specification,weights,status,n,states,estimate,low,high,percent_at_S03_cut_minus4,percent_low,percent_high)))
weighted<-filter(r,window=="interim_2025Q4_2026Q1",specification=="crop_employment_weighted")
put("The employment-weighted estimate is ",sprintf("%.2f%%",weighted$percent_at_S03_cut_minus4)," [",sprintf("%.2f%%",weighted$percent_low),", ",sprintf("%.2f%%",weighted$percent_high),"]. The equal-county primary therefore does not establish a precise employment-weighted response. The separate-quarter point estimates concentrate the decline in Q4; their broad intervals do not establish a statistically distinct timing pattern. Cash-dose and commuting-zone intervals also include zero. This is suggestive interim evidence, sensitive to specification and weighting, with continuing state-level confounding concerns.")
put("The UI classification remains frozen and incomplete; sparse covered/excluded groups cannot establish a general non-H-2A earnings effect. Animal production and all-private earnings are diagnostics, not guaranteed unaffected controls. Employment weights and commuting-zone exposure change the comparison and are reported alongside the equal-county primary.")
put("## Matched pre-policy windows")
put(table(r %>% filter(window=="matched_two_quarter_placebo") %>% select(tau,n,states,estimate,low,high)))
put("These placebo windows use the same Q4/Q1 timing, thresholds and primary model. Quiet placebos do not establish exogeneity. The post-policy and quarter-specific estimates are multiple looks, and subsequent full-window results will be updates rather than independent confirmations.")
put("## Complete declared interim result ledger")
put(table(r %>% select(window,tau,specification,weights,status,n,states,estimate,se_jackknife,low,high,p,se_crv1)))
put("## Data coverage and outstanding series")
put(table(read_csv(file.path(out,"earnings_interim_coverage.csv"),show_col_types=FALSE)))
queue<-read_csv(file.path(out,"earnings_interim_queue.csv"),show_col_types=FALSE)
put(table(queue))
put("The full 2025Q4–2026Q2 target still needs 2026Q2. No missing quarter is imputed. The complete pre-policy lead has now been estimated from the cached annual archive, removing the earlier separate-slice retrieval obstacle.")
put("## Reproduction")
put("```sh\nbash explorations/chapter_directions/run_interim.sh ",out,"\n```")
put("See the [dated amendment](../explorations/chapter_directions/interim_amendment_2026_11_14.json) and [README](../explorations/chapter_directions/README.md). Scratch contains `earnings_interim_estimates.csv`, every state deletion, coverage, input hashes and quarterly inventory. Original frozen registries and their code are preserved. No acquisition pipeline or automatic schedule was run.")
if(file.exists(file.path(out,"interim_validation.csv")))put(table(read_csv(file.path(out,"interim_validation.csv"),show_col_types=FALSE)))
writeLines(lines,path_root("markdowns/chapter_directions_earnings_interim.md"))
sources<-file.path(CD,c("09_extract_interim.py","10_estimate_interim.R","11_report_interim.R","interim_helpers.R","run_interim.sh","interim_amendment.sha256"))
jsonlite::write_json(list(amendment_sha256=digest::digest(file=file.path(CD,"interim_amendment_2026_11_14.json"),algo="sha256"),
  estimated_at=as.character(Sys.time()),post_policy_earnings_accessed=TRUE,coefficient_rows=nrow(r),inputs=jsonlite::read_json(file.path(out,"earnings_interim_inputs.json")),
  source_hashes=setNames(lapply(sources,function(f)digest::digest(file=f,algo="sha256")),sources),
  pre_policy_extract_sha256=digest::digest(file=file.path(out,"earnings_pre.parquet"),algo="sha256"),
  interim_extract_sha256=digest::digest(file=file.path(out,"earnings_interim.parquet"),algo="sha256"),
  estimates_sha256=digest::digest(file=file.path(out,"earnings_interim_estimates.csv"),algo="sha256"),
  report_sha256=digest::digest(file=path_root("markdowns/chapter_directions_earnings_interim.md"),algo="sha256")),file.path(out,"earnings_interim_provenance.json"),pretty=TRUE,auto_unbox=TRUE)
cat("Generated interim earnings report\n")
