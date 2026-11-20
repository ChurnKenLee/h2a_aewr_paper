source("explorations/bite/cross_design_helpers.R")
out_dir <- bite_output_dir()
ex <- read_parquet(file.path(out_dir,"exposures.parquet"))
p <- panel_iv_prepare_outcomes(read_parquet(path_processed("panel_iv_county_year.parquet")),bite_spec)
bite_check_keys(p)
p <- inner_join(p,ex %>% select(county_fips,all_of(bite_variants)),by="county_fips",relationship="many-to-one")
shared <- panel_iv_prepare_outcomes(read_parquet(path_processed("county_year_panel.parquet")),bite_spec)
lead <- shared %>% distinct(aewr_region_id,year,aewr_ppi) %>% transmute(aewr_region_id,year=year-1L,A_lead=aewr_ppi)
p <- left_join(p,lead,by=c("aewr_region_id","year"),relationship="many-to-one")
rows <- wild <- first <- ar <- joint <- failures <- models <- list()
for(cs in c("bea","continuing")) {
  ctl <- panel_iv_control_set(bite_spec,cs); controls <- paste(ctl$control_terms,collapse=" + ")
  for(nm in names(bite_outcomes)) {
    y <- bite_outcomes[[nm]]
    for(v in bite_variants) {
      message(cs," / ",nm," / ",v)
      d <- p %>% filter(year<=ctl$policy_end_year) %>%
        mutate(A=aewr_ppi,Z=z_wage_seasonal_composition_real,E=.data[[v]],X=A*E,ZE=Z*E,L=A_lead*E)
      d <- bite_complete(d,c(y,"A","Z","z_wage_only_real","E",ctl$control_columns,"aewr_iv_cluster_id"))
      menu <- list(
        baseline_ols=list(rhs=paste("A +",controls),fe="county_fips + year",terms="A"),
        heterogeneity=list(rhs=paste("A + X +",controls),fe="county_fips + year",terms=c("A","X")),
        bite_1b=list(rhs="X",fe="county_fips + aewr_region_id^year",terms="X"),
        bite_1a=list(rhs="X",fe="county_fips + aewr_region_id^year + year[E]",terms="X"),
        lead_1b=list(rhs="X + L",fe="county_fips + aewr_region_id^year",terms=c("X","L")),
        lead_1a=list(rhs="X + L",fe="county_fips + aewr_region_id^year + year[E]",terms=c("X","L")),
        baseline_iv=list(rhs=controls,fe="county_fips + year",iv="A ~ Z",terms="fit_A"),
        interacted_iv=list(rhs=controls,fe="county_fips + year",iv="A + X ~ Z + ZE",terms=c("fit_A","fit_X")),
        within_iv=list(rhs="1",fe="county_fips + aewr_region_id^year",iv="X ~ ZE",terms="fit_X"))
      for(id in names(menu)) {
        if(v!="E_gap" && id %in% c("baseline_ols","baseline_iv")) next
        config<-menu[[id]]; dd<-if(startsWith(id,"lead")) bite_complete(d,"L") else d
        meta<-tibble(control_set=cs,outcome=nm,exposure=v,model=id)
        key<-paste(cs,nm,v,id,sep="__")
        result<-tryCatch({
          m<-bite_fit(y,config$rhs,config$fe,dd,config$iv)
          rr<-bite_rows(m,dd,config$terms,meta)
          rows[[key]]<-rr
          if(v=="E_gap" || (nm=="cert" && id %in% c("bite_1b","bite_1a","within_iv","interacted_iv"))) {
            wild[[key]]<-bind_rows(lapply(config$terms,function(term) {
              z<-tryCatch(bite_wild(m,dd,term),error=function(e)tibble(term=term,wild_p=NA_real_,wild_status=conditionMessage(e)))
              bind_cols(meta,z)
            }))
          }
          if(!is.null(config$iv)) {
            fs<-m$iv_first_stage
            for(cl in c("state_fips","aewr_region_id","aewr_iv_cluster_id")) for(end in names(fs)) {
              fsm<-summary(fs[[end]],vcov=as.formula(paste0("~",cl)))
              instruments<-if(id=="interacted_iv") c("Z","ZE") else if(id=="within_iv") "ZE" else "Z"
              ct<-coeftable(fsm); w<-wald(fsm,keep=paste0("^(",paste(instruments,collapse="|"),")$"),print=FALSE)
              first[[paste(key,cl,end)]]<-bind_cols(meta,tibble(cluster=cl,endogenous=end,
                term=instruments,estimate=ct[instruments,1],se=ct[instruments,2],joint_F=w$stat,
                partial_r2=1-fs[[end]]$ssr/fs[[end]]$ssr_no_inst))
            }
            endogenous<-if(id=="interacted_iv") c("A","X") else if(id=="within_iv") "X" else "A"
            instruments<-if(id=="interacted_iv") c("Z","ZE") else if(id=="within_iv") "ZE" else "Z"
            # Complete-case and singleton sample used by the fitted model.
            used<-dd[obs(m),,drop=FALSE]
            r<-bite_residualize(used,c(y,endogenous,instruments),config$rhs,config$fe)
            yy<-r[,1]; XX<-r[,endogenous,drop=FALSE]; ZZ<-r[,instruments,drop=FALSE]
            for(cl in c("state_fips","aewr_region_id","aewr_iv_cluster_id")) {
              clv<-used[[cl]]; G<-n_distinct(clv)
              rf<-solve(crossprod(ZZ),crossprod(ZZ,cbind(yy,XX)))
              residual<-cbind(yy,XX)-ZZ%*%rf
              if(length(endogenous)==1) {
                s<-rowsum(residual*drop(ZZ),clv)/sum(ZZ^2)
                V<-crossprod(s)*G/(G-1)
                ci<-iv_ar_confidence_set(rf[1,1],rf[1,2],V[1,1],V[2,2],V[1,2],qf(.95,1,G-1))
                ar[[paste(key,cl)]]<-bind_cols(meta,tibble(cluster=cl,ar_type=ci$type,ar_low=ci$lower,ar_high=ci$upper))
              } else {
                sw<-bite_sw(XX,ZZ,clv)
                kp<-bite_kp(XX,ZZ,clv)
                first[[paste(key,cl,"conditional")]]<-bind_cols(meta,tibble(cluster=cl,endogenous=endogenous,
                  term="SW_conditional",estimate=NA_real_,se=NA_real_,joint_F=sw,kp_F=kp))
                if(v=="E_gap" && cl=="aewr_region_id") {
                  # Grid inversion of the joint AR test. Preserve full accepted
                  # set and boundary flags; these are not Wald intervals.
                  zinv<-solve(crossprod(ZZ))
                  scores<-lapply(seq_len(3),function(k)rowsum(ZZ*residual[,k],clv)%*%zinv)
                  beta<-coef(m)[c("fit_A","fit_X")]
                  ses<-se(summary(m,vcov=~aewr_region_id))[c("fit_A","fit_X")]
                  radii<-pmax(abs(beta)+12*ses,if(nm=="hours") 1000 else if(nm=="cert") 1 else .1)
                  axis<-function(j) sort(unique(c(seq(-radii[j],radii[j],length.out=161),
                    seq(beta[j]-8*ses[j],beta[j]+8*ses[j],length.out=161),0,beta[j])))
                  grid<-expand.grid(beta_A=axis(1),beta_X=axis(2))
                  grid$p<-vapply(seq_len(nrow(grid)),function(i){
                    b<-as.numeric(grid[i,1:2]); pi<-rf[,1]-rf[,-1,drop=FALSE]%*%b
                    ss<-scores[[1]]-b[1]*scores[[2]]-b[2]*scores[[3]]
                    V<-crossprod(ss)*G/(G-1)
                    stat<-drop(t(pi)%*%solve(V,pi))/2
                    pf(stat,2,G-1,lower.tail=FALSE)
                  },numeric(1))
                  grid$accepted<-grid$p>=.05
                  grid$boundary<-abs(grid$beta_A)==radii[1] | abs(grid$beta_X)==radii[2]
                  joint[[key]]<-bind_cols(meta[rep(1,nrow(grid)),],grid)
                }
              }
            }
          }
          if(v=="E_gap" && nm=="cert" && id %in% c("bite_1b","bite_1a")) {
            models[[key]]<-list(data=dd %>% select(county_fips,aewr_region_id,year,A,E,X,all_of(y)),config=config)
          }
          TRUE
        },error=function(e){failures[[key]]<<-bind_cols(meta,tibble(error=conditionMessage(e)));FALSE})
      }
    }
  }
}
write_csv(bind_rows(rows),file.path(out_dir,"panel_estimates.csv"))
write_csv(bind_rows(wild),file.path(out_dir,"panel_wild.csv"))
write_csv(bind_rows(first),file.path(out_dir,"iv_first_stages.csv"))
write_csv(bind_rows(ar),file.path(out_dir,"iv_ar_sets.csv"))
write_csv(bind_rows(joint),file.path(out_dir,"iv_joint_ar_grid.csv"))
write_csv(bind_rows(failures),file.path(out_dir,"panel_failures.csv"))
saveRDS(models,file.path(out_dir,"headline_models.rds"))

# Events retain 2009 outcomes without requiring an observed statutory wage.
events<-pretests<-list()
ep<-shared %>% filter(year<=2024) %>% inner_join(ex %>% select(county_fips,all_of(bite_variants)),by="county_fips")
for(v in bite_variants) for(nm in names(bite_outcomes)) {
  dd<-ep %>% mutate(E=.data[[v]]) %>% bite_complete(c(bite_outcomes[[nm]],"E"))
  m<-bite_fit(bite_outcomes[[nm]],"i(year,E,ref=2011)","county_fips + aewr_region_id^year",dd)
  for(cl in c("state_fips","aewr_region_id")) {
    sm<-summary(m,vcov=as.formula(paste0("~",cl))); ct<-coeftable(sm)
    events[[paste(v,nm,cl)]]<-tibble(exposure=v,outcome=nm,cluster=cl,year=as.integer(sub("year::([0-9]+):E","\\1",rownames(ct))),
      estimate=ct[,1],se=ct[,2],p=ct[,4],n=nobs(m))
    w<-wald(sm,keep="year::200[89]:E|year::2010:E",print=FALSE)
    pretests[[paste(v,nm,cl)]]<-tibble(exposure=v,outcome=nm,cluster=cl,F=w$stat,p=w$p,df1=w$df1,df2=w$df2)
  }
}
write_csv(bind_rows(events),file.path(out_dir,"event_study.csv"))
write_csv(bind_rows(pretests),file.path(out_dir,"pretrend_tests.csv"))

# Within-region county reassignment is a placebo distribution, not a causal
# randomization test: no assignment mechanism for bite has been established.
perms<-list(); set.seed(20261102)
for(cs in c("bea","continuing")) for(id in c("bite_1b","bite_1a")) {
  key<-paste(cs,"cert","E_gap",id,sep="__"); obj<-models[[key]]; d<-obj$data
  counties<-distinct(d,county_fips,aewr_region_id,E)
  for(b in 1:500) {
    ee<-counties %>% group_by(aewr_region_id) %>% mutate(E=sample(E)) %>% ungroup()
    d$E<-ee$E[match(d$county_fips,ee$county_fips)]; d$X<-d$A*d$E
    m<-bite_fit(bite_outcomes[["cert"]],obj$config$rhs,obj$config$fe,d)
    perms[[length(perms)+1]]<-tibble(control_set=cs,model=id,draw=b,estimate=coef(m)[["X"]])
  }
}
write_csv(bind_rows(perms),file.path(out_dir,"permutation_placebos.csv"))
