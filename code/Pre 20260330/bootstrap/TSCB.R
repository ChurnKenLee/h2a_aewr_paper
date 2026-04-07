## TSCB
## S/ource: https://urldefense.com/v3/__https://academic.oup.com/qje/article/138/1/1/6750017__;!!Mih3wA!EEhCcUDhQu1NJQfNumoJl3gCetRc1FAUQYdPClEakAbZnSJS83zHCuPCzCgSOZ2UPUFyicodyOyvt3H9Rw$
## 1/18/2024

date <- gsub("-", "", Sys.Date())

## packaged ------
library(here)
library(fixest)

## Parameters -------

popsize <- 1502
samplesize <- 1502 # must be less than pop size, multiple of 2

clusterpop <- 32 # must be less than pop size
clustersamp <- 32 # must be less than cluster pop

pairs <- samplesize / 2

bootreps <- 999

years <- 3

true_beta <- 2 # beta

alpha <- 0 # intercept

sim_reps <- 1

simresults <- NULL # store data here

for (s in 1:sim_reps) {
  ## make dataframe of paired sample -------
  
  pair_id <- seq(1, pairs)
  
  border_side <- rep(seq(1, 2), pairs)
  
  allpairs <- c(pair_id, pair_id)
  
  allpairs <- sort(allpairs)
  
  clust_list <- seq(1, clustersamp)
  
  clust_seq <- sort(sample(clust_list, size = pairs, replace = T))
  hist(clust_seq)
  
  clust <- sort(c(clust_seq, clust_seq))
  
  clustdf <- data.frame(allpairs, border_side, clust)
  dim(clustdf)
  
  # add in years
  
  if (years > 1) {
    # only run if we have multiple years
    
    clustdfyear <- rbind(clustdf, clustdf) # add year 2
    
    # years var
    
    yearsvec <- sort(rep(seq(1, years), samplesize))
    
    if (years > 2) {
      # only run if we need to add more than 2 years
      
      for (t in 3:years) {
        # add additional years, one at a
        clustdfyear <- rbind(clustdfyear, clustdf)
        #print(t)
      }
      
    }
    
    # bind
    clustdfyear$year <- yearsvec
    
  }
  hist(clustdfyear$year)
  
  # treatment
  
  clustdfyear$treat[clustdfyear$border_side == 1] <- 1
  clustdfyear$treat[clustdfyear$border_side == 2] <- 0
  
  # fixed effecs
  
  # unit
  
  units <- rep(seq(1, samplesize), years)
  hist(units)
  
  clustdfyear$unit <- units
  
  # time -- already in
  
  # pair-time
  
  # dataframe to merge on pair-year
  
  pairs_vec <- rep(allpairs, years)
  # yearsvec already exists
  
  pairfecount <- (years - 1) * pairs
  
  pv <- seq(1, pairfecount)
  
  pvc <- sort(c(pv, pv))
  
  pairyearvec <- c(rep(0, samplesize), pvc)
  
  pairfe_df <- data.frame(pairs_vec, yearsvec, pairyearvec)
  
  names(pairfe_df) <- c("allpairs", "years", "pairtime_fe")
  
  clustdfyear$pairyear_fe <- pairyearvec
  
  ## DGP -----------
  
  # let's make some (fake) data
  
  # unit FE
  
  unit_fe_vals <- rnorm(n = samplesize, mean = 0, sd = 1)
  hist(unit_fe_vals)
  
  unit_fe_vals_vec <- rep(unit_fe_vals, years)
  
  clustdfyear$unitfeval <- unit_fe_vals_vec
  
  # pair-time fe
  
  pairtime_fe_val <- rep(rnorm(n = pairs * (years - 1), mean = 0, 1), 2)
  
  pairtime_fe_val_order <- rep(seq(1, (years - 1) * pairs), 2)
  
  pairtime_val_df <-
    data.frame(pairtime_fe_val, pairtime_fe_val_order)
  
  zeros_vec <- rep(0, samplesize)
  
  zeros_df <-
    data.frame(pairtime_fe_val = zeros_vec,
               pairtime_fe_val_order = zeros_vec)
  
  pairtime_val_df <-
    pairtime_val_df[order(pairtime_val_df$pairtime_fe_val_order), ]
  
  pairtime_val_df <- rbind(zeros_df, pairtime_val_df)
  
  clustdfyear$pairyear_fe_val <- pairtime_val_df$pairtime_fe_val
  
  pairtime_fe_val <- c(zeros_vec, pairtime_fe_val)
  
  # cluster treatment assignment
  
  cluster_list <- unique(clustdfyear$clust)
  length(cluster_list)
  
  clust_treat_df <- NULL
  
  for (i in 1:years) {
    treat_temp <- rnorm(length(cluster_list), mean = 4, sd = 4)
    temp_time <- rep(i, length(cluster_list))
    
    temp_treat <-
      data.frame(year = temp_time,
                 clust = cluster_list,
                 clust_treat = treat_temp)
    
    clust_treat_df <- rbind(clust_treat_df, temp_treat)
    
    rm(temp_treat)
    
  }
  
  hist(clust_treat_df$clust_treat)
  
  clustdfyear <-
    merge(x = clustdfyear,
          y = clust_treat_df,
          by = c("clust", "year"))
  
  # Error
  
  error <- rnorm(dim(clustdfyear)[1], 0, 1)
  
  clustdfyear$error <- error
  
  # Calculate Y using True Beta
  
  # Y = a + b(treat) + C + PT + E
  
  clustdfyear$outcome <-
    alpha + (clustdfyear$clust_treat * clustdfyear$treat * true_beta) + clustdfyear$unitfeval + clustdfyear$pairyear_fe_val + clustdfyear$error
  
  clustdfyear$treatcnts <- clustdfyear$clust_treat * clustdfyear$treat
  
  ## Basic Model ----------------------------------------------------------------------------------
  
  # basic_model <- lm(outcome ~ treatcnts + factor(unit) + factor(pairyear_fe), data = clustdfyear)
  # #summary(basic_model)
  # basic_model$coefficients[2] # est of beta
  
  # try with fixest
  fixest_model  <-
    feols(
      outcome ~ treatcnts |
        factor(unit) + factor(pairyear_fe),
      cluster = clustdfyear$clust,
      data = clustdfyear
    )
  fixest_model$coeftable
  
  fe_estimate_clustfe_est <- fixest_model$coeftable[1, 1]
  fe_estimate_clustfe_se <- fixest_model$coeftable[1, 2]
  
  # way faster
  
  ## Two-Staged Cluster Boot -------------------
  
  # TSCB from Abadie, Athey, Imbens, Woodlridge, 2023, QJE.
  
  ## stage 1 (paired sample makes this trivial)
  
  # stage 1a, sample clusters with replacement 1/qk times, here, we have qk = 1.
  
  # stage 1b, calculate the assignment prob, W = 1/2 b/c sample is paired
  
  # stage 1c, sample clusters (w/repl) from cluster pop w prob qk. This is trivially satisfied b/c qk = 1
  
  # stage 1d, draw assignment prob from 1b, trivially satisfied since w = 1/2 always.
  
  ## Stage 2
  
  # 2a, sample with replacement Nkm*Akm treated units from cluster m
  
  # 2b, sample with replacement Nkm*(1-Akm) control units from cluster m
  
  # 2mod, sample with replacement Nkm*Akm pairs from cluster m
  
  ## clean up a bit --------------------------------------------------
  
  # dfs
  rm(clust_treat_df,
     clustdf,
     pairfe_df,
     pairtime_val_df,
     zeros_df)
  
  ## Bootstrap -------------------------------------------------------
  
  # clustdfyear, this is our base dataframe
  # bootreps, this is our number of reps
  
  input_data <- clustdfyear
  
  # rename
  names(input_data)
  names(input_data)[1] <- "cluster"
  names(input_data)[2] <- "time"
  names(input_data)[3] <- "pair"
  names(input_data)[4] <- "side"
  names(input_data)
  
  periods <- unique(input_data$time) # grab the period variable
  
  boot_treat_ests <- NULL # this is our storage vector
  
  # from main DF, get clusters and number of pairs per cluster
  
  cluster_list <- unique(input_data$cluster)
  
  for (b in 1:bootreps) {
    # main boot loop starts
    
    
    # for each cluster, sample pairs with replacement
    
    boot_clust_samp <- NULL # store the clustered sample here
    
    cnt <- 1 # counter for boot sample id
    
    for (i in 1:length(cluster_list)) {
      # get the unique list of pairs
      
      temp_clust_df <-
        subset(input_data, cluster == cluster_list[i]) # iterate
      
      temp_pair_vec <- unique(temp_clust_df$pair)
      
      temp_clustpairsamp <-
        sample(temp_pair_vec,
               size = length(temp_pair_vec),
               replace = TRUE)
      
      temp_bootclustid <- seq(cnt, cnt + (length(temp_pair_vec) - 1))
      
      temp_clust_df <-
        data.frame(pair = temp_clustpairsamp,
                   cluster = cluster_list[i],
                   boot_id = temp_bootclustid) # iterate
      
      boot_clust_samp <- rbind(boot_clust_samp, temp_clust_df)
      
      rm(temp_clust_df, temp_pair_vec, temp_clustpairsamp)
      
      cnt <- max(temp_bootclustid) + 1
      
      rm(temp_bootclustid)
      
      # print(i/length(cluster_list))
    }
    
    # use to make dataset for this bootstrap iteration
    
    # match years
    
    temp <- boot_clust_samp
    
    boot_clust_samp <- NULL
    
    for (i in 1:length(periods)) {
      temp$time <- periods[i]
      boot_clust_samp <- rbind(boot_clust_samp, temp)
      temp <-
        subset(temp, select = -c(time)) # remove the column we added
      # print(i)
    }
    
    dim(boot_clust_samp)
    
    # double and add treat / control indicator (or border side indicator, both would work)
    
    # match w/ 1 and 2 designating the side
    
    order_vec <- sort(rep(seq(1, 2), dim(boot_clust_samp)[1]))
    
    boot_samp_int <- rbind(boot_clust_samp, boot_clust_samp)
    
    boot_samp_int$side <- order_vec
    
    rm(boot_clust_samp)
    
    # merge with data
    
    dim(boot_samp_int)
    dim(input_data)
    
    # make bootpair id (handles replacements) #
    
    boot_samp_final <-
      merge(
        x = boot_samp_int,
        y = input_data,
        by = c("pair", "cluster", "time", "side"),
        all.x = T,
        all.y = F
      )
    dim(boot_samp_final)
    
    rm(boot_samp_int)
    
    # fix the FEs #
    
    # units will be given by boot id and side
    
    boot_samp_final$boot_unit_fe <-
      with(boot_samp_final, interaction(as.factor(boot_id),  side))
    
    # then pairs are marked by boot id, and we will use time for the year dimension
    
    boot_samp_final$boot_pairtime_fe <-
      with(boot_samp_final, interaction(as.factor(boot_id),  time))
    
    levels(boot_samp_final$boot_pairtime_fe) <-
      c(levels(boot_samp_final$boot_pairtime_fe), "0.0")
    
    boot_samp_final$boot_pairtime_fe[boot_samp_final$time == 1] <-
      "0.0" # first period is 0
    
    # run the model
    
    fixest_model  <-
      feols(outcome ~ treatcnts |
              factor(boot_unit_fe) + factor(boot_pairtime_fe),
            data = boot_samp_final)
    
    boot_est <- fixest_model$coeftable[1, 1]
    
    boot_treat_ests <- c(boot_treat_ests, boot_est)
    
    rm(boot_samp_final)
    
    # print("### Boot Percent Complete ###")
    # print(b / bootreps)
    
  } # end boot loop
  
  temp_mean <- mean(boot_treat_ests)
  temp_med <- median(boot_treat_ests)
  temp_sd <- sd(boot_treat_ests)
  temp_p95 <- quantile(boot_treat_ests, 0.975)
  temp_p05 <- quantile(boot_treat_ests, 0.025)
  
  boot_data <-
    data.frame(
      rep = s,
      boot_mean = temp_mean,
      boot_median = temp_med,
      boot_sd = temp_sd,
      boot_p05 = temp_p05,
      boot_p95 = temp_p95,
      fe_est = fe_estimate_clustfe_est,
      fe_se = fe_estimate_clustfe_se
    )
  
  simresults <- rbind(simresults, boot_data)
  
  print("### Sim Percent Complete ###")
  print(s / sim_reps)
  
  rm(boot_data, clustdfyear, fixest_model, input_data)
  
} # end sim loop

write.csv(simresults, file = here("Code", "bootstrap", paste0(folder, "tscb_sim_results_", date, ".csv")))

## Graph Results ##

library(tidyverse)

simresults <-
  read.csv(file = paste0(folder, "tscb_sim_results_20240129.csv"))

simresults <- simresults %>%
  mutate(confint = boot_p95 - boot_p05) # 90% interval

bootcomp <- ggplot(data = simresults) +
  geom_density(aes(x = boot_mean, color = "Boot mean")) +
  geom_density(aes(x = boot_median, color = "Boot median")) +
  geom_density(aes(x = fe_est, color = "Fixed Effects Estimator")) +
  scale_colour_manual(values = c("#003049", "#6e97b1", "#c1121f"),
                      name = "") +
  xlab("Estimate") +
  theme_classic()
bootcomp

bootcomp_error <- ggplot(data = simresults) +
  geom_density(aes(x = boot_sd, color = "Boot sd")) +
  geom_density(aes(x = fe_se, color = "Fixed Effects Estimator SE")) +
  scale_colour_manual(values = c("#003049", "#c1121f"), name = "") +
  xlab("Error") +
  theme_classic()
bootcomp_error
