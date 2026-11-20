# ===========================================================================
#   GOAL: Read in the FSE and visualize
# ===========================================================================

# create output dir
dir.create("plots/Figure6")

#  Read in the FSE list ----
list.est <- readRDS("in/fse/1_list_est.rds")

round_rule <- function(var, x) {
  if (var %in% c("AFL","AFR","AFZ")) return(round(x, 0))   # whole numbers
  if (var %in% c("ESLZ","ELAB")) return(round(x, 2))       # two decimals
  return(x)
}

# define groups of combos
combo_groups <- list(
  NAWS_C = c("NAWS_LCL", "NAWS_MCL", "NAWS_HCL",
             "NAWS_LCM", "NAWS_MCM", "NAWS_HCM",
             "NAWS_LCH", "NAWS_MCH", "NAWS_HCH"),
  NAWS_N = c("NAWS_LNL", "NAWS_MNL", "NAWS_HNL",
             "NAWS_LNM", "NAWS_MNM", "NAWS_HNM",
             "NAWS_LNH", "NAWS_MNH", "NAWS_HNH"),
  FLS_C  = c("FLS_LCL",  "FLS_MCL",  "FLS_HCL",
             "FLS_LCM",  "FLS_MCM",  "FLS_HCM",
             "FLS_LCH",  "FLS_MCH",  "FLS_HCH"),
  FLS_N  = c("FLS_LNL",  "FLS_MNL",  "FLS_HNL",
             "FLS_LNM",  "FLS_MNM",  "FLS_HNM",
             "FLS_LNH",  "FLS_MNH",  "FLS_HNH"))

# ---- Scaling parameters
scaling_factor <- 200 / max.ELAB

#  Plot & Process the FSE & CI for each combo ----
list.median <- vector("list", num.pr) # to save the median estimates

for (a in do.pr) {
  dir.create(paste0("temp/PR", sprintf("%02d", a)))
  name <- PR$pr[PR$PR9 == a]
  message(paste0("Generate file for saving with estimates, for PR", sprintf("%02d", a)))
  
  # Identify existing combos for this PR
  valid_indices <- which(!sapply(list.est[[a]], is.null))
  
  final_data <- do.call(rbind, lapply(valid_indices, function(c) {
    combo_elem <- list.est[[a]][[c]]
    combo_name <- names(list.est[[a]])[c]   # combo name

    # Bind all runs for this combo
    df_combo <- bind_rows(combo_elem, .id = "run") %>%
      mutate(run = as.integer(gsub("run_", "", run)),  # convert "run_1" -> 1
             combo = combo_name)

    # Keep columns of estimates: AFL, AFZ, AFR, ESLZ, ELAB, d_level
    df_combo[, c("run", "combo", "AFL", "AFZ", "AFR", "ESLZ", "ELAB", "d_level")]
  }))
  write_xlsx(final_data, paste0("temp/PR", sprintf("%02d", a),  "/PR", sprintf("%02d", a), "_pfp.xlsx"))
  
  # Process and save median and CI for FSE 
  # Loop through each of the 4 groups
  for (group_name in names(combo_groups)) {
    combos_to_use <- combo_groups[[group_name]]
    
    # Filter data for this group (all included combos)
    group_data <- final_data %>%
      filter(combo %in% combos_to_use)
    
    cat("  Calculate Median and CI:", group_name, ")\n")
    
    # ---- Data prep 
    group_data_long <- group_data %>%
      dplyr::select(run, AFL, AFR, AFZ, ESLZ, ELAB) %>%
      pivot_longer(cols = AFL:ELAB, names_to = "variable", values_to = "value") %>%
      mutate(value_scaled = ifelse(variable %in% c("ESLZ", "ELAB"), value * scaling_factor, value)) %>%
      group_by(variable) %>%
      mutate(outlier = value_scaled > (boxplot.stats(value_scaled)$stats[4] * 1.5) |
               value_scaled < (boxplot.stats(value_scaled)$stats[2] * 0.5)) %>%
      ungroup() %>%
      mutate(variable = factor(variable, levels = c("AFL", "AFR", "AFZ", "ESLZ", "ELAB")))
    
    # ---- Summary stats: Mean, Median and SD 
    summary_stats <- group_data_long %>%
      group_by(variable) %>%
      summarise(mean_raw   = mean(value, na.rm = TRUE),
                median_raw = median(value, na.rm = TRUE),
                mad_raw    = mad(value, na.rm = TRUE), 
                .groups    = "drop") %>%
      rowwise() %>%
      mutate(mean   = round_rule(variable, mean_raw),
             median = round_rule(variable, median_raw),
             mad    = round_rule(variable, mad_raw), .groups = "drop") %>%
      mutate(label = paste0("Avg.:", mean, "\nMed.:", median, "\nMAD:", mad)) %>%
      ungroup()
    
    # save a list of the summary stat
    list.median[[a]][[group_name]] <- summary_stats %>%
      #dplyr::select(combo, variable, median, sd) %>%
      dplyr::select(variable, median, mad) %>%
      pivot_wider(names_from = variable, values_from = c(median, mad), names_glue = "{variable}_{.value}") %>%
      dplyr::select(starts_with("AFL"), starts_with("AFR"), starts_with("AFZ"), starts_with("ESLZ"), starts_with("ELAB")) %>%
      mutate(group = group_name) %>%
      dplyr::select(group, everything())
  }
}

#       Save median estimates & list.plots, pfp_est_pr.xlsx ----
pfp.est <- bind_rows(
  lapply(seq_along(list.median), function(i) {
    pr_list <- list.median[[i]]
    
    # skip if nothing stored for this PR
    if (is.null(pr_list)) return(NULL)
    
    bind_rows(
      lapply(seq_along(pr_list), function(j) {
        df <- pr_list[[j]]
        if (is.null(df) || nrow(df) == 0) return(NULL)
        
        df$PR9  <- as.integer(i)
        df$group <- names(pr_list)[j]
        df
      })
    )
  })
) %>%
  dplyr::select(PR9, group, everything())

write_xlsx(pfp.est, paste0("temp/1_pfp_est_pr.xlsx"))


# Plot Figure 6, FSE
# group: source
source <- list(
  NAWS = c("NAWS_LCL", "NAWS_MCL", "NAWS_HCL",
             "NAWS_LCM", "NAWS_MCM", "NAWS_HCM",
             "NAWS_LCH", "NAWS_MCH", "NAWS_HCH",
           "NAWS_LNL", "NAWS_MNL", "NAWS_HNL",
             "NAWS_LNM", "NAWS_MNM", "NAWS_HNM",
             "NAWS_LNH", "NAWS_MNH", "NAWS_HNH"),
  FLS  = c("FLS_LCL",  "FLS_MCL",  "FLS_HCL",
             "FLS_LCM",  "FLS_MCM",  "FLS_HCM",
             "FLS_LCH",  "FLS_MCH",  "FLS_HCH",
           "FLS_LNL",  "FLS_MNL",  "FLS_HNL",
             "FLS_LNM",  "FLS_MNM",  "FLS_HNM",
             "FLS_LNH",  "FLS_MNH",  "FLS_HNH"))

# Function to read in  FSE 
load_all_pr_data <- function() {
  
  full_df <- lapply(1:9, function(a) {
    
    fpath <- file.path(paste0("temp/PR", sprintf("%02d", a), "/PR", sprintf("%02d", a), "_pfp.xlsx"))
    
    df <- read_excel(fpath)
    df$PR9 <- a   # store numeric PR code
    df
  })
  
  bind_rows(full_df)
}

# Load all PR files
first_st_estimates <- load_all_pr_data()

# Plot func with titles, axis labels, legend names
plot_box_two_panels <- function(combos_to_use, group_name) {
  
  # Join PR names
  df <- first_st_estimates %>%
    filter(combo %in% combos_to_use) %>%
    left_join(PR, by = "PR9") %>%
    filter(PR9 %in% sel_pr) %>%
    mutate(PR = factor(pr,                                   # labels
                       levels = pr[match(sel_pr, PR9)]))       # order by sel_pr
  df_long <- df %>%
    pivot_longer(cols = c(AFL, AFR, AFZ, ESLZ, ELAB),
                 names_to = "variable",
                 values_to = "value") %>%
    mutate(variable = factor(variable, levels = c("AFL","AFR","AFZ","ESLZ","ELAB")))
  
  # Panel groups
  vars_A <- c("AFL","AFR","AFZ")
  vars_B <- c("ESLZ","ELAB")
  
  # Summary table for Avg, Med, MAD ----
  df_summary <- df_long %>%
    group_by(PR, variable) %>%
    summarise(mean_raw   = mean(value, na.rm = TRUE),
              median_raw = median(value, na.rm = TRUE),
              mad_raw    = mad(value, na.rm = TRUE),  .groups    = "drop") %>%
    rowwise() %>%
    mutate(mean   = round_rule(variable, mean_raw),
           median = round_rule(variable, median_raw),
           mad     = round_rule(variable, mad_raw), .groups = "drop") %>%
    mutate(label = paste0("Avg:", mean,   "\n",
                          "Med:", median, "\n",
                          "MAD:", mad)) 
  # Panel A
  pA <- ggplot(df_long %>% filter(variable %in% vars_A),
               aes(x = variable, y = value, fill = PR)) +
    geom_boxplot(position = position_dodge(width = 0.9),
                 outlier.shape = NA) +      # REMOVE OUTLIERS
    # Mean (X) and Median (●)
    stat_summary(fun = mean, geom = "point", aes(shape = "Mean", group = PR), position = position_dodge(width = 0.9), size = 4, color = "black") +
    stat_summary(fun = median, geom = "point", aes(shape = "Median", group = PR), position = position_dodge(width = 0.9), size = 4, color = "black", fill = "yellow") +
    # Text under each box
    geom_text(data = df_summary %>% filter(variable %in% vars_A),
              aes(x = variable, y = 1, label = label, group = PR),
              size = 6, vjust = 1, lineheight = 0.8,
              position = position_dodge(width = 0.9)) +
    scale_x_discrete(labels = c(
      AFL  = expression(a[L]),
      AFR  = expression(a[R]),
      AFZ  = expression(a[I]))) +
    
    scale_y_continuous(limits = c(-4,100)) +  # Y AXIS FIXED RANGE
    labs(title = "A: PFP Estimates for Labor, Resources and Int. Inputs",
         x = "", y = "Change in PFP (%)") +
    scale_shape_manual(values = c("Mean" = 4, "Median" = 23)) +
      
    theme_bw(base_size = 18) +
    theme(legend.position = "none",   # REMOVE LEGEND
          plot.title = element_text(face = "bold", hjust = 0.5),  # CENTER TITLE
          plot.title.position = "plot",
          axis.title.y = element_text(size = 20, face = "bold"),
          axis.text.y = element_text(size = 20, face = "bold"),
          axis.text.x = element_text(size = 30, face = "bold"))
  
  # Panel B
  pB <- ggplot(df_long %>% filter(variable %in% vars_B),
               aes(x = variable, y = value, fill = PR)) +
    geom_boxplot(position = position_dodge(width = 0.9),
                 outlier.shape = NA) +      # REMOVE OUTLIERS
    # Mean (X) and Median (●)
    stat_summary(fun = mean,   geom = "point", aes(shape = "Mean",   group = PR), position = position_dodge(width = 0.9), size = 4, color = "black") +
    stat_summary(fun = median, geom = "point", aes(shape = "Median", group = PR), position = position_dodge(width = 0.9), size = 4, color = "black", fill = "yellow") +
    # Text under each box
    geom_text(data = df_summary %>% filter(variable %in% vars_B),
              aes(x = variable, y = 0.08, label = label, group = PR),
              size = 6, vjust = 1, lineheight = 0.8,
              position = position_dodge(width = 0.9)) +
    scale_x_discrete(labels = c(
      ESLZ  = expression(sigma[L]),
      ELAB  = expression(e[L]))) +
    scale_y_continuous(limits = c(0,1)) +  # Y AXIS FIXED RANGE
    labs(title = "B: Elasticity Estimates for Labor Supply and Labor Substitutability", x = "", y = "Elasticity Estimates") +
     scale_shape_manual(values = c("Mean" = 4, "Median" = 23)) +
    theme_bw(base_size = 18) +
    theme(
      legend.position = "bottom",             # LEGEND BOTTOM
      legend.direction = "horizontal",
      legend.title = element_blank(),
      legend.text = element_text(size = 20),
      legend.key.size = unit(1.5, "cm"),
      legend.margin = margin(t = 0, r = 0, b = 0, l = 0),
      legend.box.margin = margin(t = 0, r = 0, b = 0, l = 0),
      plot.title = element_text(face = "bold", hjust = 0.5),  # CENTER TITLE
      plot.title.position = "plot",
      axis.title.y = element_text(size = 20, face = "bold"),
      axis.text.y = element_text(size = 20, face = "bold"),
      axis.text.x = element_text(size = 30, face = "bold"))
  
  # Stack vertically with an overall title
  full_plot <- (pA / pB) 
  return(full_plot)
}

#       RUN EVERYTHING — CREATE ONE PNG PER COMBO GROUP
sel_pr <- c(1,2,3,7)
#sel_pr <- c(4,5,6,8,9)

for (s in names(source)) {
  
  if(1 %in% sel_pr){name = "Sel"} else {name = "Oth"}
  message("Creating plot for: ", s)
  
  pg <- plot_box_two_panels(
    #final_data = first_st_estimates,
    combos_to_use = source[[s]],
    group_name = s)
  
  outfile <- file.path(paste0("plots/Figure6/", name, "_", s, ".png"))
  
  ggsave(outfile, pg,
         width = 15, height = 18, dpi = 300, bg = "white")
} # writes reporting file

