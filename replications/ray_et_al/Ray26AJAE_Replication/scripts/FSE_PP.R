# ---- Change working directtory to mod_math ----
# Estimate unknown parameters best fitting the changes in input use for crop production 2002 - 2017
# 0. parameters for algorithm ----
alpha = 1
gamma = 2
rho   = 1/2
sigma = 1/2


# 1. define df.known.input, KNOWNINPUT.har file ----
df.known.input <- data_cz %>%
  left_join(CZ2PR) %>%
  dplyr::select(PR9, CZ, q, p_adj, q_adj, qw, pc, pz) 

for (pr in c(1:9)) {
  df.known.input.a <- df.known.input %>%
    mutate(across(c(q_adj, p_adj, qw, pc, pz), ~ if_else(PR9 != pr, 0, .))) 
  
  data.known   <- list()
  data.known[["QWAT"]] <- array(as.vector(df.known.input.a$qw),  # Data as a vector
                                dim = c(605, 1),  # Set dimensions (adjust if needed)
                                dimnames = list(dim1name = cz, dim2name = "QWAT"))
  data.known[["PLND"]] <- array(as.vector(df.known.input.a$pc),  # Data as a vector
                                dim = c(605, 1),  # Set dimensions (adjust if needed)
                                dimnames = list(dim1name = cz, dim2name = "PLND"))
  data.known[["QAUG"]] <- array(as.vector(df.known.input.a$q_adj),  # Data as a vector
                                dim = c(605, 1),  # Set dimensions (adjust if needed)
                                dimnames = list(dim1name = cz, dim2name = "QAUG"))
  data.known[["PAUG"]] <- array(as.vector(df.known.input.a$p_adj),  # Data as a vector
                                dim = c(605, 1),  # Set dimensions (adjust if needed)
                                dimnames = list(dim1name = cz, dim2name = "PAUG"))
  data.known[["PNLB"]] <- array(as.vector(df.known.input.a$pz),  # Data as a vector
                                dim = c(605, 1),  # Set dimensions (adjust if needed)
                                dimnames = list(dim1name = cz, dim2name = "PNLB"))
  
  file_name <- paste0("shf/OBS_DATA_PR", sprintf("%02d", pr),'.har')  # File name with index PR
  suppressMessages(write_har(data.known, file_name))  # Save the file with the indexed name
}

# 2. define initial values ----
df.ini <- data_pr %>%
  dplyr::select(PR9, ao_pr) %>%
  mutate(AFL = ao_pr,
         AFZ = ao_pr,
         AFR = ao_pr) %>%
  mutate(ESLZ = mean_sig) %>%
  dplyr::select(PR9, AFL, AFZ, AFR, ESLZ) # this sequence is important

# 3. define df.levels.target ----
df.levels.target <- data_cz %>%                         # goes into the obj function
  mutate(VCRP2002    = VCROP2002_ltavg / 1000,
         VCRP2017    = VCROP2017_ltavg / 1000,
         VZ2002      = VZ_02/1000,  # 1000 $s
         VZ2017      = VZ_17/1000,
         QLAND01     = HLAND2002_ltavg/1000 ,      # in 1000 ha
         QLAND16     = HLAND2017_ltavg/1000,
         VLAND01     = RENT02 * QLAND01,
         VLAND16     = RENT17 * QLAND16) %>%
  dplyr:: select(CZ, VZ2002,  VZ2017, VCRP2002, VCRP2017, QLAND01,  QLAND16, VLAND01,  VLAND16, RENT02, RENT17)  
dim(df.levels.target)
length(unique(df.levels.target$CZ))

# Functions ----
# Run simulation
run.simple.fse <- function(a, combo.name, cmf_prefix = "1-PR") {
  # Batch file name
  batfile <- paste0("run_simple_", combo.name, ".bat")
  
  # Create batch file
  sink(batfile)
  cat(paste0("call simplez.exe -cmf ", cmf_prefix, sprintf("%02d", a), "-", combo.name, ".cmf \n"))
  sink()
  
  # Run the batch file and capture output
  output <- system2(batfile, stdout = TRUE, stderr = TRUE, wait = TRUE)
  
  # Remove batch file
  file.remove(batfile)
  
  #return(success)
}

# Start parallel processing ----
numCores <- parallel::detectCores()
cl <- makeCluster(18) #(numCores - saveCores)
cl
registerDoParallel(cl) #starting parallel

# export global objects to paralle proc
clusterExport(cl, c("project_root", "temp", "dfz.out", "dfa.out", "CZ2PR", "df.known.input", "df.ini", "lower", "upper", "run.simple.fse", "obj", "run", "k",
                    "iter.mid", "iter.max", "dev.tolerance", "dev.diff.tol", "alpha", "gamma", "rho", "sigma"))
clusterEvalQ(cl, setwd(project_root))

list.est <- vector("list", num.pr) # list to store all PR and Combo

# Main script ----
for (a in do.pr) {
  name = PR$pr[PR$PR9 == a]
  print(paste0("Starting for parallel proc for PR", sprintf("%02d", a), ", ", name))
  
  # Loop over parameter combos
  list.est[[a]] <- vector("list", length(df.for.combo))
  
  # for reproduceability
  set.seed(12435687)
  
  # Parallel loop over combos (c), with reproducability
  list_c <- foreach(c = seq_along(df.for.combo), .packages = c("dplyr", "HARr", "cowplot", "ggplot2"), 
                    .options.RNG = 12435687, .errorhandling = "pass") %dorng% {
    # Grab the matching names for combo
    combo.name <- df.for.combo[c]
    
    print(paste0("PR", sprintf("%02d", a), ": Define data for combo - ", combo.name))
    
    # Create the 'plot' and 'out' subfolders, which also creates the PRXX & combo folder
    dir.create(paste0("temp/PR", sprintf("%02d", a), "/", combo.name, "/plot"), recursive = TRUE) # error if already exists
    dir.create(paste0("temp/PR", sprintf("%02d", a), "/", combo.name, "/out"))
    
    # define the combo specific QLAB and VLAB initial and targets
    dfz.lab.data <- as.data.frame(dfz.out[[combo.name]])
    dfa.lab.data <- as.data.frame(dfa.out[[combo.name]]) 
    
    df.levels.lab.target <- dfz.lab.data %>%
      dplyr::select(CZ, VLAB2002_ltavg, VLAB2017_ltavg, QLAB_Hours2002, QLAB_Hours2017, WAGE02, WAGE17) %>%
      mutate(VLAB2002    = VLAB2002_ltavg/1000, # 1000 $s
             VLAB2017    = VLAB2017_ltavg/1000,
             QLAB2002    = QLAB_Hours2002/1000, # 1000 hours
             QLAB2017    = QLAB_Hours2017/1000) %>%
      left_join(CZ2PR) %>%
      mutate(across(c(VLAB2002, VLAB2017, QLAB2002, QLAB2017, WAGE02, WAGE17), ~ if_else(PR9 != a, 0, .))) %>%
      dplyr::select(-PR9)
    
    # define the initial values for the combo, including ELAB
    df.initial <- df.ini %>%
      left_join(dfa.lab.data %>% dplyr::select(PR9, pl, ql, phil)) %>%
      mutate(ELAB = abs((ql + phil) / pl)) %>%
      filter(PR9 == a) %>%
      dplyr::select(-PR9) %>%
      dplyr::select(AFL, AFZ, AFR, ESLZ, ELAB) %>% # this sequence is important
      mutate(ELAB = ifelse(ELAB>=max.ELAB, max.ELAB-0.05, ELAB)) %>%
      reframe(x = as.numeric(t(.)))
    
    # start a list to collect results for combo z
    l.fin         <- list() # store the final run for each run,  reset at each run for within z
    
    # Initialize storage for this combo
    esti.combo <- vector("list", run) # collect best run
    
    # NOW starts run: Algorithm Starts ----
    for (r in 1:run) 
    {
      # Step 1: k random parameter values plus 1 previously calibrated
      print(paste0("PR", sprintf("%02d", a), ": For combo - ", combo.name, ", run = ", r,
                   ": Define k=", k, "(random) +1(best guess) and running simplez.tab k+1 times to calculate dev for each vector"))
      
      l <- list()
      
      # k random parameters values and their dev in level and percent terms
      for (i in 1:k) {
        # Initiate algorithm with initial values in x
        x      <- runif(k, min = lower, max = upper)
        l[[i]] <- list(x = x, dev = obj(x))}
      
      # Add initial values with mean_sig for sigma
      initial_guess <- df.initial$x
      l[[k + 1]]    <- list(x = initial_guess, dev = obj(initial_guess))
      
      df.vis = data.frame(iter = 0, 
                          AFL_best      = round( df.initial$x[1],1 ),           AFZ_best      = round( df.initial$x[2],1 ), 
                          AFR_best      = round( df.initial$x[3],1 ),     
                          ESLZ_best     = round( df.initial$x[4],1 ),           ELAB_best     = round( df.initial$x[5],1 ), 
                          d_level_min   = round( l[[1]]$dev[["d.level.a"]],5 ), d_level_max   = round( l[[k+1]]$dev[["d.level.a"]],5 ),
                          
                          d_level_QL    = round( l[[1]]$dev[["d.QLa"]],3 ),     d_level_PL    = round( l[[1]]$dev[["d.PLa"]],3 ),
                          d_level_VL    = round( l[[1]]$dev[["d.VLa"]],3 ),     d_level_VZ    = round( l[[1]]$dev[["d.VZa"]],3 ),
                          d_level_VCRP  = round( l[[1]]$dev[["d.VCRPa"]],3 ),
                          d_level_QC    = round( l[[1]]$dev[["d.QCa"]],3 ),     d_level_PC    = round( l[[1]]$dev[["d.PCa"]],3 ),
                          d_level_VC    = round( l[[1]]$dev[["d.VCa"]],3 ),
                          
                          level_QL      = round( l[[1]]$dev[["sim.QLa"]],0 ),   level_PL      = round( l[[1]]$dev[["sim.PLa"]],1 ),
                          level_VL      = round( l[[1]]$dev[["sim.VLa"]],0 ),   level_VZ      = round( l[[1]]$dev[["sim.VZa"]],0 ),
                          level_VCRP    = round( l[[1]]$dev[["sim.VCRPa"]],0 ),      
                          level_QC      = round( l[[1]]$dev[["sim.QCa"]],0 ),   level_PC      = round( l[[1]]$dev[["sim.PCa"]],1 ),
                          level_VC      = round( l[[1]]$dev[["sim.VCa"]],0 ),
                          AFL_max  = 0, AFZ_max  = 0, AFR_max  = 0,  ESLZ_max = 0, ELAB_max = 0)
      
      # For saving purpose
      saveRDS(l,        file = paste0("temp/PR", sprintf("%02d", a),  "/", combo.name, "/out/NM_PR",     sprintf("%02d", a), "_initial_r", r, ".rds"))
      write.csv(df.vis,        paste0("temp/PR", sprintf("%02d", a),  "/", combo.name, "/out/df_vis_PR", sprintf("%02d", a), "_r",         r,".csv"))
      
      iter = df.vis[nrow(df.vis),]$iter + 1
      
      # Iterative Loop starts here
      print(paste0("PR", a, ": Start algorithm loop"))
      while (iter <= iter.max)
      {
        print(paste0("PR", sprintf("%02d", a), " combo name ", combo.name, " run=", r, ": Iteration ",iter))
        
        # step 2: reorder parameters with the dev in ascending order
        l = l[order(sapply(l, function(y) y$dev[1]))]
        print(paste0("Min deviation in prev iteration: ", round(l[[1]]$dev[[1]], 5)))
        
        df.vis = rbind(df.vis, data.frame(iter = iter, 
                                          AFL_best      = round( l[[1]]$x[[1]],1 ),             AFZ_best     = round( l[[1]]$x[[2]],1 ), 
                                          AFR_best      = round( l[[1]]$x[[3]],1 ), 
                                          ESLZ_best     = round( l[[1]]$x[[4]],2 ),             ELAB_best    = round( l[[1]]$x[[5]],2 ), 
                                          d_level_min   = round( l[[1]]$dev[["d.level.a"]],5 ), d_level_max  = round( l[[k+1]]$dev[["d.level.a"]],5 ),
                                          
                                          d_level_QL    = round( l[[1]]$dev[["d.QLa"]],3 ),     d_level_PL   = round( l[[1]]$dev[["d.PLa"]],3 ),
                                          d_level_VL    = round( l[[1]]$dev[["d.VLa"]],3 ),     d_level_VZ   = round( l[[1]]$dev[["d.VZa"]],3 ),
                                          d_level_VCRP  = round( l[[1]]$dev[["d.VCRPa"]],3 ),
                                          d_level_QC    = round( l[[1]]$dev[["d.QCa"]],3 ),     d_level_PC   = round( l[[1]]$dev[["d.PCa"]],3 ),
                                          d_level_VC    = round( l[[1]]$dev[["d.VCa"]],3 ),
                                          
                                          level_QL      = round( l[[1]]$dev[["sim.QLa"]],0 ),   level_PL     = round( l[[1]]$dev[["sim.PLa"]],1 ),
                                          level_VL      = round( l[[1]]$dev[["sim.VLa"]],0 ),   level_VZ     = round( l[[1]]$dev[["sim.VZa"]],0 ),
                                          level_VCRP    = round( l[[1]]$dev[["sim.VCRPa"]],0 ),      
                                          level_QC      = round( l[[1]]$dev[["sim.QCa"]],0 ),   level_PC     = round( l[[1]]$dev[["sim.PCa"]],1 ),
                                          level_VC      = round( l[[1]]$dev[["sim.VCa"]],0 ),
                                          
                                          AFL_max  = round( max( c(l[[2]]$x[[1]], l[[3]]$x[[1]], l[[4]]$x[[1]], l[[5]]$x[[1]], l[[6]]$x[[1]]) ), 1),
                                          AFZ_max  = round( max( c(l[[2]]$x[[2]], l[[3]]$x[[2]], l[[4]]$x[[2]], l[[5]]$x[[2]], l[[6]]$x[[2]]) ), 1),
                                          AFR_max  = round( max( c(l[[2]]$x[[3]], l[[3]]$x[[3]], l[[4]]$x[[3]], l[[5]]$x[[3]], l[[6]]$x[[3]]) ), 1),
                                          ESLZ_max = round( max( c(l[[2]]$x[[4]], l[[3]]$x[[4]], l[[4]]$x[[4]], l[[5]]$x[[4]], l[[6]]$x[[4]]) ), 2),
                                          ELAB_max = round( max( c(l[[2]]$x[[5]], l[[3]]$x[[5]], l[[4]]$x[[5]], l[[5]]$x[[5]], l[[6]]$x[[5]]) ), 2)))
        
        print(paste0("Difference in dev_min of subsequent runs:", abs( diff(tail(df.vis$d_level_min ,2)) ) ))
        if(l[[1]]$dev[[1]] < dev.tolerance)
        {
          print(paste0("Optimum found for RG ", a))
          # For saving purpose
          saveRDS(l,   file=paste0("temp/PR", sprintf("%02d", a),  "/", combo.name, "/out/NM_PR",     sprintf("%02d", a), "_r", r, "_iter", iter, ".rds"))
          write.csv(df.vis, paste0("temp/PR", sprintf("%02d", a),  "/", combo.name, "/out/df_vis_PR", sprintf("%02d", a), "_r", r,                ".csv"), row.names = F)
          break
        }
        
        # step 3: calculate the centroid except n+1
        l.n = l[1:k]
        # a.o: the centroid
        a.o = array(rep(0, k))
        for (i in 1:length(l.n))
        {
          a.o = a.o + l.n[i][[1]]$x
        }
        a.o = a.o / k
        
        # step 4: calculate the reflection point
        a.r = a.o + alpha*(a.o - l[k+1][[1]]$x)
        d.r = obj(a.r)
        
        
        # step 5: compare the dev of reflection point
        if(d.r[1] < l[[k]]$dev[[1]] & d.r[1] >= l[[1]]$dev[[1]]) # reflection
        {
          print("perform reflection")
          #l[[k+1]] = c(data.frame(x = a.r), d.r)
          l[[k+1]] = list(x=a.r, dev = d.r)
        } else if (d.r[1] < l[[1]]$dev[[1]]) { # expansion
          print("perform expansion")
          a.e = a.o + gamma*(a.r - a.o)
          d.e = obj(a.e)
          if (d.e[1] < d.r[1])
          {
            #l[[k+1]] = c(data.frame(x = a.e), d.e)
            l[[k+1]] = list(x=a.e, dev = d.e)
          } else {
            #l[[k+1]] = c(data.frame(x = a.r), d.r)
            l[[k+1]] = list(x=a.r, dev = d.r)
          }
        } else { # contraction
          if (d.r[1] < l[[k+1]]$dev[[1]]) # outside contraction or shrink
          {
            a.c = a.o + rho*(a.r - a.o)
            d.c = obj(a.c)
            if (d.c[1] < d.r[1])
            {
              print("perform outside contraction")
              
              l[[k+1]] = list(x=a.c, dev = d.c)
            } else { # shrink
              print("perform shrink")
              for(i in 2:(k+1))
              {
                a.i = l[1][[1]]$x + sigma*(l[i][[1]]$x - l[1][[1]]$x)
                l[[i]] = list(x=a.i, dev = obj(a.i))
              }
            }
          } else { # inside contraction or shrink
            a.c = a.o + rho*(l[k+1][[1]]$x - a.o)
            d.c = obj(a.c)
            if (d.c[1] < l[[k+1]]$dev[[1]])
            {
              print("perform inside contraction")
              l[[k+1]] = list(x=a.c, dev = d.c)
            } else { # shrink
              print("perform shrink")
              for(i in 2:(k+1))
              {
                a.i = l[1][[1]]$x + sigma*(l[i][[1]]$x - l[1][[1]]$x)
                l[[i]] = list(x=a.i, dev = obj(a.i))
              }
            }
          }
          
          # compare dev of 2 subsequent runs
          if (iter > iter.mid ) 
          {
            l.pre    <- readRDS(paste0("temp/PR", sprintf("%02d", a),  "/", combo.name, "/out/NM_PR",     sprintf("%02d", a), "_r", r, "_iter", iter-1, ".rds"))
            dev.pre <- l.pre[[1]]$dev[[1]]
            
            l <- l[order(sapply(l, function(y) y$dev[1]))]
            if ( abs( dev.pre - l[[1]]$dev[[1]] ) != 0 &  abs( dev.pre - l[[1]]$dev[[1]] ) < dev.diff.tol )
            {
              print(paste0("PR", sprintf("%02d", a), ": @Iteration ", iter, " Differece in deviation between 2 subsequent rounds small enough"))
              break
            }
          }
        }
        # loop ends here
        
        # Save information from each iteration
        saveRDS(l,       file = paste0("temp/PR", sprintf("%02d", a),  "/", combo.name, "/out/NM_PR", sprintf("%02d", a), "_r", r, "_iter", iter, ".rds"))
        iter = iter + 1
        
        if (iter > iter.max)
        {
          print(paste0("PR", sprintf("%02d", a), ": Optimum not found with given iterations"))
        }
      }
      df.vis$d_min_diff <- c(NA, abs( diff(df.vis$d_level_min) ))  # Adds NA for the first row and difference for the rest
      
      write.csv(df.vis,        paste0("temp/PR", sprintf("%02d", a),  "/", combo.name, "/out/df_vis_PR", sprintf("%02d", a), "_r",         r,".csv"))
      
      # order list per dev
      l <- l[order(sapply(l, function(y) y$dev[1]))]
      
      # save the best run in l.fin
      l.fin[[r]] = df.vis[nrow(df.vis), ]
      # Save to combo_list
      esti.combo[[r]] <- list(AFL     = round( l.fin[[r]][["AFL_best"]],    1 ),
                              AFZ     = round( l.fin[[r]][["AFZ_best"]],    1 ),
                              AFR     = round( l.fin[[r]][["AFR_best"]],    1 ),
                              ESLZ    = round( l.fin[[r]][["ESLZ_best"]],   2 ),
                              ELAB    = round( l.fin[[r]][["ELAB_best"]],   2 ),
                              d_level = round( l.fin[[r]][["d_level_min"]], 2 ))
      
      # generate figures ----
      print(paste0("Figures for PR ", a, ", combo ", combo.name, ", run ", r))
      
      # Objective function ----
      print(paste0("Figures for PR ", a, ": Obj fn"))
      dev_level <- ggplot() +            
        geom_point(data = df.vis[2:nrow(df.vis),], aes(iter, d_level_min, color = "Dev Min"))                    +
        geom_line(data  = df.vis[2:nrow(df.vis),], aes(iter, d_level_min, color = "Dev Min"), linetype="dashed") +
        
        geom_point(data = df.vis[2:nrow(df.vis),], aes(iter, d_level_max, color = "Dev Max")) +
        geom_line(data  = df.vis[2:nrow(df.vis),], aes(iter, d_level_max, color = "Dev Max"), linetype="dashed") +
        
        geom_point(data = df.vis[2:nrow(df.vis),], aes(iter, d_min_diff, color = "Dev Difference"))                    +
        geom_line(data  = df.vis[2:nrow(df.vis),], aes(iter, d_min_diff, color = "Dev Difference"), linetype="dashed") +
        
        ylab("Error Levels: sum(z in RG(a), xshr * Xj)") +
        xlab("Iterations") +
        ggtitle("Deviation in levels") +
        scale_color_manual(values = c("Dev Min" = "blue", "Dev Max" = "red", "Dev Difference" = "green")) + # Custom colors
        theme(axis.line.x = element_line(color="black"),
              axis.line.y = element_line(color="black"),
              panel.background = element_blank(),
              axis.title = element_text(size = 15),
              axis.text.x = element_text(size = 15),
              axis.text.y = element_text(size = 15),
              plot.title = element_text(size = 15),
              legend.title = element_blank(),
              legend.position = "top")
      ggsave(paste0("temp/PR", sprintf("%02d", a),  "/", combo.name, "/plot/PR", sprintf("%02d", a), "_dev_level_min_max_diff", "_r",         r,".png"),
             plot = dev_level, units="in", width=5, height=5, dpi=200)
      
      # Levels Deviation ----
      print(paste0("Figures for PR ", a, ": d_level"))
      plot.levels.dev <- ggplot() +            
        # QLAND
        geom_point(data = df.vis[2:nrow(df.vis),],    aes(iter, d_level_QC, color = "QLAND")) +
        geom_line(data = df.vis[2:nrow(df.vis),],     aes(iter, d_level_QC, color = "QLAND"), linetype = "dashed") +
        
        # VC
        geom_point(data = df.vis[2:nrow(df.vis),],    aes(iter, d_level_VC, color = "VLAND")) +
        geom_line(data = df.vis[2:nrow(df.vis),],     aes(iter, d_level_VC, color = "VLAND"), linetype = "dashed") +
        
        # VZ
        geom_point(data = df.vis[2:nrow(df.vis),],    aes(iter, d_level_VZ, color = "VNLABOR")) +
        geom_line(data  = df.vis[2:nrow(df.vis),],    aes(iter, d_level_VZ, color = "VNLABOR"), linetype = "dashed") +
        
        # QLAB
        geom_point(data = df.vis[2:nrow(df.vis),],    aes(iter, d_level_QL,   color = "QLAB")) +
        geom_line(data = df.vis[2:nrow(df.vis),],     aes(iter, d_level_QL,   color = "QLAB"), linetype = "dashed") +
        
        # VLAB
        geom_point(data = df.vis[2:nrow(df.vis),],    aes(iter, d_level_VL,      color = "VLAB")) +
        geom_line(data = df.vis[2:nrow(df.vis),],     aes(iter, d_level_VL,      color = "VLAB"), linetype = "dashed") +
        
        ylab("Error in Levels(Q&V)") +
        xlab("Iterations") +
        ggtitle(paste0("Sum of share weighted deviations over CZs in PR:" , sprintf("%02d", a))) +
        scale_color_manual(values = c("QLAB" = "red", "QLAND" = "green", "VNLABOR" = "purple", "VLAB" = "hotpink", "VLAND" = "darkgreen")) + # Custom colors
        theme(axis.line.x = element_line(color="black"),
              axis.line.y = element_line(color="black"),
              panel.background = element_blank(),
              axis.title = element_text(size = 12),
              axis.text.x = element_text(size = 12),
              axis.text.y = element_text(size = 12),
              plot.title = element_text(size = 12),
              legend.title = element_blank(),
              legend.position = "top")
      ggsave(paste0("temp/PR", sprintf("%02d", a),  "/", combo.name, "/plot/PR", sprintf("%02d", a), "_dev", "_r",         r,".png"),
             plot = plot.levels.dev, units="in", width=5, height=5, dpi=200)
      
      # Levels vs Non-lab Target at PR ----
      print(paste0("Figures for PR ", a, ": Levels vs target for non-lab"))
      
      plot.targ.a <- df.levels.target %>%
        left_join(df.levels.lab.target) %>% #has PR9
        left_join(CZ2PR) %>%
        filter(PR9==a) %>%
        mutate(RENT_TOT = (QLAND16) * RENT17,
               WAGE_TOT = (QLAB2017) * WAGE17) %>%
        dplyr::select(PR9, VZ2017, VCRP2017, QLAND16, VLAND16, VLAB2017, QLAB2017, WAGE_TOT, RENT_TOT) %>%
        group_by(PR9) %>%
        summarise(across(everything(), ~ sum(.x, na.rm = TRUE))) %>%
        mutate(RENT17 = RENT_TOT / QLAND16,
               WAGE17 = WAGE_TOT / QLAB2017)
      
      # Calculate scaling factor and bounds
      primary_max_nlab    <- max(plot.targ.a$QLAND16*100, plot.targ.a$VLAND16, na.rm = TRUE)
      primary_min_nlab    <- min(df.vis$level_QC*100,     df.vis$level_VC,     na.rm = TRUE)
      
      secondary_max_nlab  <- max(plot.targ.a$VZ2017, na.rm = TRUE)
      secondary_min_nlab  <- min(df.vis$level_VZ,      na.rm = TRUE)
      
      scaling_factor_nlab <- primary_max_nlab / secondary_max_nlab
      
      # Compute combined y-limits after scaling (to start both at comparable minima)
      primary_lower_nlab  <- min(primary_min_nlab, secondary_min_nlab * scaling_factor_nlab, na.rm = TRUE)
      primary_upper_nlab  <- max(primary_max_nlab, secondary_max_nlab * scaling_factor_nlab, na.rm = TRUE)
      
      plot.levels.v.target <- ggplot() +
        # QLAND
        geom_point(data = df.vis[2:nrow(df.vis),],   aes(iter, level_QC*100,                color = "QLAND")) +
        geom_line(data  = df.vis[2:nrow(df.vis),],   aes(iter, level_QC*100,                color = "QLAND"), linetype = "dashed") +
        geom_point(data = plot.targ.a            ,   aes(-1,   QLAND16 *100,                color = "QLAND"), shape = 18, size = 4) +
        
        # VLAND
        geom_point(data = df.vis[2:nrow(df.vis),],    aes(iter, level_VC,                    color = "VLAND")) +
        geom_line(data = df.vis[2:nrow(df.vis),],     aes(iter, level_VC,                    color = "VLAND"), linetype = "dashed") +
        geom_point(data = plot.targ.a            ,    aes(-1,   VLAND16,                     color = "VLAND"), shape = 18, size = 4) +
        
        # VZ
        geom_point(data = df.vis[2:nrow(df.vis),],    aes(iter, level_VZ * scaling_factor_nlab,   color = "VNLABOR")) +
        geom_line(data  = df.vis[2:nrow(df.vis),],    aes(iter, level_VZ * scaling_factor_nlab,   color = "VNLABOR"), linetype = "dashed") +
        geom_point(data = plot.targ.a            ,    aes(nrow(df.vis)+1, VZ2017* scaling_factor_nlab,    color = "VNLABOR"), shape = 18, size = 4) +
        
        # Define primary and scaled secondary y-axis
        scale_y_continuous(name = "QC (/10), VC (/1000)",
                           limits = c(primary_lower_nlab, primary_upper_nlab),
                           sec.axis = sec_axis(~ . / scaling_factor_nlab, name = "VZ (/1000), VCRP (/1000)")) +
        ylab("Non-labor targets") +
        xlab("Iterations") +
        ggtitle("QLAND, VLAND, VNLABOR, VCRP") +
        scale_color_manual(values = c("QLAND" = "green", "VLAND" = "cyan", "VNLABOR" = "purple")) + # Custom colors
        theme(axis.line.x = element_line(color="black"),
              axis.line.y = element_line(color="black"),
              panel.background = element_blank(),
              axis.title = element_text(size = 12),
              axis.text.x = element_text(size = 12),
              axis.text.y = element_text(size = 12),
              plot.title = element_text(size = 12),
              legend.title = element_blank(),
              legend.position = "top")
      ggsave(paste0("temp/PR", sprintf("%02d", a),  "/", combo.name, "/plot/PR", sprintf("%02d", a), "_ldev_NonL", "_r",         r,".png"),
             plot = plot.levels.v.target, units="in", width=5, height=5, dpi=200)
      
      # Levels vs Target at PR ----
      print(paste0("Figures for PR ", a, ": Levels vs lab target lab"))
      
      # plotting units
      # units in df.known.target / 1000
      
      # Calculate scaling factor and bounds
      primary_max    <- max(plot.targ.a$QLAB2017/ 1000, na.rm = TRUE)
      primary_min    <- min(df.vis$level_QL/ 1000, na.rm = TRUE)
      
      secondary_max  <- max(plot.targ.a$VLAB2017/ 1000, na.rm = TRUE)
      secondary_min  <- min(df.vis$level_VL/ 1000, na.rm = TRUE)
      
      scaling_factor <- primary_max / secondary_max
      
      # Compute combined y-limits after scaling (to start both at comparable minima)
      primary_lower  <- min(primary_min, secondary_min * scaling_factor, na.rm = TRUE)
      primary_upper  <- max(primary_max, secondary_max * scaling_factor, na.rm = TRUE)
      
      plot.levels.v.lab.target <- ggplot() +
        # QLAB
        geom_point(data = df.vis[2:nrow(df.vis),],    aes(iter, level_QL/ 1000,      color = "QLAB")) +
        geom_line(data = df.vis[2:nrow(df.vis),],     aes(iter, level_QL/ 1000,      color = "QLAB"), linetype = "dashed") +
        geom_point(data = plot.targ.a            ,    aes(-1, QLAB2017/ 1000,    color = "QLAB"), shape = 18, size = 4) +
        
        # VLAB
        geom_point(data = df.vis[2:nrow(df.vis),],    aes(iter, level_VL/ 1000 *  scaling_factor,      color = "VLAB")) +
        geom_line(data = df.vis[2:nrow(df.vis),],     aes(iter, level_VL/ 1000 * scaling_factor,      color = "VLAB"), linetype = "dashed") +
        geom_point(data = plot.targ.a            ,   aes(nrow(df.vis)+1,    VLAB2017/ 1000 * scaling_factor,    color = "VLAB"), shape = 18, size = 4) +
        
        # Define primary and scaled secondary y-axis with dynamic limits
        scale_y_continuous(name = "QLAB (/1,000,000)",
          limits = c(primary_lower, primary_upper),
          sec.axis = sec_axis(~ . / scaling_factor, name = "VLABOR (/1,000,000)")) +
        
        ylab("Labor outcomes") +
        xlab("Iterations") +
        ggtitle("QLABOR, VLAB (scaled), both /1000") +
        scale_color_manual(values = c("QLAB" = "red", "VLAB" = "hotpink")) + # Custom colors
        theme(axis.line.x = element_line(color="black"),
              axis.line.y = element_line(color="black"),
              panel.background = element_blank(),
              axis.title = element_text(size = 12),
              axis.text.x = element_text(size = 12),
              axis.text.y = element_text(size = 12),
              plot.title = element_text(size = 12),
              legend.title = element_blank(),
              legend.position = "top")
      ggsave(paste0("temp/PR", sprintf("%02d", a),  "/", combo.name, "/plot/PR", sprintf("%02d", a), "_ldev_LAB", "_r",         r,".png"),
             plot = plot.levels.v.lab.target, units="in", width=5, height=5, dpi=200)
      
      # PFP, ESLZ and ELAB figure ----
      print(paste0("Figures for PR ", a, ": Estimates"))
      scaling_factor <- 205 / max.ELAB
      
      parm.plot <- ggplot() +
        geom_point(data = df.vis[2:nrow(df.vis),],  aes(iter, AFL_best,    color = "AFL"),     shape = 18, size = 1) +
        geom_line(data =  df.vis[2:nrow(df.vis),],  aes(iter, AFL_best,    color = "AFL"),     linetype="dashed") +
        
        geom_point(data = df.vis[2:nrow(df.vis),],  aes(iter, AFZ_best,     color = "AFZ"),     shape = 18, size = 1) +
        geom_line(data =  df.vis[2:nrow(df.vis),],  aes(iter, AFZ_best,     color = "AFZ"),     linetype="dashed") +
        
        geom_point(data = df.vis[2:nrow(df.vis),],  aes(iter, AFR_best,     color = "AFR"),     shape = 18, size = 1) +
        geom_line(data =  df.vis[2:nrow(df.vis),],  aes(iter, AFR_best,     color = "AFR"),     linetype="dashed") +
        
        geom_point(data = df.vis[2:nrow(df.vis),],   aes(iter, ESLZ_best * scaling_factor, color = "ESLZ"), shape = 17, size = 1) +
        geom_line(data =  df.vis[2:nrow(df.vis),],   aes(iter, ESLZ_best * scaling_factor, color = "ESLZ"), linetype="dashed") +
        
        geom_point(data = df.vis[2:nrow(df.vis),],   aes(iter, ELAB_best * scaling_factor,  color = "ELAB"), shape = 17, size = 1) +
        geom_line(data =  df.vis[2:nrow(df.vis),],   aes(iter, ELAB_best * scaling_factor,  color = "ELAB"), linetype="dashed") +
        
        geom_point(data = subset(data_pr, PR9 == a),  aes(-1, ao_pr,                             color = "AO"),  shape = 7, size = 4) +
        geom_point(data = NULL,                       aes(-1, as.numeric(df.initial[4,1] * scaling_factor), color = "ESLZ"), shape = 7, size = 4) +
        geom_point(data = NULL,                       aes(-1, as.numeric(df.initial[5,1] * scaling_factor), color = "ELAB"), shape = 7, size = 4) +
        
        # Define primary and secondary y-axis labels# Add y-axis: primary and secondary(ESLZ)
        scale_y_continuous(limits = c(0, 205),  # Set limits for the primary y-axis
                           sec.axis = sec_axis(~ . / scaling_factor, name = "ESLZ & ELAB")) +  # Adjust scaling factor for secondary axisscale_y_continuous(name = "AF", sec.axis = sec_axis(~ ., name = "ESLZ")) +
        xlab("Iterations") +
        ggtitle("PFP & ESLZ & ELAB") +
        scale_color_manual(values = c("AFL" = "red", "AFZ" = "purple", "AFR" = "green", "ESLZ" = "blue", "ELAB" = "cyan", "AO" = "black")) + # Custom colors
        theme(axis.line.x = element_line(color="black"),
              axis.line.y = element_line(color="black"),
              panel.background = element_blank(),
              axis.title = element_text(size = 15),
              axis.text.x = element_text(size = 15),
              axis.text.y = element_text(size = 15),
              plot.title = element_text(size = 15),
              legend.title = element_blank(),
              legend.position = "top")
      ggsave(paste0("temp/PR", sprintf("%02d", a),  "/", combo.name, "/plot/PR", sprintf("%02d", a), "_AF_r",         r,".png"),
             plot = parm.plot, units="in", width=5, height=5, dpi=200)
      
      # ---- combine ----
      combined_plot5 <- gridExtra::grid.arrange(dev_level, plot.levels.dev,  plot.levels.v.target, plot.levels.v.lab.target, parm.plot, ncol = 3, nrow = 2)
      ggsave(paste0("temp/PR", sprintf("%02d", a), "/", combo.name, "/plot/PR", sprintf("%02d", a), "_combined_plot5_r", r, ".png"),
             plot = combined_plot5, units="in", width=15, height=10, dpi=200)
      
      print(paste0("Individual run plots done for RG", a, ", RUN ", r))
    } #end r, which returns esti.combo
    names(esti.combo) <- paste0("run_", 1:run)
    esti.combo
  } # end foreach %dopar%, which returns list_c
  
  # Name the combo elements in list_c
  names(list_c) <- df.for.combo
  
  # Save to the main list
  list.est[[a]] <- list_c
  
} # end a
stopCluster(cl)

# save the list of first stage estimates ----
#saveRDS(list.est, file = "temp/1_list_est.rds")