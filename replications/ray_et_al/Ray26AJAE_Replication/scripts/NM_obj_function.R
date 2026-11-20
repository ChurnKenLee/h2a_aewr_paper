# THIS FILE DEFINES THE OBJECTIVE FUNCTION FOR NELDER-MEAD, EQUATION 9, based on NMalgorithm101725_dopar.R

obj = function(input)
{
  # Ensure that each element of input is within its respective bounds
  if (any(input >= upper | input < lower)) # calibrated parameters should be within the defined range
  {
    # dev = Inf
    obj = c(Inf, seq(1, length(l[[k]]$dev)-1))
  } else {
    df.input = data.frame(AFL   = input[1 ],
                          AFZ   = input[2 ],
                          AFR   = input[3 ],
                          ESLZ  = input[4 ],
                          ELAB  = input[5 ])
    
    # AFL, AFZ, AFR as scalars
    AFL_array <- array(df.input[1], dim = c(1,1), dimnames = list(dim1name = "parm", dim2name = "AFL"))
    AFZ_array <- array(df.input[2], dim = c(1,1), dimnames = list(dim1name = "parm", dim2name = "AFZ"))
    AFR_array <- array(df.input[3], dim = c(1,1), dimnames = list(dim1name = "parm", dim2name = "AFR"))
    
    # Turn into (24 × 1) arrays
    ESLZ_array <- array(rep(df.input[4], 24), dim = c(24,1), dimnames = list(dim1name = pr_names, dim2name = "ESLZ"))
    ELAB_array <- array(rep(df.input[5], 24), dim = c(24,1), dimnames = list(dim1name = pr_names, dim2name = "ELAB"))
    
    # Collect into a list
    data.to.cali <- list(AFL  = AFL_array, AFZ  = AFZ_array, AFR  = AFR_array, ESLZ = ESLZ_array, ELAB = ELAB_array)
    
    # Write to HAR file
    write_har(data.to.cali, paste0("shf/ESTIPARM-",combo.name,".har"))
    
    run.simple.fse(a,combo.name)
    result = read_SL4(paste0(          'out/1-PR', sprintf("%02d", a), "-",combo.name, '.sl4'), toLowerCase = F)
    
    sim.pQz  = as.vector(result$p_QAUGLANDz)
    sim.pPz  = as.vector(result$p_PAUGLANDz)
    sim.pQLz = as.vector(result$p_QLABORz)
    sim.pPLz = as.vector(result$p_PLABORz)
    sim.pQZz = as.vector(result$p_QNLABORz)
    sim.pPZz = as.vector(result$p_PNLABORz)
    sim.pQCz = as.vector(result$p_QLANDz)
    sim.pPCz = as.vector(result$p_PLANDz)
    
    sim <- data.frame(sim.pQz  = sim.pQz,
                      sim.pPz  = sim.pPz,
                      sim.pQLz = sim.pQLz,
                      sim.pPLz = sim.pPLz,
                      sim.pQZz = sim.pQZz,
                      sim.pPZz = sim.pPZz,
                      sim.pQCz = sim.pQCz,
                      sim.pPCz = sim.pPCz) %>%
      
      slice(1:605, ) %>%
      mutate(CZ=1:605) %>%
      left_join(CZ2PR) %>%
      left_join(df.levels.target) %>%
      left_join(df.levels.lab.target) %>%
      left_join(shares2010) %>% # used to weight the deviations (p.error)
      filter(PR9==a) %>%
      mutate(num_cz = n()) %>% # no. of CZ in a CZ
      mutate(sim.VCRP2017 = VCRP2002*1.37 * (1 + ( sim.pQz + sim.pPz )/100), #VCRPXXX, VLABXXXX, QLABXXXX, VZXX, QLANDXX, VLAND are all /1000
             sim.QLAB2017 = QLAB2002      * (1 + sim.pQLz/100),
             sim.WAGE2017 = (WAGE02*1.37) * (1 + sim.pPLz/100),
             #sim.VLAB2017 = sim.WAGE2017  * sim.QLAB2017,
             sim.VLAB2017 = WAGE17  * sim.QLAB2017,
             sim.VZ2017   = VZ2002*1.37   * (1 + ( sim.pQZz + sim.pPZz )/100),
             sim.QLAND16  = QLAND01       * (1 + sim.pQCz/100),
             sim.RENT17   = (RENT02*1.37) * (1 + sim.pPCz/100),
             #sim.VLAND16  = sim.RENT17    * sim.QLAND16,
             sim.VLAND16  = RENT17        * sim.QLAND16) %>%
      mutate(p.error.VCRP = abs( ( VCRP2017 - sim.VCRP2017) / VCRP2017 ) ,
             p.error.QLAB = abs( ( QLAB2017 - sim.QLAB2017) / QLAB2017 ) ,
             p.error.WAGE = abs( ( WAGE17   - sim.WAGE2017) / WAGE17   ) ,
             p.error.VLAB = abs( ( VLAB2017 - sim.VLAB2017) / VLAB2017 ),
             p.error.VZ   = abs( ( VZ2017   - sim.VZ2017)   / VZ2017 )  ,
             p.error.QLND = abs( ( QLAND16  - sim.QLAND16)  / QLAND16 )   ,
             p.error.RENT = abs( ( RENT17   - sim.RENT17)   / RENT17  )   ,
             p.error.VLND = abs( ( VLAND16  - sim.VLAND16)  / VLAND16 )) # %>%
    
    d.VCRPa       <-   sum(sim$p.error.VCRP   * sim$vshr_crp , na.rm=T)
    d.QLa         <-   sum(sim$p.error.QLAB   * sim$vshr_lab , na.rm=T)
    d.PLa         <-   sum(sim$p.error.WAGE   * sim$vshr_lab , na.rm=T)
    d.VLa         <-   sum(sim$p.error.VLAB   * sim$vshr_lab , na.rm=T) 
    d.VZa         <-   sum(sim$p.error.VZ     * sim$vshr_nlb , na.rm=T)
    d.QCa         <-   sum(sim$p.error.QLND   * sim$qshr_lnd , na.rm=T)
    d.PCa         <-   sum(sim$p.error.RENT   * sim$vshr_lw  , na.rm=T)
    d.VCa         <-   sum(sim$p.error.VLND   * sim$vshr_lw  , na.rm=T)
    
    sim.VCRPa     <- sum(sim$sim.VCRP2017, na.rm=T)
    sim.QLa       <- sum(sim$sim.QLAB2017, na.rm=T)
    sim.PLa       <- sum(sim$sim.WAGE2017 * sim$vshr_lab, na.rm=T)
    sim.VLa       <- sum(sim$sim.VLAB2017, na.rm=T)
    sim.VZa       <- sum(sim$sim.VZ2017,   na.rm=T)
    sim.QCa       <- sum(sim$sim.QLAND16,  na.rm=T)
    sim.PCa       <- sum(sim$sim.RENT17   * sim$vshr_lw ,   na.rm=T)
    sim.VCa       <- sum(sim$sim.VLAND16,  na.rm=T)
    
    #d.level.a   = (2*d.VCRPa) + d.QLa  + d.VZa  + d.QCa  
    # updated 092225 - include ratio of VLAB:VNLAB - not doing any better
    # updated 092325 - include VLAB and allow calibration of ELAB too, track how QLAB does 
    # d.level.a   = d.VCRPa + d.VLa  + d.VZa  + d.QCa  
    # tryng square!
    d.level.a   <- d.VLa^2 + d.QLa^2  + d.VZa^2  + d.QCa^2  # remove d.VCRPa, explicitly include d.QLa & d.VLa so implicitly d.PLa also 
    d.level.a
    dev<-c(d.level.a = d.level.a, 
          d.VCRPa   = d.VCRPa,   d.VZa   = d.VZa,    
          d.QLa     = d.QLa,     d.PLa   = d.PLa,   d.VLa = d.VLa,   
          d.QCa     = d.QCa,     d.PCa   = d.PCa,   d.VCa = d.VCa,
          sim.VCRPa = sim.VCRPa, sim.VZa = sim.VZa,  
          sim.QLa   = sim.QLa,   sim.PLa = sim.PLa, sim.VLa = sim.VLa, 
          sim.QCa   = sim.QCa,   sim.PCa = sim.PCa, sim.VCa = sim.VCa)
    return(dev) } }