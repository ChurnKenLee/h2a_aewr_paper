 ! Closure
 ! -------
 !                   KEY EXOGENOUS VARIABLES 
! Exogenous  p_INC_PC   ;!per capita income by inc. reg.
! Exogenous  p_POP      ;!population by inc. reg.
! Exogenous  p_QCRPBIOFr ;!crop demand for global biofuel use
! Exogenous  s_QCRPBIOFr ;!crop demand for biofuel use | alternative

! Exogenous  p_P(NFOOD_COMM,REG) ;!price of non-food commodities by inc. reg.
! Exogenous  p_PNCRPFEED ;!price of non-feed inputs used in livestock prod.
! Exogenous  p_PNCRPFOOD ;!price of non-crop inputs used in proc. food prod.

 !                      Livestock Sector
! Exogenous  p_AOCRPFEED   ;!hicks-neutral eff. index in livestock prod.
! Exogenous  p_AFCRPFEEDW  ;!sub-component of feed eff. index by inc. reg. (global)
! Exogenous  p_AFCRPFEEDr  ;!sub-component of feed eff. index by inc. reg. (reg. specific)
! Exogenous  p_AFNCRPFEEDW ;!sub-component of non-feed eff. index by inc. reg. (global)
! Exogenous  p_AFNCRPFEEDr ;!sub-component of non-feed eff. index by inc. reg. (reg. specific)
 
 !                      Processed Food Sector
! Exogenous  p_AOCRPFOOD   ;!hicks-neutral eff. index in proc. food prod.
! Exogenous  p_AFNCRPFOODW ;!sub-component of eff. index of non-crop inputs in proc. food prod. by inc. reg. (global)
! Exogenous  p_AFCRPFOODW  ;!sub-component of eff. index of crops in proc. food prod. by inc. reg. (global)
! Exogenous  p_AFCRPFOODr  ;!sub-component of eff. index of crops in proc. food prod. by inc. reg. (reg. specific)
! Exogenous  p_AFNCRPFOODr ;!sub-component of eff. index of non-crop inputs in proc. food prod. by inc. reg. (reg. specific)

 !                    EFFICIENCY VARIABLES (also exogenous)
Exogenous  p_AOCROPz  ;! efficiency index by LMS for CROP

Exogenous  p_AFNITROa ;
Exogenous  p_AFNLANDa  ;! efficiency index by reg for NLAND
Exogenous  p_AFLABORa  ;
Exogenous  p_AFNLABORa  ;
Exogenous  p_AFLANDWTRa  ;! efficiency index by reg for LAND+WTR

Exogenous  p_AFLABORz  ;
Exogenous  p_AFNLABORz  ;
Exogenous  p_AFLANDWTRz  ;! efficiency index by reg for LAND+WTR
Exogenous  p_AFNITROz  ;
Exogenous  p_AFLANDz  ;
Exogenous  p_AFNLANDz  ;
Exogenous  p_AFWATERz  ;
Exogenous  p_AFWATSGz  ;
Exogenous  p_AFWEQPTz  ;

Exogenous  p_AOCROPr  ;! efficiency index by reg for CROP
Exogenous  p_AFNITROr  ;! efficiency index by reg for NITRO
Exogenous  p_AFLANDr  ;! efficiency index by reg for LAND
Exogenous  p_AFNLANDr  ;! efficiency index by reg for NLAND
Exogenous  p_AFLANDWTRr  ;! efficiency index by reg for LAND+WTR
Exogenous  p_AFWATERr  ;! efficiency index by reg for WATER
Exogenous  p_AFWATSGr  ;! efficiency index by reg for WATSG
Exogenous  p_AFWEQPTr  ;! efficiency index by reg for WEQPT
Exogenous  p_AFLABORr  ;! efficiency index by grid & ltype for LABOR
Exogenous  p_AFNLABORr  ;! efficiency index by grid & ltype for NLABOR
 
 !                   POLICY VARIABLES (also exogenous)
! turn on slack and off p_PCROPr when putting back national model
!Exogenous  p_PCROPr;
Exogenous  s_PCROPz;

!   Land water levers                                             !
Exogenous  s_QWATERz   ;! Slack to control water use by grid !
Exogenous  s_QLANDz     ;! Slack to target corn-soy harvested area by grid (in 1000 ha) !;
Exogenous t_PNLABORz;
Exogenous t_PNITROz;

!Exogenous  s_QLANDWTRz ;
Exogenous  s_QAUGLANDz ;

! labor!
!Exogenous  s_QNLANDz;
!Exogenous  s_PNLANDz;
Exogenous  s_QLABORz;
Exogenous  s_QLABORa;

Exogenous t_PAUGLANDz; ! added 02/20
REST Endogenous;

! hold production constant in all CZ !
Swap p_QAUGLANDz       = s_QAUGLANDz;
Swap p_PAUGLANDz       = t_PAUGLANDz;
!Swap p_PLABORz      = s_QLABORz;
Swap p_QWATERz      = s_QWATERz;
Swap p_PNLABORz     = t_PNLABORz;