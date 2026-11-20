# Focused numerical checks for new exploratory inference, using synthetic data.
source("explorations/bite/cross_design_helpers.R")
set.seed(47)
d<-expand.grid(unit=seq_len(32),year=2011:2018)
d$county_fips<-sprintf("%05d",d$unit)
d$aewr_region_id<-as.character(1+(d$unit-1)%/%4)
d$w<-rnorm(nrow(d));d$Z<-rnorm(nrow(d));d$ZE<-rnorm(nrow(d))
d$A<-d$Z+.3*d$ZE+rnorm(nrow(d));d$X<-.2*d$Z+2*d$ZE+rnorm(nrow(d))
d$y<-.2*d$A-.1*d$X+.3*d$w+rnorm(nrow(d))+.1*d$unit+.2*d$year
for(iv in c(FALSE,TRUE)) {
  m<-if(iv) bite_fit("y","w","county_fips + year",d,"A + X ~ Z + ZE") else bite_fit("y","A + X + w","county_fips + year",d)
  term<-if(iv) "fit_A" else "A"
  B<-199; seed<-842
  fast<-bite_wild(m,d,term,B=B,seed=seed)
  # Independent literal restricted bootstrap: refit full raw-row FE model for
  # each artificial outcome, including the global year effects.
  restricted<-if(iv) bite_fit("y","w","county_fips + year",d,"X ~ Z + ZE") else bite_fit("y","X + w","county_fips + year",d)
  u0<-resid(restricted);f0<-d$y-u0
  cl<-factor(d$aewr_region_id); G<-nlevels(cl)
  set.seed(seed); W<-matrix(sample(c(-sqrt(1.5),-1,-sqrt(.5),sqrt(.5),1,sqrt(1.5)),G*B,TRUE),G)
  observed<-coeftable(m)[term,3]
  ts<-vapply(seq_len(B),function(i){
    dd<-d;dd$y<-f0+u0*W[as.integer(cl),i]
    mm<-if(iv) bite_fit("y","w","county_fips + year",dd,"A + X ~ Z + ZE") else bite_fit("y","A + X + w","county_fips + year",dd)
    coeftable(mm)[term,3]
  },numeric(1))
  literal<-(1+sum(abs(ts)>=abs(observed)))/(B+1)
  stopifnot(abs(literal-fast$wild_p)<1e-12)
}
r<-bite_residualize(d,c("A","X","Z","ZE"),"w")
X<-r[,c("A","X")];Z<-r[,c("Z","ZE")];cl<-d$aewr_region_id
kp<-bite_kp(X,Z,cl)
stopifnot(is.finite(kp),kp>0,abs(kp-bite_kp(X[,2:1],Z[,2:1],cl))<1e-8,
  abs(kp-bite_kp(sweep(X,2,c(10,.2),"*"),sweep(Z,2,c(.1,3),"*"),cl))<1e-8,
  all(is.finite(bite_sw(X,Z,cl))))
x<-X[,1,drop=FALSE];z<-Z[,1,drop=FALSE];pi<-sum(x*z)/sum(z^2);u<-x-z*pi
F<-pi^2/(sum(rowsum(drop(z*u),cl)^2)/sum(z^2)^2*n_distinct(cl)/(n_distinct(cl)-1))
stopifnot(abs(F-bite_kp(x,z,cl))<1e-8)
cat("PASS: OLS and IV restricted bootstrap match literal FE refits; KP scalar Wald, permutation/scaling invariance; finite SW.\n")
