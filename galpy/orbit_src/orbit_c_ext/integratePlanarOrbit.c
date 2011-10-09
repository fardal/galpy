/*
  Wrappers around the C integration code for planar Orbits
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <bovy_symplecticode.h>
#include <bovy_rk.h>
//Potentials
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
/*
  Function Declarations
*/
void evalPlanarRectForce(double, double *, double *,
			 int, struct leapFuncArg *);
void evalPlanarRectDeriv(double, double *, double *,
			 int, struct leapFuncArg *);
double calcPlanarRforce(double, double, double, 
			int, struct leapFuncArg *);
double calcPlanarphiforce(double, double, double, 
			int, struct leapFuncArg *);
/*
  Actual functions
*/
void integratePlanarOrbit(double *yo,
			  int nt, 
			  double *t,
			  int npot,
			  int * pot_type,
			  double * pot_args,
			  double rtol,
			  double atol,
			  double *result,
			  int odeint_type){
  //Set up the forces, first count
  int ii, jj;
  int dim;
  struct leapFuncArg * leapFuncArgs= (struct leapFuncArg *) malloc ( npot * sizeof (struct leapFuncArg) );
  for (ii=0; ii < npot; ii++){
    switch ( *pot_type++ ) {
    case 0: //LogarithmicHaloPotential, 2 arguments
      leapFuncArgs->planarRforce= &LogarithmicHaloPotentialPlanarRforce;
      leapFuncArgs->planarphiforce= &ZeroPlanarForce;
      leapFuncArgs->nargs= 2;
      break;
    case 1: //DehnenBarPotential, 7 arguments
      leapFuncArgs->planarRforce= &DehnenBarPotentialRforce;
      leapFuncArgs->planarphiforce= &DehnenBarPotentialphiforce;
      leapFuncArgs->nargs= 7;
      break;
    case 2: //TransientLogSpiralPotential, 8 arguments
      leapFuncArgs->planarRforce= &TransientLogSpiralPotentialRforce;
      leapFuncArgs->planarphiforce= &TransientLogSpiralPotentialphiforce;
      leapFuncArgs->nargs= 8;
      break;
    case 3: //SteadyLogSpiralPotential, 8 arguments
      leapFuncArgs->planarRforce= &SteadyLogSpiralPotentialRforce;
      leapFuncArgs->planarphiforce= &SteadyLogSpiralPotentialphiforce;
      leapFuncArgs->nargs= 8;
      break;
    }
    leapFuncArgs->args= (double *) malloc( leapFuncArgs->nargs * sizeof(double));
    for (jj=0; jj < leapFuncArgs->nargs; jj++){
      *(leapFuncArgs->args)= *pot_args++;
      leapFuncArgs->args++;
    }
    leapFuncArgs->args-= leapFuncArgs->nargs;
    leapFuncArgs++;
  }
  leapFuncArgs-= npot;
  //Integrate
  void (*odeint_func)(void (*func)(double, double *, double *,
			   int, struct leapFuncArg *),
		      int,
		      double *,
		      int, double *,
		      int, struct leapFuncArg *,
		      double, double,
		      double *);
  void (*odeint_deriv_func)(double, double *, double *,
			    int,struct leapFuncArg *);
  switch ( odeint_type ) {
  case 0: //leapfrog
    odeint_func= &leapfrog;
    odeint_deriv_func= &evalPlanarRectForce;
    dim= 2;
    break;
  case 1: //RK4
    odeint_func= &bovy_rk4;
    odeint_deriv_func= &evalPlanarRectDeriv;
    dim= 4;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    odeint_deriv_func= &evalPlanarRectDeriv;
    dim= 4;
    break;
  }
  odeint_func(odeint_deriv_func,dim,yo,nt,t,npot,leapFuncArgs,rtol,atol,result);
  //Free allocated memory
  for (ii=0; ii < npot; ii++) {
    free(leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= npot;
  free(leapFuncArgs);
  //Done!
}

void evalPlanarRectForce(double t, double *q, double *a,
			 int nargs, struct leapFuncArg * leapFuncArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phiforce;
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcPlanarRforce(R,phi,t,nargs,leapFuncArgs);
  phiforce= calcPlanarphiforce(R,phi,t,nargs,leapFuncArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a--= sinphi*Rforce+1./R*cosphi*phiforce;
}
void evalPlanarRectDeriv(double t, double *q, double *a,
			 int nargs, struct leapFuncArg * leapFuncArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phiforce;
  //first two derivatives are just the velocities
  *a++= *(q+2);
  *a++= *(q+3);
  //Rest is force
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcPlanarRforce(R,phi,t,nargs,leapFuncArgs);
  phiforce= calcPlanarphiforce(R,phi,t,nargs,leapFuncArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a= sinphi*Rforce+1./R*cosphi*phiforce;
}

double calcPlanarRforce(double R, double phi, double t, 
			int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double Rforce= 0.;
  for (ii=0; ii < nargs; ii++){
    Rforce+= leapFuncArgs->planarRforce(R,phi,t,
					leapFuncArgs->nargs,
					leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return Rforce;
}
double calcPlanarphiforce(double R, double phi, double t, 
			  int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double phiforce= 0.;
  for (ii=0; ii < nargs; ii++){
    phiforce+= leapFuncArgs->planarphiforce(R,phi,t,
					    leapFuncArgs->nargs,
					    leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return phiforce;
}