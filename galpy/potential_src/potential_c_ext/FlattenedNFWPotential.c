#include <math.h>
#include <galpy_potentials.h>
//FlattenedNFWPotential
//2 arguments: amp, a
double FlattenedNFWPotentialEval(double R,double Z, double phi,
			  double t,
			struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double q= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z/q/q,0.5);
  return - amp * log ( 1. + sqrtRz / a ) / sqrtRz;
}
double FlattenedNFWPotentialRforce(double R,double Z, double phi,
			  double t,
			  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double q= *args;
  //Calculate Rforce
  double Rz= R*R+Z*Z/q/q;
  double sqrtRz= pow(Rz,0.5);
  return amp * R * (1. / Rz / (a + sqrtRz)-log(1.+sqrtRz / a)/sqrtRz/Rz);
}
double FlattenedNFWPotentialPlanarRforce(double R,double phi,
					    double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double q= *args;
  //Calculate Rforce
  exit(-1);  //not implemented
  return amp / R * (1. / (a + R)-log(1.+ R / a)/ R);
}
double FlattenedNFWPotentialzforce(double R,double Z,double phi,
			  double t,
			  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double q= *args;
  //Calculate Rforce
  double Rz= R*R+Z*Z/q/q;
  double sqrtRz= pow(Rz,0.5);
  return amp * Z * (1. / Rz / (a + sqrtRz)-log(1.+sqrtRz / a)/sqrtRz/Rz) / (q*q);
}
double FlattenedNFWPotentialPlanarR2deriv(double R,double phi,
				 double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double q= *args;
  //Calculate R2deriv
  double aR= a+R;
  double aR2= aR*aR;
  exit(-1);   //not implemented
  return amp * (((R*(2.*a+3.*R))-2.*aR2*log(1.+R/a))/R/R/R/aR2);
}
