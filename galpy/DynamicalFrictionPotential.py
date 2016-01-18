
'this needs documentation - maf'

#this is an early implementation of dynamical friction using
#the random-keyword interface to galpy potential routines.

import numpy as np
from scipy import special
from scipy import interpolate
from scipy import integrate
#from Potential import Potential
from galpy.potential import Potential, evaluateRforces, evaluatezforces, \
    evaluateDensities

FOURPI = 4.*np.pi
_INVSQRTTWO= 1./np.sqrt(2.)
_INVSQRTPI= 1./np.sqrt(np.pi)



# def isothermalrhor(r):
#     return 1./r**2.


# def isothermalsigmar(r):
#     return _INVSQRTTWO


class ChandrasekharDynamicalFriction(Potential):
    """Class that implements the Chandrasekhar dynamical friction."""
    def __init__(self, 
                 amp=1., 
                 lnLambda='Petts16', 
                 Ms=None, 
                 rsat=None, 
                 rhor=None,
                 sigmar=None,
                 rhoscalelength=None,
                 satsize=None):
#                 rhor=isothermalrhor,
#                 sigmar=isothermalsigmar):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a Chandrasekhar dynamical friction force.
           Assumes spherical potential.  Can apply for nonspherical but
               will be approximate.  As will any dynamical friction, really.
           Requires galpy version with arbitrary keywords in equations of motion.
           Currently, input velocity is actually dR/dt, dphi/dt, dz/dt.
        INPUT:
           amp - amplitude to be applied to the potential (default: 1)
           Ms - satellite mass (assuming G=1)
           rsat - satellite half-mass radius (system units)
           lnLambda - Coulomb integral.  Default is 'Petts16', which uses
                equation 5 of Petts et al 2016. May also give a float constant.
           rhor - function that gives SPHERICALIZED density as a function of r
                 (as function of one variable, like lambda x: pot.dens(x,0)
           sigmar - function that gives velocity dispersion as a function of r
           rhoscalelength - function that gives 1/(-dln(rho)/dr) as function of r
           (these last three functions can be approximate.  rhoscalelength only
            used with lnLambda=Petts16.)
        OUTPUT:
           (none)
        HISTORY:
           2011-12-26 - Started - Bovy (NYU)
           2016-01-14 - Continued - MAF
        """
        Potential.__init__(self,amp=amp)
        self._lnLambda= lnLambda
        self._ms= Ms
        self._satsize = rsat
        self._rhor= rhor
        assert rhor is not None
        assert sigmar is not None
        if (lnLambda=='Petts16'):
            assert rhoscalelength is not None
        else:
            assert type(lnLambda)==type(1.)
        self._rhoscalelength = rhoscalelength
        self._sigmar= sigmar
        self.isDynamicalFriction = True  #actually just test attribute existence
        return None

    def _evaluate(self,R,z,phi=0.,t=0.,dR=0,dphi=0,v=None):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
           dR=, dphi=
           v= current velocity as [vR,vphi,vz]
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2010-04-02 - Started - Bovy (NYU)
           2010-04-30 - Adapted for R,z - Bovy (NYU)
        """
        raise Exception('Dynamical friction not implemented as potential field.')
        if dR == 0 and dphi == 0:
            return 0.
        elif dR == 1 and dphi == 1:
            return self._Rforce(R,z,phi=phi,t=t,v=v)
        elif dR == 0 and dphi == 1:
            return self._phiforce(R,z,phi=phi,t=t,v=v)
        else:
            raise NotImplementedError("'_evaluate' not implemented for ChandrasekharDynamicalFriction")
            pass

    def _Rforce(self, R, z, v=None, **kwargs):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
           v - current coordinate velocity, required
        OUTPUT:
           the radial force
        HISTORY:
        """
        ffac = self.forcefac(R, z, v)
        return ffac*v[0]

    def _phiforce(self, R, z, v=None, **kwargs):
        """
        NAME:
           _phiforce
        PURPOSE:
           evaluate the azimuthal force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
           v - current coordinate velocity, required
        OUTPUT:
           the azimuthal force (note this is dPhi/dphi, the "coordinate" or
           generalized force, so picks up an extra factor of R)
        HISTORY:
        """
        vT= v[1]*R
        ffac = self.forcefac(R, z, v)
        return ffac*vT*R

    def _zforce(self, R, z, v=None, **kwargs):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
           v - current coordinate velocity, required
        OUTPUT:
           the vertical force
        HISTORY:
        """
        ffac = self.forcefac(R, z, v)
        return ffac*v[2]

    def forcefac(self, R, z, v):
        'compute common factor in force'
        #for speed, could store last arg. values and just check whether same.
        #Jo has noted that this could be stored in a hash.
        r = np.sqrt(R**2.+z**2.)
        vT = v[1]*R
        vs = np.sqrt(v[0]**2.+vT**2+v[2]**2.)
        if (self._lnLambda=='Petts16'):
            bmax = min(self._rhoscalelength(r), r)
            bmin = max(self._satsize, self._ms/vs**2) #assumes G=1
            lnLambda = np.log(max(1.,bmax/bmin))
        else:
            lnLambda = self._lnLambda
        X = vs*_INVSQRTTWO/self._sigmar(r)
        Xfactor = special.erf(X)-2.*X*_INVSQRTPI*np.exp(-X**2.)
        fac = -FOURPI*lnLambda*self._ms*self._rhor(r)/vs**3.*Xfactor
        return fac


class SphericalHost:
    'Helper class for dynamical friction calculations'
    def __init__(self, pot, rmin, rmax):
        """pot should be a galpy potential or list thereof.
        rmin, rmax should span the possible range of radii that will
        be queried.  rmax should also be large enough that the contribution
        to the velocity dispersion from larger radii can be estimated easily
        from extrapolation of local power-law behavior.  e.g., well outside 
        scale radius of an NFW potential.
        """
        assert(rmax > rmin)
        assert(rmin > 0.)

        #set up radius
        n = max(1000, int(np.log(rmax/rmin)*100))
        logr = np.linspace(np.log(rmin), np.log(rmax), n)
        r = np.exp(logr)
        self._logr = logr

        #set up force
        planeforce = evaluateRforces(r,0.,pot)
        poleforce = evaluatezforces(0.,r,pot)
#        rforce = 0.5 * (pot.rforce(r,0.) + pot.rforce(0.,r)) #sphericalized
        rforce = 0.5 * (planeforce + poleforce) #sphericalized

        #get sphericalized density, either directly or inferred from force
        #should there be sphericaldensity attribute for potential??
        if (np.abs(planeforce-poleforce) / rforce > 2.e-5).any():
            spherical = False
            mass = -r*r * rforce
            mspline = interpolate.InterpolatedUnivariateSpline(
                logr, np.log(mass), k=3)
            dmdr = mass / r * mspline.derivative()(logr)
            rho = dmdr / (4.*np.pi * r*r)
            self._logrhospline = interpolate.InterpolatedUnivariateSpline(
                logr, np.log(rho), k=1)
            self.dens = self.interpdens
        else:
            #spherical
            spherical = True
            #rho = pot.dens(r,0)
            rho = evaluateDensities(r,0.,pot)
            self.pot = pot
            self.dens = self.evaldens

        #start with integral to infinity 
        pintegrand = interpolate.InterpolatedUnivariateSpline(
            logr, np.log(-r * rho * rforce), k=3)
        logrmax = logr[-1]
        q0 = np.exp(pintegrand(logrmax))
        qpower = pintegrand.derivative()(logrmax)
        Pext = -q0 / qpower

        #then compute cumulative integral over array range
        q = -r * rho * rforce
        press = r*0
        press[::-1] = -integrate.cumtrapz(q[::-1], logr[::-1], initial=0.)
        press += Pext
        sigma_r = np.sqrt(press / rho)

        #set up required spline
        self._sigspline = interpolate.InterpolatedUnivariateSpline(
            logr, sigma_r, k=2) #lower order since possibly noisy

        #also set up density derivative in case needed for scale length
        if spherical:
            logrhospline = interpolate.InterpolatedUnivariateSpline(
                logr, np.log(rho), k=1)
            self._densderivspline = logrhospline.derivative()
        else:
            #already have this stored
            self._densderivspline = self._logrhospline.derivative()

    def evaldens(self, r):
        #return self.pot.dens(r,0)
        return evaluateDensities(r,0.,self.pot)

    def interpdens(self, r):
        return np.exp(self._logrhospline(np.log(r)))

    def sigma_r(self, r):
        return self._sigspline(np.log(r))
    
    def rhoscalelength(self, r):
        'Return 1/|-dln(rho)/dr|'
        return r / (-self._densderivspline(r))


