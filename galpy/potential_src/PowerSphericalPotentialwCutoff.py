###############################################################################
#   PowerSphericalPotentialwCutoff.py: spherical power-law potential w/ cutoff
#
#                                     amp
#                          rho(r)= ---------   e^{-(r/rc)^2}
#                                   r^\alpha
###############################################################################
import numpy as nu
from scipy import special, integrate
from galpy.potential_src.Potential import Potential, kms_to_kpcGyrDecorator
class PowerSphericalPotentialwCutoff(Potential):
    """Class that implements spherical potentials that are derived from 
    power-law density models

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{r^\\alpha}\\,\\exp\\left(-(r/rc)^2\\right)

    """
    def __init__(self,amp=1.,alpha=1.,rc=1.,normalize=False):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a power-law-density potential

        INPUT:

           amp= amplitude to be applied to the potential (default: 1)

           alpha= inner power

           rc= cut-off radius

           normalize= if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

        OUTPUT:

           (none)

        HISTORY:

           2013-06-28 - Written - Bovy (IAS)

        """
        Potential.__init__(self,amp=amp)
        self.alpha= alpha
        self.rc= rc
        self._scale= self.rc
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)
        self.hasC= True
        self.hasC_dxdv= True
        self._nemo_accname= 'PowSphwCut'

    def _evaluate(self,R,z,**kwargs):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2013-06-28 - Started - Bovy (IAS)
        """
        r= nu.sqrt(R**2.+z**2.)
        #return 2.*nu.pi*self.rc**(3.-self.alpha)/r*(r/self.rc*special.gamma(1.-self.alpha/2.)*special.gammainc(1.-self.alpha/2.,(r/self.rc)**2.)-special.gamma(1.5-self.alpha/2.)*special.gammainc(1.5-self.alpha/2.,(r/self.rc)**2.))
        w = r / self.rc
        return -2.*nu.pi*self.rc**(2.-self.alpha) * (special.gamma(1.-self.alpha/2.)*special.gammaincc(1.-self.alpha/2.,w*w)+special.gamma(1.5-self.alpha/2.)*special.gammainc(1.5-self.alpha/2.,w*w)/w)

    def _Rforce(self,R,z,**kwargs):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the radial force
        HISTORY:
           2013-06-26 - Written - Bovy (IAS)
        """
        r= nu.sqrt(R*R+z*z)
        return -self._mass(r)*R/r**3.

    def _zforce(self,R,z,**kwargs):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the vertical force
        HISTORY:
           2013-06-26 - Written - Bovy (IAS)
        """
        r= nu.sqrt(R*R+z*z)
        return -self._mass(r)*z/r**3.

    def _R2deriv(self,R,z,**kwargs):
        """
        NAME:
           _Rderiv
        PURPOSE:
           evaluate the second radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the second radial derivative
        HISTORY:
           2013-06-28 - Written - Bovy (IAS)
        """
        r= nu.sqrt(R*R+z*z)
        return 4.*nu.pi*r**(-2.-self.alpha)*nu.exp(-(r/self.rc)**2.)*R**2.\
            +self._mass(r)/r**5.*(z**2.-2.*R**2.)

    def _z2deriv(self,R,z,**kwargs):
        """
        NAME:
           _z2deriv
        PURPOSE:
           evaluate the second vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2013-06-28 - Written - Bovy (IAS)
        """
        r= nu.sqrt(R*R+z*z)
        return 4.*nu.pi*r**(-2.-self.alpha)*nu.exp(-(r/self.rc)**2.)*z**2.\
            +self._mass(r)/r**5.*(R**2.-2.*z**2.)

    def _Rzderiv(self,R,z,**kwargs):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed R,z derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2013-08-28 - Written - Bovy (IAS)
        """
        r= nu.sqrt(R*R+z*z)
        return R*z*(4.*nu.pi*r**(-2.-self.alpha)*nu.exp(-(r/self.rc)**2.)
                    -3.*self._mass(r)/r**5.)

    def _dens(self,R,z,**kwargs):
        """
        NAME:
           _dens
        PURPOSE:
           evaluate the density force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the density
        HISTORY:
           2013-06-28 - Written - Bovy (IAS)
        """
        r= nu.sqrt(R**2.+z**2.)
        return 1./r**self.alpha*nu.exp(-(r/self.rc)**2.)

    def _mass(self,R,z=0.,**kwargs):
        """
        NAME:
           _mass
        PURPOSE:
           evaluate the mass within R for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
        OUTPUT:
           the mass enclosed
        HISTORY:
           2013-XX-XX - Written - Bovy (IAS)
        """
        if z is None: r= R
        else: r= nu.sqrt(R**2.+z**2.)
        return 2.*nu.pi*self.rc**(3.-self.alpha)*special.gammainc(1.5-self.alpha/2.,(r/self.rc)**2.)*special.gamma(1.5-self.alpha/2.)

    @kms_to_kpcGyrDecorator
    def _nemo_accpars(self,vo,ro):
        """
        NAME:

           _nemo_accpars

        PURPOSE:

           return the accpars potential parameters for use of this potential with NEMO

        INPUT:

           vo - velocity unit in km/s

           ro - length unit in kpc

        OUTPUT:

           accpars string

        HISTORY:

           2014-12-18 - Written - Bovy (IAS)

        """
        ampl= self._amp*vo**2.*ro**(self.alpha-2.)
        return "0,%s,%s,%s" % (ampl,self.alpha,self.rc*ro)
