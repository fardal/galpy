
#maf change - ultimately just add qhalo to twopowerlaw

import math as m
import numpy
from scipy import special, optimize
from galpy.util import bovy_conversion
from TwoPowerSphericalPotential import NFWPotential

class FlattenedNFWPotential(NFWPotential):
    """Class that implements a flattened NFW potential.
    Everything is same except for term z/q in potential instead of z.

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\\,\\pi\\,a^3}\\,\\frac{1}{(r/a)\\,(1+r/a)^{2}}

where r^2 = R^2 + (z/q)^2

    """
    def __init__(self,amp=1.,a=1.,normalize=False,
                 conc=None,mvir=None,
                 vo=220.,ro=8.,
                 H=70.,Om=0.3,overdens=200.,wrtcrit=False,
                 q=1.0):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a NFW potential

        INPUT:

           amp - amplitude to be applied to the potential

           a - "scale" (in terms of Ro)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.


           Alternatively, NFW potentials can be initialized using 

              conc= concentration

              mvir= virial mass in 10^12 Msolar

           in which case you also need to supply the following keywords
           
              vo= (220.) velocity unit in km/s

              ro= (8.) length unit in kpc

              H= (default: 70) Hubble constant in km/s/Mpc
           
              Om= (default: 0.3) Omega matter
       
              overdens= (200) overdensity which defines the virial radius

              wrtcrit= (False) if True, the overdensity is wrt the critical density rather than the mean matter density
           
        OUTPUT:

           (none)

        HISTORY:

           2010-07-09 - Written - Bovy (NYU)

           2014-04-03 - Initialization w/ concentration and mass - Bovy (IAS)

        """
#        Potential.__init__(self,amp=amp)
        self.q = q
        NFWPotential.__init__(self,
                 amp=amp, a=a, normalize=normalize,
                 conc=conc, mvir=mvir,
                 vo=vo, ro=ro,
                 H=H, Om=Om, overdens=overdens,
                 wrtcrit=wrtcrit)
#        self.hasC= False  #until I write it
        self.hasC_dxdv= False  #until I write it
        self.hasC= True
#        self.hasC_dxdv= True
        return None

    def _evaluate(self,R,z,phi=0.,t=0.):
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
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2010-07-09 - Started - Bovy (NYU)
        """
        r= numpy.sqrt(R**2.+(z/self.q)**2.)
        return -numpy.log(1.+r/self.a)/r

    def _Rforce(self,R,z,phi=0.,t=0.):
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
        OUTPUT:
           the radial force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        Rz= R**2.+(z/self.q)**2.
        sqrtRz= numpy.sqrt(Rz)
        return R*(1./Rz/(self.a+sqrtRz)-numpy.log(1.+sqrtRz/self.a)/sqrtRz/Rz)

    def _zforce(self,R,z,phi=0.,t=0.):
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
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-09 - Written - Bovy (NYU)
        """
        Rz= R**2.+(z/self.q)**2.
        sqrtRz= numpy.sqrt(Rz)
        return z*(1./Rz/(self.a+sqrtRz)
                  -numpy.log(1.+sqrtRz/self.a)/sqrtRz/Rz) / self.q**2

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the second radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second radial derivative
        HISTORY:
           2011-10-09 - Written - Bovy (IAS)
        """
        zq = z / self.q
        r = numpy.hypot(R, zq)  #r in potential formula, not physical r
        x = r / self.a
        lnx1 = numpy.log(1.+x)
        dphidr = -1./self.a**2 * (lnx1 / x**2 - 1./x/(1.+x))
        ddphidrdr = 1./self.a**3 * (2.*lnx1/x**3 - (2.+3.*x)/x**2/(1.+x)**2)
        return -(zq**2/r**3 * dphidr + R**2/r**2 * ddphidrdr)

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed R,z derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2013-08-28 - Written - Bovy (IAS)
        """
        raise Exception("unimplemented.") 
        Rz= R**2.+z**2.
        sqrtRz= numpy.sqrt(Rz)
        return -R*z*(-4.*Rz-3.*self.a*sqrtRz+3.*(self.a**2.+Rz+2.*self.a*sqrtRz)*numpy.log(1.+sqrtRz/self.a))*Rz**-2.5*(self.a+sqrtRz)**-2.

    def _z2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _z2deriv
        PURPOSE:
           evaluate the second vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2012-07-26 - Written - Bovy (IAS@MPIA)
        """
        zq = z / self.q
        r = numpy.hypot(R, zq)  #r in potential formula, not physical r
        x = r / self.a
        lnx1 = numpy.log(1.+x)
        dphidr = -1./self.a**2 * (lnx1 / x**2 - 1./x/(1.+x))
        ddphidrdr = 1./self.a**3 * (2.*lnx1/x**3 - (2.+3.*x)/x**2/(1.+x)**2)
        return -(R**2/r**3 * dphidr + zq**2/r**2 * ddphidrdr) / self.q**2
        #raise Exception('unimplemented.')


    def _mass(self,R,z=0.,t=0.):
        """
        NAME:
           _mass
        PURPOSE:
           calculate the mass out to a given radius
        INPUT:
           R - radius at which to return the enclosed mass
           z - (don't specify this) vertical height
        OUTPUT:
           mass in natural units
        HISTORY:
           2014-01-29 - Written - Bovy (IAS)
        """
        raise Exception("unimplemented.") #not just z -> z/q substitution 
        if z is None: r= R
        else: r= numpy.sqrt(R**2.+z**2.)
        return numpy.log(1+r/self.a)-r/self.a/(1.+r/self.a)

    # I can just let superclass handle this right?
    # def rvir(self,vo,ro,H=70.,Om=0.3,overdens=200.,wrtcrit=False):
    #     """
    #     NAME:

    #        rvir

    #     PURPOSE:

    #        calculate the virial radius for this density distribution

    #     INPUT:

    #        vo - velocity unit in km/s

    #        ro - length unit in kpc

    #        H= (default: 70) Hubble constant in km/s/Mpc
           
    #        Om= (default: 0.3) Omega matter
       
    #        overdens= (200) overdensity which defines the virial radius

    #        wrtcrit= (False) if True, the overdensity is wrt the critical density rather than the mean matter density
           
    #     OUTPUT:
        
    #        virial radius in natural units
        
    #     HISTORY:

    #        2014-01-29 - Written - Bovy (IAS)

    #     """
    #     return NFWPotential.__init__(self,
    #         vo, ro, H=H, Om=Om, overdens=overdens, wrtcrit=wrtcrit)


