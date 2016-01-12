.. galpy documentation master file, created by
   sphinx-quickstart on Sun Jul 11 15:58:27 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. ifconfig:: not_on_rtd

   .. WARNING:: You are looking at the rarely updated, GitHub version of this documentation, **please go to** `galpy.readthedocs.org <http://galpy.readthedocs.org>`_ **for the latest documentation**.

Welcome to galpy's documentation
=================================

galpy is a Python 2 and 3 package for galactic dynamics. It supports
orbit integration in a variety of potentials, evaluating and sampling
various distribution functions, and the calculation of action-angle
coordinates for all static potentials.

Quick-start guide
-----------------

.. toctree::
   :maxdepth: 2

   installation.rst

   getting_started.rst

   potential.rst

   basic_df.rst

   orbit.rst

   actionAngle.rst

   diskdf.rst

Tutorials
---------

.. toctree::
   :maxdepth: 2

   streamdf.rst

Library reference
-----------------

.. toctree::
   :maxdepth: 2

   reference/orbit.rst

   reference/potential.rst

   reference/aa.rst

   reference/df.rst

   reference/util.rst


Acknowledging galpy
--------------------

If you use galpy in a publication, please cite the following paper

* *galpy: A Python Library for Galactic Dynamics*, Jo Bovy (2015), *Astrophys. J. Supp.*, **216**, 29 (`arXiv/1412.3451 <http://arxiv.org/abs/1412.3451>`_).

and link to ``http://github.com/jobovy/galpy``. Some of the code's
functionality is introduced in separate papers (like
``galpy.df.streamdf`` and ``galpy.df.streamgapdf``, see above), so
please also cite those papers when using these functions. Please also
send me a reference to the paper or send a pull request including your
paper in the list of galpy papers on this page (this page is at
doc/source/index.rst). Thanks!

When using the ``galpy.actionAngle.actionAngleAdiabatic`` and ``galpy.actionAngle.actionAngleStaeckel`` modules, please cite `2013ApJ...779..115B <http://adsabs.harvard.edu/abs/2013ApJ...779..115B>`_ in addition to the papers describing the algorithm used. When using ``galpy.actionAngle.actionAngleIsochroneApprox``, please cite `2014ApJ...795...95B <http://adsabs.harvard.edu/abs/2014ApJ...795...95B>`_, which introduced this technique.


Papers using galpy
--------------------

``galpy`` is described in detail in this publication:

* *galpy: A Python Library for Galactic Dynamics*, Jo Bovy (2015), *Astrophys. J. Supp.*, **216**, 29 (`arXiv/1412.3451 <http://arxiv.org/abs/1412.3451>`_).

The following is a list of publications using ``galpy``; please let me (bovy -at- ias.edu) know if you make use of ``galpy`` in a publication.

#. *Tracing the Hercules stream around the Galaxy*, Jo Bovy (2010), *Astrophys. J.* **725**, 1676 (`2010ApJ...725.1676B <http://adsabs.harvard.edu/abs/2010ApJ...725.1676B>`_): 
  	   Uses what later became the orbit integration routines and Dehnen and Shu disk distribution functions.
#. *The spatial structure of mono-abundance sub-populations of the Milky Way disk*, Jo Bovy, Hans-Walter Rix, Chao Liu, et al. (2012), *Astrophys. J.* **753**, 148 (`2012ApJ...753..148B <http://adsabs.harvard.edu/abs/2012ApJ...753..148B>`_):
       Employs galpy orbit integration in ``galpy.potential.MWPotential`` to characterize the orbits in the SEGUE G dwarf sample.
#. *On the local dark matter density*, Jo Bovy & Scott Tremaine (2012), *Astrophys. J.* **756**, 89 (`2012ApJ...756...89B <http://adsabs.harvard.edu/abs/2012ApJ...756...89B>`_):
      Uses ``galpy.potential`` force and density routines to characterize the difference between the vertical force and the surface density at large heights above the MW midplane.
#. *The Milky Way's circular velocity curve between 4 and 14 kpc from APOGEE data*, Jo Bovy, Carlos Allende Prieto, Timothy C. Beers, et al. (2012), *Astrophys. J.* **759**, 131 (`2012ApJ...759..131B <http://adsabs.harvard.edu/abs/2012ApJ...759..131B>`_):
       Utilizes the Dehnen distribution function to inform a simple model of the velocity distribution of APOGEE stars in the Milky Way disk and to create mock data.
#. *A direct dynamical measurement of the Milky Way's disk surface density profile, disk scale length, and dark matter profile at 4 kpc < R < 9 kpc*, Jo Bovy & Hans-Walter Rix (2013), *Astrophys. J.* **779**, 115 (`2013ApJ...779..115B <http://adsabs.harvard.edu/abs/2013ApJ...779..115B>`_):
     Makes use of potential models, the adiabatic and Staeckel actionAngle modules, and the quasiisothermal DF to model the dynamics of the SEGUE G dwarf sample in mono-abundance bins.
#. *The peculiar pulsar population of the central parsec*, Jason Dexter & Ryan M. O'Leary (2013), *Astrophys. J. Lett.*, **783**, L7 (`2014ApJ...783L...7D <http://adsabs.harvard.edu/abs/2014ApJ...783L...7D>`_):
     Uses galpy for orbit integration of pulsars kicked out of the Galactic center.
#. *Chemodynamics of the Milky Way. I. The first year of APOGEE data*, Friedrich Anders, Christina Chiappini, Basilio X. Santiago, et al. (2013), *Astron. & Astrophys.*, **564**, A115 (`2014A&A...564A.115A <http://adsabs.harvard.edu/abs/2014A%26A...564A.115A>`_):
  		 Employs galpy to perform orbit integrations in ``galpy.potential.MWPotential`` to characterize the orbits of stars in the APOGEE sample.

#. *Dynamical modeling of tidal streams*, Jo Bovy (2014), *Astrophys. J.*, **795**, 95 (`2014ApJ...795...95B <http://adsabs.harvard.edu/abs/2014ApJ...795...95B>`_):
    Introduces ``galpy.df.streamdf`` and ``galpy.actionAngle.actionAngleIsochroneApprox`` for modeling tidal streams using simple models formulated in action-angle space (see the tutorial above).
#. *The Milky Way Tomography with SDSS. V. Mapping the Dark Matter Halo*, Sarah R. Loebman, Zeljko Ivezic Thomas R. Quinn, Jo Bovy, Charlotte R. Christensen, Mario Juric, Rok Roskar, Alyson M. Brooks, & Fabio Governato (2014), *Astrophys. J.*, **794**, 151 (`2014ApJ...794..151L <http://adsabs.harvard.edu/abs/2014ApJ...794..151L>`_):
    Uses ``galpy.potential`` functions to calculate the acceleration field of the best-fit potential in Bovy & Rix (2013) above.
#. *The power spectrum of the Milky Way: Velocity fluctuations in the Galactic disk*, Jo Bovy, Jonathan C. Bird, Ana E. Garcia Perez, Steven M. Majewski, David L. Nidever, & Gail Zasowski (2015), *Astrophys. J.*, **800**, 83 (`arXiv/1410.8135 <http://arxiv.org/abs/1410.8135>`_):
    Uses ``galpy.df.evolveddiskdf`` to calculate the mean non-axisymmetric velocity field due to different non-axisymmetric perturbations and compares it to APOGEE data.
#. *The LMC geometry and outer stellar populations from early DES data*, Eduardo Balbinot, B. X. Santiago, L. Girardi, et al. (2015), *Mon. Not. Roy. Astron. Soc.*, **449**, 1129 (`arXiv/1502.05050 <http://arxiv.org/abs/1502.05050>`_):
    Employs ``galpy.potential.MWPotential`` as a mass model for the Milky Way to constrain the mass of the LMC.
#. *Generation of mock tidal streams*, Mark A. Fardal, Shuiyao Huang, & Martin D. Weinberg (2014), *Mon. Not. Roy. Astron. Soc.*, **452**, 301 (`arXiv/1410.1861 <http://arxiv.org/abs/1410.1861>`_):
    Uses ``galpy.potential`` and ``galpy.orbit`` for orbit integration in creating a *particle-spray* model for tidal streams.
#. *The nature and orbit of the Ophiuchus stream*, Branimir Sesar, Jo Bovy, Edouard J. Bernard, et al. (2015), *Astrophys. J.*, **809**, 59 (`2015ApJ...809...59S <http://adsabs.harvard.edu/abs/2015ApJ...809...59S>`_):
    Uses the ``Orbit.fit`` routine in ``galpy.orbit`` to fit the orbit of the Ophiuchus stream to newly obtained observational data and the routines in ``galpy.df.streamdf`` to model the creation of the stream.
#. *Young Pulsars and the Galactic Center GeV Gamma-ray Excess*, Ryan M. O’Leary, Matthew D. Kistler, Matthew Kerr, & Jason Dexter (2015), *Phys. Rev. Lett.*, submitted (`arXiv/1504.02477 <http://arxiv.org/abs/1504.02477>`_):
     Uses galpy orbit integration  and ``galpy.potential.MWPotential2014`` as part of a Monte Carlo simulation of the Galactic young-pulsar population.
#. *Phase Wrapping of Epicyclic Perturbations in the Wobbly Galaxy*, Alexander de la Vega, Alice C. Quillen, Jeffrey L. Carlin, Sukanya Chakrabarti, & Elena D'Onghia (2015), *Mon. Not. Roy. Astron. Soc.*, **454**, 933 (`2015MNRAS.454..933D <http://adsabs.harvard.edu/abs/2015MNRAS.454..933D>`_):
     Employs galpy orbit integration, ``galpy.potential`` functions, and ``galpy.potential.MWPotential2014`` to investigate epicyclic motions induced by the pericentric passage of a large dwarf galaxy and how these motions give rise to streaming motions in the vertical velocities of Milky Way disk stars.
#. *Chemistry of the Most Metal-poor Stars in the Bulge and the z ≳ 10 Universe*, Andrew R. Casey & Kevin C. Schlaufman (2015), *Astrophys. J.*, **809**, 110 (`2015ApJ...809..110C <http://adsabs.harvard.edu/abs/2015ApJ...809..110C>`_):
     This paper employs galpy orbit integration in ``MWPotential`` to characterize the orbits of three very metal-poor stars in the Galactic bulge.
#. *The Phoenix stream: a cold stream in the Southern hemisphere*, E. Balbinot, B. Yanny, T. S. Li, et al. (2015), *Astrophys. J.*, submitted (`arXiv/1509.04283 <http://arxiv.org/abs/1509.04283>`_).
#. *Discovery of a Stellar Overdensity in Eridanus-Phoenix in the Dark Energy Survey*, T. S. Li, E. Balbinot, N. Mondrik, et al. (2015), *Astrophys. J.*, submitted (`arXiv/1509.04296 <http://arxiv.org/abs/1509.04296>`_):
     Both of these papers use galpy orbit integration to integrate the orbit of NGC 1261 to investigate a possible association of this cluster with the newly discovered Phoenix stream and Eridanus-Phoenix overdensity.
#. *The Proper Motion of Palomar 5*, T. K. Fritz & N. Kallivayalil (2015), *Astrophys. J.*, **811**, 123 (`2015ApJ...811..123F <http://adsabs.harvard.edu/abs/2015ApJ...811..123F>`_):
     This paper makes use of the ``galpy.df.streamdf`` model for tidal streams to constrain the Milky Way's gravitational potential using the kinematics of the Palomar 5 cluster and stream.
#. *Spiral- and bar-driven peculiar velocities in Milky Way-sized galaxy simulations*, Robert J. J. Grand, Jo Bovy, Daisuke Kawata, Jason A. S. Hunt, Benoit Famaey, Arnaud Siebert, Giacomo Monari, & Mark Cropper (2015), *Mon. Not. Roy. Astron. Soc.*, **453**, 1867 (`2015MNRAS.453.1867G <http://adsabs.harvard.edu/abs/2015MNRAS.453.1867G>`_):
     Uses ``galpy.df.evolveddiskdf`` to calculate the mean non-axisymmetric velo\city field due to the bar in different parts of the Milky Way.
#. *Vertical kinematics of the thick disc at 4.5 ≲ R ≲ 9.5 kpc*, Kohei Hattori & Gerard Gilmore (2015), *Mon. Not. Roy. Astron. Soc.*, **454**, 649 (`2015MNRAS.454..649H <http://adsabs.harvard.edu/abs/2015MNRAS.454..649H>`_):
     This paper uses ``galpy.potential`` functions to set up a realistic Milky-Way potential for investigating the kinematics of stars in the thick disk.
#. *Local Stellar Kinematics from RAVE data - VI. Metallicity Gradients Based on the F-G Main-sequence Stars*, O. Plevne, T. Ak, S. Karaali, S. Bilir, S. Ak, Z. F. Bostanci (2015), *Pub. Astron. Soc. Aus.*, **32**, 43 (`2015PASA...32...43P <http://adsabs.harvard.edu/abs/2015PASA...32...43P>`_):
     This paper employs galpy orbit integration in ``MWPotential2014`` to calculate orbital parameters for a sample of RAVE F and G dwarfs to investigate the metallicity gradient in the Milky Way.
#. *Dynamics of stream-subhalo interactions*, Jason L. Sanders, Jo Bovy, & Denis Erkal (2015), *Mon. Not. Roy. Astron. Soc.*, submitted (`arXiv/1510.03426 <http://arxiv.org/abs/1510.03426>`_):
     Uses and extends ``galpy.df.streamdf`` to build a generative model of the dynamical effect of sub-halo impacts on tidal streams. This new functionality is contained in ``galpy.df.streamgapdf``, a subclass of ``galpy.df.streamdf``, and can be used to efficiently model the effect of impacts on the present-day structure of streams in position and velocity space.
#. *Extremely metal-poor stars from the cosmic dawn in the bulge of the Milky Way*, L. M. Howes, A. R. Casey, M. Asplund et al. (2015), *Nature*, in press (`arXiv/1511.03930 <http://arxiv.org/abs/1511.03930>`_):
     Employs galpy orbit integration in ``MWPotential2014`` to characterize the orbits of a sample of extremely metal-poor stars found in the bulge of the Milky Way. This analysis demonstrates that the orbits of these metal-poor stars are always close to the center of the Milky Way and that these stars are therefore true bulge stars rather than halo stars passing through the bulge.
#. *Detecting the disruption of dark-matter halos with stellar streams*, Jo Bovy (2015), *Phys. Rev. Lett.*, submitted (`arXiv/1512.00452 <http://arxiv.org/abs/1512.00452>`_):
     Uses galpy functions in ``galpy.df`` to estimate the velocity kick imparted by a disrupting dark-matter halo on a stellar stream. Also employs ``galpy.orbit`` integration and ``galpy.actionAngle`` functions to analyze *N*-body simulations of such an interaction.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

