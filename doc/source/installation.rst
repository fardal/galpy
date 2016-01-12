Installation
==============

galpy can be installed using pip as::

      > pip install galpy

or to upgrade without upgrading the dependencies::

      > pip install -U --no-deps galpy

Some advanced features require the GNU Scientific Library (GSL; see below). If you want to use these, install the GSL first (or install it later and re-install using the upgrade command above).

The latest updates in galpy can be installed using::
    
    > pip install -U --no-deps git+git://github.com/jobovy/galpy.git#egg=galpy

or::

    > pip install -U --no-deps --install-option="--prefix=~/local" git+git://github.com/jobovy/galpy.git#egg=galpy

for a local installation. The latest updates can also be installed from the source code downloaded from github using the standard python ``setup.py`` installation::

      > python setup.py install

or::

	> python setup.py install --prefix=~/local

for a local installation. A basic installation works with just the
numpy/scipy/matplotlib stack. Some basic tests can be performed by executing::

		       > nosetests -v -w nose/


Installation FAQ
-----------------

What is the required ``numpy`` version?
++++++++++++++++++++++++++++++++++++++++

``galpy`` should mostly work for any relatively recent version of
``numpy``, but some advanced features, including calculating the
normalization of certain distribution functions using Gauss-Legendre
integration require ``numpy`` version 1.7.0 or higher.

How do I install the GSL?
++++++++++++++++++++++++++

Certain advanced features require the GNU Scientific Library (`GSL
<http://www.gnu.org/software/gsl/>`_), with action calculations
requiring version 1.14 or higher. On a Mac, the easiest way to install
the GSL is using `Homebrew <http://brew.sh/>`_ as::

		> brew install gsl --universal

You should be able to check your version  using::

   > gsl-config --version

On Linux distributions with ``apt-get``, the GSL can be installed using::

   apt-get install libgsl0-dev


The ``galpy`` installation fails because of C compilation errors
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``galpy``'s installation can fail due to compilation errors, which look like::

	    error: command 'gcc' failed with exit status 1

or::

	error: command 'clang' failed with exit status 1

or::

	error: command 'cc' failed with exit status 1

This is typically because the compiler cannot locate the GSL header
files or the GSL library. You can tell the installation about where
you've installed the GSL library by defining (for example, when the
GSL was installed under ``/usr``)::

       export CFLAGS=-I/usr/include
       export LDFLAGS=-L/usr/lib

or::

	setenv CFLAGS -I/usr/include
	setenv LDFLAGS -L/usr/lib

depending on your shell type (change the actual path to the include
and lib directories that have the gsl directory). If you already have
``CFLAGS`` and ``LDFLAGS`` defined you just have to add the
``'-I/usr/include'`` and ``'-L/usr/lib'`` to them.

I'm having issues with OpenMP
+++++++++++++++++++++++++++++++

galpy uses `OpenMP <http://www.openmp.org/>`_ to parallelize various
of the computations done in C. galpy can be installed without OpenMP
by specifying the option ``--no-openmp`` when running the ``python
setup.py`` commands above::

	   python setup.py install --no-openmp

or when using pip as follows::

    > pip install -U --no-deps --install-option="--no-openmp" git+git://github.com/jobovy/galpy.git#egg=galpy 

or::

    > pip install -U --no-deps --install-option="--prefix=~/local" --install-option="--no-openmp" git+git://github.com/jobovy/galpy.git#egg=galpy 

for a local installation. This might be useful if one is using the
``clang`` compiler, which is the new default on macs with OS X (>=
10.8), but does not support OpenMP. ``clang`` might lead to errors in the
installation of galpy such as::

  ld: library not found for -lgomp

  clang: error: linker command failed with exit code 1 (use -v to see invocation)

If you get these errors, you can use the commands given above to
install without OpenMP, or specify to use ``gcc`` by specifying the
``CC`` and ``LDSHARED`` environment variables to use ``gcc``. Note
that ``clang`` does not seem to have this issue anymore in more recent
versions, but it still does not support ``OpenMP``.