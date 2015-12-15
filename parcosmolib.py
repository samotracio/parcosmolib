# -*- coding: utf-8 -*-
"""
Paralell Wrappers for Cosmology Functions in Astropy
----------------------------------------------------

Defines wrappers to run **in paralell** various functions (e.g. comoving_distance) 
from astropy.cosmology package.

It does so by slicing the input data (such as a large array of redshifts) and
distributing the chunks among all available cores using the ``multiprocessing``
package.

Worker functions are passed to the paralell wrapper ``run_anyfz()``. 
Available workers are:

   lumdis_worker, comdis_worker, angular_diameter_distance_worker,
   comoving_volume_worker,differential_comoving_volume_worker,
   distmod_worker, arcsec_per_kpc_comoving_worker, arcsec_per_kpc_proper_worker

Basic Usage
-----------
::

    z = np.random.random(200000)
    import parcosmolib as par
    from astropy.cosmology import FlatLambdaCDM
    cosmo=FlatLambdaCDM(H0=70,Om0=0.3)
    par.run_anyfz(par.lumdis_worker,z,cosmo)

Note that astropy.cosmology is probably more efficient for small samples. However,
this wrapper is still faster for large samples (~3.5x for the example above in
a quad-core Core i7)

Dependencies
------------
1. numpy
2. multiprocessing
3. astropy

Credits
-------
Emilio Donoso (this package)
"""

import numpy as np
from multiprocessing import Pool, cpu_count

#==============================================================================
# WORKER FUNCTIONS
# These are worker functions that can be passed to the wrapper run_anyfz()
# If you need another function from astropy.cosmolgy, just add it here
#==============================================================================
def comdis_worker(z,cosmo):
    """ Worker function for comoving distance (in Mpc). Note that uses passed cosmology
    and strips any units. For more info see astropy.cosmology documentation.
    """
    return cosmo.comoving_distance(z).value

def lumdis_worker(z,cosmo):
    """ Worker function for luminosity distance (in Mpc). Note that uses passed cosmology
    and strips any units. For more info see astropy.cosmology documentation.
    """
    return cosmo.luminosity_distance(z).value

def angular_diameter_distance_worker(z,cosmo):
    """ Worker function for angular diameter distance (in Mpc). Note that uses 
    passed cosmology and strips any units. For more info see astropy.cosmology 
    documentation.
    """
    return cosmo.angular_diameter_distance(z).value

def comoving_volume_worker(z,cosmo):
    """ Worker function for luminosity distance (in Mpc^3). Note that uses passed cosmology
    and strips any units. For more info see astropy.cosmology documentation.
    """
    return cosmo.comoving_volume(z).value

def differential_comoving_volume_worker(z,cosmo):
    """ Worker function for differential comoving volume (Mpc^3/sr). Note that 
    uses passed cosmology and strips any units. For more info see 
    astropy.cosmology documentation.
    """
    return cosmo.differential_comoving_volume(z).value

def distmod_worker(z,cosmo):
    """ Worker function for distance modulus (in mag). Note that uses passed 
    cosmology and strips any units. For more info see astropy.cosmology 
    documentation.
    """
    return cosmo.distmod(z).value

def arcsec_per_kpc_comoving_worker(z,cosmo):
    """ Worker function for angular separation per comoving kpc at z. Note 
    that uses passed cosmology and strips any units. For more info see 
    astropy.cosmology documentation.
    """
    return cosmo.arcsec_per_kpc_comoving(z).value

def arcsec_per_kpc_proper_worker(z,cosmo):
    """ Worker function for angular separation per proper kpc at z. Note 
    that uses passed cosmology and strips any units. For more info see 
    astropy.cosmology documentation.
    """
    return cosmo.arcsec_per_kpc_proper(z).value


#==============================================================================
# WRAPPER FOR FUNCTIONS F(z)
# This the wrapper for worker functions that take 1-parameter (z) and a given 
# cosmology
#==============================================================================
def run_anyfz(f,z,cosmo):
    """ PARALELL calculation of any (1-parameter) function of z in a 
    given cosmology.
    
    Parameters
    ----------
    f : function
        Any 1-parameter worker function f(z) defined the library. Currently:
        lumdis_worker, comdis_worker, angular_diameter_distance_worker,
        comoving_volume_worker,differential_comoving_volume_worker,
        ditsmod_worker, arcsec_per_kpc_comoving_worker, arcsec_per_kpc_proper_worker

    z : ndarray/list
        1D-array of input redshifts

    cosmo : object
        An astropy.cosmology object

    Returns
    -------
    res : ndarray
        1D-array of results returned by f(z)
    
    Examples
    --------
    ::
    
        z = np.random.random(200000)
        import parcosmolib as par
        from astropy.cosmology import FlatLambdaCDM
        cosmo=FlatLambdaCDM(H0=70,Om0=0.3)
        par.run_anyfz(par.lumdis_worker,z,cosmo)
    """
    nchunks = cpu_count()
    pool = Pool(processes=nchunks)  #create pool of workers
    zz=np.array_split(z,nchunks)    #split input in chunks
    
    res=[pool.apply_async(f, [chk,cosmo]) for chk in zz]  #execute async
    pool.close()
    pool.join()
    res=[chk.get() for chk in res]  #get results
    res=np.concatenate(res)         #convert list of arrays into a single array
    return res



def comdis(z,cosmo):
    """ 
    Same as run_anyfz(), but with comdis_worker() hardcoded 
    """
    nchunks = cpu_count()
    pool = Pool(processes=nchunks)
    zz=np.array_split(z,nchunks)  #split input
    
    res=[pool.apply_async(comdis_worker, [chk,cosmo]) for chk in zz]
    pool.close()
    pool.join()
    res=[chk.get() for chk in res]
    res=np.concatenate(res)
    return res



    