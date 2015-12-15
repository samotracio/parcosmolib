Paralell Wrappers for Cosmology Functions in Astropy
====================================================

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
