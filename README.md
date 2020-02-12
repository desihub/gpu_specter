This is a README file for the DESI gpu hackathon code.

#######################

For the cpu version:

`ssh cori.nersc.gov`

Get an interactive Haswell node:

`salloc -N 1 -t 30 -C haswell -q interactive`

for now use the desi environment (if you can)
if you don't have permission to access please let me know

`source activate /global/cfs/cdirs/desi/software/desi_environment.sh master`

`cd $SCRATCH`

`git clone https://github.com/sbailey/gpu_specter`

`cd gpu_specter`

`git fetch`

`git checkout hackathon`

`time srun -u -n 20 -c 2 python -u cpu_wrapper_specter.py -o test.fits`

`cpu_wrapper_specter.py` divides the ccd frame into 20 bundles and launches
20 mpi ranks then calls `cpu_extract.py` which does the prep for the projection
matrix, actual projection matrix, and the extraction kernel.

The answers are wrong and some bookkeeping issues need to be fixed, but this is
good enough to get started for our purposes of moving this to the gpu. 


#####################

For the gpu version:

GPU conversion is in progress. `cache_spots` and `projection_matrix` are
now completely on the gpu. We have an error in the gpu function `projection_matrix`
that results in zeros for the final wavelength patch. This causes singular matrices
and the whole program crashes. We are actively trying to solve this.

After this point, we can flip our numpy functions in `ex2d_patch` over to cupy functions.
This part has been previously tested so it should hopefully work as desired.

To run the gpu version (everyone should be able to run, no linux group necessary)

`ssh cori.nersc.gov`

`module load esslurm`

Get a gpu node:

`salloc -C gpu -N 1 -t 60 -c 10 --gres=gpu:1 -A <account>`

`module load python`

`module load cuda/10.1.243`

Cuda must be version 10.1 to be compatible with the latest release of CuPy (also 10.1)

Source our special desi gpu conda environment:

`source activate /global/cfs/cdirs/m1759/desi/desi_gpu`

Then source the custom desi modules 

`source /global/cfs/cdirs/m1759/desi/desi_libs.sh`

`cd /global/cfs/cdirs/m1759/desi/gpu_specter`

And launch the program which currently runs for a few seconds and then fails:

`time srun -u -n 5 -c 2 python -u gpu_wrapper_specter.py -o test.fits`

Here is the current failure message you can expect to see:

```
INFO:wrapper_specter.py:278:main: extract:  Rank 4 extracting pix-r0-00003578.fits spectra 425:450 at Tue Feb 11 11:35:40 2020
ERROR:wrapper_specter.py:362:main: extract:  FAILED bundle 12, spectrum range 300:325
ERROR:wrapper_specter.py:365:main: Traceback (most recent call last):
  File "wrapper_specter.py", line 287, in main
    full_output=True, nsubbundles=args.nsubbundles)
  File "/global/cscratch1/sd/stephey/git_repo/gpu_specter/gpu_extract.py", line 482, in ex2d
    full_output=True, use_cache=True)
  File "/global/cscratch1/sd/stephey/git_repo/gpu_specter/gpu_extract.py", line 782, in ex2d_patch
    f_cpu = np.linalg.solve(iCov_cpu.todense(), y_cpu).reshape((nspec, nwave)) #requires array, not sparse object
  File "/global/homes/s/stephey/.conda/envs/desi_gpu_default/lib/python3.7/site-packages/numpy/linalg/linalg.py", line 403, in solve
    r = gufunc(a, b, signature=signature, extobj=extobj)
  File "/global/homes/s/stephey/.conda/envs/desi_gpu_default/lib/python3.7/site-packages/numpy/linalg/linalg.py", line 97, in _raise_linalgerror_singular
    raise LinAlgError("Singular matrix")
numpy.linalg.LinAlgError: Singular matrix
```

