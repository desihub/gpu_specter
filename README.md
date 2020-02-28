# This is a README file for the DESI gpu hackathon code

Here is the `original`, cpu version of the desi code we are working to adapt:

https://github.com/desihub/desispec/blob/master/py/desispec/scripts/extract.py

and

https://github.com/desihub/specter/blob/master/py/specter/extract/ex2d.py

The answers are wrong and some bookkeeping issues need to be fixed (so it can't be swapped directly into the desi pipeline), but this is good enough to get started for our purposes of moving this to the gpu. 

# For both the cpu and gpu:

To run the both versions on our cori gpu skylakes/v100s (everyone should be able to run, no linux group necessary)

`ssh cori.nersc.gov`

For the hackathon, everyone should have their own checkout/working directory for profiling and development. 

Here is how to set yours up:

```
cd /global/cfs/cdirs/m1759/desi/
mkdir <yourname>
cd <yourname>
git clone https://github.com/sbailey/gpu_specter
cd gpu_specter
git fetch
git checkout hackathon
```
This is the hackathon branch. You may want to create your own branch based on the hackathon branch if you plan to submit changes:
```
git branch <yourbranch>
git checkout <yourbranch>
```

Now that you are ready with your development directory/branch, let's get a GPU node and get started:

`module load esslurm python cuda/10.1.243`

Cuda must be version 10.1 to be compatible with the latest release of CuPy (also 10.1)

`salloc -C gpu -N 1 -t 60 -c 10 --gres=gpu:1 -A m1759`

Source our special desi gpu conda environment:

`source activate /global/cfs/cdirs/m1759/desi/desi_gpu`

Then source the custom desi modules 

`source /global/cfs/cdirs/m1759/desi/desi_libs.sh`

And now make sure you are in your directory:

`cd /global/cfs/cdirs/m1759/desi/<yourname>`

# To run the cpu version (mpi has been removed):

`time srun -u python -u cpu_wrapper_specter.py -o out.fits`

This runs in about 9 mins on the skylake using 1/8 of a cpu.

# To run the gpu version (mpi has been removed):

`time srun -u python -u gpu_wrapper_specter.py -o out.fits`

Right now it successfully runs on 1 cori gpu (and 1/8 skylake). The runtime is relatively long (~6 minutes) because the bundles are currently computed serially. For debugging/profiling we can add `--nspec 50` to process only two bundles, for example. 

# Correctness checking

Since the answers are currently wrong with respect the real version of specter, unit/correctness testing is tricky. 

Our current solultion:

1) Check that the gpu and cpu versions get the same results (they do)
2) Continue to compare the gpu output to the reference output files captured 02/25/2020. This will at least let us know that we have made some change that affected the output.

To use enable this feature, you can append `--test` to the end of the cpu or gpu version of the code:

`time srun -u python -u gpu_wrapper_specter.py -o out.fits --test`

Note that you must run with the entire frame for these comparisons to work. (Running with `--nspec 50` will cause the check to fail.)

You will find the cpu and gpu reference files in 

`/global/cfs/cdirs/m1759/desi/ref_files`

# We have added a decorator for nvtx profiling: 

`@nvtx_profile(profile=nvtx_collect,name='function_name')`

# To profile using nvprof

On cori gpu run nvprof and have it write an output file:

```
srun nvprof --log-file desi_nvprof_02252020.log python -u gpu_wrapper_specter.py -o test.fits --nspec 50 --nwavestep 50
```

# To profile using nsys

On cori gpu run nsys and write .qdrep file, move to laptop for local analysis.

```
srun nsys profile -s none -o desi_nsys_02252020 -t cuda,nvtx --force-overwrite true python -u gpu_wrapper_specter.py -o test.fits --nspec 50 --nwavestep 50
```

# Plans for Hackathon (3/3/2020 - 3/6/2020)

## Goals for the hackathon:

* Pre-hackathon-- get correctness testing in place. Done!
* Optimize overall structure of code to fully occupy GPU. Use CUDA/CuPy streams instead of computing bundles serially.
* Get rid of unnecessary HtD and DtH transfer. Understand mystery overhead shown in nsys. May need kernel fusion to prevent CuPy from moving data back to the host. May need to do all memory management manually.
* Overlap data transfer (like at the end of `ex2d_patch`) and compute. Try pinned memory.
* GPU-ize code in ex2d (still largely on CPU).
* Open to other goals too, so please suggest something!


## In the meantime, what can you do?

* Please make sure you can log in to cori/corigpu and are able to run our code!
* Take a look at our hackathon code
* Take a look at Mark's CPU profiling in our dropbox folder
* Please make sure you are familiar with CuPy and Numba CUDA basics
* Try to use nvprof, nsight systems, and nsight compute GPU profiling tools
* Check out M. Nicely's CuPy examples that include streams and pinned memory in `nicely_cupy_examples.py` in this repo. 
* Keep an eye on our hackathon README (this page!)-- we'll continue to update it with our progress throughout the week


