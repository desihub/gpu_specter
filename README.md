# This is a README file for the DESI gpu hackathon code, March 2020

It is also useful in general for the ongoing progress of porting the DESI
spectral extraction code to GPUs.

Here is the original cpu version of the desi code we are working to adapt:

https://github.com/desihub/desispec/blob/master/py/desispec/scripts/extract.py

and

https://github.com/desihub/specter/blob/master/py/specter/extract/ex2d.py

The answers are wrong and some bookkeeping issues need to be fixed (so it can't
be swapped directly into the desi pipeline), but this is good enough to get
started for our purposes of moving this to the gpu. 

# Getting set up

To run the both versions on our cori gpu skylakes/v100s

`ssh cori.nersc.gov`

For the hackathon, everyone should have their own checkout/working directory
for profiling and development. 

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

This is the hackathon branch. You may want to create your own branch based on
the hackathon branch if you plan to submit changes:

```
git branch <yourbranch>
git checkout <yourbranch>
```

Now that you are ready with your development directory/branch, let's get a GPU
node and get started:

# Getting ready to run

`module load esslurm python cuda/10.1.243`

Cuda and CuPy versions must be compatible (in this case, both are 10.1)

`salloc -C gpu -N 1 -t 60 -c 10 --gres=gpu:1 -A m1759`

Source our special desi gpu conda environment:

`source activate /global/cfs/cdirs/m1759/desi/desi_gpu`

Then source the custom desi modules 

`source /global/cfs/cdirs/m1759/desi/desi_libs.sh`

And now make sure you are in your directory:

`cd /global/cfs/cdirs/m1759/desi/<yourname>`

# To run the cpu version:

`time srun -n 20 -c 2 -u python -u cpu_wrapper_specter.py -o out.fits`

Our time to beat (cpu time on Haswell with 20 mpi ranks) is `~1:49`. 

```
1m49.842s
1m46.758s
1m49.340s
```
Baseline data taken on Cori Haswell using `$SCRATCH` at 4:30PM Friday, Feb 28.

```
module load python
source /global/cfs/cdirs/desi/software/desi_environment.sh master
time srun -n 20 -c 2 -u python -u cpu_wrapper_specter.py -o out.fits
```

# To run the gpu version:

`time srun -n 4 -c 2 -u python -u gpu_wrapper_specter.py -o out.fits`

For more info about running gpu version with mpi see our slides
(desi_gpu_hackathon_slides.pdf).

# Correctness checking

**This is now broken afer our changes for pinned memory.** 

Since the answers are currently wrong with respect the real version of specter,
unit/correctness testing is tricky. 

Our current solultion:

1) Check that the gpu and cpu versions get the same results (they do) 
2) Continue to compare the gpu output to the reference output files captured
02/25/2020. This will at least let us know that we have made some change that
affected the output.

To use enable this feature, you can append `--test` to the end of the cpu or gpu version of the code:

Note that on the cpu this should be done without mpi. If this is a bottleneck it's fixable.

`time srun -u python -u gpu_wrapper_specter.py -o out.fits --test`

Note that you must run with the entire frame for these comparisons to work.
(Running with `--nspec 50` will cause the check to fail.)

You will find the cpu and gpu reference files in 

`/global/cfs/cdirs/m1759/desi/ref_files`

# Profiling

## We have added a decorator for nvtx profiling: 

`@nvtx_profile(profile=nvtx_collect,name='function_name')`

## To profile using nvprof

On cori gpu run nvprof and have it write an output file:

```
srun nvprof --log-file desi_nvprof_02252020.log python -u gpu_wrapper_specter.py -o test.fits --nspec 50 --nwavestep 50
```

## To profile using nsys

On cori gpu run nsys and write .qdrep file, move to laptop for local analysis.

```
srun nsys profile -s none -o desi_nsys_02252020 -t cuda,nvtx --force-overwrite true --stats=true python -u gpu_wrapper_specter.py -o test.fits --nspec 50 --nwavestep 50
```

## To profile using nsight compute (really slow!)

Here the kernel name `-k` is what the compiler calls the kernel. You see this by looking in `nsys`. 

```
time srun nv-nsight-cu-cli -k dlaed4 -o desi_ncom_02282020 -f python -u gpu_wrapper_specter.py -o out.fits --nspec 50 --nwavestep 50
```

`-s` specifies how many samples to skip (for example, skip the first 5)

`-c` controls the number of samples (~10 is a good number)

# Hackathon summary

The NERSC COE GPU hackathon took place March 3-6, 2020. The PDF that summarizes
our progress is in this repo (desi_gpu_hackathon_slides.pdf)



