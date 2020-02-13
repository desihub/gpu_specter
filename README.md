This is a README file for the DESI gpu hackathon code.

The answers are wrong and some bookkeeping issues need to be fixed (so it can't be swapped directly into the desi pipeline), but this is good enough to get started for our purposes of moving this to the gpu. 

# For both the cpu and gpu:

To run the both versions on our cori gpu skylakes/v100s (everyone should be able to run, no linux group necessary)

`ssh cori.nersc.gov`

`module load esslurm`

Get a cori gpu node:

`salloc -C gpu -N 1 -t 60 -c 10 --gres=gpu:1 -A m1759`

`module load python`

`module load cuda/10.1.243`

Cuda must be version 10.1 to be compatible with the latest release of CuPy (also 10.1)

Source our special desi gpu conda environment:

`source activate /global/cfs/cdirs/m1759/desi/desi_gpu`

Then source the custom desi modules 

`source /global/cfs/cdirs/m1759/desi/desi_libs.sh`

`cd /global/cfs/cdirs/m1759/desi/gpu_specter`

# To run the cpu version:

`time srun -u -n 5 -c 2 python -u cpu_wrapper_specter.py -o test.fits`

This runs in about 3 mins on the skylake using 1/8 of a cpu.

You'll see some mpi warning messages at the end but don't panic, apparently these are `normal` on corigpu for mpi4py

Everything ran correctly if you see all 20 ranks report:

```
INFO:cpu_wrapper_specter.py:351:main: extract:  Done pix-r0-00003578.fits spectra 375:400 at Wed Feb 12 12:41:56 2020
INFO:cpu_wrapper_specter.py:351:main: extract:  Done pix-r0-00003578.fits spectra 275:300 at Wed Feb 12 12:41:56 2020
INFO:cpu_wrapper_specter.py:351:main: extract:  Done pix-r0-00003578.fits spectra 175:200 at Wed Feb 12 12:41:57 2020
INFO:cpu_wrapper_specter.py:351:main: extract:  Done pix-r0-00003578.fits spectra 475:500 at Wed Feb 12 12:41:57 2020
INFO:cpu_wrapper_specter.py:351:main: extract:  Done pix-r0-00003578.fits spectra 75:100 at Wed Feb 12 12:41:58 2020
```

# To run the gpu version:

`time srun -u -n 5 -c 2 python -u gpu_wrapper_specter.py -o test.fits`

Right now it successfully runs on 1 cori gpu (and 1/8 skylake) in about 4 mins (slower than the cpu version).
`cache_spots` and `projection_matrix` are now on the gpu.

# Next steps

* cpu profiling
* gpu profiling
* Flipping cpu functions to gpu functions in `ex2d_patch` extraction kernel (be careful, probably will OOM)
* Removing mpi, running whole frame on a single gpu


