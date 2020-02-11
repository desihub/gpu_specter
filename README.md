This is a README file for the DESI gpu hackathon code.

#######################

For the cpu version:

`ssh cori.nersc.gov`

`cd $SCRATCH`

`git clone https://github.com/sbailey/gpu_specter/`

`cd gpu_specter`

`git fetch`

`git checkout hackathon`

`source /global/cfs/cdirs/desi/software/desi_environment.sh master`

`salloc -N 1 -t 30 -C haswell -q interactive`

To run the cpu version of the code you'll need to change the line
`from gpu_extract import ex2d`
to 
`from cpu_extract import ex2d`

`time srun -u -n 20 -c 2 python -u wrapper_specter.py -o test.fits`

`wrapper_specter.py` divides the ccd frame into 20 bundles and launches
20 mpi ranks then calls `gpu_extract.py` which does the prep for the projection
matrix, actual projection matrix, and the extraction kernel.

The answers are wrong and some bookkeeping issues need to be fixed, but this is
good enough to get started for our purposes of moving this to the gpu. 

Right now this runs in about 2 mins on Haswell (although it is missing the
somewhat expensive reassembly steps at the end). 


#####################

For the gpu version:

GPU conversion is in progress. `cache_spots` and `projection_matrix` are
now completely on the gpu. We have an error in the gpu function `projection_matrix`
that results in zeros for the final wavelength patch. This causes singular matrices
and the whole program crashes. We hope to solve this soon. 

After this point, we can flip our numpy functions in `ex2d_patch` over to cupy functions.
This part has been previously tested so it should hopefully work as desired.

To run the gpu version

`ssh cori.nersc.gov`

`cd $SCRATCH`

`git clone https://github.com/sbailey/gpu_specter/`

You'll also need desi's `desispec` and `desiutil` libraries

`git clone https://github.com/desihub/desispec`
`git glone https://github.com/desihub/desiutil`

Don't source the desi environment on the gpu. desi's mpi will not work on our gpus! this
is why we manually install and source the packages we need instead.

`cd gpu_specter`

`git fetch`

`git checkout hackathon`

You'll need to construct a custom conda environment with numba and cupy. You'll need to follow our directions here:

https://docs-dev.nersc.gov/cgpu/software/#python

You'll also have to install mpi4py into your conda environment. Follow the directions here:

https://docs-dev.nersc.gov/cgpu/software/#mpi4py

When you have your custom conda environment ready to go

`module load python`
`module load cuda/10.1.243`

`source activate <your_custom_conda_env>`

And source the desi libraries (you'll need to edit this file to point to wherever you installed
desispec and desiutil):

`source desi_libs.sh`

Get a gpu node:

`salloc -C gpu -N 1 -t 60 -c 10 --gres=gpu:1 -A <account>`

Make sure that the line in `wrapper_specter.py` calls the gpu version of the code:
`from cpu_extract import ex2d`
to
`from gpu_extract import ex2d`

And launch the program, which currently runs for a few seconds and then fails:

`time srun -u -n 5 -c 2 python -u wrapper_specter.py -o test.fits`


