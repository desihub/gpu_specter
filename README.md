This is a README file for the DESI gpu hackathon code.

Directions to run (now working!) mini app

`ssh cori.nersc.gov`
`cd $SCRATCH`
`git clone https://github.com/sbailey/gpu_specter/`
`cd gpu_specter`
`git fetch
`git checkout hackathon`

`source /global/cfs/cdirs/desi/software/desi_environment.sh master`
`salloc -N 1 -t 30 -C haswell -q interactive`

`time srun -u -n 20 -c 2 python -u wrapper_specter.py -o test.fits`

`wrapper_specter.py` which divides the ccd frame into 20 bundles and launches 20 mpi ranks
then calls `gpu_extract.py` which does the prep for the projection matrix, actual projection matrix, and the extraction kernel

The answers are wrong and some bookkeeping issues need to be fixed, but this is good enough to get started for our purposes of moving this to the gpu
