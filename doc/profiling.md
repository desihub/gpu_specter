# Profiling

## CPU 

### Python cProfile

```
cd gpu_specter
salloc -N 1 -C haswell -t 60 -q interactive
source /global/cfs/cdirs/desi/software/desi_environment.sh master

export PATH=$(pwd)/bin:$PATH
export PYTHONPATH=$(pwd)/py:$PYTHONPATH

basedir=/global/cfs/cdirs/desi/spectro/redux/andes
args="-w 5760.0,7620.0,0.8 -i $basedir/preproc/20200219/00051060/preproc-r0-00051060.fits -p $basedir/exposures/20200219/00051060/psf-r0-00051060.fits"

export OMP_NUM_THREADS=1

export PROFILEOUT=$SCRATCH/profile-$(date +'%Y%m%d')-$(git rev-parse --short HEAD)

# Full frame (single process, no MPI)
srun -n 1 -c 2 --cpu-bind=cores python -m cProfile -o $PROFILEOUT.pstats bin/spex -o $SCRATCH/spex-haswell.fits $args
```

Convert the profile output to an image:

```
gprof2dot -f pstats filename.pstats | dot -Tpng -o filename.png
```

### ARM Performance Report

```
module load allinea-forge

perf-report srun -n 32 -c 2 bin/spex --mpi -o $SCRATCH/spex_haswell_mpi32_gpu0.fits $args
```

## GPU

### NVIDA NSight Sys

The following instructions assume you have a conda env named `gpu-specter-dev` with:
 * CUDA 11.1 compatible cupy (cupy-cuda111)
 * mpi4py built with openmpi
 * installed gpu_specter (`cd gpu_specter && pip install -e .`)

#### Single GPU

```
# Request one GPU node with one GPU
module purge
module load cgpu
salloc -C gpu -N 1 -G 1 -c 10 -t 60 -q interactive

# Setup environment
module load python cuda/11.1.1 gcc openmpi
source activate gpu-specter-dev
export OMP_NUM_THREADS=1

INDIR=/global/cfs/cdirs/desi/spectro/redux/andes
IMAGE=$INDIR/preproc/20200315/00055672/preproc-r0-00055672.fits
PSF=$INDIR/exposures/20200315/00055672/psf-r0-00055672.fits
OUTPUT=frame-r0-00055672.fits
WLEN=5760.0,7620.0,0.8

time nsys profile --sample none --trace cuda,nvtx --stats=true --force-overwrite true --output profile-cgpu \
    srun -n 2 -c 2 --cpu-bind=cores \
    mps-wrapper \
    spex --mpi --gpu -i $IMAGE -p $PSF -o $OUTPUT -w $WLEN --nsubbundles 5 --nwavestep 50 --nspec 100
```

#### Multi GPU

```
# Request one GPU node with multiple GPUs
module purge
module load cgpu
salloc -C gpu -N 1 -G 2 -c 20 -t 60 -q interactive

# Setup environment
module load python cuda/11.1.1 gcc openmpi
source activate gpu-specter-dev
export OMP_NUM_THREADS=1

INDIR=/global/cfs/cdirs/desi/spectro/redux/andes
IMAGE=$INDIR/preproc/20200315/00055672/preproc-r0-00055672.fits
PSF=$INDIR/exposures/20200315/00055672/psf-r0-00055672.fits
OUTPUT=frame-r0-00055672.fits
WLEN=5760.0,7620.0,0.8

# Determine number of MPI ranks based on number of GPUs
NRANKSPERGPU=2
NGPUS=$SLURM_GPUS
NRANKS=$((NRANKSPERGPU * NGPUS))

PROFILE=profile-cgpu-n${NRANKS}-g${NGPUS}

time nsys profile --sample none --trace cuda,nvtx --stats=true --force-overwrite true --output ${PROFILE} \
    srun -n ${NRANKS} -c 2 --cpu-bind=cores \
    mps-wrapper \
    spex --mpi --gpu -i ${IMAGE} -p ${PSF} -o ${OUTPUT} -w ${WLEN} --nsubbundles 5 --nwavestep 50 --nspec 100
```

If desispec with gpu_specter support is installed, you can replace `spex` with `desi_extract_spectra --gpu-specter`.

To view the profile timeline, copy the profile output file `$PROFILE.qdrep` to your local environment and open the file with `NVIDIA Nsight Systems`.
