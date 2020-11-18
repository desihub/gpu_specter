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

#### Single GPU (No MPI)

**Note the following instructions are out of date**

```
cd gpu_specter
module load python esslurm cuda/10.2.89
salloc -C gpu -N 1 -G 1 -c 10 -t 60 -A m1759

source activate desi-gpu
export PATH=$(pwd)/bin:$PATH
export PYTHONPATH=$(pwd)/py:$PYTHONPATH

module load gcc/7.3.0 mvapich2
export MV2_USE_CUDA=1
export MV2_ENABLE_AFFINITY=0
export LD_PRELOAD=$MVAPICH2_DIR/lib/libmpi.so

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

basedir=/global/cfs/cdirs/desi/spectro/redux/andes
args="-w 5760.0,7620.0,0.8 -i $basedir/preproc/20200219/00051060/preproc-r0-00051060.fits -p $basedir/exposures/20200219/00051060/psf-r0-00051060.fits"

# 1 GPU (single process, no MPI)
cmd_spex="python -O bin/spex --gpu -o $SCRATCH/spex-gpu.fits $args --nspec 100"

export PROFILEOUT=$SCRATCH/profile-$(date +'%Y%m%d')-$(git rev-parse --short HEAD)
cmd_nsys="nsys profile --sample none --trace cuda,nvtx --stats=true --force-overwrite true --output $PROFILEOUT"

time srun -n 1 -c 2 --cpu-bind=cores $cmd_nsys $cmd_spex
```

#### Multiple MPI Ranks with Single GPU

The following instructions assume you have a conda env named `profile-gpu-specter` with:
 * CUDA 11.1 compatible cupy (cupy-cuda111)
 * mpi4py built with openmpi
 * installed gpu_specter (`cd gpu_specter && pip install -e .`)

```
# Request one GPU node with one GPU
module purge
module load cgpu
salloc -C gpu -N 1 -G 1 -c 10 -t 60

# Setup environment
module load python cuda/11.1.1 openmpi
source activate profile-gpu-specter
export OMP_NUM_THREADS=1

# Setup spex command
OUTDIR=${SCRATCH}
INDIR=/global/cfs/cdirs/desi/spectro/redux/andes
NIGHT=20200315
EXPID=00055672
CAM=r0
IMAGE=${INDIR}/preproc/${NIGHT}/${EXPID}/preproc-${CAM}-${EXPID}.fits
PSF=${INDIR}/exposures/${NIGHT}/${EXPID}/psf-${CAM}-${EXPID}.fits
FRAME=${OUTDIR}/frame-${CAM}-${EXPID}.fits
SPEXCMD="spex -i ${IMAGE} -p ${PSF} -o ${FRAME} -w 5760.0,7620.0,0.8 --nsubbundles 5 --gpu-specter --gpu --mpi --nspec 100"

# Setup srun command and mps-wrapper
NRANKS=2
NGPU=1
SRUNCMD="srun -n ${NRANKS} -c 2 --cpu-bind=cores mps-wrapper"

# Setup profile command
PROFILE=${OUTDIR}/profile-cgpu-n${NRANKS}-g{NGPU}
NSYSCMD="nsys profile --sample none --trace cuda,nvtx --stats=true --force-overwrite true --output ${PROFILE}"

# Run everything together
time ${NSYSCMD} ${SRUNCMD} ${SPEXCMD}
```

To view the profile timeline, copy the profile output file `$PROFILE.qdrep` to your local environment and open the file with `NVIDIA Nsight Systems`.
