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
