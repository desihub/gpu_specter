# gpu_specter refactor README

Scratch work for porting spectroperfectionism extractions to GPUs

# Example usage at NERSC

## CPU

```
# these instructions assume we're at the top level of this repo
cd gpu_specter

# request an interactive node
salloc -N 1 -C haswell -t 60 -q interactive

# setup environment
source /global/cfs/cdirs/desi/software/desi_environment.sh master
export PATH=$(pwd)/bin:$PATH
export PYTHONPATH=$(pwd)/py:$PYTHONPATH

# prepare command line arguments
basedir=/global/cfs/cdirs/desi/spectro/redux/andes
args="-w 5760.0,7620.0,0.8 -i $basedir/preproc/20200219/00051060/preproc-r0-00051060.fits -p $basedir/exposures/20200219/00051060/psf-r0-00051060.fits"

# run spex
srun -n 32 -c 2 bin/spex --mpi -o $SCRATCH/spex_haswell_mpi32_gpu0.fits $args

# for comparison, run desi_extract_spectra
srun -n 32 -c 2 desi_extract_spectra --mpi -o $SCRATCH/desi_haswell_mpi32_gpu0.fits $args
```

## GPU

Use the [these instructions](doc/how-to-build-gpu-conda-env.md) for building a gpu conda environment.

### Single GPU

```
cd gpu_specter
module load python esslurm cuda
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

# 0 GPU (CPU only)
srun -n 5 -c 2 --cpu-bind=cores bin/spex --mpi -o $SCRATCH/spex_skylake_mpi5_gpu0.fits $args

# 1 GPU (single process, no MPI)
srun -n 1 -c 2 --cpu-bind=cores bin/spex --gpu -o $SCRATCH/spex_skylake_mpi0_gpu1.fits $args

# 1 GPU + MPI
srun -n 2 -c 2 --cpu-bind=cores bin/spex --mpi --gpu -o $SCRATCH/spex_skylake_mpi2_gpu1.fits $args

# 1 GPU + MPI + MPS
srun -n 2 -c 2 --cpu-bind=cores bin/mps-wrapper bin/spex --mpi --gpu -o $SCRATCH/spex_skylake_mpi2_gpu1_mps.fits $args
```

### Multi GPU 

```
cd gpu_specter
module load python esslurm cuda
salloc -C gpu -N 1 -G 2 -c 20 -t 60 -A m1759

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

### 2 GPU + MPI
srun -n 2 -c 2 --cpu-bind=cores bin/spex --mpi --gpu -o $SCRATCH/spex_skylake_mpi2_gpu2.fits $args

### 2 GPU + MPI + MPS
srun -n 4 -c 2 --cpu-bind=cores bin/mps-wrapper bin/spex --mpi --gpu -o $SCRATCH/spex_skylake_mpi4_gpu2_mps.fits $args
```
