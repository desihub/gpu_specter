# Typical benchmark configurations

Combine a configuration and a benchmark

## Configuration

### CGPU / V100

Half Cori GPU Node (4 GPUs, perlmutter-like), can use up to 5 ranks per GPU:

```
module purge
module load cgpu
salloc -C gpu -N 1 --gpus-per-node=4 --ntasks-per-node=20 --cpus-per-task=2 -t 60

module load python cuda/11.1.1 gcc openmpi
source activate gpu-specter-dev
export OMP_NUM_THREADS=1
```

### DGX / A100

Half DGX Node (4 GPUs, perlmutter-like), can use up to 8 ranks per GPU:

```
module purge
module load dgx
salloc -C dgx -N 1 --gpus-per-node=4 --ntasks-per-node=32 --cpus-per-task=2 -t 60

module load python cuda/11.1.1 gcc openmpi
source activate gpu-specter-dev-dgx
export OMP_NUM_THREADS=1
```

## Benchmarks

### Single frame benchmark

```
INDIR=/global/cfs/cdirs/desi/spectro/redux/andes
IMAGE=$INDIR/preproc/20200315/00055672/preproc-r0-00055672.fits
PSF=$INDIR/exposures/20200315/00055672/psf-r0-00055672.fits
OUTPUT=frame-r0-00055672.fits
WLEN=5760.0,7620.0,0.8

ARGS="--gpu-specter --mpi --gpu -i ${IMAGE} -p ${PSF} -o ${OUTPUT} -w ${WLEN} --nsubbundles 5"
```

On Cori GPU (2 ranks per GPU):

```
time srun -n 8 -c 2 --cpu-bind=cores mps-wrapper desi_extract_spectra $ARGS
```

On DGX (5 ranks per GPU):

```
time srun -n 20 -c 2 --cpu-bind=cores mps-wrapper desi_extract_spectra $ARGS
```

#### 30 frame benchmark

```
INDIR=/global/cfs/cdirs/desi/spectro/redux/andes
NIGHT=20200315
EXPID=00055672
OUTDIR=/global/cfs/cdirs/m1759/dmargala
JOBOUTDIR=${OUTDIR}/temp-${SLURM_JOB_ID}
mkdir -p ${JOBOUTDIR}

ARGS="${INDIR} ${JOBOUTDIR} $(date +%s) --night ${NIGHT} --expid ${EXPID} --gpu"
```

On Cori GPU (2 ranks per GPU):

```
time srun -n 8 -c 2 --cpu-bind=cores mps-wrapper desi-extract-exposure $ARGS
```

On DGX (5 ranks per GPU):

```
time srun -n 20 -c 2 --cpu-bind=cores mps-wrapper desi-extract-exposure $ARGS
```
