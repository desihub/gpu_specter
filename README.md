# gpu_specter refactor README

Scratch work for porting spectroperfectionism extractions to GPUs

# Directions to run cpu version (currently only working version)

## Set up the environment; there are multiple correct ways to do this

```
source /global/cfs/cdirs/desi/software/desi_environment.sh master
git clone https://github.com/sjbailey/gpu_specter
cd gpu_specter
git checkout refactor
export PATH=$(pwd)/bin:$PATH
export PYTHONPATH=$(pwd)/py:$PYTHONPATH
```

## Get an interactive cpu node

```
salloc -N 1 -C haswell -t 60 -q interactive
basedir=/global/cfs/cdirs/desi/spectro/redux/daily/
```

## Run spex (just over 1 min for a full frame)

```
time srun -n 32 -c 2 spex --mpi -w 5760.0,7620.0,0.8 \
-i $basedir/preproc/20200219/00051060/preproc-r0-00051060.fits \
-p $basedir/exposures/20200219/00051060/psf-r0-00051060.fits \
-o $SCRATCH/blat.fits
```

# TODO: directions to run gpu version
