#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --time=15
#SBATCH --qos=debug

echo start $(date)

source /global/cfs/cdirs/desi/software/desi_environment.sh master

export PATH=$(pwd)/bin:$PATH
export PYTHONPATH=$(pwd)/py:$PYTHONPATH

export basedir=/global/cfs/cdirs/desi/spectro/redux/daily

time srun -n 32 -c 2 spex --mpi -w 5761.0,7620.0,0.8 \
    -i $basedir/preproc/20200219/00051060/preproc-r0-00051060.fits \
    -p $basedir/exposures/20200219/00051060/psf-r0-00051060.fits \
    -o $SCRATCH/blat.fits

echo end $(date)
