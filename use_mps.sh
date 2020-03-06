#!/bin/bash
if [ $SLURM_PROCID -eq 0 ]; then
    nvidia-cuda-mps-control -d
fi

sleep 10

python -u gpu_wrapper_specter.py -o out.fits --nwavestep 75 --nspec 100
