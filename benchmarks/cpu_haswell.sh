#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --time=15
#SBATCH --qos=debug

echo "Start time: $(date --iso-8601=seconds)"

# Returns the number of seconds since unix epoch
_time() {
    echo $(date +%s.%N)
}

# Returns the elapsed time in seconds between start and end
# setting scale=3 and dividing by 1 to limit to millisecond precision
_elapsed_time () {
    echo "scale=3; ($2 - $1) / 1" | bc
}

# Check environement
python -c "import fitsio" 2&> /dev/null
if [ $? != 0 ]; then
    echo "ERROR: fitsio not in PYTHONPATH"
    echo "Try loading an environment such as:"
    echo "    source /global/cfs/cdirs/desi/software/desi_environment.sh master"
    echo "exiting"
    exit 1
fi

# See if gpu_specter is already in PYTHONPATH
python -c "import gpu_specter" 2&> /dev/null
if [ $? != 0 ]; then
    # try adding current directory
    export PATH=$(pwd)/bin:$PATH
    export PYTHONPATH=$(pwd)/py:$PYTHONPATH
    python -c "import gpu_specter" 2&> /dev/null

    # did that work?
    if [ $? != 0 ]; then
        echo "ERROR: gpu_specter not in PYTHONPATH; exiting"
        exit 1
    fi
fi

# Assemble command with arguments
basedir="/global/cfs/cdirs/desi/spectro/redux/andes"
input="$basedir/preproc/20200219/00051060/preproc-r0-00051060.fits"
psf="$basedir/exposures/20200219/00051060/psf-r0-00051060.fits"
output="$SCRATCH/frame-r0-00051060.fits"
cmd="spex --mpi -w 5761.0,7620.0,0.8 -i $input -p $psf -o $output"

# Perform benchmark
start_time=$(_time)
srun -n 32 -c 2 $cmd
end_time=$(_time)

elapsed_time=$(_elapsed_time ${start_time} ${end_time})

nnodes=1
nframes=1
nodehours=$(echo "$nnodes * $elapsed_time / (60 * 60)" | bc -l)
frames_per_nodehour=$(echo "scale=1; $nframes / $nodehours" | bc)

echo "Elapsed time (seconds): ${elapsed_time}"
echo "Frames per node hour: ${frames_per_nodehour}"
echo "End time: $(date --iso-8601=seconds)"

