## Unit testing

Run unit test suite using:

```
python setup.py test
```

## Strict regression test
    
Use to confirm that changes do not affect the output. 
This is a good test to run before any merge to master. It can also be helpful to use while refactoring code.
Differences in the output might be expected depending on the nature of the code changes. 

Example:

```
cd gpu_specter
git checkout master
args=...
bin/spex $args -o output-master.fits 
git checkout branch
bin/spex $args -o output-branch.fits

fitsdiff output-master.fits output-branch.fits
```

## Science parity test

When we do expect differences in the output, we want to confirm that they will not adversely impact science.
This test is more nuanced than the strict regression test. 

Example comparison between `spex` and `desi_extract_spectra`

Generate output files:

```
cd gpu_specter
source /global/cfs/cdirs/desi/software/desi_environment.sh master
export PATH=$(pwd)/bin:$PATH
export PYTHONPATH=$(pwd)/py:$PYTHONPATH

salloc -N 1 -C haswell -t 60 -q interactive

basedir=/global/cfs/cdirs/desi/spectro/redux/andes
args="-w 5760.0,7620.0,0.8 -i $basedir/preproc/20200219/00051060/preproc-r0-00051060.fits -p $basedir/exposures/20200219/00051060/psf-r0-00051060.fits"

srun -n 32 -c 2 desi_extract_spectra --mpi -o $SCRATCH/desi_extract.fits $args
srun -n 32 -c 2 bin/spex --mpi -o $SCRATCH/spex_extract.fits $args
```

Example parity test output:

```
$ python bin/compare-frame -a $SCRATCH/spex_extract.fits -b $SCRATCH/desi_extract.fits
(f_a, f_b):
  isclose:   70.17%
(f_a - f_b)/sqrt(var_a + var_b):
    1e-05:   60.38%
   0.0001:   73.73%
    0.001:   84.80%
     0.01:   94.74%
      0.1:   99.88%
      1.0:  100.00%
```
