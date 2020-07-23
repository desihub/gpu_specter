## Unit testing

Run unit test suite using:

```
python setup.py test
```

Note that `python setup.py test` triggers a deprecation warning. An altenative:

```
python -m unittest --verbose gpu_specter.test.test_suite
```

On a gpu node:

```
srun python -m unittest --verbose gpu_specter.test.test_suite
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

Use the instructions from the top-level README to generate the output "frame" files:
 * `$SCRATCH/spex_haswell_mpi32_gpu0.fits` 
 * `$SCRATCH/desi_haswell_mpi32_gpu0.fits`

Compare the results using `bin/compare-frame`:

```
$ python bin/compare-frame -a $SCRATCH/spex_haswell_mpi32_gpu0.fits -b $SCRATCH/desi_haswell_mpi32_gpu0.fits
wave (allclose): True
(f_a, f_b):
   isclose:  70.17%
(f_a - f_b)/sqrt(var_a + var_b):
     1e-05:  60.38%
    0.0001:  73.73%
     0.001:  84.80%
      0.01:  94.74%
       0.1:  99.88%
       1.0: 100.00%
(ivar_a, ivar_b):
   isclose:  81.98%
(sigma_a - sigma_b)/sqrt(var_a + var_b):
     1e-05:  84.95%
    0.0001:  92.98%
     0.001:  99.14%
      0.01:  99.99%
       0.1:  99.99%
       1.0: 100.00%
(resolution_a, resolution_b):
   isclose:  81.34%
```
