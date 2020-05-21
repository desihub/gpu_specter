# Instructions for building a GPU-enabled DESI conda environment for corigpu

## Build your conda environment on a corigpu node

```
module load esslurm python cuda
salloc -C gpu -N 1 -t 120 -c 10 -G 1 -A m1759
conda create -n desi-gpu python=3.7
source activate desi-gpu
```

## Now install DESI libraries and GPU libraries

```
conda install numpy scipy numba cudatoolkit pyyaml astropy
pip install fitsio
pip install speclite
pip install cupy-cuda102
```

!!! warning "CuPy/CUDA must be same version"
    Note that the CUDA version and the CuPy version must match. As
    of May 2020 the default CUDA version is 10.2, hence we need
    `cupy-cuda102`.

## Now build mpi4py

```
module load gcc/7.3.0 mvapich2
wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.3.tar.gz
tar zxvf mpi4py-3.0.3.tar.gz
cd mpi4py-3.0.3
python setup.py build --mpicc=mpicc
python setup.py install
```

## Finished desi-gpu conda env should have all of the following:

```
(desi-gpu) stephey@cgpu08:/global/cscratch1/sd/stephey/gpu_specter> conda list
# packages in environment at /global/homes/s/stephey/.conda/envs/desi-gpu:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
astropy                   4.0.1.post1      py37h7b6447c_1  
blas                      1.0                         mkl  
ca-certificates           2020.1.1                      0  
certifi                   2020.4.5.1               py37_0  
cudatoolkit               10.2.89              hfd86e86_1  
cupy-cuda102              7.4.0                    pypi_0    pypi
fastrlock                 0.4                      pypi_0    pypi
fitsio                    1.1.2                    pypi_0    pypi
intel-openmp              2020.1                      217  
ld_impl_linux-64          2.33.1               h53a641e_7  
libedit                   3.1.20181209         hc058e9b_0  
libffi                    3.3                  he6710b0_1  
libgcc-ng                 9.1.0                hdf63c60_0  
libgfortran-ng            7.3.0                hdf63c60_0  
libstdcxx-ng              9.1.0                hdf63c60_0  
llvmlite                  0.32.1           py37hd408876_0  
mkl                       2020.1                      217  
mkl-service               2.3.0            py37he904b0f_0  
mkl_fft                   1.0.15           py37ha843d7b_0  
mkl_random                1.1.0            py37hd6b4f25_0  
mpi4py                    3.0.3                    pypi_0    pypi
ncurses                   6.2                  he6710b0_1  
numba                     0.49.1           py37h0573a6f_0  
numpy                     1.18.1           py37h4f9e942_0  
numpy-base                1.18.1           py37hde5b4d6_1  
openssl                   1.1.1g               h7b6447c_0  
pip                       20.0.2                   py37_3  
python                    3.7.7                hcff3b4d_5  
pyyaml                    5.3.1            py37h7b6447c_0  
readline                  8.0                  h7b6447c_0  
scipy                     1.4.1            py37h0b6359f_0  
setuptools                46.4.0                   py37_0  
six                       1.14.0                   py37_0  
speclite                  0.8                      pypi_0    pypi
sqlite                    3.31.1               h62c20be_1  
tbb                       2020.0               hfd86e86_0  
tk                        8.6.8                hbc83047_0  
wheel                     0.34.2                   py37_0  
xz                        5.2.5                h7b6447c_0  
yaml                      0.1.7                had09818_2  
zlib                      1.2.11               h7b6447c_3  

```

## Additional info and caveats

On 5/21/2020 I was able to run spex (cpu only) using this environment.

On 5/21/2020 I was able to run the hackathon branch cpu version and gpu version
using this environment. I think it is suitable for both cpu and gpu Numba (my
chief concern). Note that this version has dependencies on desispec, desiutil,
specter, and desitarget, all of which are loaded via
```
source /global/cfs/cdirs/m1759/desi/desi_libs.sh
```


Per this [page](https://docs-dev.nersc.gov/cgpu/software/#mvapich2-ptmalloc-warnings-with-python-mpi-codes),
the warning:
```
WARNING: Error in initializing MVAPICH2 ptmalloc library.Continuing without InfiniBand registration cache support.
```

is a known issue. To silence the warning we can set

```
export LD_PRELOAD=$MVAPICH2_DIR/lib/libmpi.so
```


