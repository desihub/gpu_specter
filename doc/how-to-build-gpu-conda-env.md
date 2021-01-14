# Instructions for building a GPU-enabled DESI conda environment at NERSC

For reference:
 * NERSC corigpu docs: https://docs-dev.nersc.gov/cgpu/software/python/
 * CuPy release info: https://github.com/cupy/cupy/releases/latest
 * CUDA release info: https://developer.nvidia.com/cuda-toolkit-archive
 * mpi4py info: https://bitbucket.org/mpi4py/mpi4py/src/master/CHANGES.rst

## Start an interactive session on a GPU node

As of Nov 2020, the latest CUDA toolkit version available at NERSC is `11.1.1`. Note that CuPy and CUDA versions must match.

On Cori GPU:

```
module purge
module load cgpu
salloc -C gpu -N 1 -G 1 -c 10 -t 60 -A m1759
module load python cuda/11.1.1 gcc openmpi

conda create -n gpu-specter-dev python=3.8
source activate gpu-specter-dev
```

On DGX:

```
module purge
module load dgx
salloc -C dgx -N 1 -G 1 -c 16 -t 60
module load python cuda/11.1.1 gcc openmpi

conda create -n gpu-specter-dev-dgx python=3.8
source activate gpu-specter-dev-dgx
```

## Install required DESI, GPU, and MPI python libraries

```
conda install -y numpy scipy numba pyyaml matplotlib
pip install astropy==4.1
pip install fitsio
pip install healpy
pip install speclite
pip install cupy-cuda111
# pip install mpi4py
```

Make sure mpi4py builds from source using the loaded mpi module.

Build mpi4py from development source. mpi4py [v3.1.0](https://bitbucket.org/mpi4py/mpi4py/src/master/CHANGES.rst) will support passing Python GPU arrays but is not officially released yet.

```
git clone https://bitbucket.org/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build
python setup.py install
```

Once mpi4py v3.1.0 is released, use the following:

```
# MPICC="$(which mpicc)" pip install --no-binary mpi4py mpi4py
```

## Install gpu_specter

I recommend installing gpu_specter in dev/edit mode:

```
git clone https://github.com/sbailey/gpu_specter.git
cd gpu_specter
pip install -e .
```

or you can add gpu_specter libary and executables to paths:

```
git clone https://github.com/sbailey/gpu_specter.git
cd gpu_specter
export PYTHONPATH=$(pwd)/py/gpu_specter:$PYTHONPATH
export PATH=$(pwd)/bin:$PATH
```

Note that you will have to set these vars each time you load the conda environment.

## Test installation

On a gpu node with environment loaded:

```
srun -n 1 -c 2 python -m unittest --verbose gpu_specter.test.test_suite
```

Tests requiring `specter` are skipped unless the package is installed (see below).

## Install additional DESI dependencies

These are required to run the 30-frame benchmark.

```
pip install git+https://github.com/desihub/specter.git
pip install git+https://github.com/desihub/desiutil.git
pip install git+https://github.com/desihub/desitarget.git
# pip install git+https://github.com/desihub/desispec.git
```

As of Jan 2020, gpu_specter support is not yet merged into desispec so install from fork:

```
pip install git+https://github.com/dmargala/desispec.git@add-gpu-specter-support
```

## Finished gpu-specter-dev conda env should have all of the following:

```
> conda list
# packages in environment at /global/homes/d/dmargala/.conda/envs/gpu-specter-dev:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
astropy                   4.0.2            py38h7b6447c_0  
blas                      1.0                         mkl  
ca-certificates           2020.10.14                    0  
certifi                   2020.6.20          pyhd3eb1b0_3  
cupy-cuda111              8.1.0                    pypi_0    pypi
cycler                    0.10.0                   py38_0  
dbus                      1.13.18              hb2f20db_0  
desispec                  0.34.4.dev4119           pypi_0    pypi
desitarget                0.44.0.dev4250           pypi_0    pypi
desiutil                  3.0.3.dev823             pypi_0    pypi
expat                     2.2.10               he6710b0_2  
fastrlock                 0.5                      pypi_0    pypi
fitsio                    1.1.3                    pypi_0    pypi
fontconfig                2.13.0               h9420a91_0  
freetype                  2.10.4               h5ab3b9f_0  
glib                      2.66.1               h92f7085_0  
gpu-specter               0.0.0                     dev_0    <develop>
gst-plugins-base          1.14.0               hbbd80ab_1  
gstreamer                 1.14.0               hb31296c_0  
icu                       58.2                 he6710b0_3  
intel-openmp              2020.2                      254  
jpeg                      9b                   h024ee3a_2  
kiwisolver                1.3.0            py38h2531618_0  
lcms2                     2.11                 h396b838_0  
ld_impl_linux-64          2.33.1               h53a641e_7  
libedit                   3.1.20191231         h14c3975_1  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 9.1.0                hdf63c60_0  
libgfortran-ng            7.3.0                hdf63c60_0  
libllvm10                 10.0.1               hbcb73fb_5  
libpng                    1.6.37               hbc83047_0  
libstdcxx-ng              9.1.0                hdf63c60_0  
libtiff                   4.1.0                h2733197_1  
libuuid                   1.0.3                h1bed415_2  
libxcb                    1.14                 h7b6447c_0  
libxml2                   2.9.10               hb55368b_3  
llvmlite                  0.34.0           py38h269e1b5_4  
lz4-c                     1.9.2                heb0550a_3  
matplotlib                3.3.2                         0  
matplotlib-base           3.3.2            py38h817c723_0  
mkl                       2020.2                      256  
mkl-service               2.3.0            py38he904b0f_0  
mkl_fft                   1.2.0            py38h23d657b_0  
mkl_random                1.1.1            py38h0573a6f_0  
mpi4py                    3.0.3                    pypi_0    pypi
ncurses                   6.2                  he6710b0_1  
numba                     0.51.2           py38h0573a6f_1  
numpy                     1.19.2           py38h54aff64_0  
numpy-base                1.19.2           py38hfa32c7d_0  
olefile                   0.46                       py_0  
openssl                   1.1.1h               h7b6447c_0  
pcre                      8.44                 he6710b0_0  
pillow                    8.0.1            py38he98fc37_0  
pip                       20.2.4           py38h06a4308_0  
pyparsing                 2.4.7                      py_0  
pyqt                      5.9.2            py38h05f1152_4  
python                    3.8.5                h7579374_1  
python-dateutil           2.8.1                      py_0  
pyyaml                    5.3.1            py38h7b6447c_1  
qt                        5.9.7                h5867ecd_1  
readline                  8.0                  h7b6447c_0  
scipy                     1.5.2            py38h0b6359f_0  
setuptools                50.3.1           py38h06a4308_1  
sip                       4.19.13          py38he6710b0_0  
six                       1.15.0           py38h06a4308_0  
speclite                  0.10.dev547              pypi_0    pypi
specter                   0.9.4.dev601             pypi_0    pypi
sqlite                    3.33.0               h62c20be_0  
tbb                       2020.3               hfd86e86_0  
tk                        8.6.10               hbc83047_0  
tornado                   6.0.4            py38h7b6447c_1  
wheel                     0.35.1             pyhd3eb1b0_0  
xz                        5.2.5                h7b6447c_0  
yaml                      0.2.5                h7b6447c_0  
zlib                      1.2.11               h7b6447c_3  
zstd                      1.4.5                h9ceee32_0  
```

## Install environment as a jupyter kernel

For details, see: https://docs.nersc.gov/services/jupyter/

```
conda install ipykernel
python -m ipykernel install --user --name gpu-specter-dev --display-name gpu-specter-dev
```

## Cleanup

Remove the environment:

```
conda remove --name gpu-specter-dev --all
```

Remove jupyter kernel:

```
jupyter kernelspec uninstall gpu-specter-dev
```
