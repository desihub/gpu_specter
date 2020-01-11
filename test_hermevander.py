import cupy as cp
from numba import cuda
import numpy as np
import numba, pytest
from time import time
from astropy.table import Table

@cuda.jit
def hermevander(x, deg, output_matrix):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(i, x.shape[0], stride):
        output_matrix[i][0] = 1
        if deg > 0:
            output_matrix[i][1] = x[i]
            for j in range(2, deg + 1):
                output_matrix[i][j] = output_matrix[i][j-1]*x[i] - output_matrix[i][j-2]*(j-1)

def hermevander_wrapper(x, deg):
    """Temprorary wrapper the allocates memory and calls hermevander_gpu
    """
    output = cp.ndarray((len(x), deg + 1))
    blocksize = 256
    numblocks = (len(x) + blocksize - 1) // blocksize
    hermevander[numblocks, blocksize](x, deg, output)
    return output

def test_hermevander():
    # Generate dummy input
    degree = 10
    np.random.seed = 1
    x_cpu = np.random.rand(100)
    x_gpu = cp.array(x_cpu)

    # Calculate on cpu
    hermevander_cpu = np.polynomial.hermite_e.hermevander(x_cpu, degree)

    # Calculate on gpu
    hermevander_gpu = hermevander_wrapper(x_gpu, degree)

    # Compare
    assert np.allclose(hermevander_cpu, hermevander_gpu.get())

