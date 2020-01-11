import cupy as cp
from numba import cuda
import numpy as np
import numba, pytest
from time import time
from astropy.table import Table

@cuda.jit
def hermevander(x, deg, output_matrix):
    i, j = cuda.grid(2)
    _, stride = cuda.gridsize(2)
    for j in range(j, x.shape[1], stride):
        output_matrix[i][j][0] = 1
        if deg > 0:
            output_matrix[i][j][1] = x[i][j]
            for k in range(2, deg + 1):
                output_matrix[i][j][k] = output_matrix[i][j][k-1]*x[i][j] - output_matrix[i][j][k-2]*(k-1)

def hermevander_wrapper(x, deg):
    """Temprorary wrapper that allocates memory and calls hermevander_gpu
    """
    if x.ndim == 1:
        x = cp.expand_dims(x, 0)
    output = cp.ndarray(x.shape + (deg+1,))
    blocksize = 256
    numblocks = (x.shape[0], (x.shape[1] + blocksize - 1) // blocksize)
    hermevander[numblocks, blocksize](x, deg, output)
    return cp.squeeze(output)

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

    print(hermevander_cpu, hermevander_gpu.get())

    # Compare
    assert np.allclose(hermevander_cpu, hermevander_gpu.get())

