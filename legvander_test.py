import cupy as cp
from numba import cuda
import numpy as np
import numba, pytest

@cuda.jit
def legvander(x, deg, output_matrix):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(i, x.shape[0], stride):
        output_matrix[i][0] = 1
        output_matrix[i][1] = x[i]
        for j in range(2, deg + 1):
            output_matrix[i][j] = (output_matrix[i][j-1]*x[i]*(2*j - 1) - output_matrix[i][j-2]*(j - 1)) / j

def test_main():
    # Generate dummy input
    degree = 10
    np.random.seed(1)
    x_cpu = np.random.rand(100)
    x_gpu = numba.cuda.to_device(x_cpu)

    # Calculate on cpu
    legvander_cpu = np.polynomial.legendre.legvander(x_cpu, degree)

    # Calculate on gpu
    legvander_gpu = cuda.device_array((len(x_cpu), degree + 1))
    blocksize = 256
    numblocks = (len(x_cpu) + blocksize - 1) // blocksize
    legvander[numblocks, blocksize](x_gpu, degree, legvander_gpu)

    #Compare
    assert np.allclose(legvander_cpu, legvander_gpu.copy_to_host())

