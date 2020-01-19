import cupy as cp
from numba import cuda

@cuda.jit
def hermevander(x, deg, output_matrix):
    i = cuda.blockIdx.x
    _, j = cuda.grid(2)
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

