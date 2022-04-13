"""This module provides functions for performing linear algebra operations.
"""

import numpy
import cupy
import cupyx

from cupyx.profiler import time_range

#- _posv and _batch_posv are superseded by cupyx.lapack.posv as of CuPy v9.0.0


@time_range("cholesky_solve")
def cholesky_solve(a, b):
    return cupyx.lapack.posv(a, b)

@time_range("clipped_eigh")
def clipped_eigh(a, clip_scale=1e-14):
    assert clip_scale >= 0
    w, v = cupy.linalg.eigh(a)
    #- clip eigenvalues relative to maximum eigenvalue
    #- TODO: assuming w is sorted, can skip cupy.max and use the appropriate index
    w = cupy.clip(w, a_min=clip_scale*cupy.max(w))
    return w, v

@time_range("compose_eigh")
def compose_eigh(w, v):
    return cupy.einsum('...ik,...k,...jk->...ij', v, w, v)

@time_range("matrix_sqrt")
def matrix_sqrt(a):
    #- eigen decomposition
    w, v = clipped_eigh(a)
    #- compose sqrt from eigen decomposition
    q = compose_eigh(cupy.sqrt(w), v)
    return q

@time_range("diag_block_matrix_sqrt")
def diag_block_matrix_sqrt(a, block_size):
    a_shape = a.shape
    n, m = a_shape[-2:]
    batch_size = numpy.prod(a_shape[:-2], dtype=int)
    nblocks, remainder = divmod(n, block_size)
    assert n == m
    assert remainder == 0
    #- flatten batch dimensions
    a = a.reshape(batch_size, n, m)
    #- eigen decomposition
    w, v = clipped_eigh(a)
    #- compose inverse from eigen decomposition
    ainv = compose_eigh(1.0/w, v)
    #- extract diagonal blocks
    #- TODO: use a view of diagonal blocks instead of copy?
    ainv_diag_blocks = cupy.empty(
        (batch_size * nblocks, block_size, block_size),
        dtype=a.dtype
    )
    for i in range(batch_size):
        for j, s in enumerate(range(0, n, block_size)):
            bs = slice(s, s + block_size)
            ainv_diag_blocks[i*nblocks + j] = ainv[i, bs, bs]
    #- eigen decomposition
    w, v = clipped_eigh(ainv_diag_blocks)
    #- compose inverse sqrt from eigen decomposition
    q_diag_blocks = compose_eigh(cupyx.rsqrt(w), v)
    #- insert block sqrts into result
    #- TODO: is there a way to avoid this new alloc/copy?
    q = cupy.zeros_like(a)
    for i in range(batch_size):
        for j, s in enumerate(range(0, n, block_size)):
            bs = slice(s, s + block_size)
            q[i, bs, bs] = q_diag_blocks[i*nblocks + j]
    return q.reshape(a_shape)

