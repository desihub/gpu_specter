"""This module provides functions for performing linear algebra operations.
"""

import numpy
import cupy
import cupy.prof
import cupyx


@cupy.prof.TimeRangeDecorator("_batch_posv")
def _batch_posv(a, b):
    """Solve the linear equations A x = b via Cholesky factorization of A, where A
    is a real symmetric or complex Hermitian positive-definite matrix.

    If matrix ``a[i]`` is not positive definite, Cholesky factorization fails and
    it raises an error.

    Args:
        a (cupy.ndarray): Array of real symmetric or complex hermitian
            matrices with dimension (..., N, N).
        b (cupy.ndarray): right-hand side (..., N).
    Returns:
        x (cupy.ndarray): Array of solutions (..., N).
    """
    if not cupy.cusolver.check_availability('potrsBatched'):
        raise RuntimeError('potrsBatched is not available')

    dtype = numpy.promote_types(a.dtype, b.dtype)
    dtype = numpy.promote_types(dtype, 'f')

    if dtype == 'f':
        potrfBatched = cupy.cuda.cusolver.spotrfBatched
        potrsBatched = cupy.cuda.cusolver.spotrsBatched
    elif dtype == 'd':
        potrfBatched = cupy.cuda.cusolver.dpotrfBatched
        potrsBatched = cupy.cuda.cusolver.dpotrsBatched
    elif dtype == 'F':
        potrfBatched = cupy.cuda.cusolver.cpotrfBatched
        potrsBatched = cupy.cuda.cusolver.cpotrsBatched
    elif dtype == 'D':
        potrfBatched = cupy.cuda.cusolver.zpotrfBatched
        potrsBatched = cupy.cuda.cusolver.zpotrsBatched
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(a.dtype))
        raise ValueError(msg)

    # Cholesky factorization
    a = a.astype(dtype, order='C', copy=True)
    ap = cupy.core._mat_ptrs(a)
    lda, n = a.shape[-2:]
    batch_size = int(numpy.prod(a.shape[:-2]))

    handle = cupy.cuda.device.get_cusolver_handle()
    uplo = cupy.cuda.cublas.CUBLAS_FILL_MODE_LOWER
    dev_info = cupy.empty(batch_size, dtype=numpy.int32)

    potrfBatched(handle, uplo, n, ap.data.ptr, lda, dev_info.data.ptr,
                 batch_size)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrfBatched, dev_info)

    # Cholesky solve
    b_shape = b.shape
    b = b.conj().reshape(batch_size, n, -1).astype(dtype, order='C', copy=True)
    bp = cupy.core._mat_ptrs(b)
    ldb, nrhs = b.shape[-2:]
    dev_info = cupy.empty(1, dtype=numpy.int32)

    potrsBatched(handle, uplo, n, nrhs, ap.data.ptr, lda, bp.data.ptr, ldb,
                 dev_info.data.ptr, batch_size)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrsBatched, dev_info)

    return b.conj().reshape(b_shape)

@cupy.prof.TimeRangeDecorator("_posv")
def _posv(a, b):
    """Solve the linear equations A x = b via Cholesky factorization of A,
    where A is a real symmetric or complex Hermitian positive-definite matrix.

    If matrix ``A`` is not positive definite, Cholesky factorization fails
    and it raises an error.

    Note: For batch input, NRHS > 1 is not currently supported.

    Args:
        a (cupy.ndarray): Array of real symmetric or complex hermitian
            matrices with dimension (..., N, N).
        b (cupy.ndarray): right-hand side (..., N) or (..., N, NRHS).
    Returns:
        x (cupy.ndarray): The solution (shape matches b).
    """

    cupy.linalg._util._assert_cupy_array(a, b)
    cupy.linalg._util._assert_nd_squareness(a)

    if a.ndim > 2:
        return _batch_posv(a, b)

    dtype = numpy.promote_types(a.dtype, b.dtype)
    dtype = numpy.promote_types(dtype, 'f')

    if dtype == 'f':
        potrf = cupy.cuda.cusolver.spotrf
        potrf_bufferSize = cupy.cuda.cusolver.spotrf_bufferSize
        potrs = cupy.cuda.cusolver.spotrs
    elif dtype == 'd':
        potrf = cupy.cuda.cusolver.dpotrf
        potrf_bufferSize = cupy.cuda.cusolver.dpotrf_bufferSize
        potrs = cupy.cuda.cusolver.dpotrs
    elif dtype == 'F':
        potrf = cupy.cuda.cusolver.cpotrf
        potrf_bufferSize = cupy.cuda.cusolver.cpotrf_bufferSize
        potrs = cupy.cuda.cusolver.cpotrs
    elif dtype == 'D':
        potrf = cupy.cuda.cusolver.zpotrf
        potrf_bufferSize = cupy.cuda.cusolver.zpotrf_bufferSize
        potrs = cupy.cuda.cusolver.zpotrs
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(a.dtype))
        raise ValueError(msg)

    a = a.astype(dtype, order='F', copy=True)
    lda, n = a.shape

    handle = cupy.cuda.device.get_cusolver_handle()
    uplo = cupy.cuda.cublas.CUBLAS_FILL_MODE_LOWER
    dev_info = cupy.empty(1, dtype=numpy.int32)

    worksize = potrf_bufferSize(handle, uplo, n, a.data.ptr, lda)
    workspace = cupy.empty(worksize, dtype=dtype)

    # Cholesky factorization
    potrf(handle, uplo, n, a.data.ptr, lda, workspace.data.ptr,
          worksize, dev_info.data.ptr)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrf, dev_info)

    b_shape = b.shape
    b = b.reshape(n, -1).astype(dtype, order='F', copy=True)
    ldb, nrhs = b.shape

    # Solve: A * X = B
    potrs(handle, uplo, n, nrhs, a.data.ptr, lda, b.data.ptr, ldb,
          dev_info.data.ptr)
    cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrs, dev_info)

    return cupy.ascontiguousarray(b.reshape(b_shape))

@cupy.prof.TimeRangeDecorator("cholesky_solve")
def cholesky_solve(a, b):
    return _posv(a, b)

@cupy.prof.TimeRangeDecorator("clipped_eigh")
def clipped_eigh(a, clip_scale=1e-14):
    assert clip_scale >= 0
    w, v = cupy.linalg.eigh(a)
    #- clip eigenvalues relative to maximum eigenvalue
    #- TODO: assuming w is sorted, can skip cupy.max and use the appropriate index
    w = cupy.clip(w, a_min=clip_scale*cupy.max(w))
    return w, v

@cupy.prof.TimeRangeDecorator("compose_eigh")
def compose_eigh(w, v):
    return cupy.einsum('...ik,...k,...jk->...ij', v, w, v)

@cupy.prof.TimeRangeDecorator("matrix_sqrt")
def matrix_sqrt(a):
    #- eigen decomposition
    w, v = clipped_eigh(a)
    #- compose sqrt from eigen decomposition
    q = compose_eigh(cupy.sqrt(w), v)
    return q

@cupy.prof.TimeRangeDecorator("diag_block_matrix_sqrt")
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

