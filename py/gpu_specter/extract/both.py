"""
Tools for DESI spectroperfectionism extractions implemented in a generic fashion
for both CPU and GPU by leveraging the compatible API of NumPy and CuPy.
"""

import numpy as np

from ..util import Timer
from ..util import get_array_module
from .cpu import get_spec_padding

def safe_range_push(xp, message, id_color=-1):
    if xp.__name__ == 'cupy':
        xp.cuda.nvtx.RangePush(message, id_color)

def safe_range_pop(xp):
    if xp.__name__ == 'cupy':
        xp.cuda.nvtx.RangePop()

def xp_deconvolve(pixel_values, pixel_ivar, A, debug=False):
    """Calculate the weighted linear least-squares flux solution for an observed trace.

    Args:
        pixel_values (ny*nx,): 1D array of pixel values
        pixel_ivar (ny*nx,): 1D array of pixel inverse variances to use for weighting
        A (ny*nx, nspec*nwave): projection matrix that transforms a 1D spectrum into a 2D image

    Returns:
        deconvolved (nspec*nwave): the best-fit 1D array of flux values
        iCov (nspec*nwave, nspec*nwave): the correlated inverse covariance matrix of the deconvolved flux

    """
    xp = get_array_module(A)
    assert not debug or xp == get_array_module(pixel_values)
    assert not debug or xp == get_array_module(pixel_ivar)
    #- Set up the equation to solve (B&S eq 4)
    safe_range_push(xp, 'BS Eq 4 Setup')
    ATNinv = A.T * pixel_ivar
    iCov = ATNinv.dot(A)
    iCov += 1e-12*xp.eye(iCov.shape[0])
    safe_range_pop(xp)
    #- Solve the linear least-squares problem.
    #- Force rcond to be the same when using cupy/numpy 
    #- See: https://github.com/numpy/numpy/blob/v1.18.4/numpy/linalg/linalg.py#L2247
    # rcond = np.core.finfo(iCov.dtype).eps * max(iCov.shape)
    # deconvolved, res, rank, sing = xp.linalg.lstsq(iCov, ATNinv.dot(pixel_values), rcond=rcond)
    # if rank < len(deconvolved):
    #     print('WARNING: deconvolved inverse-covariance is not positive definite.')
    safe_range_push(xp, 'BS Eq 4 Solve')
    deconvolved = xp.linalg.solve(iCov, ATNinv.dot(pixel_values))
    safe_range_pop(xp)
    return deconvolved, iCov

def xp_decorrelate(iCov, debug=False):
    """Calculate the decorrelated errors and resolution matrix via BS Eq 10-13

    Args:
        iCov (nspec*nwave, nspec*nwave): the inverse covariance matrix

    Returns:
        ivar (ny*nx,): uncorrelated flux inverse variances
        R (nspec*nwave, nspec*nwave): resoultion matrix
    """
    xp = get_array_module(iCov)
    # Calculate the matrix square root of iCov to diagonalize the flux errors.
    u, v = xp.linalg.eigh(iCov)
    # Check that all eigenvalues are positive.
    assert not debug or xp.all(u > 0), 'Found some negative iCov eigenvalues.'
    # Check that the eigenvectors are orthonormal so that vt.v = 1
    assert not debug or xp.allclose(xp.eye(len(u)), v.T.dot(v))
    Q = v.dot(xp.diag(xp.sqrt(u)).dot(v.T))
    # Check BS eqn.10
    assert not debug or xp.allclose(iCov, Q.dot(Q))
    #- Calculate the corresponding resolution matrix and diagonal flux errors. (BS Eq 11-13)
    s = xp.sum(Q, axis=1)
    R = Q/s[:, xp.newaxis]
    ivar = s**2
    # Check BS eqn.14
    assert not debug or xp.allclose(iCov, R.T.dot(xp.diag(ivar).dot(R)))
    return ivar, R

def xp_decorrelate_blocks(iCov, block_size, debug=False):
    """Calculate the decorrelated errors and resolution matrix via BS Eq 19

    Args:
        iCov (nspec*nwave, nspec*nwave): the inverse covariance matrix
        block_size (int): size of the block corresponding to a single spectrum (i.e. nwave)

    Returns:
        ivar (ny*nx,): uncorrelated flux inverse variances
        R (nspec*nwave, nspec*nwave): resoultion matrix
    """
    xp = get_array_module(iCov)
    size = iCov.shape[0]
    assert not debug or size % block_size == 0
    #- Invert iCov (B&S eq 17)
    safe_range_push(xp, 'eigh icov', id_color=0)
    # u, v = xp.linalg.eigh((iCov + iCov.T)/2.)
    u, v = xp.cusolver.syevj((iCov + iCov.T)/2.)
    assert not debug or xp.all(u > 0), 'Found some negative iCov eigenvalues.'
    # Check that the eigenvectors are orthonormal so that vt.v = 1
    assert not debug or xp.allclose(xp.eye(len(u)), v.T.dot(v))
    safe_range_pop(xp) # eigh icov
    safe_range_push(xp, 'compose cov', id_color=1)
    C = (v * (1.0/u)).dot(v.T)
    safe_range_pop(xp) # compose cov
    #- Calculate C^-1 = QQ (B&S eq 17-19)
    safe_range_push(xp, 'batch eigh setup', id_color=1)
    A = xp.empty((size // block_size, block_size, block_size), dtype=iCov.dtype)
    for i, s in enumerate(range(0, size, block_size)):
        A[i] = C[s:s + block_size, s:s + block_size]
    safe_range_pop(xp) # batch eigh setup
    safe_range_push(xp, 'batch eigh solve', id_color=0)
    w, v = xp.linalg.eigh(A)
    safe_range_pop(xp) # batch eigh
    safe_range_push(xp, 'batch compose q', id_color=1)
    # v.dot(diag(w)).dot(v.T)
    q = xp.einsum('lij,ljk->lik', (v * xp.sqrt(1.0/w)[:, xp.newaxis, :]), v.transpose(0, 2, 1))
    safe_range_pop(xp) # batch compose q
    safe_range_push(xp, 'expand q', id_color=1)
    Q = xp.zeros_like(iCov)
    for i, s in enumerate(range(0, size, block_size)):
        Q[s:s + block_size, s:s + block_size] = q[i]
    safe_range_pop(xp) # expand q
    #- Calculate the corresponding resolution matrix and diagonal flux errors. (BS Eq 11-13)
    safe_range_push(xp, 'resolution and ivar')
    s = xp.sum(Q, axis=1)
    R = Q/s[:, xp.newaxis]
    ivar = s**2
    safe_range_pop(xp) # resolution and ivar
    #- Check BS eqn.14
    assert not debug or xp.allclose(Q.dot(Q), R.T.dot(xp.diag(ivar).dot(R)))
    return ivar, R

def xp_ex2d_patch(img, ivar, A4, decorrelate='signal', debug=False):
    """Perform spectroperfectionism extractions returning flux, ivar, and resolution matrix

    Args:
        img (ny, nx): 2D array of image pixel values
        imgivar (ny, nx): 2D array of independent pixel inverse variances
        A4 (ny, nx, spex, nwave): the projection matrix

    Returns:
        flux (nspec, nwave): extracted resolution convolved flux
        ivar (nspec, nwave): uncorrelated flux inverse variances
        R (nspec*nwave, nspec*nwave): dense resolution matrix
    """
    # timer = Timer()
    xp = get_array_module(A4)
    safe_range_push(xp, 'xp_ex2d_patch')
    assert not debug or decorrelate in ('signal', 'noise')
    ny, nx, nspec, nwave = A4.shape
    assert not debug or img.shape == (ny, nx)
    # Flatten arrays
    pixel_values = img.ravel()
    pixel_ivar = ivar.ravel()
    A = A4.reshape(ny*nx, nspec*nwave)
    # timer.split('init')
    # Deconvole fiber traces
    safe_range_push(xp, 'deconvolve')
    deconvolved, iCov = xp_deconvolve(pixel_values, pixel_ivar, A, debug=debug)
    safe_range_pop(xp)
    # timer.split('deconvolve')
    # Calculate the decorrelated errors and resolution matrix.
    safe_range_push(xp, 'decorrelate')
    if decorrelate == 'signal':
        fluxivar, resolution = xp_decorrelate_blocks(iCov, nwave, debug=debug)
    elif decorrelate == 'noise':
        fluxivar, resolution = xp_decorrelate(iCov, debug=debug)
    else:
        raise ValueError(f'{decorrelate} is not a valid value for decorrelate')
    safe_range_pop(xp)
    # timer.split('decorrelate')
    # Convolve the reduced flux (BS eq 16)
    safe_range_push(xp, 'reconvolve')
    flux = resolution.dot(deconvolved).reshape(nspec, nwave)
    fluxivar = fluxivar.reshape(nspec, nwave)
    safe_range_pop(xp)
    # timer.split('reconvolve')
    # timer.print_splits()
    safe_range_pop(xp)
    return flux, fluxivar, resolution
