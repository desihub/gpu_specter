import numpy as np

try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cupy_available = False

def xp_deconvolve(pixel_values, pixel_ivar, A):
    """
    """
    try:
        xp = cp.get_array_module(A)
    except NameError:
        #- If the cupy module is unavailble, default to numpy
        xp = np
    #- Set up the equation to solve (B&S eq 4)
    ATNinv = A.T.dot(xp.diag(pixel_ivar))
    iCov = ATNinv.dot(A)
    iCov += 1e-12*xp.eye(iCov.shape[0])
    #- Solve the linear least-squares problem.
    #- Force rcond to be the same when using cupy/numpy 
    #- See: https://github.com/numpy/numpy/blob/v1.18.4/numpy/linalg/linalg.py#L2247
    rcond = np.core.finfo(iCov.dtype).eps * max(iCov.shape)
    deconvolved, res, rank, sing = xp.linalg.lstsq(iCov, ATNinv.dot(pixel_values), rcond=rcond)
    if rank < len(deconvolved):
        print('WARNING: deconvolved inverse-covariance is not positive definite.')
    return deconvolved, iCov

def xp_decorrelate(iCov):
    """
    """
    try:
        xp = cp.get_array_module(A)
    except NameError:
        # If the cupy module is unavailble, default to numpy
        xp = np
    # Calculate the matrix square root of iCov to diagonalize the flux errors.
    u, v = xp.linalg.eigh((iCov + iCov.T)/2.)
    # Check that all eigenvalues are positive.
    assert xp.all(u > 0), 'Found some negative iCov eigenvalues.'
    # Check that the eigenvectors are orthonormal so that vt.v = 1
    assert xp.allclose(xp.eye(len(u)), v.T.dot(v))
    Q = v.dot(xp.diag(xp.sqrt(u)).dot(v.T))
    # Check BS eqn.10
    assert xp.allclose(iCov, Q.dot(Q))
    #- Calculate the corresponding resolution matrix and diagonal flux errors. (BS Eq 11-13)
    s = xp.sum(Q, axis=1)
    R = Q/s[:, xp.newaxis]
    ivar = s**2
    # Check BS eqn.14
    assert xp.allclose(iCov, R.T.dot(xp.diag(ivar).dot(R)))
    return ivar, R
    

def xp_decorrelate_blocks(iCov, block_size):
    """
    """
    try:
        xp = cp.get_array_module(iCov)
    except NameError:
        # If the cupy module is unavailble, default to numpy
        xp = np
    size = iCov.shape[0]
    assert size % block_size == 0
    #- Invert iCov (B&S eq 17)
    u, v = xp.linalg.eigh((iCov + iCov.T)/2.)
    assert xp.all(u > 0), 'Found some negative iCov eigenvalues.'
    # Check that the eigenvectors are orthonormal so that vt.v = 1
    assert xp.allclose(xp.eye(len(u)), v.T.dot(v))
    C = (v * (1.0/u)).dot(v.T)
    #- Calculate C^-1 = QQ (B&S eq 17-19)
    Q = xp.zeros_like(iCov)
    #- Proceed one block at a time
    for i in range(0, size, block_size):
        s = np.s_[i:i+block_size, i:i+block_size]
        #- Invert this block
        bu, bv = xp.linalg.eigh(C[s])
        assert xp.all(bu > 0), 'Found some negative iCov eigenvalues.'
        # Check that the eigenvectors are orthonormal so that vt.v = 1
        assert xp.allclose(xp.eye(len(bu)), bv.T.dot(bv))
        bQ = (bv * xp.sqrt(1.0/bu)).dot(bv.T)
        Q[s] = bQ
    #- Calculate the corresponding resolution matrix and diagonal flux errors. (BS Eq 11-13)
    s = xp.sum(Q, axis=1)
    R = Q/s[:, xp.newaxis]
    ivar = s**2
    #- Check BS eqn.14
    assert xp.allclose(Q.dot(Q), R.T.dot(xp.diag(ivar).dot(R)))
    return ivar, R

def xp_ex2d_patch(img, ivar, A4, decorrelate='signal'):
    """
    """
    assert decorrelate in ('signal', 'noise')
    ny, nx, nspec, nwave = A4.shape
    assert img.shape == (ny, nx)
    # Flatten arrays
    pixel_values = img.ravel()
    pixel_ivar = ivar.ravel()
    A = A4.reshape(ny*nx, nspec*nwave)
    # Deconvole fiber traces
    deconvolved, iCov = xp_deconvolve(pixel_values, pixel_ivar, A)
    # Calculate the decorrelated errors and resolution matrix.
    if decorrelate == 'signal':
        ivar, resolution = xp_decorrelate_blocks(iCov, nwave)
    elif decorrelate == 'noise':
        ivar, resolution = xp_decorrelate(iCov)
    else:
        raise ValueError(f'{decorrelate} is not a valid value for decorrelate')
    # Convolve the reduced flux (BS eq 16)
    flux = resolution.dot(deconvolved).reshape(nspec, nwave)
    ivar = ivar.reshape(nspec, nwave)
    return flux, ivar, resolution
