"""
NOTE: these functions are copied from "gpu_extract.py" in the hackathon branch;
the pieces have not yet been put together into a working GPU extraction
in this branch.
"""

import math

import numpy as np
import numpy.polynomial.legendre
from numba import cuda
import cupy as cp
import cupyx.scipy.special

from .cpu import get_spec_padding
from .both import xp_ex2d_patch
from ..io import native_endian
from ..util import Timer

@cuda.jit
def _hermevander(x, deg, output_matrix):
    i = cuda.blockIdx.x
    _, j = cuda.grid(2)
    _, stride = cuda.gridsize(2)
    for j in range(j, x.shape[1], stride):
        output_matrix[i][j][0] = 1
        if deg > 0:
            output_matrix[i][j][1] = x[i][j]
            for k in range(2, deg + 1):
                output_matrix[i][j][k] = output_matrix[i][j][k-1]*x[i][j] - output_matrix[i][j][k-2]*(k-1)

def hermevander(x, deg):
    """Temprorary wrapper that allocates memory and calls hermevander_gpu
    """
    if x.ndim == 1:
        x = cp.expand_dims(x, 0)
    output = cp.ndarray(x.shape + (deg+1,))
    blocksize = 256
    numblocks = (x.shape[0], (x.shape[1] + blocksize - 1) // blocksize)
    _hermevander[numblocks, blocksize](x, deg, output)
    return cp.squeeze(output)

@cuda.jit
def _legvander(x, deg, output_matrix):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(i, x.shape[0], stride):
        output_matrix[i][0] = 1
        output_matrix[i][1] = x[i]
        for j in range(2, deg + 1):
            output_matrix[i][j] = (output_matrix[i][j-1]*x[i]*(2*j - 1) - output_matrix[i][j-2]*(j - 1)) / j

def legvander(x, deg):
    """Temporary wrapper that allocates memory and defines grid before calling legvander.
    Probably won't be needed once cupy has the correpsponding legvander function.

    Input: Same as cpu version of legvander
    Output: legvander matrix, cp.ndarray
    """
    output = cp.ndarray((len(x), deg + 1))
    blocksize = 256
    numblocks = (len(x) + blocksize - 1) // blocksize
    _legvander[numblocks, blocksize](x, deg, output)
    return output

def evalcoeffs(psfdata, wavelengths, specmin=0, nspec=None):
    '''
    evaluate PSF coefficients parameterized as Legendre polynomials

    Args:
        psfdata: PSF data from io.read_psf() of Gauss Hermite PSF file
        wavelengths: 1D array of wavelengths

    Options:
        specmin: first spectrum to include
        nspec: number of spectra to include (default: all)

    Returns a dictionary params[paramname] = value[nspec, nwave]

    The Gauss Hermite coefficients are treated differently:

        params['GH'] = value[i,j,nspec,nwave]

    The dictionary also contains scalars with the recommended spot size
    2*(HSIZEX, HSIZEY)+1 and Gauss-Hermite degrees GHDEGX, GHDEGY
    (which is also derivable from the dimensions of params['GH'])
    '''
    if nspec is None:
        nspec = psfdata['PSF']['COEFF'].shape[1]

    p = dict(WAVE=wavelengths)

    #- Evaluate X and Y which have different dimensionality from the
    #- PSF coefficients (and might have different WAVEMIN, WAVEMAX)
    meta = psfdata['XTRACE'].meta
    wavemin, wavemax = meta['WAVEMIN'], meta['WAVEMAX']
    ww = (wavelengths - wavemin) * (2.0 / (wavemax - wavemin)) - 1.0
    # TODO: Implement cuda legval
    p['X'] = cp.asarray(numpy.polynomial.legendre.legval(ww, psfdata['XTRACE']['X'][specmin:specmin+nspec].T))

    meta = psfdata['YTRACE'].meta
    wavemin, wavemax = meta['WAVEMIN'], meta['WAVEMAX']
    ww = (wavelengths - wavemin) * (2.0 / (wavemax - wavemin)) - 1.0
    # TODO: Implement cuda legval
    p['Y'] = cp.asarray(numpy.polynomial.legendre.legval(ww, psfdata['YTRACE']['Y'][specmin:specmin+nspec].T))

    #- Evaluate the remaining PSF coefficients with a shared dimensionality
    #- and WAVEMIN, WAVEMAX
    meta = psfdata['PSF'].meta
    wavemin, wavemax = meta['WAVEMIN'], meta['WAVEMAX']
    ww = (wavelengths - wavemin) * (2.0 / (wavemax - wavemin)) - 1.0
    L = legvander(ww, meta['LEGDEG'])

    nparam = psfdata['PSF']['COEFF'].shape[0]
    ndeg = psfdata['PSF']['COEFF'].shape[2]

    nwave = L.shape[0]
    nghx = meta['GHDEGX']+1
    nghy = meta['GHDEGY']+1
    p['GH'] = cp.zeros((nghx, nghy, nspec, nwave))
    coeff_gpu = cp.array(native_endian(psfdata['PSF']['COEFF']))
    for name, coeff in zip(psfdata['PSF']['PARAM'], coeff_gpu):
        name = name.strip()
        coeff = coeff[specmin:specmin+nspec]
        if name.startswith('GH-'):
            i, j = map(int, name.split('-')[1:3])
            p['GH'][i,j] = L.dot(coeff.T).T
        else:
            p[name] = L.dot(coeff.T).T

    #- Include some additional keywords that we'll need
    for key in ['HSIZEX', 'HSIZEY', 'GHDEGX', 'GHDEGY']:
        p[key] = meta[key]

    return p

def calc_pgh(ispec, wavelengths, psfparams):
    '''
    Calculate the pixelated Gauss Hermite for all wavelengths of a single spectrum

    ispec : integer spectrum number
    wavelengths : array of wavelengths to evaluate
    psfparams : dictionary of PSF parameters returned by evalcoeffs

    returns pGHx, pGHy

    where pGHx[ghdeg+1, nwave, nbinsx] contains the pixel-integrated Gauss-Hermite polynomial
    for all degrees at all wavelengths across nbinsx bins spaning the PSF spot, and similarly
    for pGHy.  The core PSF will then be evaluated as

    PSFcore = sum_ij c_ij outer(pGHy[j], pGHx[i])
    '''

    #- shorthand
    p = psfparams

    #- spot size (ny,nx)
    nx = 2*p['HSIZEX'] + 1
    ny = 2*p['HSIZEY'] + 1
    nwave = len(wavelengths)
    #- convert to cupy arrays
    for k in ['X', 'Y', 'GHSIGX', 'GHSIGY']:
        p[k] = cp.asarray(p[k])

    #- x and y edges of bins that span the center of the PSF spot
    xedges = cp.repeat(cp.arange(nx+1) - nx//2 - 0.5, nwave).reshape(nx+1, nwave)
    yedges = cp.repeat(cp.arange(ny+1) - ny//2 - 0.5, nwave).reshape(ny+1, nwave)

    #- Shift to be relative to the PSF center and normalize
    #- by the PSF sigma (GHSIGX, GHSIGY).
    #- Note: x,y = 0,0 is center of pixel 0,0 not corner
    #- Dimensions: xedges[nx+1, nwave], yedges[ny+1, nwave]
    dx = (p['X'][ispec]+0.5)%1 - 0.5
    dy = (p['Y'][ispec]+0.5)%1 - 0.5
    xedges = ((xedges - dx)/p['GHSIGX'][ispec])
    yedges = ((yedges - dy)/p['GHSIGY'][ispec])

    #- Degree of the Gauss-Hermite polynomials
    ghdegx = p['GHDEGX']
    ghdegy = p['GHDEGY']

    #- Evaluate the Hermite polynomials at the pixel edges
    #- HVx[ghdegx+1, nwave, nx+1]
    #- HVy[ghdegy+1, nwave, ny+1]
    HVx = hermevander(xedges, ghdegx).T
    HVy = hermevander(yedges, ghdegy).T

    #- Evaluate the Gaussians at the pixel edges
    #- Gx[nwave, nx+1]
    #- Gy[nwave, ny+1]
    Gx = cp.exp(-0.5*xedges**2).T / cp.sqrt(2. * cp.pi)
    Gy = cp.exp(-0.5*yedges**2).T / cp.sqrt(2. * cp.pi)

    #- Combine into Gauss*Hermite
    GHx = HVx * Gx
    GHy = HVy * Gy

    #- Integrate over the pixels using the relationship
    #  Integral{ H_k(x) exp(-0.5 x^2) dx} = -H_{k-1}(x) exp(-0.5 x^2) + const

    #- pGHx[ghdegx+1, nwave, nx]
    #- pGHy[ghdegy+1, nwave, ny]
    pGHx = cp.zeros((ghdegx+1, nwave, nx))
    pGHy = cp.zeros((ghdegy+1, nwave, ny))
    pGHx[0] = 0.5 * cp.diff(cupyx.scipy.special.erf(xedges/cp.sqrt(2.)).T)
    pGHy[0] = 0.5 * cp.diff(cupyx.scipy.special.erf(yedges/cp.sqrt(2.)).T)
    pGHx[1:] = GHx[:ghdegx,:,0:nx] - GHx[:ghdegx,:,1:nx+1]
    pGHy[1:] = GHy[:ghdegy,:,0:ny] - GHy[:ghdegy,:,1:ny+1]
    
    return pGHx, pGHy


@cuda.jit()
def _multispot(pGHx, pGHy, ghc, spots):
    nx = pGHx.shape[-1]
    ny = pGHy.shape[-1]
    nwave = pGHx.shape[1]

    #this is the magic step
    iwave = cuda.grid(1)

    n = pGHx.shape[0]
    m = pGHy.shape[0]

    if (0 <= iwave < nwave):
    #yanked out the i and j loops in lieu of the cuda grid of threads
        for i in range(pGHx.shape[0]):
            px = pGHx[i,iwave]
            for j in range(0, pGHy.shape[0]):
                py = pGHy[j,iwave]
                c = ghc[i,j,iwave]
                for iy in range(len(py)):
                    for ix in range(len(px)):
                        spots[iwave, iy, ix] += c * py[iy] * px[ix]

def multispot(pGHx, pGHy, ghc):
    nx = pGHx.shape[-1]
    ny = pGHy.shape[-1]
    nwave = pGHx.shape[1]
    blocksize = 256
    numblocks = (nwave + blocksize - 1) // blocksize
    spots = cp.zeros((nwave, ny, nx)) #empty every time!
    _multispot[numblocks, blocksize](pGHx, pGHy, ghc, spots)
    return spots


def get_spots(specmin, nspec, wavelengths, psfdata):
    '''Calculate PSF spots for the specified spectra and wavelengths

    Args:
        specmin: first spectrum to include
        nspec: number of spectra to evaluate spots for
        wavelengths: 1D array of wavelengths
        psfdata: PSF data from io.read_psf() of Gauss Hermite PSF file

    Returns:
        spots: 4D array[ispec, iwave, ny, nx] of PSF spots
        corners: (xc,yc) where each is 2D array[ispec,iwave] lower left corner of spot

    '''
    nwave = len(wavelengths)
    p = evalcoeffs(psfdata, wavelengths, specmin, nspec)
    nx = 2*p['HSIZEX']+1
    ny = 2*p['HSIZEY']+1
    spots = cp.zeros((nspec, nwave, ny, nx))
    for ispec in range(nspec):
        pGHx, pGHy = calc_pgh(ispec, wavelengths, p)
        spots[ispec] = multispot(pGHx, pGHy, p['GH'][:,:,ispec,:])

    #- ensure positivity and normalize
    #- TODO: should this be within multispot itself?
    spots = spots.clip(0.0)
    norm = cp.sum(spots, axis=(2,3))  #- norm[nspec, nwave] = sum over each spot
    spots = (spots.T / norm.T).T      #- transpose magic for numpy array broadcasting

    #- Define corners of spots
    #- extra 0.5 is because X and Y are relative to center of pixel not edge
    xc = np.floor(p['X'] - p['HSIZEX'] + 0.5).astype(int)
    yc = np.floor(p['Y'] - p['HSIZEY'] + 0.5).astype(int)

    corners = (xc, yc)

    return spots, corners


@cuda.jit()
def _cuda_projection_matrix(A, xc, yc, xmin, ymin, ispec, iwave, nspec, nwave, spots):
    #this is the heart of the projection matrix calculation
    ny, nx = spots.shape[2:4]
    i, j = cuda.grid(2)
    #no loops, just a boundary check
    if (0 <= i < nspec) and (0 <= j <nwave):
        ixc = xc[ispec+i, iwave+j] - xmin
        iyc = yc[ispec+i, iwave+j] - ymin
        #A[iyc:iyc+ny, ixc:ixc+nx, i, j] = spots[ispec+i,iwave+j]
        #this fancy indexing is not allowed in numba gpu (although it is in numba cpu...)
        #try this instead
        for iy, y in enumerate(range(iyc,iyc+ny)):
            for ix, x in enumerate(range(ixc,ixc+nx)):
                temp_spot = spots[ispec+i, iwave+j][iy, ix]
                A[y, x, i, j] += temp_spot

def get_xyrange(ispec, nspec, iwave, nwave, spots, corners):
    """
    Find xy ranges that these spectra cover

    Args:
        ispec: starting spectrum index
        nspec: number of spectra
        iwave: starting wavelength index
        nwave: number of wavelengths
        spots: 4D array[ispec, iwave, ny, nx] of PSF spots
        corners: (xc,yc) where each is 2D array[ispec,iwave] lower left corner of spot

    Returns (xmin, xmax, ymin, ymax)

    spots[ispec:ispec+nspec,iwave:iwave+nwave] touch pixels[ymin:ymax,xmin:xmax]
    """
    ny, nx = spots.shape[2:4]

    # Note: transfer corners back to host
    xc = corners[0][ispec:ispec+nspec, iwave:iwave+nwave].get()
    yc = corners[1][ispec:ispec+nspec, iwave:iwave+nwave].get()

    xmin = np.min(xc)
    xmax = np.max(xc) + nx
    ymin = np.min(yc)
    ymax = np.max(yc) + ny

    return xmin, xmax, ymin, ymax

def projection_matrix(ispec, nspec, iwave, nwave, spots, corners):
    '''
    Create the projection matrix A for p = Af

    Args:
        ispec: starting spectrum index
        nspec: number of spectra
        iwave: starting wavelength index
        nwave: number of wavelengths
        spots: 4D array[ispec, iwave, ny, nx] of PSF spots
        corners: (xc,yc) where each is 2D array[ispec,iwave] lower left corner of spot

    Returns (A[iy, ix, ispec, iwave], (xmin, xmax, ymin, ymax))
    '''
    xc, yc = corners
    xmin, xmax, ymin, ymax = get_xyrange(ispec, nspec, iwave, nwave, spots, corners)
    A = cp.zeros((ymax-ymin,xmax-xmin,nspec,nwave), dtype=np.float64)

    threads_per_block = (16, 16)
    blocks_per_grid_y = math.ceil(A.shape[0] / threads_per_block[0])
    blocks_per_grid_x = math.ceil(A.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    _cuda_projection_matrix[blocks_per_grid, threads_per_block](
        A, xc, yc, xmin, ymin, ispec, iwave, nspec, nwave, spots)

    return A, (xmin, xmax, ymin, ymax)


def get_resolution_diags(R, ndiag, ispec, nspec, nwave, wavepad):
    """Returns the diagonals of R in a form suited for creating scipy.sparse.dia_matrix

    Args:
        R: dense resolution matrix
        ndiag: number of diagonal elements to keep in the resolution matrix
        ispec: starting spectrum index relative to padding
        nspec: number of spectra to extract (not including padding)
        nwave: number of wavelengths to extract (not including padding)
        wavepad: number of extra wave bins to extract (and discard) on each end

    Returns:
        Rdiags (nspec,  2*ndiag+1, nwave): resolution matrix diagonals
    """
    nwavetot = 2*wavepad + nwave
    Rdiags = cp.zeros( (nspec, 2*ndiag+1, nwave) )
    mask = (
        ~cp.tri(nwave, nwavetot, (wavepad-ndiag-1), dtype=bool) &
        cp.tri(nwave, nwavetot, (wavepad+ndiag), dtype=bool)
    )
    for i in range(ispec, ispec+nspec):
        ii = slice(nwavetot*i, nwavetot*(i+1))
        Rdiags[i-ispec] = R[ii, ii][:,wavepad:-wavepad].T[mask].reshape(nwave, 2*ndiag+1).T
    return Rdiags

def prepare_patch(image, imageivar, ispec, nspec, iwave, nwave, spots, corners, wavepad, bundlesize):
    specmin, nspecpad = get_spec_padding(ispec, nspec, bundlesize)
    nwavetot = nwave+2*wavepad
    A4, xyrange = projection_matrix(specmin, nspecpad, iwave-wavepad, nwave+2*wavepad, spots, corners)
    xmin, xmax, ypadmin, ypadmax = xyrange

    xlo, xhi, ymin, ymax = get_xyrange(specmin, nspecpad, iwave, nwave, spots, corners)

    ypadlo = ymin - ypadmin
    ypadhi = ypadmax - ymax
    A4 = A4[ypadlo:-ypadhi]

    ny, nx = A4.shape[0:2]

    if (0 <= ymin) & (ymin+ny <= image.shape[0]):
        xyslice = np.s_[ymin:ymin+ny, xmin:xmin+nx]
        patchpixels = image[xyslice]
        patchivar = imageivar[xyslice]
    else:
        xyslice = None
        patchivar = cp.zeros((ny, nx))
        patchpixels = cp.zeros((ny, nx))

    return patchpixels, patchivar, A4, xyslice

def apply_weights(pixel_values, pixel_ivar, A, regularize=0, weight_scale=1e-4):
    ATNinv = A.T * pixel_ivar
    icov = ATNinv.dot(A)
    y = ATNinv.dot(pixel_values)
    fluxweight = ATNinv.sum(axis=1)

    minweight = weight_scale*cp.max(fluxweight)
    ibad = fluxweight <= minweight
    lambda_squared = regularize*regularize*cp.ones_like(y)
    lambda_squared[ibad] = minweight - fluxweight[ibad]
    if np.any(lambda_squared):
        icov += cp.diag(lambda_squared)

    return icov, y

def batch_apply_weights(batch_pixels, batch_ivar, batch_A4, regularize=0, weight_scale=1e-4):

    batch_size = len(batch_A4)
    ny, nx, nspecpad, nwavetot = batch_A4[0].shape
    nbin = nspecpad * nwavetot

    batch_icov = cp.zeros((batch_size, nbin, nbin))
    batch_y = cp.zeros((batch_size, nbin))
    for i, (pix, ivar, A4) in enumerate(zip(batch_pixels, batch_ivar, batch_A4)):
        # Note that each patch can have a different number of pixels
        batch_icov[i], batch_y[i] = apply_weights(
            pix.ravel(), ivar.ravel(), A4.reshape(-1, nbin),
            regularize=regularize, weight_scale=weight_scale)

    return batch_icov, batch_y

def batch_cholesky_solve(a, b):
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
    if not cp.cusolver.check_availability('potrsBatched'):
        raise RuntimeError('potrsBatched is not available')

    dtype = numpy.promote_types(a.dtype, b.dtype)
    dtype = numpy.promote_types(dtype, 'f')

    if dtype == 'f':
        potrfBatched = cp.cuda.cusolver.spotrfBatched
        potrsBatched = cp.cuda.cusolver.spotrsBatched
    elif dtype == 'd':
        potrfBatched = cp.cuda.cusolver.dpotrfBatched
        potrsBatched = cp.cuda.cusolver.dpotrsBatched
    elif dtype == 'F':
        potrfBatched = cp.cuda.cusolver.cpotrfBatched
        potrsBatched = cp.cuda.cusolver.cpotrsBatched
    elif dtype == 'D':
        potrfBatched = cp.cuda.cusolver.zpotrfBatched
        potrsBatched = cp.cuda.cusolver.zpotrsBatched
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(a.dtype))
        raise ValueError(msg)

    # Cholesky factorization
    a = a.astype(dtype, order='C', copy=True)
    ap = cp.core._mat_ptrs(a)
    lda, n = a.shape[-2:]
    batch_size = int(numpy.prod(a.shape[:-2]))

    handle = cp.cuda.device.get_cusolver_handle()
    uplo = cp.cuda.cublas.CUBLAS_FILL_MODE_LOWER
    dev_info = cp.empty(batch_size, dtype=numpy.int32)

    potrfBatched(handle, uplo, n, ap.data.ptr, lda, dev_info.data.ptr,
                 batch_size)
    cp.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrfBatched, dev_info)

    # Cholesky solve
    b_shape = b.shape
    b = b.conj().reshape(batch_size, n, -1).astype(dtype, order='C', copy=True)
    bp = cp.core._mat_ptrs(b)
    ldb, nrhs = b.shape[-2:]
    dev_info = cp.empty(1, dtype=numpy.int32)

    potrsBatched(handle, uplo, n, nrhs, ap.data.ptr, lda, bp.data.ptr, ldb,
                 dev_info.data.ptr, batch_size)
    cp.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrsBatched, dev_info)

    return b.conj().reshape(b_shape)

import cupyx

def batch_decorrelate(batch_icov, block_size, clip_scale=0):

    batch_size, n, m = batch_icov.shape
    nblocks, remainder = divmod(n, block_size)
    assert n == m
    assert remainder == 0

    cp.cuda.nvtx.RangePush('batch_invert_icov')
    # invert icov
    # cov = cp.linalg.inv(batch_icov)
    cp.cuda.nvtx.RangePush('eigh')
    w, v = cp.linalg.eigh(batch_icov)
    cp.cuda.nvtx.RangePop() # eigh


    cp.cuda.nvtx.RangePush('compose')
    if clip_scale > 0:
        w = cp.clip(w, a_min=clip_scale*cp.max(w))
    cov = cp.einsum('...ik,...k,...jk->...ij', v, 1.0/w, v)
    cp.cuda.nvtx.RangePop() # compose
    cp.cuda.nvtx.RangePop() # batch_invert_icov

    cp.cuda.nvtx.RangePush('extract_blocks')
    cov_block_diags = cp.empty(
        (batch_size * nblocks, block_size, block_size),
        dtype=batch_icov.dtype
    )
    for i in range(batch_size):
        for j, s in enumerate(range(0, n, block_size)):
            cov_block_diags[i*nblocks + j] = cov[i, s:s + block_size, s:s + block_size]
    cp.cuda.nvtx.RangePop() # extract_blocks

    cp.cuda.nvtx.RangePush('eigh')
    ww, vv = cp.linalg.eigh(cov_block_diags)
    cp.cuda.nvtx.RangePop() # eigh
    cp.cuda.nvtx.RangePush('compose')
    if clip_scale > 0:
        ww = cp.clip(ww, a_min=clip_scale*cp.max(ww))
    q = cp.einsum('...ik,...k,...jk->...ij', vv, cupyx.rsqrt(ww), vv)
    cp.cuda.nvtx.RangePop() # compose

    cp.cuda.nvtx.RangePush('replace_blocks')
    Q = cp.zeros_like(batch_icov)
    for i in range(batch_size):
        for j, s in enumerate(range(0, n, block_size)):
            Q[i, s:s + block_size, s:s + block_size] = q[i*nblocks + j]
    cp.cuda.nvtx.RangePop() # replace_blocks

    return Q

def batch_extraction(batch_pixels, batch_ivar, batch_A4, regularize=0, clip_scale=0):

    batch_size = len(batch_A4)
    ny, nx, nspecpad, nwavetot = batch_A4[0].shape

    cp.cuda.nvtx.RangePush('apply_weights')
    batch_icov, batch_y = batch_apply_weights(batch_pixels, batch_ivar, batch_A4, regularize=regularize)
    cp.cuda.nvtx.RangePop() # apply_weights

    cp.cuda.nvtx.RangePush('deconvolve')
    deconvolved = batch_cholesky_solve(batch_icov, batch_y)
    cp.cuda.nvtx.RangePop() # deconvolve

    cp.cuda.nvtx.RangePush('decorrelate')
    batch_Q = batch_decorrelate(batch_icov, nwavetot, clip_scale=clip_scale)

    cp.cuda.nvtx.RangePush('apply_resolution')
    s = cp.einsum('...ij->...i', batch_Q)
    batch_resolution = batch_Q/s[..., cp.newaxis]
    batch_fluxivar = (s*s).reshape(-1, nspecpad, nwavetot)
    batch_flux = cp.einsum('...ij,...j->...i', batch_resolution, deconvolved).reshape(-1, nspecpad, nwavetot)
    cp.cuda.nvtx.RangePop() # apply_resolution

    cp.cuda.nvtx.RangePop() # decorrelate

    return batch_flux, batch_fluxivar, batch_resolution

def finalize_patch(patchpixels, patchivar, A4, xyslice, fx, ivarfx, R,
    ispec, nspec, bundlesize, nwave, wavepad, ndiag, psferr, model=None):

    specmin, nspecpad = get_spec_padding(ispec, nspec, bundlesize)

    specslice = np.s_[ispec-specmin:ispec-specmin+nspec,wavepad:wavepad+nwave]
    specflux = fx[specslice]
    specivar = ivarfx[specslice]
    Rdiags = get_resolution_diags(R, ndiag, ispec-specmin, nspec, nwave, wavepad)

    ny, nx, nspecpad, nwavetot = A4.shape

    Apadded = A4.reshape(ny*nx, nspecpad*nwavetot)
    Apatch = A4[:, :, ispec-specmin:ispec-specmin+nspec, wavepad:wavepad+nwave]
    Apatch = Apatch.reshape(ny*nx, nspec*nwave)

    pixmask_fraction = Apatch.T.dot(patchivar.ravel() == 0)
    pixmask_fraction = pixmask_fraction.reshape(nspec, nwave)

    modelpadded = Apadded.dot(fx.ravel()).reshape(ny, nx)
    modelivar = (modelpadded*psferr + 1e-32)**-2
    ii = (modelivar > 0 ) & (patchivar > 0)
    totpix_ivar = cp.zeros((ny, nx))
    totpix_ivar[ii] = 1.0 / (1.0/modelivar[ii] + 1.0/patchivar[ii])

    #- Weighted chi2 of pixels that contribute to each flux bin;
    #- only use unmasked pixels and avoid dividing by 0
    chi = (patchpixels - modelpadded)*cp.sqrt(totpix_ivar)
    psfweight = Apadded.T.dot(totpix_ivar.ravel() > 0)
    bad = psfweight == 0

    #- Compute chi2pix and reshape
    chi2pix = (Apadded.T.dot(chi.ravel()**2) * ~bad) / (psfweight + bad)
    chi2pix = chi2pix.reshape(nspecpad, nwavetot)[specslice]

    if model:
        modelimage = Apatch.dot(specflux.ravel()).reshape(ny, nx)
    else:
        #modelimage = cp.zeros((ny, nx))
        modelimage = None

    result = dict(
        flux = specflux,
        ivar = specivar,
        Rdiags = Rdiags,
        modelimage = modelimage,
        xyslice = xyslice,
        pixmask_fraction = pixmask_fraction,
        chi2pix = chi2pix,
    )

    return result


def ex2d_padded(image, imageivar, ispec, nspec, iwave, nwave, spots, corners, psferr,
                wavepad, bundlesize=25, model=None, regularize=0):
    """
    Extracted a patch with border padding, but only return results for patch

    Args:
        image: full image (not trimmed to a particular xy range)
        imageivar: image inverse variance (same dimensions as image)
        ispec: starting spectrum index relative to `spots` indexing
        nspec: number of spectra to extract (not including padding)
        iwave: starting wavelength index
        nwave: number of wavelengths to extract (not including padding)
        spots: array[nspec, nwave, ny, nx] pre-evaluated PSF spots
        corners: tuple of arrays xcorners[nspec, nwave], ycorners[nspec, nwave]
        wavepad: number of extra wave bins to extract (and discard) on each end

    Options:
        bundlesize: size of fiber bundles; padding not needed on their edges
    """
    # timer = Timer()

    specmin, nspecpad = get_spec_padding(ispec, nspec, bundlesize)

    #- Total number of wavelengths to be extracted, including padding
    nwavetot = nwave+2*wavepad

    # timer.split('init')

    #- Get the projection matrix for the full wavelength range with padding
    cp.cuda.nvtx.RangePush('projection_matrix')
    A4, xyrange = projection_matrix(specmin, nspecpad,
        iwave-wavepad, nwave+2*wavepad, spots, corners)
    cp.cuda.nvtx.RangePop()
    # timer.split('projection_matrix')

    xmin, xmax, ypadmin, ypadmax = xyrange

    #- But we only want to use the pixels covered by the original wavelengths
    #- TODO: this unnecessarily also re-calculates xranges
    cp.cuda.nvtx.RangePush('get_xyrange')
    xlo, xhi, ymin, ymax = get_xyrange(specmin, nspecpad, iwave, nwave, spots, corners)
    cp.cuda.nvtx.RangePop()
    # timer.split('get_xyrange')

    ypadlo = ymin - ypadmin
    ypadhi = ypadmax - ymax
    A4 = A4[ypadlo:-ypadhi]

    #- Number of image pixels in y and x
    ny, nx = A4.shape[0:2]

    #- Check dimensions
    assert A4.shape[2] == nspecpad
    assert A4.shape[3] == nwave + 2*wavepad

    #- Diagonals of R in a form suited for creating scipy.sparse.dia_matrix
    ndiag = spots.shape[2]//2
    cp.cuda.nvtx.RangePush('Rdiags allocation')
    Rdiags = cp.zeros( (nspec, 2*ndiag+1, nwave) )
    cp.cuda.nvtx.RangePop()

    specslice = np.s_[ispec-specmin:ispec-specmin+nspec,wavepad:wavepad+nwave]
    if (0 <= ymin) & (ymin+ny <= image.shape[0]):
        xyslice = np.s_[ymin:ymin+ny, xmin:xmin+nx]
        patchpixels = image[xyslice]
        patchivar = imageivar[xyslice]
        # timer.split('ready for extraction')
        cp.cuda.nvtx.RangePush('extract patch')
        fx, ivarfx, R = xp_ex2d_patch(patchpixels, patchivar, A4, regularize=regularize)
        cp.cuda.nvtx.RangePop()
        # timer.split('extracted patch')

        #- Select the non-padded spectra x wavelength core region
        cp.cuda.nvtx.RangePush('select slices to keep')
        cp.cuda.nvtx.RangePush('slice flux')
        specflux = fx[specslice]
        cp.cuda.nvtx.RangePop()
        cp.cuda.nvtx.RangePush('slice ivar')
        specivar = ivarfx[specslice]
        cp.cuda.nvtx.RangePop()

        cp.cuda.nvtx.RangePush('slice R')
        Rdiags = get_resolution_diags(R, ndiag, ispec-specmin, nspec, nwave, wavepad)
        # timer.split('saved Rdiags')
        cp.cuda.nvtx.RangePop()
        cp.cuda.nvtx.RangePop()

    else:
        #- TODO: this zeros out the entire patch if any of it is off the edge
        #- of the image; we can do better than that
        fx = cp.zeros((nspecpad, nwavetot))
        specflux = cp.zeros((nspec, nwave))
        specivar = cp.zeros((nspec, nwave))
        Rdiags = cp.zeros( (nspec, 2*ndiag+1, nwave) )
        xyslice = None
        patchivar = cp.zeros((ny, nx))
        patchpixels = cp.zeros((ny, nx))

    if cp.any(cp.isnan(specflux)):
        raise RuntimeError('Found NaN in extracted flux')

    Apadded = A4.reshape(ny*nx, nspecpad*nwavetot)
    Apatch = A4[:, :, ispec-specmin:ispec-specmin+nspec, wavepad:wavepad+nwave]
    Apatch = Apatch.reshape(ny*nx, nspec*nwave)

    pixmask_fraction = Apatch.T.dot(patchivar.ravel() == 0)
    pixmask_fraction = pixmask_fraction.reshape(nspec, nwave)

    modelpadded = Apadded.dot(fx.ravel()).reshape(ny, nx)
    modelivar = (modelpadded*psferr + 1e-32)**-2
    ii = (modelivar > 0 ) & (patchivar > 0)
    totpix_ivar = cp.zeros((ny, nx))
    totpix_ivar[ii] = 1.0 / (1.0/modelivar[ii] + 1.0/patchivar[ii])

    #- Weighted chi2 of pixels that contribute to each flux bin;
    #- only use unmasked pixels and avoid dividing by 0
    chi = (patchpixels - modelpadded)*cp.sqrt(totpix_ivar)
    psfweight = Apadded.T.dot(totpix_ivar.ravel() > 0)
    bad = psfweight == 0

    #- Compute chi2pix and reshape
    chi2pix = (Apadded.T.dot(chi.ravel()**2) * ~bad) / (psfweight + bad)
    chi2pix = chi2pix.reshape(nspecpad, nwavetot)[specslice]

    if model:
        modelimage = Apatch.dot(specflux.ravel()).reshape(ny, nx)
    else:
        #modelimage = cp.zeros((ny, nx))
        modelimage = None

    #- TODO: add chi2pix, pixmask_fraction, optionally modelimage; see specter
    cp.cuda.nvtx.RangePush('prepare result')
    result = dict(
        flux = specflux,
        ivar = specivar,
        Rdiags = Rdiags,
        modelimage = modelimage,
        xyslice = xyslice,
        pixmask_fraction = pixmask_fraction,
        chi2pix = chi2pix,
    )
    cp.cuda.nvtx.RangePop()
    # timer.split('done')
    # timer.print_splits()

    return result
