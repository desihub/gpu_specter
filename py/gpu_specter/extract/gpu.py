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
import cupy.prof
import cupyx
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

@cupy.prof.TimeRangeDecorator("hermevander")
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

@cupy.prof.TimeRangeDecorator("legvander")
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

@cupy.prof.TimeRangeDecorator("evalcoeffs")
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

@cupy.prof.TimeRangeDecorator("calc_pgh")
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

@cupy.prof.TimeRangeDecorator("multispot")
def multispot(pGHx, pGHy, ghc):
    nx = pGHx.shape[-1]
    ny = pGHy.shape[-1]
    nwave = pGHx.shape[1]
    blocksize = 256
    numblocks = (nwave + blocksize - 1) // blocksize
    spots = cp.zeros((nwave, ny, nx)) #empty every time!
    _multispot[numblocks, blocksize](pGHx, pGHy, ghc, spots)
    return spots

@cupy.prof.TimeRangeDecorator("get_spots")
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

        # spots[ispec] = cp.einsum('lmk,mkj,lki->kji',
        #     p['GH'][:,:,ispec,:], pGHy, pGHx, optimize='greedy')

    # spots = cp.einsum('lmnk,mkj,lki->nkji', p['GH'], pGHy, pGHx, optimize='greedy')

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

@cupy.prof.TimeRangeDecorator("get_xyrange")
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

    xc = corners[0][ispec:ispec+nspec, iwave:iwave+nwave]
    yc = corners[1][ispec:ispec+nspec, iwave:iwave+nwave]

    xmin = np.min(xc)
    xmax = np.max(xc) + nx
    ymin = np.min(yc)
    ymax = np.max(yc) + ny

    return xmin, xmax, ymin, ymax

@cupy.prof.TimeRangeDecorator("projection_matrix")
def projection_matrix(ispec, nspec, iwave, nwave, spots, corners, corners_cpu):
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
    xmin, xmax, ymin, ymax = get_xyrange(ispec, nspec, iwave, nwave, spots, corners_cpu)
    A = cp.zeros((ymax-ymin,xmax-xmin,nspec,nwave), dtype=np.float64)

    threads_per_block = (16, 16)
    blocks_per_grid_y = math.ceil(A.shape[0] / threads_per_block[0])
    blocks_per_grid_x = math.ceil(A.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    _cuda_projection_matrix[blocks_per_grid, threads_per_block](
        A, xc, yc, xmin, ymin, ispec, iwave, nspec, nwave, spots)

    return A, (xmin, xmax, ymin, ymax)

@cp.memoize()
def _rdiags_mask(ndiag, nspecpad, nwave, wavepad):
    nwavetot = 2*wavepad + nwave
    n = nspecpad*nwavetot
    ii = cp.c_[cp.arange(n)]
    # select elements near diagonal
    mask = cp.abs(ii + -ii.T) <= ndiag
    # select elements in core wavelength regions
    mask &= (cp.abs((2 * (ii % nwavetot) - (nwavetot - 0.5))) <= nwave)
    return mask

@cupy.prof.TimeRangeDecorator("get_resolution_diags")
def get_resolution_diags(R, ndiag, nspecpad, nwave, wavepad):
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
    mask = _rdiags_mask(ndiag, nspecpad, nwave, wavepad)
    Rdiags = R.T[mask].reshape(nspecpad, nwave, -1).swapaxes(-2, -1)
    return Rdiags

@cupy.prof.TimeRangeDecorator("ex2d_padded")
def ex2d_padded(image, imageivar, ispec, nspec, iwave, nwave, spots, corners, psferr,
                wavepad, bundlesize=25, model=None, regularize=0):
    """
    Extracts a patch with border padding, but only return results for patch

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

    #- Yikes, pulling this out from get_xyrange
    corners_cpu = (corners[0].get(), corners[1].get())
    #- Get patch pixels and projection matrix
    patchpixels, patchivar, patchA4, xyslice = _prepare_patch(
        image, imageivar, ispec, nspec, iwave, nwave, spots, corners, corners_cpu, wavepad, bundlesize,
    )
    #- Standardize problem size
    icov, y = _apply_weights(
        patchpixels.ravel(), patchivar.ravel(), patchA4.reshape(patchpixels.size, -1),
        regularize=regularize
    )
    #- Perform the extraction
    nwavetot = nwave + 2*wavepad
    flux, fluxivar, resolution = _batch_extraction(icov, y, nwavetot, clip_scale=0)
    #- Finalize the output for this patch
    ndiag = spots.shape[2]//2
    result = _finalize_patch(
        patchpixels, patchivar, patchA4, xyslice,
        flux, fluxivar, resolution,
        ispec, nspec, bundlesize,
        nwave, wavepad, ndiag, psferr, model=model
    )

    return result

@cupy.prof.TimeRangeDecorator("_prepare_patch")
def _prepare_patch(image, imageivar, ispec, nspec, iwave, nwave, spots, corners, corners_cpu, wavepad, bundlesize):
    """This is essentially the preamble of `gpu_specter.extract.gpu.ex2d_padded`"""

    #- Get the projection matrix for the full wavelength range with padding
    specmin, nspecpad = get_spec_padding(ispec, nspec, bundlesize)
    wavemin, nwavepad = iwave-wavepad, nwave+2*wavepad
    A4, xyrange = projection_matrix(specmin, nspecpad, wavemin, nwavepad, spots, corners, corners_cpu)
    xmin, xmax, ypadmin, ypadmax = xyrange

    #- But we only want to use the pixels covered by the original wavelengths
    #- TODO: this unnecessarily also re-calculates xranges
    xlo, xhi, ymin, ymax = get_xyrange(specmin, nspecpad, iwave, nwave, spots, corners_cpu)

    ypadlo = ymin - ypadmin
    ypadhi = ypadmax - ymax
    A4 = A4[ypadlo:-ypadhi]

    #- Number of image pixels in y and x
    ny, nx = A4.shape[0:2]

    #- Check dimensions
    assert A4.shape[2] == nspecpad
    assert A4.shape[3] == nwavepad

    if (0 <= ymin) & (ymin+ny <= image.shape[0]):
        xyslice = np.s_[ymin:ymin+ny, xmin:xmin+nx]
        patchpixels = image[xyslice]
        patchivar = imageivar[xyslice]
    else:
        #- TODO: this zeros out the entire patch if any of it is off the edge
        #- of the image; we can do better than that
        xyslice = None
        patchivar = cp.zeros((ny, nx))
        patchpixels = cp.zeros((ny, nx))

    return patchpixels, patchivar, A4, xyslice

@cupy.fuse()
def _regularize(ATNinv, regularize, weight_scale):
    fluxweight = ATNinv.sum(axis=1)
    minweight = weight_scale*cp.max(fluxweight)
    ii = fluxweight > minweight
    lambda_squared = ~ii*(minweight - fluxweight) + ii*regularize*regularize
    return lambda_squared

@cupy.prof.TimeRangeDecorator("_apply_weights")
def _apply_weights(pixel_values, pixel_ivar, A, regularize=0, weight_scale=1e-4):
    """This is essentially the preamble of of `gpu_specter.extract.both.xp_deconvolve`
    The outputs of this will be uniform shape for a subbundle.
    """
    ATNinv = A.T * pixel_ivar
    icov = ATNinv.dot(A)
    y = ATNinv.dot(pixel_values)
    fluxweight = ATNinv.sum(axis=1)

    minweight = weight_scale*cp.max(fluxweight)
    ibad = fluxweight <= minweight
    lambda_squared = regularize*regularize*cp.ones_like(y)
    lambda_squared[ibad] = minweight - fluxweight[ibad]
    # if np.any(lambda_squared):
    icov += cp.diag(lambda_squared)

    # icov += cp.diag(_regularize(ATNinv, regularize, weight_scale))

    return icov, y

@cupy.prof.TimeRangeDecorator("_batch_apply_weights")
def _batch_apply_weights(batch_pixels, batch_ivar, batch_A4, regularize=0, weight_scale=1e-4):
    """Turns a list of subbundle patch inputs into batch arrays of unifom shape
    """

    batch_size = len(batch_A4)
    ny, nx, nspecpad, nwavetot = batch_A4[0].shape
    nbin = nspecpad * nwavetot

    batch_icov = cp.zeros((batch_size, nbin, nbin))
    batch_y = cp.zeros((batch_size, nbin))
    for i, (pix, ivar, A4) in enumerate(zip(batch_pixels, batch_ivar, batch_A4)):
        # Note that each patch can have a different number of pixels
        batch_icov[i], batch_y[i] = _apply_weights(
            pix.ravel(), ivar.ravel(), A4.reshape(-1, nbin),
            regularize=regularize, weight_scale=weight_scale)

    return batch_icov, batch_y

@cupy.prof.TimeRangeDecorator("_batch_cholesky_solve")
def _batch_cholesky_solve(a, b):
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

@cupy.prof.TimeRangeDecorator("cholesky_solve")
def cholesky_solve(a, b):
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

    cp.linalg._util._assert_cupy_array(a, b)
    cp.linalg._util._assert_nd_squareness(a)

    if a.ndim > 2:
        return _batch_cholesky_solve(a, b)

    dtype = np.promote_types(a.dtype, b.dtype)
    dtype = np.promote_types(dtype, 'f')

    if dtype == 'f':
        potrf = cp.cuda.cusolver.spotrf
        potrf_bufferSize = cp.cuda.cusolver.spotrf_bufferSize
        potrs = cp.cuda.cusolver.spotrs
    elif dtype == 'd':
        potrf = cp.cuda.cusolver.dpotrf
        potrf_bufferSize = cp.cuda.cusolver.dpotrf_bufferSize
        potrs = cp.cuda.cusolver.dpotrs
    elif dtype == 'F':
        potrf = cp.cuda.cusolver.cpotrf
        potrf_bufferSize = cp.cuda.cusolver.cpotrf_bufferSize
        potrs = cp.cuda.cusolver.cpotrs
    elif dtype == 'D':
        potrf = cp.cuda.cusolver.zpotrf
        potrf_bufferSize = cp.cuda.cusolver.zpotrf_bufferSize
        potrs = cp.cuda.cusolver.zpotrs
    else:
        msg = ('dtype must be float32, float64, complex64 or complex128'
               ' (actual: {})'.format(a.dtype))
        raise ValueError(msg)

    a = a.astype(dtype, order='F', copy=True)
    lda, n = a.shape

    handle = cp.cuda.device.get_cusolver_handle()
    uplo = cp.cuda.cublas.CUBLAS_FILL_MODE_LOWER
    dev_info = cp.empty(1, dtype=np.int32)

    worksize = potrf_bufferSize(handle, uplo, n, a.data.ptr, lda)
    workspace = cp.empty(worksize, dtype=dtype)

    # Cholesky factorization
    potrf(handle, uplo, n, a.data.ptr, lda, workspace.data.ptr,
          worksize, dev_info.data.ptr)
    cp.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrf, dev_info)

    b_shape = b.shape
    b = b.reshape(n, -1).astype(dtype, order='F', copy=True)
    ldb, nrhs = b.shape

    # Solve: A * X = B
    potrs(handle, uplo, n, nrhs, a.data.ptr, lda, b.data.ptr, ldb,
          dev_info.data.ptr)
    cp.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(
        potrs, dev_info)

    return cp.ascontiguousarray(b.reshape(b_shape))

@cupy.prof.TimeRangeDecorator("_batch_decorrelate_noise")
def _batch_decorrelate_noise(icov):
    """Batch version of the simple decorrelation method"""
    cp.cuda.nvtx.RangePush('batch_sqrt_icov')
    cp.cuda.nvtx.RangePush('eigh')
    w, v = xp.linalg.eigh(iCov)
    cp.cuda.nvtx.RangePop() # eigh
    cp.cuda.nvtx.RangePush('compose')
    Q = cp.einsum('...ik,...k,...jk->...ij', v, cp.sqrt(w), v)
    cp.cuda.nvtx.RangePop() # compose
    cp.cuda.nvtx.RangePop() # batch_sqrt_icov
    return Q

@cupy.prof.TimeRangeDecorator("_batch_decorrelate")
def _batch_decorrelate(icov, block_size, clip_scale=0):
    """Batch version of the better decorrelation method"""

    icov_shape = icov.shape
    n, m = icov_shape[-2:]
    batch_size = np.prod(icov_shape[:-2], dtype=int)
    nblocks, remainder = divmod(n, block_size)
    assert n == m
    assert remainder == 0
    icov = icov.reshape(batch_size, n, m)

    cp.cuda.nvtx.RangePush('batch_invert_icov')
    # invert icov
    # cov = cp.linalg.inv(icov)
    cp.cuda.nvtx.RangePush('eigh')
    w, v = cp.linalg.eigh(icov)
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
        dtype=icov.dtype
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
    Q = cp.zeros_like(icov)
    for i in range(batch_size):
        for j, s in enumerate(range(0, n, block_size)):
            Q[i, s:s + block_size, s:s + block_size] = q[i*nblocks + j]
    cp.cuda.nvtx.RangePop() # replace_blocks

    return Q.reshape(icov_shape)

@cupy.prof.TimeRangeDecorator("_batch_apply_resolution")
def _batch_apply_resolution(deconvolved, Q):
    """Compute and apply resolution to deconvolved flux"""
    s = cp.einsum('...ij->...i', Q)
    resolution = Q/s[..., cp.newaxis]
    fluxivar = s*s
    flux = cp.einsum('...ij,...j->...i', resolution, deconvolved)
    return flux, fluxivar, resolution

@cupy.prof.TimeRangeDecorator("_batch_extraction")
def _batch_extraction(icov, y, nwavetot, clip_scale=0):
    """Performs batch extraction given a batch of patches from a subbundle.

    Note that the inputs are lists of ndarrays because the patches on the ccd are not
    the same size.
    """

    # batch_size = len(A4)
    # ny, nx, nspecpad, nwavetot = A4[0].shape

    # cp.cuda.nvtx.RangePush('apply_weights')
    # icov, y = _batch_apply_weights(pixel_values, pixel_ivar, A4, regularize=regularize)
    # cp.cuda.nvtx.RangePop() # apply_weights

    cp.cuda.nvtx.RangePush('deconvolve')
    deconvolved = cholesky_solve(icov, y)
    cp.cuda.nvtx.RangePop() # deconvolve

    cp.cuda.nvtx.RangePush('decorrelate')
    Q = _batch_decorrelate(icov, nwavetot, clip_scale=clip_scale)
    # Q = _batch_decorrelate_noise(icov)
    cp.cuda.nvtx.RangePop() # decorrelate

    cp.cuda.nvtx.RangePush('apply_resolution')
    flux, fluxivar, resolution = _batch_apply_resolution(deconvolved, Q)
    cp.cuda.nvtx.RangePop() # apply_resolution

    return flux, fluxivar, resolution

@cp.fuse()
def compute_chisq(patchpixels, patchivar, patchmodel, psferr):
    modelsigma = psferr*patchmodel
    ii = (modelsigma > 0 ) & (patchivar > 0)
    totpix_ivar = ii*cp.reciprocal(~ii + ii*modelsigma*modelsigma + ii*cp.reciprocal(ii*patchivar+~ii))
    chi = (patchpixels - patchmodel)*cp.sqrt(totpix_ivar)
    return chi*chi

@cp.fuse()
def reweight_chisq(chi2pix, weight):
    bad = weight == 0
    return (chi2pix * ~bad) / (weight + bad)

@cupy.prof.TimeRangeDecorator("_finalize_patch")
def _finalize_patch(patchpixels, patchivar, A4, xyslice, fx, ivarfx, R,
    ispec, nspec, bundlesize, nwave, wavepad, ndiag, psferr, model=None):
    """This is essentially the postamble of gpu_specter.extract.gpu.ex2d_padded."""

    specmin, nspecpad = get_spec_padding(ispec, nspec, bundlesize)

    ny, nx, nspecpad, nwavetot = A4.shape

    #- Select the non-padded spectra x wavelength core region
    specslice = np.s_[ispec-specmin:ispec-specmin+nspec,wavepad:wavepad+nwave]
    specflux = fx.reshape(nspecpad, nwavetot)[specslice]
    specivar = ivarfx.reshape(nspecpad, nwavetot)[specslice]
    #- Diagonals of R in a form suited for creating scipy.sparse.dia_matrix
    Rdiags = get_resolution_diags(R, ndiag, nspecpad, nwave, wavepad)[specslice[0]]

    # if cp.any(cp.isnan(specflux)):
    #     raise RuntimeError('Found NaN in extracted flux')

    patchpixels = patchpixels.ravel()
    patchivar = patchivar.ravel()

    cp.cuda.nvtx.RangePush('pixmask_fraction')
    Apatch = A4[:, :, specslice[0], specslice[1]]
    Apatch = Apatch.reshape(ny*nx, nspec*nwave)
    pixmask_fraction = Apatch.T.dot(patchivar == 0)
    pixmask_fraction = pixmask_fraction.reshape(nspec, nwave)
    cp.cuda.nvtx.RangePop() # pixmask_fraction

    #- Weighted chi2 of pixels that contribute to each flux bin;
    #- only use unmasked pixels and avoid dividing by 0
    cp.cuda.nvtx.RangePush('chi2pix')
    cp.cuda.nvtx.RangePush('modelpadded')
    Apadded = A4.reshape(ny*nx, nspecpad*nwavetot)
    patchmodel = Apadded.dot(fx.ravel())
    cp.cuda.nvtx.RangePop()
    cp.cuda.nvtx.RangePush('chi2')
    chi2 = compute_chisq(patchpixels, patchivar, patchmodel, psferr)
    cp.cuda.nvtx.RangePop()
    cp.cuda.nvtx.RangePush('Apadded dot chi2')
    chi2pix = Apadded.T.dot(chi2)
    cp.cuda.nvtx.RangePop()
    cp.cuda.nvtx.RangePush('psfweight')
    psfweight = Apadded.T.dot(chi2 > 0)
    chi2pix = reweight_chisq(chi2pix, psfweight)
    cp.cuda.nvtx.RangePop()
    chi2pix = chi2pix.reshape(nspecpad, nwavetot)[specslice]
    cp.cuda.nvtx.RangePop() # chi2pix

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

def ex2d_subbundle(image, imageivar, patches, spots, corners, bundlesize, regularize, clip_scale, psferr, model):
    """Extract an entire subbundle of patches. The patches' output shape (nspec, nwave) must be aligned.

    Args:
        image: full image (not trimmed to a particular xy range)
        imageivar: image inverse variance (same dimensions as image)
        patches: list contain gpu_specter.core.Patch objects for extraction
        spots: array[nspec, nwave, ny, nx] pre-evaluated PSF spots
        corners: tuple of arrays xcorners[nspec, nwave], ycorners[nspec, nwave]
        bundlesize: size of fiber bundles
        clip_scale: scale factor to use when clipping eigenvalues
        psferr: value of error to assume in psf model
        model: compute image pixel model using extracted flux

    Returns:
        results: list of (patch, result) tuples
    """
    batch_pixels = list()
    batch_ivar = list()
    batch_A4 = list()
    batch_xyslice = list()
    batch_icov = list()
    batch_y = list()

    corners_cpu = (corners[0].get(), corners[1].get())

    cp.cuda.nvtx.RangePush('batch_prepare')
    for patch in patches:
        patchpixels, patchivar, patchA4, xyslice = _prepare_patch(
            image, imageivar, patch.ispec-patch.bspecmin, patch.nspectra_per_patch,
            patch.iwave, patch.nwavestep, spots, corners, corners_cpu, wavepad=patch.wavepad, bundlesize=bundlesize,
        )
        icov, y = _apply_weights(
            patchpixels.ravel(), patchivar.ravel(), patchA4.reshape(patchpixels.size, -1),
            regularize=regularize
        )
        patch.xyslice = xyslice
        batch_pixels.append(patchpixels)
        batch_ivar.append(patchivar)
        batch_A4.append(patchA4)
        batch_xyslice.append(xyslice)
        batch_icov.append(icov)
        batch_y.append(y)
    batch_icov = cp.array(batch_icov)
    batch_y = cp.array(batch_y)
    cp.cuda.nvtx.RangePop()

    # perform batch extraction
    cp.cuda.nvtx.RangePush('batch_extraction')
    # batch_flux, batch_fluxivar, batch_resolution = _batch_extraction(
    #     batch_pixels, batch_ivar, batch_A4, regularize=regularize, clip_scale=clip_scale
    # )
    nwavetot = patches[0].nwavestep + 2*patches[0].wavepad
    batch_flux, batch_fluxivar, batch_resolution = _batch_extraction(
        batch_icov, batch_y, nwavetot, clip_scale=clip_scale
    )
    cp.cuda.nvtx.RangePop()

    # finalize patch results
    cp.cuda.nvtx.RangePush('batch_finalize')
    results = list()
    for i, patch in enumerate(patches):
        result = _finalize_patch(
            batch_pixels[i], batch_ivar[i], batch_A4[i], batch_xyslice[i],
            batch_flux[i], batch_fluxivar[i], batch_resolution[i],
            patch.ispec-patch.bspecmin, patch.nspectra_per_patch, bundlesize,
            patch.nwavestep, patch.wavepad, patch.ndiag, psferr, model=model
        )
        results.append( (patches[i], result) )
    cp.cuda.nvtx.RangePop()

    return results


