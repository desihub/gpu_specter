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
from cupyx.profiler import time_range
import cupyx
import cupyx.scipy.special

from .cpu import get_spec_padding
from .both import xp_ex2d_patch
from ..io import native_endian
from ..util import Timer
from ..linalg import (
    cholesky_solve,
    matrix_sqrt,
    diag_block_matrix_sqrt,
)
from ..polynomial import (
    hermevander,
    legvander,
)


default_weight_scale = 1e-4


@time_range("evalcoeffs")
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

    if wavelengths.ndim == 2:
        wave2d = True
    else:
        wave2d = False
    nwave = wavelengths.shape[-1]

    p = dict()

    #- Evaluate X and Y which have different dimensionality from the
    #- PSF coefficients (and might have different WAVEMIN, WAVEMAX)
    for k in ['X', 'Y']:
        meta = psfdata[k + 'TRACE'].meta
        wavemin, wavemax = meta['WAVEMIN'], meta['WAVEMAX']
        ww = (wavelengths - wavemin) * (2.0 / (wavemax - wavemin)) - 1.0
        # TODO: Implement cuda legval
        if wave2d:
            ww = ww[specmin:specmin+nspec]
            cur_pk = np.zeros((nspec, ww.shape[-1]))
            for i in range(nspec):
                cur_pk[i] = numpy.polynomial.legendre.legval(ww[i], psfdata[k+'TRACE'][k][specmin+i])
            p[k] = cp.asarray(cur_pk)
        else:
            p[k] = cp.asarray(numpy.polynomial.legendre.legval(ww.T,
                                                               psfdata[k+'TRACE'][k][specmin:specmin+nspec].T))

    #- Evaluate the remaining PSF coefficients with a shared dimensionality
    #- and WAVEMIN, WAVEMAX
    meta = psfdata['PSF'].meta
    wavemin, wavemax = meta['WAVEMIN'], meta['WAVEMAX']
    ww = (wavelengths - wavemin) * (2.0 / (wavemax - wavemin)) - 1.0

    if wave2d:
        L = cp.zeros((nspec, nwave, meta['LEGDEG']+1))
        for i in range(nspec):
            L[i] = legvander(ww[specmin+i], meta['LEGDEG'])
    else:
        L = legvander(ww, meta['LEGDEG'])
    # L has a shape of either nspec,nwave,ndeg or nwave, ndeg
    nghx = meta['GHDEGX']+1
    nghy = meta['GHDEGY']+1
    p['GH'] = cp.zeros((nghx, nghy, nspec, nwave))
    coeff_gpu = cp.array(native_endian(psfdata['PSF']['COEFF']))
    for name, coeff in zip(psfdata['PSF']['PARAM'], coeff_gpu):
        name = name.strip()
        coeff = coeff[specmin:specmin+nspec]
        if wave2d:
            curv = cp.einsum('kji,ki->kj', L, coeff) # L.dot(coeff.T).T
        else:
            curv = cp.einsum('ji,ki->kj', L, coeff) # L.dot(coeff.T).T

        if name.startswith('GH-'):
            i, j = map(int, name.split('-')[1:3])
            p['GH'][i, j] = curv
        else:
            p[name] = curv

    #- Include some additional keywords that we'll need
    for key in ['HSIZEX', 'HSIZEY', 'GHDEGX', 'GHDEGY']:
        p[key] = meta[key]

    return p

@time_range("calc_pgh")
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
    nwave = wavelengths.shape[-1]
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

@time_range("multispot")
def multispot(pGHx, pGHy, ghc):
    nx = pGHx.shape[-1]
    ny = pGHy.shape[-1]
    nwave = pGHx.shape[1]
    blocksize = 256
    numblocks = (nwave + blocksize - 1) // blocksize
    spots = cp.zeros((nwave, ny, nx)) #empty every time!
    _multispot[numblocks, blocksize](pGHx, pGHy, ghc, spots)
    cuda.synchronize()
    return spots

@time_range("get_spots")
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

    return spots, corners, p

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

@time_range("get_xyrange")
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

@time_range("projection_matrix")
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
    cp.cuda.nvtx.RangePush('allocate A')
    A = cp.zeros((ymax-ymin,xmax-xmin,nspec,nwave), dtype=np.float64)
    cp.cuda.nvtx.RangePop()


    cp.cuda.nvtx.RangePush('blocks_per_grid')
    threads_per_block = (16, 16)
    blocks_per_grid_y = math.ceil(A.shape[0] / threads_per_block[0])
    blocks_per_grid_x = math.ceil(A.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    cp.cuda.nvtx.RangePop()

    cp.cuda.nvtx.RangePush('_cuda_projection_matrix')
    _cuda_projection_matrix[blocks_per_grid, threads_per_block](
        A, xc, yc, xmin, ymin, ispec, iwave, nspec, nwave, spots)
    cuda.synchronize()
    cp.cuda.nvtx.RangePop()

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

@time_range("get_resolution_diags")
def get_resolution_diags(R, ndiag, nspecpad, nwave, wavepad):
    """Returns the diagonals of R in a form suited for creating scipy.sparse.dia_matrix

    Args:
        R: dense resolution matrix
        ndiag: number of diagonal elements to keep in the resolution matrix
        nspec: number of spectra to extract (not including padding)
        nwave: number of wavelengths to extract (not including padding)
        wavepad: number of extra wave bins to extract (and discard) on each end

    Returns:
        Rdiags (nspec,  2*ndiag+1, nwave): resolution matrix diagonals
    """
    mask = _rdiags_mask(ndiag, nspecpad, nwave, wavepad)
    Rdiags = R.T[mask].reshape(nspecpad, nwave, -1).swapaxes(-2, -1)
    # NOTE: I think this is actually correct but need to compare with specter
    # Rdiags = R[mask].reshape(nspecpad, nwave, -1).swapaxes(-2, -1)
    return Rdiags

@time_range("ex2d_padded")
def ex2d_padded(image, imageivar, patch, spots, corners, pixpad_frac, regularize, model, psferr):
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
    ispec = patch.ispec - patch.bspecmin
    nspec = patch.nspectra_per_patch
    iwave = patch.iwave
    nwave = patch.nwavestep
    wavepad = patch.wavepad
    
    #- Yikes, pulling this out from get_xyrange
    corners_cpu = (corners[0].get(), corners[1].get())
    #- Get patch pixels and projection matrix
    specmin, nspectot = get_spec_padding(ispec, nspec, patch.bundlesize)
    patchpixels, patchivar, patchA4, xyslice = _prepare_patch(
        image, imageivar, specmin, nspectot, iwave, nwave, wavepad, spots, corners, corners_cpu, pixpad_frac
    )
    #- Standardize problem size
    icov, y = _apply_weights(
        patchpixels.ravel(), patchivar.ravel(), patchA4.reshape(patchpixels.size, -1),
        regularize=regularize
    )
    #- Perform the extraction
    nwavetot = nwave + 2*wavepad
    flux, fluxivar, resolution, xflux = _batch_extraction(icov, y, nwavetot)
    #- Finalize the output for this patch
    ndiag = spots.shape[2]//2
    result = _finalize_patch(
        patchpixels, patchivar, patchA4, xyslice,
        flux, fluxivar, resolution, xflux,
        ispec-specmin, nspec,
        nwave, wavepad, ndiag, psferr, patch, model=model
    )

    return result

@time_range("_prepare_patch")
def _prepare_patch(image, imageivar, specmin, nspectot, iwave, nwave, wavepad, spots, corners, corners_cpu, pixpad_frac):
    """This is essentially the preamble of `gpu_specter.extract.gpu.ex2d_padded`"""

    #- Get the projection matrix for the full wavelength range with padding
    # specmin, nspectot = get_spec_padding(ispec, nspec, bundlesize)
    wavemin, nwavetot = iwave-wavepad, nwave+2*wavepad
    A4, xyrange = projection_matrix(specmin, nspectot, wavemin, nwavetot, spots, corners, corners_cpu)
    xmin, xmax, ypadmin, ypadmax = xyrange

    #- But we only want to use the pixels covered by the original wavelengths
    #- TODO: this unnecessarily also re-calculates xranges
    xlo, xhi, ymin, ymax = get_xyrange(specmin, nspectot, iwave, nwave, spots, corners_cpu)

    # ypadlo = ymin - ypadmin
    # ypadhi = ypadmax - ymax
    # A4 = A4[ypadlo:-ypadhi]
    #- TODO: for ypadmax=ymax the above logic will not work
    ypadlo = int((ymin - ypadmin) * (1 - pixpad_frac))
    ypadhi = int((ymax - ypadmin) + (ypadmax - ymax) * (pixpad_frac))
    A4 = A4[ypadlo:ypadhi]
    
    #- use padded pixel boundaries
    # ymin, ymax = ypadmin, ypadmax

    #- Number of image pixels in y and x
    ny, nx = A4.shape[0:2]
    ymin = ypadmin+ypadlo
    ymax = ypadmin+ypadhi

    #- Check dimensions
    assert A4.shape[2] == nspectot
    assert A4.shape[3] == nwavetot

    if (0 <= ymin) & (ymin+ny <= image.shape[0]):
        xyslice = np.s_[ymin:ymin+ny, xmin:xmin+nx]
        patchpixels = image[xyslice]
        patchivar = imageivar[xyslice]
    elif ymin+ny > image.shape[0]:
        ny = image.shape[0] - ymin
        A4 = A4[:ny]
        xyslice = np.s_[ymin:ymin+ny, xmin:xmin+nx]
        patchpixels = image[xyslice]
        patchivar = imageivar[xyslice]
    else:
        #- TODO: this zeros out the entire patch if any of it is off the edge
        #- of the image; we can do better than that
        #print('offedge:', ymin, ymin+ny, image.shape[0], flush=True)
        xyslice = None
        patchivar = cp.zeros((ny, nx))
        patchpixels = cp.zeros((ny, nx))

    return patchpixels, patchivar, A4, xyslice

@cp.fuse()
def _regularize(ATNinv, regularize, weight_scale):
    fluxweight = ATNinv.sum(axis=-1)
    minweight = weight_scale*cp.max(fluxweight)
    ibad = fluxweight <= minweight
    lambda_squared = ibad*(minweight - fluxweight) + ~ibad*regularize*regularize
    return lambda_squared

@time_range("_apply_weights")
def _apply_weights(pixel_values, pixel_ivar, A, regularize, weight_scale=default_weight_scale):
    """This is essentially the preamble of of `gpu_specter.extract.both.xp_deconvolve`
    The outputs of this will be uniform shape for a subbundle.
    """
    ATNinv = A.T * pixel_ivar
    icov = ATNinv.dot(A)
    y = ATNinv.dot(pixel_values)
    fluxweight = ATNinv.sum(axis=-1)

    minweight = weight_scale*cp.max(fluxweight)
    ibad = fluxweight <= minweight
    alpha = regularize*cp.ones_like(y)
    alpha[ibad] = minweight - fluxweight[ibad]
    icov += cp.diag(alpha*alpha + 1e-15)
    
    #- TODO: is cupy.fuse() faster?
    # icov += cp.diag(_regularize(ATNinv, regularize, weight_scale))

    return icov, y

@time_range("_batch_apply_weights")
def _batch_apply_weights(batch_pixels, batch_ivar, batch_A4, regularize, weight_scale=default_weight_scale):
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

@time_range("_batch_apply_resolution")
def _batch_apply_resolution(deconvolved, Q):
    """Compute and apply resolution to deconvolved flux"""
    s = cp.einsum('...ij->...i', Q)
    resolution = Q/s[..., cp.newaxis]
    fluxivar = s*s
    flux = cp.einsum('...ij,...j->...i', resolution, deconvolved)
    return flux, fluxivar, resolution

@time_range("_batch_extraction")
def _batch_extraction(icov, y, nwavetot):
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
    xflux = cholesky_solve(icov, y)
    cp.cuda.nvtx.RangePop() # deconvolve

    cp.cuda.nvtx.RangePush('decorrelate')
    Q = diag_block_matrix_sqrt(icov, nwavetot)
    #- TODO: implement alternate noise decorrelation path
    # Q = matrix_sqrt(icov)
    cp.cuda.nvtx.RangePop() # decorrelate

    cp.cuda.nvtx.RangePush('apply_resolution')
    flux, fluxivar, resolution = _batch_apply_resolution(xflux, Q)
    cp.cuda.nvtx.RangePop() # apply_resolution

    return flux, fluxivar, resolution, xflux

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

@time_range("_finalize_patch")
def _finalize_patch(patchpixels, patchivar, A4, xyslice, flux, fluxivar, R, xflux,
    ispec, nspec, nwave, wavepad, ndiag, psferr, patch, model=None):
    """This is essentially the postamble of gpu_specter.extract.gpu.ex2d_padded."""

    ny, nx, nspectot, nwavetot = A4.shape

    #- Select the non-padded spectra x wavelength core region
    specslice = np.s_[ispec:ispec+nspec,wavepad:wavepad+nwave]
    specflux = flux.reshape(nspectot, nwavetot)[specslice]
    specivar = fluxivar.reshape(nspectot, nwavetot)[specslice]
    #- Diagonals of R in a form suited for creating scipy.sparse.dia_matrix
    Rdiags = get_resolution_diags(R, ndiag, nspectot, nwave, wavepad)[specslice[0]]

    # if cp.any(cp.isnan(specflux[:, patch.keepslice])):
    #     # raise RuntimeError('Found NaN in extracted flux')
    #     print(f'nanflux: {patch.bspecmin}, {patch.ispec}, {patch.iwave}, {xyslice}', flush=True)

    # if cp.any(specflux[:, patch.keepslice] == 0):
    #     # raise RuntimeError('Found zero in extracted flux')
    #     print(specflux.shape, patch.keepslice, flush=True)
    #     print(f'zeroflux: ({patch.bspecmin}, {patch.ispec}, {ispec}), {patch.iwave}, {xyslice}', flush=True)
    #     where = np.where(specflux[:, patch.keepslice] == 0)
    #     print(f'where: {where}', flush=True)

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
    Apadded = A4.reshape(ny*nx, nspectot*nwavetot)
    patchmodel = Apadded.dot(xflux.ravel())
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
    chi2pix = chi2pix.reshape(nspectot, nwavetot)[specslice]
    cp.cuda.nvtx.RangePop() # chi2pix

    if model:
        #TODO: divide flux by wavelength grid spacing?
        modelimage = Apatch.dot(specflux.ravel()*(specivar.ravel() > 0)).reshape(ny, nx)
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

def ex2d_subbundle(image, imageivar, patches, spots, corners, pixpad_frac, regularize, model, psferr):
    """Extract an entire subbundle of patches. The patches' output shape (nspec, nwave) must be aligned.

    Args:
        image: full image (not trimmed to a particular xy range)
        imageivar: image inverse variance (same dimensions as image)
        patches: list contain gpu_specter.core.Patch objects for extraction
        spots: array[nspec, nwave, ny, nx] pre-evaluated PSF spots
        corners: tuple of arrays xcorners[nspec, nwave], ycorners[nspec, nwave]
        pixpad_frac: padded pixel fraction to use (value between 0 and 1)
        regularize: added to diagonal of icov
        model: compute image pixel model using extracted flux
        psferr: value of error to assume in psf model

    Returns:
        results: list of (patch, result) tuples
    """
    batch_pixels = list()
    batch_ivar = list()
    batch_A4 = list()
    batch_xyslice = list()
    # batch_icov = list()
    # batch_y = list()

    corners_cpu = (corners[0].get(), corners[1].get())

    cp.cuda.nvtx.RangePush('batch_prepare')

    # Use the first patch to determine spec padding, must be the same for all patches
    # in this subbundle
    p = patches[0]
    specmin, nspectot = get_spec_padding(p.ispec-p.bspecmin, p.nspectra_per_patch, p.bundlesize)
    nwavetot = p.nwavestep + 2*p.wavepad

    batch_size = len(patches)
    n = nspectot*nwavetot
    batch_icov = cp.zeros((batch_size, n, n))
    batch_y = cp.zeros((batch_size, n))

    for i, patch in enumerate(patches):
        patchpixels, patchivar, patchA4, xyslice = _prepare_patch(
            image, imageivar, specmin, nspectot,
            patch.iwave, patch.nwavestep, patch.wavepad,
            spots, corners, corners_cpu, pixpad_frac
        )
        patch.xyslice = xyslice
        batch_pixels.append(patchpixels)
        batch_ivar.append(patchivar)
        batch_A4.append(patchA4)
        batch_xyslice.append(xyslice)
        batch_icov[i], batch_y[i] = _apply_weights(
            patchpixels.ravel(),
            patchivar.ravel(),
            patchA4.reshape(patchpixels.size, nspectot*nwavetot),
            regularize=regularize
        )
        # batch_icov.append(icov)
        # batch_y.append(y)
    # batch_icov = cp.array(batch_icov)
    # batch_y = cp.array(batch_y)
    cp.cuda.nvtx.RangePop()

    # perform batch extraction
    cp.cuda.nvtx.RangePush('batch_extraction')
    batch_flux, batch_fluxivar, batch_resolution, batch_xflux = _batch_extraction(
        batch_icov, batch_y, nwavetot
    )
    cp.cuda.nvtx.RangePop()

    # finalize patch results
    cp.cuda.nvtx.RangePush('batch_finalize')
    results = list()
    for i, patch in enumerate(patches):
        result = _finalize_patch(
            batch_pixels[i], batch_ivar[i], batch_A4[i], batch_xyslice[i],
            batch_flux[i], batch_fluxivar[i], batch_resolution[i], batch_xflux[i],
            patch.ispec-patch.bspecmin-specmin, patch.nspectra_per_patch,
            patch.nwavestep, patch.wavepad, patch.ndiag, psferr, patch, model=model
        )
        results.append( (patches[i], result) )
    cp.cuda.nvtx.RangePop()

    return results


