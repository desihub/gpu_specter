"""
Tools for DESI spectroperfectionism extractions implemented for a CPU
"""

import sys
import numpy as np
from numpy.polynomial.legendre import legvander, legval
from numpy.polynomial import hermite_e as He
import scipy.special
import numba

#-------------------------------------------------------------------------
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
    p['X'] = legval(ww, psfdata['XTRACE']['X'][specmin:specmin+nspec].T)

    meta = psfdata['YTRACE'].meta
    wavemin, wavemax = meta['WAVEMIN'], meta['WAVEMAX']
    ww = (wavelengths - wavemin) * (2.0 / (wavemax - wavemin)) - 1.0
    p['Y'] = legval(ww, psfdata['YTRACE']['Y'][specmin:specmin+nspec].T)

    #- Evaluate the remaining PSF coefficients with a shared dimensionality
    #- and WAVEMIN, WAVEMAX
    meta = psfdata['PSF'].meta
    wavemin, wavemax = meta['WAVEMIN'], meta['WAVEMAX']
    ww = (wavelengths - wavemin) * (2.0 / (wavemax - wavemin)) - 1.0
    L = np.polynomial.legendre.legvander(ww, meta['LEGDEG'])

    nparam = psfdata['PSF']['COEFF'].shape[0]
    ndeg = psfdata['PSF']['COEFF'].shape[2]

    nwave = L.shape[0]
    nghx = meta['GHDEGX']+1
    nghy = meta['GHDEGY']+1
    p['GH'] = np.zeros((nghx, nghy, nspec, nwave))
    for name, coeff in zip(psfdata['PSF']['PARAM'], psfdata['PSF']['COEFF']):
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
    Calculate pixelated Gauss Hermite for all wavelengths of a single spectrum
    
    Args:
        ispec : integer spectrum number
        wavelengths : array of wavelengths to evaluate
        psfparams : dictionary of PSF parameters returned by evalcoeffs

    returns pGHx, pGHy

    where pGHx[ghdeg+1, nwave, nbinsx] contains the pixel-integrated
    Gauss-Hermite polynomial for all degrees at all wavelengths across
    nbinsx bins spaning the PSF spot, and similarly for pGHy.  The core
    PSF will then be evaluated as

    PSFcore = sum_ij c_ij outer(pGHy[j], pGHx[i])
    '''

    #- shorthand
    p = psfparams

    #- spot size (ny,nx)
    nx = 2*p['HSIZEX']+1
    ny = 2*p['HSIZEY']+1
    nwave = len(wavelengths)
    # print('Spot size (ny,nx) = {},{}'.format(ny, nx))
    # print('nwave = {}'.format(nwave))

    #- x and y edges of bins that span the center of the PSF spot
    xedges = np.repeat(np.arange(nx+1) - nx//2 - 0.5, nwave).reshape(nx+1, nwave)
    yedges = np.repeat(np.arange(ny+1) - ny//2 - 0.5, nwave).reshape(ny+1, nwave)

    #- Shift to be relative to the PSF center and normalize
    #- by the PSF sigma (GHSIGX, GHSIGY).
    #- Note: x,y = 0,0 is center of pixel 0,0 not corner
    #- Dimensions: xedges[nx+1, nwave], yedges[ny+1, nwave]
    dx = (p['X'][ispec]+0.5)%1 - 0.5
    dy = (p['Y'][ispec]+0.5)%1 - 0.5
    xedges = ((xedges - dx)/p['GHSIGX'][ispec])
    yedges = ((yedges - dy)/p['GHSIGY'][ispec])
    # print('xedges.shape = {}'.format(xedges.shape))
    # print('yedges.shape = {}'.format(yedges.shape))

    #- Degree of the Gauss-Hermite polynomials
    ghdegx = p['GHDEGX']
    ghdegy = p['GHDEGY']

    #- Evaluate the Hermite polynomials at the pixel edges
    #- HVx[ghdegx+1, nwave, nx+1]
    #- HVy[ghdegy+1, nwave, ny+1]
    HVx = He.hermevander(xedges, ghdegx).T
    HVy = He.hermevander(yedges, ghdegy).T
    # print('HVx.shape = {}'.format(HVx.shape))
    # print('HVy.shape = {}'.format(HVy.shape))

    #- Evaluate the Gaussians at the pixel edges
    #- Gx[nwave, nx+1]
    #- Gy[nwave, ny+1]
    Gx = np.exp(-0.5*xedges**2).T / np.sqrt(2. * np.pi)   # (nwave, nedges)
    Gy = np.exp(-0.5*yedges**2).T / np.sqrt(2. * np.pi)
    # print('Gx.shape = {}'.format(Gx.shape))
    # print('Gy.shape = {}'.format(Gy.shape))

    #- Combine into Gauss*Hermite
    GHx = HVx * Gx
    GHy = HVy * Gy

    #- Integrate over the pixels using the relationship
    #  Integral{ H_k(x) exp(-0.5 x^2) dx} = -H_{k-1}(x) exp(-0.5 x^2) + const

    #- pGHx[ghdegx+1, nwave, nx]
    #- pGHy[ghdegy+1, nwave, ny]
    pGHx = np.zeros((ghdegx+1, nwave, nx))
    pGHy = np.zeros((ghdegy+1, nwave, ny))
    pGHx[0] = 0.5 * np.diff(scipy.special.erf(xedges/np.sqrt(2.)).T)
    pGHy[0] = 0.5 * np.diff(scipy.special.erf(yedges/np.sqrt(2.)).T)
    pGHx[1:] = GHx[:ghdegx,:,0:nx] - GHx[:ghdegx,:,1:nx+1]
    pGHy[1:] = GHy[:ghdegy,:,0:ny] - GHy[:ghdegy,:,1:ny+1]
    # print('pGHx.shape = {}'.format(pGHx.shape))
    # print('pGHy.shape = {}'.format(pGHy.shape))

    return pGHx, pGHy

# @numba.jit(nopython=True)
def multispot(pGHx, pGHy, ghc, out):
    '''
    TODO: Document
    '''
    nx = pGHx.shape[-1]
    ny = pGHy.shape[-1]
    nwave = pGHx.shape[1]
    # spots = np.zeros((nwave, ny, nx))

    # 2.5s
    # spots = np.einsum('lmk,mkj,lki->kji', ghc, pGHy, pGHx)

    # 1.4s
    spots = np.einsum('lmk,mkj,lki->kji', ghc, pGHy, pGHx, optimize=True)

    # np.einsum('lmk,mkj,lki->kji', ghc, pGHy, pGHx, out=out, optimize=True)

    # for iwave in range(nwave):
    #     for i in range(pGHx.shape[0]):
    #         px = pGHx[i,iwave]
    #         for j in range(0, pGHy.shape[0]):
    #             py = pGHy[j,iwave]
    #             c = ghc[i,j,iwave]
    #             #- c * outer(py, px)
    #             # 1.2s
    #             for iy in range(len(py)):
    #                 for ix in range(len(px)):
    #                     spots[iwave, iy, ix] += c * py[iy] * px[ix]
    #             # 3.6s
    #             # spots[iwave] = c*np.outer(py, px)
    #             # 2.5s
    #             # np.outer(c * py, px, spots[iwave])

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
    spots = np.zeros((nspec, nwave, ny, nx))

    for ispec in range(nspec):
        pGHx, pGHy = calc_pgh(ispec, wavelengths, p)
        # spots[ispec] = multispot(pGHx, pGHy, p['GH'][:,:,ispec,:])
        np.einsum('lmk,mkj,lki->kji',
            p['GH'][:,:,ispec,:], pGHy, pGHx, out=spots[ispec], optimize='greedy')

    #- ensure positivity and normalize
    #- TODO: should this be within multispot itself?
    spots = spots.clip(0.0)
    norm = np.sum(spots, axis=(2,3))  #- norm[nspec, nwave] = sum over each spot
    spots = (spots.T / norm.T).T      #- transpose magic for numpy array broadcasting

    #- Define corners of spots
    #- extra 0.5 is because X and Y are relative to center of pixel not edge
    xc = np.floor(p['X'] - p['HSIZEX'] + 0.5).astype(int)
    yc = np.floor(p['Y'] - p['HSIZEY'] + 0.5).astype(int)

    corners = (xc, yc)

    return spots, corners

@numba.jit
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
    

@numba.jit
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

    Cast to 2D for using with linear algebra:

        nypix, nxpix, nspec, nwave = A.shape
        A2D = A.reshape((nypix*nxpix, nspec*nwave))
        pix1D = A2D.dot(flux1D)
    '''
    ny, nx = spots.shape[2:4]
    xc, yc = corners
    xmin, xmax, ymin, ymax = get_xyrange(ispec, nspec, iwave, nwave, spots, corners)
    A = np.zeros((ymax-ymin,xmax-xmin,nspec,nwave))
    for i in range(nspec):
        for j in range(nwave):
            ixc = xc[ispec+i, iwave+j] - xmin
            iyc = yc[ispec+i, iwave+j] - ymin
            A[iyc:iyc+ny, ixc:ixc+nx, i, j] = spots[ispec+i,iwave+j]

    return A, (xmin, xmax, ymin, ymax)

def get_spec_padding(ispec, nspec, bundlesize):
    """
    Calculate padding needed for boundary spectra

    Args:
        ispec: starting spectrum index
        nspec: number of spectra to extract (not including padding)
        bundlesize: size of fiber bundles; padding not needed on their edges

    returns specmin, nspecpad
    """    
    #- if not at upper boundary, extract one additional spectrum
    if (ispec+nspec) % bundlesize == 0:
        nspecpad = nspec
    else:
        nspecpad = nspec + 1

    #- if not at lower boundary, start one lower and extract one more
    if ispec % bundlesize == 0:
        specmin = ispec
    else:
        specmin = ispec-1
        nspecpad += 1
    
    assert nspecpad <= nspec+2
    assert specmin >= ispec-1
    assert specmin+nspecpad <= ispec+nspec+1
    
    return specmin, nspecpad

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
    Rdiags = np.zeros( (nspec, 2*ndiag+1, nwave) )
    #- TODO: check indexing
    for i in np.arange(ispec, ispec+nspec):
        #- subregion of R for this spectrum
        ii = slice(nwavetot*i, nwavetot*(i+1))
        Rx = R[ii, ii]
        #- subregion of non-padded wavelengths for this spectrum
        for j in range(wavepad,wavepad+nwave):
            # Rdiags dimensions [nspec, 2*ndiag+1, nwave]
            Rdiags[i-ispec, :, j-wavepad] = Rx[j-ndiag:j+ndiag+1, j]
    return Rdiags

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

    specmin, nspecpad = get_spec_padding(ispec, nspec, bundlesize)

    #- Total number of wavelengths to be extracted, including padding
    nwavetot = nwave+2*wavepad

    #- Get the projection matrix for the full wavelength range with padding
    A4, xyrange = projection_matrix(specmin, nspecpad,
        iwave-wavepad, nwave+2*wavepad, spots, corners)

    xmin, xmax, ypadmin, ypadmax = xyrange

    #- But we only want to use the pixels covered by the original wavelengths
    #- TODO: this unnecessarily also re-calculates xranges
    xlo, xhi, ymin, ymax = get_xyrange(specmin, nspecpad, iwave, nwave, spots, corners)
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

    specslice = np.s_[ispec-specmin:ispec-specmin+nspec,wavepad:wavepad+nwave]
    if (0 <= ymin) & (ymin+ny <= image.shape[0]):
        xyslice = np.s_[ymin:ymin+ny, xmin:xmin+nx]
        patchpixels = image[xyslice]
        patchivar = imageivar[xyslice]
        fx, ivarfx, R = ex2d_patch(patchpixels, patchivar, A4, regularize=regularize)

        #- Select the non-padded spectra x wavelength core region
        specflux = fx[specslice]
        specivar = ivarfx[specslice]

        #- Diagonals of R in a form suited for creating scipy.sparse.dia_matrix
        Rdiags = get_resolution_diags(R, ndiag, ispec-specmin, nspec, nwave, wavepad)

    else:
        #- TODO: this zeros out the entire patch if any of it is off the edge
        #- of the image; we can do better than that
        fx = np.zeros((nspecpad, nwavetot))
        specflux = np.zeros((nspec, nwave))
        specivar = np.zeros((nspec, nwave))
        Rdiags = np.zeros( (nspec, 2*ndiag+1, nwave) )
        # xyslice = np.s_[
        #     max(0, ymin):min(ymin+ny, image.shape[0]),
        #     max(0, xmin):min(xmin+nx, image.shape[1])
        # ]
        xyslice = None
        patchivar = np.zeros((ny, nx))
        patchpixels = np.zeros((ny, nx))

    if np.any(np.isnan(specflux)):
        raise RuntimeError('Found NaN in extracted flux')

    Apadded = A4.reshape(ny*nx, nspecpad*nwavetot)
    Apatch = A4[:, :, ispec-specmin:ispec-specmin+nspec, wavepad:wavepad+nwave]
    Apatch = Apatch.reshape(ny*nx, nspec*nwave)

    pixmask_fraction = Apatch.T.dot(patchivar.ravel() == 0)
    pixmask_fraction = pixmask_fraction.reshape(nspec, nwave)

    modelpadded = Apadded.dot(fx.ravel()).reshape(ny, nx)
    modelivar = (modelpadded*psferr + 1e-32)**-2
    ii = (modelivar > 0 ) & (patchivar > 0)
    totpix_ivar = np.zeros((ny, nx))
    totpix_ivar[ii] = 1.0 / (1.0/modelivar[ii] + 1.0/patchivar[ii])

    #- Weighted chi2 of pixels that contribute to each flux bin;
    #- only use unmasked pixels and avoid dividing by 0
    chi = (patchpixels - modelpadded)*np.sqrt(totpix_ivar)
    psfweight = Apadded.T.dot(totpix_ivar.ravel() > 0)
    bad = psfweight == 0

    #- Compute chi2pix and reshape
    chi2pix = (Apadded.T.dot(chi.ravel()**2) * ~bad) / (psfweight + bad)
    chi2pix = chi2pix.reshape(nspecpad, nwavetot)[specslice]

    if model:
        modelimage = Apatch.dot(specflux.ravel()).reshape(ny, nx)
    else:
        modelimage = None

    #- TODO: add chi2pix, pixmask_fraction, optionally modelimage; see specter
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

#- Simplest form of A.T.dot( Diag(w).dot(A) )
def dotdot1(A, w):
    '''
    return A.T.dot( Diag(w).dot(A) ) = (A.T * w).dot(A)
    '''
    return (A.T * w).dot(A)

#- 2x faster than dotdot1 by using sparse arrays
def dotdot2(A, w):
    '''
    return A.T.dot( Diag(w).dot(A) ) when A is sparse
    '''
    import scipy.sparse
    W = scipy.sparse.spdiags(data=w, diags=[0,], m=len(w), n=len(w))
    Ax = scipy.sparse.csc_matrix(A)
    return Ax.T.dot(W.dot(Ax)).toarray()

#- 3x faster than dotdot1 by using numba and sparsity
@numba.jit(nopython=True)
def dotdot3(A, w):
    '''
    return A.T.dot( Diag(w).dot(A) ) when A is sparse using numba
    '''
    n, m = A.shape
    B = np.zeros((m,m))
    for i in range(n):
        for j1 in range(m):
            Aw = w[i] * A[i,j1]
            if Aw != 0.0:
                for j2 in range(j1, m):
                    tmp = Aw * A[i,j2]
                    B[j1, j2] += tmp

    #- fill in other half
    for j1 in range(m-1):
        for j2 in range(j1+1, m):
            B[j2, j1] = B[j1, j2]

    return B

@numba.jit(nopython=True)
def dotall(p, w, A):
    '''Compute icov, y and fluxweight in the same loop(s)

        icov = A^T W A
        y = A^T W p
        fluxweight = (A^T W).sum(axis=1)

    Arguments:
        pixel_values: pixel values
        pixel_ivar: pixel weights
        A: projection matrix

    Returns:
        icov, y, fluxweight
    '''
    n, m = A.shape
    icov = np.zeros((m,m))
    y = np.zeros(m)
    fluxweight = np.zeros(m)
    for i in range(n):
        for j1 in range(m):
            Aw = w[i] * A[i,j1]
            if Aw != 0.0:
                for j2 in range(j1, m):
                    tmp = Aw * A[i,j2]
                    icov[j1, j2] += tmp
                fluxweight[j1] += Aw
                y[j1] += Aw * p[i]

    #- fill in other half
    for j1 in range(m-1):
        for j2 in range(j1+1, m):
            icov[j2, j1] = icov[j1, j2]

    return icov, y, fluxweight

def deconvolve(pixel_values, pixel_ivar, A, regularize=0, debug=False):
    """Calculate the weighted linear least-squares flux solution for an observed trace.

    Args:
        pixel_values (ny*nx,): 1D array of pixel values
        pixel_ivar (ny*nx,): 1D array of pixel inverse variances to use for weighting
        A (ny*nx, nspec*nwave): projection matrix that transforms a 1D spectrum into a 2D image

    Returns:
        deconvolved (nspec*nwave): the best-fit 1D array of flux values
        iCov (nspec*nwave, nspec*nwave): the correlated inverse covariance matrix of the deconvolved flux

    """
    #- Set up the equation to solve (B&S eq 4)
    iCov, y, fluxweight = dotall(pixel_values, pixel_ivar, A)
    #- Add a weak flux=0 prior to avoid singular matrices
    #- TODO: review this; compare to specter
    minweight = 1e-4*np.max(fluxweight)
    ibad = fluxweight < minweight
    lambda_squared = regularize*regularize*np.ones_like(y)
    lambda_squared[ibad] = minweight - fluxweight[ibad]
    if np.any(lambda_squared):
        iCov += np.diag(lambda_squared)
    #- Solve the linear least-squares problem.
    deconvolved = scipy.linalg.solve(iCov, y)
    return deconvolved, iCov

def decorrelate_noise(iCov, debug=False):
    """Calculate the decorrelated errors and resolution matrix via BS Eq 10-13

    Args:
        iCov (nspec*nwave, nspec*nwave): the inverse covariance matrix

    Returns:
        ivar (ny*nx,): uncorrelated flux inverse variances
        R (nspec*nwave, nspec*nwave): resoultion matrix
    """
    # Calculate the matrix square root of iCov to diagonalize the flux errors.
    u, v = np.linalg.eigh(iCov)
    # Check that all eigenvalues are positive.
    assert not debug or np.all(u > 0), 'Found some negative iCov eigenvalues.'
    # Check that the eigenvectors are orthonormal so that vt.v = 1
    assert not debug or np.allclose(np.eye(len(u)), v.T.dot(v))
    Q = (v * np.sqrt(u)).dot(v.T)
    # Check BS eqn.10
    assert not debug or np.allclose(iCov, Q.dot(Q))
    #- Calculate the corresponding resolution matrix and diagonal flux errors. (BS Eq 11-13)
    s = np.sum(Q, axis=1)
    R = Q/s[:, np.newaxis]
    ivar = s**2
    # Check BS eqn.14
    assert not debug or np.allclose(iCov, R.T.dot(np.diag(ivar).dot(R)))
    return ivar, R

def decorrelate_blocks(iCov, block_size, debug=False):
    """Calculate the decorrelated errors and resolution matrix via BS Eq 19

    Args:
        iCov (nspec*nwave, nspec*nwave): the inverse covariance matrix
        block_size (int): size of the block corresponding to a single spectrum (i.e. nwave)

    Returns:
        ivar (ny*nx,): uncorrelated flux inverse variances
        R (nspec*nwave, nspec*nwave): resoultion matrix
    """
    size = iCov.shape[0]
    assert not debug or size % block_size == 0
    #- Invert iCov (B&S eq 17)
    u, v = np.linalg.eigh((iCov + iCov.T)/2.)
    assert not debug or np.all(u > 0), 'Found some negative iCov eigenvalues.'
    # Check that the eigenvectors are orthonormal so that vt.v = 1
    assert not debug or np.allclose(np.eye(len(u)), v.T.dot(v))

    if debug:
        threshold = 10.0 * sys.float_info.epsilon
        maxval = np.max(u)
        minval = maxval * threshold
        i = u > minval
        if np.any(~i):
            raise RuntimeError(f'Eigenvalue below minval {minval}: {u[i]}')
        # u = np.clip(u, minval, None)

    C = (v * (1.0/u)).dot(v.T)
    #- Calculate C^-1 = QQ (B&S eq 17-19)
    Q = np.zeros_like(iCov)
    #- Proceed one block at a time
    for i in range(0, size, block_size):
        s = np.s_[i:i+block_size, i:i+block_size]
        #- Invert this block
        bu, bv = np.linalg.eigh(C[s])
        assert not debug or np.all(bu > 0), 'Found some negative iCov eigenvalues.'
        # Check that the eigenvectors are orthonormal so that vt.v = 1
        assert not debug or np.allclose(np.eye(len(bu)), bv.T.dot(bv))

        if debug:
            threshold = 10.0 * sys.float_info.epsilon
            maxval = np.max(bu)
            minval = np.sqrt(maxval) * threshold
            i = np.sqrt(bu) > minval
            if np.any(~i):
                raise RuntimeError(f'Eigenvalue below minval {minval}: {bu[i]}')
            # bu = np.clip(bu, minval, None)

        bQ = (bv * np.sqrt(1.0/bu)).dot(bv.T)
        Q[s] = bQ
    #- Calculate the corresponding resolution matrix and diagonal flux errors. (BS Eq 11-13)
    s = np.sum(Q, axis=1)
    R = Q/s[:, np.newaxis]
    ivar = s**2
    #- Check BS eqn.14
    assert not debug or np.allclose(Q.dot(Q), R.T.dot(np.diag(ivar).dot(R)))
    return ivar, R

# @profile
def ex2d_patch(noisyimg, imgweights, A4, decorrelate='signal', regularize=0, debug=False):
    '''
    Perform spectroperfectionism extractions returning flux, varflux, R

    Inputs:
        noisyimage[ny,nx] : input image
        imgweights[ny,nx] : inverse variance weights of input image
        A4[ny, nx, nspec, nwave] : projection matrix for p = A f

    Returns (f, vf, R) where
        flux (nspec, nwave): extracted resolution convolved flux
        ivar (nspec, nwave): uncorrelated flux inverse variances
        R (nspec*nwave, nspec*nwave): dense resolution matrix
    '''
    ny, nx, nspec, nwave = A4.shape
    assert noisyimg.shape == (ny, nx)
    npix = ny*nx

    assert decorrelate in ('signal', 'noise')

    A = A4.reshape(ny*nx, nspec*nwave)

    #- Solve f (B&S eq 4)
    deconvolved, iCov = deconvolve(noisyimg.ravel(), imgweights.ravel(), A, regularize=regularize)

    #- Calculate the decorrelated errors and resolution matrix.
    if decorrelate == 'signal':
        fluxivar, resolution = decorrelate_blocks(iCov, nwave, debug=debug)
    elif decorrelate == 'noise':
        fluxivar, resolution = decorrelate_noise(iCov, debug=debug)
    else:
        raise ValueError(f'{decorrelate} is not a valid value for decorrelate')
    
    #- Convolve the reduced flux (BS eq 16)
    flux = resolution.dot(deconvolved).reshape(nspec, nwave)
    fluxivar = fluxivar.reshape(nspec, nwave)
    
    return flux, fluxivar, resolution
