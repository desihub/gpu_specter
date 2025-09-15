import os, sys, time
import numpy as np
from astropy.table import Table
import numba
import scipy.special
import pytest
import cupy as cp

def evalcoeffs(wavelengths, psfdata):
    '''
    wavelengths: 1D array of wavelengths to evaluate all coefficients for all wavelengths of all spectra
    psfdata: Table of parameter data ready from a GaussHermite format PSF file

    Returns a dictionary params[paramname] = value[nspec, nwave]

    The Gauss Hermite coefficients are treated differently:

        params['GH'] = value[i,j,nspec,nwave]

    The dictionary also contains scalars with the recommended spot size HSIZEX, HSIZEY
    and Gauss-Hermite degrees GHDEGX, GHDEGY (which is also derivable from the dimensions
    of params['GH'])
    '''
    wavemin, wavemax = psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0]
    wx = (wavelengths - wavemin) * (2.0 / (wavemax - wavemin)) - 1.0
    L = np.polynomial.legendre.legvander(wx, psfdata.meta['LEGDEG'])

    p = dict(WAVE=wavelengths)
    nparam, nspec, ndeg = psfdata['COEFF'].shape
    nwave = L.shape[0]
    p['GH'] = np.zeros((psfdata.meta['GHDEGX']+1, psfdata.meta['GHDEGY']+1, nspec, nwave))
    for name, coeff in zip(psfdata['PARAM'], psfdata['COEFF']):
        name = name.strip()
        if name.startswith('GH-'):
            i, j = map(int, name.split('-')[1:3])
            p['GH'][i,j] = L.dot(coeff.T).T
        else:
            p[name] = L.dot(coeff.T).T

    #- Include some additional keywords that we'll need
    for key in ['HSIZEX', 'HSIZEY', 'GHDEGX', 'GHDEGY']:
        p[key] = psfdata.meta[key]

    return p

from numpy.polynomial import hermite_e as He

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
    nx = p['HSIZEX']
    ny = p['HSIZEY']
    nwave = len(wavelengths)
    # print('Spot size (ny,nx) = {},{}'.format(ny, nx))
    # print('nwave = {}'.format(nwave))

    #- x and y edges of bins that span the center of the PSF spot
    xedges = np.repeat(np.arange(nx+1) - nx//2, nwave).reshape(nx+1, nwave)
    yedges = np.repeat(np.arange(ny+1) - ny//2, nwave).reshape(ny+1, nwave)

    #- Shift to be relative to the PSF center at 0 and normalize
    #- by the PSF sigma (GHSIGX, GHSIGY)
    #- xedges[nx+1, nwave]
    #- yedges[ny+1, nwave]
    xedges = ((xedges - p['X'][ispec]%1)/p['GHSIGX'][ispec])
    yedges = ((yedges - p['Y'][ispec]%1)/p['GHSIGY'][ispec])
#     print('xedges.shape = {}'.format(xedges.shape))
#     print('yedges.shape = {}'.format(yedges.shape))

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

@numba.jit(nopython=True)
def multispot(pGHx, pGHy, ghc):
    '''
    TODO: Document
    '''
    nx = pGHx.shape[-1]
    ny = pGHy.shape[-1]
    nwave = pGHx.shape[1]
    spots = np.zeros((nwave, ny, nx))

    tmpspot = np.zeros((ny,nx))
    for iwave in range(nwave):
        for i in range(pGHx.shape[0]):
            px = pGHx[i,iwave]
            for j in range(0, pGHy.shape[0]):
                py = pGHy[j,iwave]
                c = ghc[i,j,iwave]
                #- c * outer(py, px)
                for iy in range(len(py)):
                    for ix in range(len(px)):
                        spots[iwave, iy, ix] += c * py[iy] * px[ix]
    return spots



def test_main():
    psfdata = Table.read('psf.fits')
    assert np.any(psfdata['WAVEMIN'] == psfdata['WAVEMIN'][0]), 'Different values of WAVEMIN are not supported'
    assert np.any(psfdata['WAVEMAX'] == psfdata['WAVEMAX'][0]), 'Different values of WAVEMAX are not supported'
    wavelengths = np.arange(psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0], 0.8)
    p = evalcoeffs(wavelengths, psfdata)
    pGHx, pGHy = calc_pgh(0, wavelengths, p)
    ispec = 0
    mspots = multispot(pGHx, pGHy, p['GH'][:,:,ispec,:])


