from numpy.polynomial import hermite_e as He
import numpy as np
from astropy.table import Table
import scipy.special
import cupy as cp
from test_hermevander import hermevander_wrapper # importing a temporary wrapper function from a temporary test file

def evalcoeffs(wavelengths, psfdata):
    '''Directly copied from the notebook. Testing use only'''
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
    p_gpu = {}
    p_gpu['X'], p_gpu['Y'], p_gpu['GHSIGX'], p_gpu['GHSIGY'] = \
    cp.array(p['X']), cp.array(p['Y']), cp.array(p['GHSIGX']), cp.array(p['GHSIGY'])

    #- x and y edges of bins that span the center of the PSF spot
    # CPU
    xedges = np.repeat(np.arange(nx+1) - nx//2, nwave).reshape(nx+1, nwave)
    yedges = np.repeat(np.arange(ny+1) - ny//2, nwave).reshape(ny+1, nwave)
    # GPU
    xedges_gpu = cp.repeat(cp.arange(nx+1) - nx//2, nwave).reshape(nx+1, nwave)
    yedges_gpu = cp.repeat(cp.arange(ny+1) - ny//2, nwave).reshape(ny+1, nwave)
    assert np.allclose(xedges, xedges_gpu.get())
    assert np.allclose(yedges, yedges_gpu.get())

    #- Shift to be relative to the PSF center at 0 and normalize
    #- by the PSF sigma (GHSIGX, GHSIGY)
    #- xedges[nx+1, nwave]
    #- yedges[ny+1, nwave]
    # CPU
    xedges = (xedges - p['X'][ispec]%1)/p['GHSIGX'][ispec]
    yedges = (yedges - p['Y'][ispec]%1)/p['GHSIGY'][ispec]
    # GPU
    xedges_gpu = (xedges_gpu - p_gpu['X'][ispec]%1)/p_gpu['GHSIGX'][ispec]
    yedges_gpu = (yedges_gpu - p_gpu['Y'][ispec]%1)/p_gpu['GHSIGY'][ispec]
    assert np.allclose(xedges, xedges_gpu.get())
    assert np.allclose(yedges, yedges_gpu.get())

    #- Degree of the Gauss-Hermite polynomials
    ghdegx = p['GHDEGX']
    ghdegy = p['GHDEGY']

    #- Evaluate the Hermite polynomials at the pixel edges
    #- HVx[ghdegx+1, nwave, nx+1]
    #- HVy[ghdegy+1, nwave, ny+1]
    HVx = He.hermevander(xedges, ghdegx).T
    HVy = He.hermevander(yedges, ghdegy).T
    HVx_gpu = hermevander_wrapper(xedges_gpu, ghdegx).T
    HVy_gpu = hermevander_wrapper(yedges_gpu, ghdegy).T
    assert np.allclose(HVx, HVx_gpu.get())
    assert np.allclose(HVy, HVy_gpu.get())

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

def test_calc_pgh():
    # Generate inputs
    psfdata = Table.read('psf.fits')
    wavelengths = np.arange(psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0], 0.8)
    p = evalcoeffs(wavelengths, psfdata)

    # Call pgh function
    pGHx, pGHy = calc_pgh(0, wavelengths, p)
