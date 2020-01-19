import cupy as cp
from numba import cuda
import numpy as np
import numba, pytest
from time import time
from astropy.table import Table

def native_endian(data):
    """Temporary function, sourced from desispec.io
    Convert numpy array data to native endianness if needed.
    Returns new array if endianness is swapped, otherwise returns input data
    Context:
    By default, FITS data from astropy.io.fits.getdata() are not Intel
    native endianness and scipy 0.14 sparse matrices have a bug with
    non-native endian data.
    """
    if data.dtype.isnative:
        return data
    else:
        return data.byteswap().newbyteorder()

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
    # Initialization
    wavemin, wavemax = psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0]
    wx = (wavelengths - wavemin) * (2.0 / (wavemax - wavemin)) - 1.0

    # Call legvander
    # CPU
    start = time()
    L = np.polynomial.legendre.legvander(wx, psfdata.meta['LEGDEG'])
    print('CPU Legvander took:', time() - start, 's')
    # GPU
    start = time()
    L_gpu = legvander_wrapper(wx, psfdata.meta['LEGDEG'])
    print('GPU Legvander took:', time() - start, 's')
    assert np.allclose(L, L_gpu.get())

    # More initialization
    p = dict(WAVE=wavelengths)
    p_gpu = dict(WAVE=wavelengths) # p_gpu doesn't live on the gpu, but it's last-level values do
    nparam, nspec, ndeg = psfdata['COEFF'].shape
    nwave = L.shape[0]

    # Init zeros
    p['GH'] = np.zeros((psfdata.meta['GHDEGX']+1, psfdata.meta['GHDEGY']+1, nspec, nwave))
    p_gpu['GH'] = cp.zeros((psfdata.meta['GHDEGX']+1, psfdata.meta['GHDEGY']+1, nspec, nwave))

    # Init coeff on GPU
    coeff_gpu = cp.array(native_endian(psfdata['COEFF']))

    # Main loop
    # CPU
    start = time()
    for name, coeff in zip(psfdata['PARAM'], psfdata['COEFF']):
        name = name.strip()
        if name.startswith('GH-'):
            i, j = map(int, name.split('-')[1:3])
            p['GH'][i,j] = L.dot(coeff.T).T
        else:
            p[name] = L.dot(coeff.T).T
    print('CPU Dot products took', time() - start, 's')
    # GPU
    start = time()
    k = 0
    for name, coeff in zip(psfdata['PARAM'], psfdata['COEFF']):
        name = name.strip()
        if name.startswith('GH-'):
            i, j = map(int, name.split('-')[1:3])
            p_gpu['GH'][i,j] = L_gpu.dot(coeff_gpu[k].T).T
        else:
            p_gpu[name] = L_gpu.dot(coeff_gpu[k].T).T
        k += 1
    print('GPU Dot products took', time() - start, 's')

    # Test if results are the same
    assert np.allclose(p['GH'], p_gpu['GH'].get())

    #- Include some additional keywords that we'll need
    for key in ['HSIZEX', 'HSIZEY', 'GHDEGX', 'GHDEGY']:
        p[key] = psfdata.meta[key]

    return p

def test_evalcoeffs():
    # Read data
    psfdata = Table.read('psf.fits')
    wavelengths = np.arange(psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0], 0.8)

    # Call evalcoeffs
    start = time()
    p = evalcoeffs(wavelengths, psfdata)
    print('Time:', time() - start, 's')

test_legvander()
test_evalcoeffs()
