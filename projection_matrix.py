import cupy as cp
from numba import cuda
import numpy as np
import numba, pytest
from time import time
from astropy.table import Table

@cuda.jit
def legvander(x, deg, output_matrix):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(i, x.shape[0], stride):
        output_matrix[i][0] = 1
        output_matrix[i][1] = x[i]
        for j in range(2, deg + 1):
            output_matrix[i][j] = (output_matrix[i][j-1]*x[i]*(2*j - 1) - output_matrix[i][j-2]*(j - 1)) / j

def legvander_wrapper(x, deg):
    """Temporary wrapper that allocates memory and defines grid before calling legvander.
    Probably won't be needed once cupy has the correpsponding legvander function.

    Input: Same as legvander
    Output: legvander matrix, cp.ndarray
    """
    output = cp.ndarray((len(x), deg + 1))
    blocksize = 256
    numblocks = (len(x) + blocksize - 1) // blocksize
    legvander[numblocks, blocksize](x, deg, output)
    return output

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
    coeff_gpu = cp.array(psfdata['COEFF'])

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
    assert np.allclose(L, L_gpu.get())
    for name, coeff in zip(psfdata['PARAM'], psfdata['COEFF']):
        L_gpu_new = cp.array(L)
        np.save('L', L)
        np.save('coeff', coeff.T)
        coeff_gpu_new = cp.array(coeff.byteswap().newbyteorder())
        print('L.dtype.isnative', L.dtype.isnative)
        print('coeff.dtype.isnative', coeff.dtype.isnative)
        assert np.allclose(L.dot(coeff.T), L_gpu_new.dot(coeff_gpu_new.T).get())
        assert np.allclose(coeff, coeff_gpu[k].get())
        assert np.allclose(coeff.T, coeff_gpu[k].T.get())
        assert np.allclose(L.dot(coeff.T), L_gpu.dot(coeff_gpu[k].T).get())
        name = name.strip()
        if name.startswith('GH-'):
            i, j = map(int, name.split('-')[1:3])
            p_gpu['GH'][i,j] = L_gpu.dot(coeff_gpu[k].T).T
        else:
            p_gpu[name] = L_gpu.dot(coeff_gpu[k].T).T
        k += 1
    print('GPU Dot products took', time() - start, 's')

    # Test if results are the same
    # gh_gpu = p_gpu['GH'].get()
    # for i in range(len(p['GH'])):
    #     for j in range(len(p['GH'][i])):
    #         print('CPU', p['GH'][i][j], 'GPU', gh_gpu[i][j])
    # assert np.allclose(p['GH'], p_gpu['GH'].get())

    #- Include some additional keywords that we'll need
    for key in ['HSIZEX', 'HSIZEY', 'GHDEGX', 'GHDEGY']:
        p[key] = psfdata.meta[key]

    return p

def test_legvander():
    # Generate dummy input
    degree = 10
    np.random.seed(1)
    x_cpu = np.random.rand(100)
    x_gpu = cp.array(x_cpu)

    # Calculate on cpu
    legvander_cpu = np.polynomial.legendre.legvander(x_cpu, degree)

    # Calculate on gpu
    legvander_gpu = legvander_wrapper(x_gpu, degree)

    #Compare
    assert np.allclose(legvander_cpu, legvander_gpu.get())

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
