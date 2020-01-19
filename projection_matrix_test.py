import cupy as cp
import numpy as np
from projectin_matrix_gpu import hermevander_wrapper, legvander_wrapper
from projectin_matrix_gpu import evalcoeffs as evalcoeffs_gpu
from projectin_matrix_reference import evalcoeffs as evalcoeffs_cpu

def test_hermevander():
    # Generate dummy input
    degree = 10
    np.random.seed = 1
    x_cpu = np.random.rand(10, 100)
    x_gpu = cp.array(x_cpu)
    
    # Calculate on cpu
    hermevander_cpu = np.polynomial.hermite_e.hermevander(x_cpu, degree)
    # Calculate on gpu
    hermevander_gpu = hermevander_wrapper(x_gpu, degree)
    # Compare
    assert np.allclose(hermevander_cpu, hermevander_gpu.get())

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

def test_evalcoeffs():
    # Read data
    psfdata_cpu = Table.read('psf.fits')
    psfdata_gpu = psfdata_cpu
    psfdata_gpu['COEFF'] = cp.array(native_endian(psfdata['COEFF']))

    wavelengths_cpu = np.arange(psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0], 0.8)
    wavelengths_gpu = cp.arange(psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0], 0.8)
    
    # Call evalcoeffs
    p_cpu = evalcoeffs_cpu(wavelengths_cpu, psfdata)
    p_gpu = evalcoeffs_gpu(wavelengths_gpu, psfdata)
    # Compare
    assert np.allclose(p_cpu, p_gpu.get())
