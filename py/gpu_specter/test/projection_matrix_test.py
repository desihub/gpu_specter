import cupy as cp
import numpy as np
from astropy.table import Table
from projection_matrix_gpu import hermevander_wrapper, legvander_wrapper
from projection_matrix_gpu import evalcoeffs as evalcoeffs_gpu
from projection_matrix_gpu import calc_pgh as calc_pgh_gpu
from projection_matrix_reference import evalcoeffs as evalcoeffs_cpu
from projection_matrix_reference import calc_pgh as calc_pgh_cpu

##########################################
# Unit Tests Start
##########################################

def test_hermevander():
    # Generate dummy input
    degree = 10
    np.random.seed(1)
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

def test_evalcoeffs(capsys):
    # Read data
    psfdata = Table.read('psf.fits')
    wavelengths_cpu = np.arange(psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0], 0.8)
    wavelengths_gpu = cp.arange(psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0], 0.8)
    
    # Call evalcoeffs
    p_cpu = evalcoeffs_cpu(wavelengths_cpu, psfdata)
    p_gpu = evalcoeffs_gpu(wavelengths_gpu, psfdata)
    # Compare
    for k in p_cpu:
        assert np.allclose(p_cpu[k], p_gpu[k])

def test_calc_pgh():
    # Generate inputs
    psfdata = Table.read('psf.fits')
    wavelengths = np.arange(psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0], 0.8)
    p = evalcoeffs_cpu(wavelengths, psfdata) # Use the cpu version to prevent bugs from cascading

    # Call pgh function
    pGHx_cpu, pGHy_cpu = calc_pgh_cpu(0, wavelengths, p)
    pGHx_gpu, pGHy_gpu = calc_pgh_gpu(0, cp.array(wavelengths), p)
    # Compare
    assert np.allclose(pGHx_cpu, pGHx_gpu)
    assert np.allclose(pGHy_cpu, pGHy_gpu)

##########################################
# Unit Tests End
##########################################

##########################################
# Integration Tests Start
##########################################

##########################################
# Integration Tests End
##########################################
