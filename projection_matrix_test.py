import cupy as cp
import numpy as np
from projectin_matrix_gpu import hermevander_wrapper, legvander_wrapper

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
