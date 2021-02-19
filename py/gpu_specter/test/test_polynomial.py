import unittest
import numpy as np

try:
    import cupy as cp
    cupy_available = cp.is_available()
except ImportError:
    cupy_available = False

class TestPolynomial(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basics(self):
        pass

    @unittest.skipIf(not cupy_available, 'cupy not available')
    def test_gpu_hermevander(self):
        from gpu_specter.polynomial import hermevander
        degree = 10
        rng = np.random.default_rng(12345)
        cpu_x = rng.random((10, 100))
        gpu_x = cp.array(cpu_x)
        cpu_result = np.polynomial.hermite_e.hermevander(cpu_x, degree)
        gpu_result = hermevander(gpu_x, degree)
        self.assertTrue(np.allclose(cpu_result, gpu_result.get()))


    @unittest.skipIf(not cupy_available, 'cupy not available')
    def test_gpu_legvander(self):
        from gpu_specter.polynomial import legvander
        degree = 10
        rng = np.random.default_rng(12345)
        cpu_x = rng.random(100)
        gpu_x = cp.array(cpu_x)
        cpu_result = np.polynomial.legendre.legvander(cpu_x, degree)
        gpu_result = legvander(gpu_x, degree)
        self.assertTrue(np.allclose(cpu_result, gpu_result.get()))
