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
        from numpy.polynomial.hermite_e import hermevander as numpy_hermevander
        from gpu_specter.polynomial import hermevander as gpu_hermevander
        x = np.array([-1, 0, 1])
        result = numpy_hermevander(x, 3)
        gpu_x = cp.array(x)
        gpu_result = gpu_hermevander(gpu_x, 3)
        ok = np.allclose(cp.asnumpy(gpu_result), result)
        self.assertTrue(ok, f'{gpu_result} != {result}')


    @unittest.skipIf(not cupy_available, 'cupy not available')
    def test_gpu_legvander(self):
        from numpy.polynomial.legendre import legvander as numpy_legvander
        from gpu_specter.polynomial import legvander as gpu_legvander
        x = np.array([-1, 0, 1])
        result = numpy_legvander(x, 3)
        gpu_x = cp.array(x)
        gpu_result = gpu_legvander(gpu_x, 3)
        ok = np.allclose(cp.asnumpy(gpu_result), result)
        self.assertTrue(ok, f'{gpu_result} != {result}')
        