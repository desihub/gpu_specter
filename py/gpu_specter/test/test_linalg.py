import unittest
import numpy as np

try:
    import cupy as cp
    cupy_available = cp.is_available()

    from gpu_specter.linalg import (
        cholesky_solve,
        matrix_sqrt,
        diag_block_matrix_sqrt
    )
except ImportError:
    cupy_available = False

class TestLinalg(unittest.TestCase):

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
    def test_gpu_matrix_sqrt(self):
        nblocks, block_size = 2, 3
        n = nblocks * block_size
        diagsqrt = 1 + np.arange(n)
        diag = diagsqrt**2
        a = np.diag(diag).astype(np.float32)
        a_device = cp.asarray(a)

        # test basic matrix sqrt
        q_device = matrix_sqrt(a_device)
        qdiag = np.diag(cp.asnumpy(q_device))
        ok = np.allclose(qdiag, diagsqrt)
        self.assertTrue(ok, f'{qdiag} != {diagsqrt}')

    @unittest.skipIf(not cupy_available, 'cupy not available')
    def test_gpu_diag_block_matrix_sqrt(self):
        nblocks, block_size = 2, 3
        n = nblocks * block_size
        diagsqrt = 1 + np.arange(n)
        diag = diagsqrt**2
        a = np.diag(diag).astype(np.float32)
        a_device = cp.asarray(a)

        # test diag block matrix sqrt
        q_device = diag_block_matrix_sqrt(a_device, block_size)
        qdiag = np.diag(cp.asnumpy(q_device))
        ok = np.allclose(qdiag, diagsqrt)
        self.assertTrue(ok, f'{qdiag} != {diagsqrt}')

    @unittest.skipIf(not cupy_available, 'cupy not available')
    def test_gpu_cholesky_solve(self):

        n = 10
        diag = 1 + np.arange(n)
        a = np.diag(diag).astype(dtype=np.float32)
        b = np.ones(n).astype(dtype=np.float32)
        x_device = cholesky_solve(cp.asarray(a), cp.asarray(b))
        x = cp.asnumpy(x_device)

        ok = np.allclose(x, 1.0/diag)
        self.assertTrue(ok, f'{x} != {diag}')


