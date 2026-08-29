import unittest, os, shutil, uuid
from astropy.table import Table
import numpy as np
from scipy.ndimage import center_of_mass

from gpu_specter.io import read_psf
from gpu_specter.extract.cpu import evalcoeffs, get_spots
from .util import find_test_file

try:
    import specter.psf
    specter_available = True
except ImportError:
    specter_available = False

try:
    import cupy as cp
    from numba import cuda
    from gpu_specter.extract.gpu import get_spots as gpu_get_spots
    gpu_available = cp.is_available()
except ImportError:
    gpu_available = False


class TestPSFSpots(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.psffile = find_test_file('psf')
        cls.psfdata = read_psf(cls.psffile)
        meta = cls.psfdata['PSF'].meta
        nwave = 15
        cls.wavelengths = np.linspace(meta['WAVEMIN'] + 100,
                                      meta['WAVEMAX'] - 100, nwave)
        cls.psfparams = evalcoeffs(cls.psfdata, cls.wavelengths)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basics(self):
        nspec = 50
        spots, corners, psfparams = get_spots(0, nspec, self.wavelengths,
                                              self.psfdata)
        cx, cy = corners

        #- Dimensions
        _nspec, nwave, ny, nx = spots.shape
        self.assertEqual(_nspec, nspec)
        self.assertEqual(nwave, len(self.wavelengths))
        self.assertEqual(cx.shape, (nspec, nwave))
        self.assertEqual(cy.shape, (nspec, nwave))

        #- Spots should have an odd number of pixels in each dimension so
        #- that there is a well defined central pixel
        self.assertEqual(ny % 2, 1)
        self.assertEqual(nx % 2, 1)

        #- positivity and normalization
        self.assertTrue(np.all(spots >= 0.0))
        norm = spots.sum(axis=(2, 3))
        self.assertTrue(np.allclose(norm, 1.0))

        #- The PSF centroid should be within that central pixel
        #- Note: X,Y relative to pixel center, not edge
        dx = self.psfparams['X'][0:nspec, :] - cx - nx // 2
        dy = self.psfparams['Y'][0:nspec, :] - cy - ny // 2

        self.assertTrue(np.all((-0.5 <= dx) & (dx < 0.5)))
        self.assertTrue(np.all((-0.5 <= dy) & (dy < 0.5)))

        #- The actual centroid of the spot should be within that pixel
        #- Allow some buffer for asymmetric tails
        for ispec in range(nspec):
            for iwave in range(len(self.wavelengths)):
                yy, xx = center_of_mass(spots[ispec, iwave])
                dx = xx - nx // 2
                dy = yy - ny // 2
                msg = f'ispec={ispec}, iwave={iwave}'
                self.assertTrue((-0.7 <= dx) and (dx < 0.7), msg + f' dx={dx}')
                self.assertTrue((-0.7 <= dy) and (dy < 0.7), msg + f' dy={dy}')

    @unittest.skipIf(not gpu_available, 'gpu not available')
    def test_compare_gpu(self):
        for ispec in np.linspace(0, 499, 20).astype(int):
            spots, corners, psfparams = get_spots(ispec, 1, self.wavelengths,
                                                  self.psfdata)
            xc, yc = corners

            spots_gpu, corners_gpu, psfparams_gpu = gpu_get_spots(
                ispec, 1, self.wavelengths, self.psfdata)
            xc_gpu, yc_gpu = corners_gpu

            # compare corners
            self.assertTrue(np.array_equal(xc, cp.asnumpy(xc_gpu)))
            self.assertTrue(np.array_equal(yc, cp.asnumpy(yc_gpu)))

            # compare spots
            self.assertEqual(spots.shape, spots_gpu.shape)
            eps_double = np.finfo(np.float64).eps
            self.assertTrue(
                np.allclose(spots,
                            cp.asnumpy(spots_gpu),
                            rtol=eps_double,
                            atol=eps_double))

    @unittest.skipIf(not gpu_available, 'gpu not available')
    def test_compare2d_gpu(self):
        nspec0 = 500
        for ispec, nspec in [(1, 1), (10, 20)]:
            spots_gpu, corners_gpu, psfparams_gpu = gpu_get_spots(
                ispec, nspec, self.wavelengths, self.psfdata)
            spots_gpu1, corners_gpu1, psfparams_gpu1 = gpu_get_spots(
                ispec, nspec,
                self.wavelengths[None, :] + np.zeros(nspec0)[:, None],
                self.psfdata)
            xc_gpu1, yc_gpu1 = corners_gpu1
            xc_gpu, yc_gpu = corners_gpu

            # compare corners
            self.assertTrue(
                np.array_equal(cp.asnumpy(xc_gpu), cp.asnumpy(xc_gpu1)))
            self.assertTrue(
                np.array_equal(cp.asnumpy(yc_gpu), cp.asnumpy(yc_gpu1)))

            # compare spots
            self.assertEqual(spots_gpu.shape, spots_gpu1.shape)
            eps_double = np.finfo(np.float64).eps
            self.assertTrue(
                np.allclose(cp.asnumpy(spots_gpu),
                            cp.asnumpy(spots_gpu1),
                            rtol=eps_double,
                            atol=eps_double))

    def test_compare2d_cpu(self):
        nspec0 = 500
        for ispec, nspec in [(1, 1), (10, 20)]:
            spots, corners, psfparams = get_spots(ispec, nspec,
                                                  self.wavelengths,
                                                  self.psfdata)
            xc, yc = corners
            spots1, corners1, psfparams1 = get_spots(
                ispec, nspec,
                self.wavelengths[None, :] + np.zeros(nspec0)[:, None],
                self.psfdata)
            xc1, yc1 = corners1

            # compare corners
            self.assertTrue(np.array_equal(xc, xc1))
            self.assertTrue(np.array_equal(yc, yc1))

            # compare spots
            self.assertEqual(spots.shape, spots1.shape)
            eps_double = np.finfo(np.float64).eps
            self.assertTrue(
                np.allclose(spots, spots1, rtol=eps_double, atol=eps_double))

    @unittest.skipIf(not specter_available, 'specter not available')
    def test_compare_specter(self):
        #- specter version
        psf = specter.psf.load_psf(self.psffile)

        for ispec in np.linspace(0, 499, 20).astype(int):
            spots, corners, psfparams = get_spots(ispec, 1, self.wavelengths,
                                                  self.psfdata)
            xc, yc = corners

            ny, nx = spots.shape[2:4]

            for iwave, w in enumerate(self.wavelengths):
                xslice, yslice, pix = psf.xypix(ispec, w)

                self.assertEqual(pix.shape, (ny, nx))
                self.assertEqual(xc[0, iwave], xslice.start)
                self.assertEqual(yc[0, iwave], yslice.start)
                self.assertTrue(np.allclose(spots[0, iwave], pix))


if __name__ == '__main__':
    unittest.main()
