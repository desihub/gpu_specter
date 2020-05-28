import unittest, os, shutil, uuid
import pkg_resources
from astropy.table import Table
import numpy as np

from gpu_specter.io import read_psf
from gpu_specter.extract.cpu import projection_matrix, get_spots

try:
    import specter.psf
    specter_available = True
except ImportError:
    specter_available = False

try:
    import cupy as cp
    from numba import cuda
    from gpu_specter.extract.gpu import projection_matrix as gpu_projection_matrix
    gpu_available = True
except ImportError:
    gpu_available = False

class TestProjectionMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.psffile = pkg_resources.resource_filename(
            'gpu_specter', 'test/data/psf-r0-00051060.fits')
        cls.psfdata = read_psf(cls.psffile)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basics(self):
        wavelengths = np.arange(6000.0, 6050.0, 1.0)
        spots, corners = get_spots(0, 25, wavelengths, self.psfdata)
        
        #- Projection matrix for a subset of spots and wavelenghts
        A4, (xmin, xmax, ymin, ymax) = projection_matrix(
            ispec=0, nspec=5, iwave=0, nwave=25, spots=spots, corners=corners)
        self.assertEqual(A4.shape[2:4], (5,25))
        self.assertEqual(A4.shape[0:2], (ymax-ymin, xmax-xmin))

        #- Another subset using same spots, but not starting from (0,0)
        A4, (xmin, xmax, ymin, ymax) = projection_matrix(
            ispec=10, nspec=5, iwave=20, nwave=25, spots=spots, corners=corners)
        self.assertEqual(A4.shape[2:4], (5,25))
        self.assertEqual(A4.shape[0:2], (ymax-ymin, xmax-xmin))

    @unittest.skipIf(not gpu_available, 'gpu not available')
    def test_compare_gpu(self):
        wavelengths = np.arange(6000.0, 6050.0, 1.0)
        spots, corners = get_spots(0, 25, wavelengths, self.psfdata)

        spots_gpu = cp.asarray(spots)
        corners_gpu = [cp.asarray(c) for c in corners]

        #- Compare projection matrix for a few combos of spectra & waves
        for ispec, nspec, iwave, nwave in (
            (0, 5, 0, 25),
            (10, 5, 20, 25),
            (7, 3, 10, 12),
            ):

            #- cpu projection matrix
            A4, xyrange = projection_matrix(
                ispec=ispec, nspec=nspec, iwave=iwave, nwave=nwave, spots=spots, corners=corners)

            #- gpu projection matrix
            A4_gpu, xyrange_gpu = gpu_projection_matrix(
                ispec=ispec, nspec=nspec, iwave=iwave, nwave=nwave, spots=spots_gpu, corners=corners_gpu)

            self.assertEqual(xyrange, xyrange_gpu)
            self.assertEqual(A4.shape, A4_gpu.shape)
            self.assertTrue(cp.allclose(A4, A4_gpu))


    @unittest.skipIf(not specter_available, 'specter not available')
    def test_compare_specter(self):
        
        #- gpu_specter
        wavelengths = np.arange(6000.0, 6050.0, 1.0)
        spots, corners = get_spots(0, 25, wavelengths, self.psfdata)

        #- Load specter PSF
        psf = specter.psf.load_psf(self.psffile)
        
        #- Compare projection matrix for a few combos of spectra & waves
        for ispec, nspec, iwave, nwave in (
            (0, 5, 0, 25),
            (10, 5, 20, 25),
            (7, 3, 10, 12),
            ):
        
            A4, xyrange = projection_matrix(
                ispec=ispec, nspec=nspec, iwave=iwave, nwave=nwave,
                spots=spots, corners=corners)
            A2 = A4.reshape(A4.shape[0]*A4.shape[1], A4.shape[2]*A4.shape[3])

            #- Specter
            A = psf.projection_matrix(
                (ispec, ispec+nspec), wavelengths[iwave:iwave+nwave],
                xyrange).toarray()
        
            self.assertEqual(A.shape, A2.shape)
            self.assertTrue(np.allclose(A, A2))

if __name__ == '__main__':
    unittest.main()
