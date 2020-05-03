import unittest, os, shutil, uuid
import pkg_resources
from astropy.table import Table
import numpy as np

from gpu_specter.io import read_psf
from gpu_specter.extract.cpu import projection_matrix, get_spots, ex2d_patch

try:
    import specter.psf
    import specter.extract
    specter_available = True
except ImportError:
    specter_available = False

class TestEx2dPatch(unittest.TestCase):

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
        nwave = len(wavelengths)
        nspec = 5

        spots, corners = get_spots(0, nspec, wavelengths, self.psfdata)
        A4, xyrange = projection_matrix(0, nspec, 0, nwave, spots, corners)

        ny, nx = A4.shape[0:2]

        image = np.random.randn(ny, nx)
        imageivar = np.ones((ny, nx))

        flux, varflux, R = ex2d_patch(image, imageivar, A4)

        self.assertEqual(flux.shape, (nspec, nwave))
        self.assertEqual(varflux.shape, (nspec, nwave))
        self.assertEqual(R.shape, (nspec*nwave, nspec*nwave))


    @unittest.skipIf(not specter_available, 'specter not available')
    def test_compare_specter(self):

        wavelengths = np.arange(6000, 6050, 1)
        nwave = len(wavelengths)
        nspec = 5
        phot = np.zeros((nspec, nwave))
        phot[0] = 100
        phot[1] = 5*np.arange(nwave)
        phot[2] = 50
        phot[4] = 100*(1+np.sin(np.arange(nwave)/10.))
        phot[0,10] += 500
        phot[1,15] += 200
        phot[2,20] += 300
        phot[3,25] += 1000
        phot[4,30] += 600

        spots, corners = get_spots(0, nspec, wavelengths, self.psfdata)
        A4, xyrange = projection_matrix(0, nspec, 0, nwave, spots, corners)

        psf = specter.psf.load_psf(self.psffile)
        img = psf.project(wavelengths, phot, xyrange=xyrange)

        readnoise = 3.0
        noisyimg = np.random.normal(loc=0.0, scale=readnoise, size=img.shape)
        noisyimg += np.random.poisson(img)
        #- for test, cheat by using noiseless img instead of noisyimg to estimate variance
        imgivar = 1.0/(img + readnoise**2)   

        #- TODO: set ndecorr=True to disable?
        flux0, ivar0, R0 = specter.extract.ex2d_patch(
            noisyimg, imgivar, psf, 0, nspec, wavelengths, xyrange=xyrange)

        #- TODO: use the exact same projection matrix?
        # A = psf.projection_matrix((0, nspec), wavelengths, xyrange).toarray()
        # A4 = A.reshape(A4.shape)

        flux1, varflux1, R1 = ex2d_patch(noisyimg, imgivar, A4)

        self.assertTrue(np.allclose(flux0, flux1))
        self.assertTrue(np.allclose(ivar0, 1.0/varflux1))
        self.assertTrue(np.allclose(R0, R1))


if __name__ == '__main__':
    unittest.main()
