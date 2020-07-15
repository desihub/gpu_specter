import unittest, os
import pkg_resources
from astropy.table import Table
import numpy as np

from gpu_specter.io import read_img, read_psf
from gpu_specter.core import extract_frame

try:
    import specter.psf
    import specter.extract
    specter_available = True
except ImportError:
    specter_available = False

try:
    import cupy as cp
    from numba import cuda
    from gpu_specter.extract.gpu import projection_matrix as gpu_projection_matrix
    gpu_available = cp.is_available()
except ImportError:
    gpu_available = False

imgfile = pkg_resources.resource_filename('gpu_specter', 'test/data/preproc-r0-00051060.fits')
try:
    img = read_img(imgfile)
    preproc_available = True
except:
    preproc_available = False


@unittest.skipIf(not preproc_available, f'{imgfile} not available')
class TestCore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.psffile = pkg_resources.resource_filename(
            'gpu_specter', 'test/data/psf-r0-00051060.fits')
        cls.psfdata = read_psf(cls.psffile)

        cls.imgdata = read_img(imgfile)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_extract_frame(self):

        bundlesize = 10
        wavelength = '6000.0,6100.0,1.0'
        nwave = 101

        specmin = 0
        nspec = 10
        nwavestep = 50
        nsubbundles = 2

        frame = extract_frame(
            self.imgdata, self.psfdata, bundlesize,
            specmin, nspec,
            wavelength=wavelength,
            nwavestep=nwavestep, nsubbundles=nsubbundles,
            comm=None, rank=0, size=1,
            gpu=None,
            loglevel='WARN',
        )

        self.assertEqual(frame['specflux'].shape, (nspec, nwave))

    @unittest.skipIf(not specter_available, 'specter not available')
    def test_compare_specter(self):

        bundlesize = 10
        wavelength = '5760.0,7620.0,0.8'

        specmin = 0
        nspec = 10
        nwavestep = 50
        nsubbundles = 2

        frame_spex = extract_frame(
            self.imgdata, self.psfdata, bundlesize,
            specmin, nspec,
            wavelength=wavelength,
            nwavestep=nwavestep, nsubbundles=nsubbundles,
            comm=None, rank=0, size=1,
            gpu=None,
            loglevel='WARN',
        )

        self.assertEqual(frame_spex['specflux'].shape[0], nspec)

        psf = specter.psf.load_psf(self.psffile)

        wavelengths = frame_spex['wave']

        frame_specter = specter.extract.ex2d(
            self.imgdata['image'], self.imgdata['ivar'], psf, 
            specmin, nspec, wavelengths, 
            xyrange=None, regularize=0.0, ndecorr=False,
            bundlesize=bundlesize, nsubbundles=nsubbundles,
            wavesize=nwavestep, 
            full_output=True, verbose=False,
            debug=False, psferr=None,
        )

        self.assertEqual(frame_spex['specflux'].shape, frame_specter['flux'].shape)

        diff = frame_spex['specflux'] - frame_specter['flux']
        norm = np.sqrt(1.0/frame_spex['specivar'] + 1.0/frame_specter['ivar'])
        pull = diff/norm
        isclose_threshold = 0.01
        isclose_fraction = np.average(np.abs(pull).ravel() < isclose_threshold)

        self.assertGreaterEqual(isclose_fraction, 0.95)

if __name__ == '__main__':
    unittest.main()
