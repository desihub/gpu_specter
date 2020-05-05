import unittest, os, shutil, uuid
import pkg_resources
from astropy.table import Table
import numpy as np
from scipy.ndimage.measurements import center_of_mass

from gpu_specter.io import read_psf
from gpu_specter.extract.cpu import evalcoeffs, get_spots

try:
    import specter.psf
    specter_available = True
except ImportError:
    specter_available = False

class TestPSFSpots(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.psffile = pkg_resources.resource_filename(
            'gpu_specter', 'test/data/psf-r0-00051060.fits')
        cls.psfdata = read_psf(cls.psffile)
        meta = cls.psfdata['PSF'].meta
        nwave = 15
        cls.wavelengths = np.linspace(meta['WAVEMIN']+100, meta['WAVEMAX']-100, nwave)
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
        spots, corners = get_spots(0, nspec, self.wavelengths, self.psfdata)
        cx, cy = corners
        
        #- Dimensions
        _nspec, nwave, ny, nx = spots.shape
        self.assertEqual(_nspec, nspec)
        self.assertEqual(nwave, len(self.wavelengths))
        self.assertEqual(cx.shape, (nspec, nwave))
        self.assertEqual(cy.shape, (nspec, nwave))

        #- Spots should have an odd number of pixels in each dimension so
        #- that there is a well defined central pixel
        self.assertEqual(ny%2, 1)
        self.assertEqual(nx%2, 1)
        
        #- positivity and normalization
        self.assertTrue(np.all(spots >= 0.0))
        norm = spots.sum(axis=(2,3))
        self.assertTrue(np.allclose(norm, 1.0))

        #- The PSF centroid should be within that central pixel
        #- Note: X,Y relative to pixel center, not edge
        dx = self.psfparams['X'][0:nspec, :] - cx - nx//2
        dy = self.psfparams['Y'][0:nspec, :] - cy - ny//2
        
        self.assertTrue(np.all((-0.5 <= dx) & (dx < 0.5)))
        self.assertTrue(np.all((-0.5 <= dy) & (dy < 0.5)))
        
        #- The actual centroid of the spot should be within that pixel
        #- Allow some buffer for asymmetric tails
        for ispec in range(nspec):
            for iwave in range(len(self.wavelengths)):
                yy, xx = center_of_mass(spots[ispec, iwave])
                dx = xx - nx//2
                dy = yy - ny//2
                msg = f'ispec={ispec}, iwave={iwave}'
                self.assertTrue((-0.7 <= dx) and (dx < 0.7), msg + f' dx={dx}')
                self.assertTrue((-0.7 <= dy) and (dy < 0.7), msg + f' dy={dy}')

    @unittest.skipIf(not specter_available, 'specter not available')
    def test_compare_specter(self):
        #- specter version
        psf = specter.psf.load_psf(self.psffile)

        for ispec in np.linspace(0, 499, 20).astype(int):
            spots, corners = get_spots(ispec, 1, self.wavelengths, self.psfdata)
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