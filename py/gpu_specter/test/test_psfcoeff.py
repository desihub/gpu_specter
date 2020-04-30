import unittest, os, shutil, uuid
import pkg_resources
from astropy.table import Table
import numpy as np

from gpu_specter.io import read_psf
from gpu_specter.extract.cpu import evalcoeffs

try:
    import specter.psf
    specter_available = True
except ImportError:
    specter_available = False

class TestPSFCoeff(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.psffile = pkg_resources.resource_filename(
            'gpu_specter', 'test/data/psf-r0-00051060.fits')

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basics(self):
        psfdata = read_psf(self.psffile)
        for key in ('XTRACE', 'YTRACE', 'PSF'):
            self.assertTrue(key in psfdata.keys(), f'{key} is not in psfdata')
            meta = psfdata[key].meta
            self.assertTrue('WAVEMIN' in meta.keys(), f'{key}.meta missing WAVEMIN')
            self.assertTrue('WAVEMAX' in meta.keys(), f'{key}.meta missing WAVEMAX')

        #- wavelengths outside original range are allowed
        meta = psfdata['PSF'].meta
        nwave = 30
        wavelengths = np.linspace(meta['WAVEMIN']-10, meta['WAVEMAX']+10, nwave)
        psfparams = evalcoeffs(psfdata, wavelengths)
        psfparams = evalcoeffs(psfdata, wavelengths, specmin=0, nspec=25)
        self.assertEqual(psfparams['X'].shape, (25, nwave))
        psfparams = evalcoeffs(psfdata, wavelengths, specmin=25, nspec=5)
        self.assertEqual(psfparams['Y'].shape, (5, nwave))        

    @unittest.skipIf(not specter_available, 'specter not available')
    def test_compare_specter(self):
        #- gpu_specter version
        psfdata = read_psf(self.psffile)
        meta = psfdata['PSF'].meta
        wavelengths = np.linspace(meta['WAVEMIN'], meta['WAVEMAX'], 30)
        psfparams = evalcoeffs(psfdata, wavelengths)
        
        #- specter version
        psf = specter.psf.load_psf(self.psffile)
        iispec = np.arange(500)
        
        # print('Comparing X and Y')
        self.assertTrue(np.allclose(psf.x(iispec, wavelengths), psfparams['X']))
        self.assertTrue(np.allclose(psf.y(iispec, wavelengths), psfparams['Y']))

        common_keys = sorted(set(psfparams.keys()) & set(psf.coeff.keys()))
        self.assertTrue(len(common_keys) > 0)
        
        for key in common_keys:
            # print(f'Comparing {key}')
            ok = np.allclose(psfparams[key], psf.coeff[key].eval(iispec, wavelengths))
            self.assertTrue(ok, key)

        for i in range(psfparams['GH'].shape[0]):
            for j in range(psfparams['GH'].shape[1]):
                # print(f'Comparing GH-{i}-{j}')
                ok = np.allclose(psfparams['GH'][i,j],
                                 psf.coeff[f'GH-{i}-{j}'].eval(iispec, wavelengths))
                self.assertTrue(ok, f'GH-{i}-{j}')

if __name__ == '__main__':
    unittest.main()
