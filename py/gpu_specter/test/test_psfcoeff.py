import unittest, os, shutil, uuid
from astropy.table import Table
import numpy as np

from gpu_specter.io import read_psf
from gpu_specter.extract.cpu import evalcoeffs
from .util import find_test_file

try:
    import specter.psf
    specter_available = True
except ImportError:
    specter_available = False

try:
    import cupy as cp
    from numba import cuda
    from gpu_specter.extract.gpu import evalcoeffs as gpu_evalcoeffs
    gpu_available = cp.is_available()
except ImportError:
    gpu_available = False

class TestPSFCoeff(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.psffile = find_test_file('psf')
        cls.psfdata = read_psf(cls.psffile)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basics(self):
        psfdata = self.psfdata
        for key in ('XTRACE', 'YTRACE', 'PSF'):
            self.assertTrue(key in psfdata.keys(), f'{key} is not in psfdata')
            meta = psfdata[key].meta
            self.assertTrue('WAVEMIN' in meta.keys(), f'{key}.meta missing WAVEMIN')
            self.assertTrue('WAVEMAX' in meta.keys(), f'{key}.meta missing WAVEMAX')

        #- wavelengths outside original range are allowed
        meta = psfdata['PSF'].meta
        nspec = psfdata['PSF']['COEFF'].shape[1]
        nwave = 30
        wavelengths = np.linspace(meta['WAVEMIN']-10, meta['WAVEMAX']+10, nwave)
        psfparams = evalcoeffs(psfdata, wavelengths)
        self.assertEqual(psfparams['X'].shape, (nspec, nwave))
        psfparams = evalcoeffs(psfdata, wavelengths, specmin=0, nspec=25)
        self.assertEqual(psfparams['X'].shape, (25, nwave))
        psfparams = evalcoeffs(psfdata, wavelengths, specmin=25, nspec=5)
        self.assertEqual(psfparams['Y'].shape, (5, nwave))
        wavelengths = np.linspace(meta['WAVEMIN']-10, meta['WAVEMAX']+10, nwave)
        psfparams = evalcoeffs(psfdata, wavelengths[None, :] + np.zeros(nspec)[:,
                                                                            None], specmin=0) #, nspec=25)
        self.assertEqual(psfparams['X'].shape, (nspec, nwave))

    @unittest.skipIf(not gpu_available, 'gpu not available')
    def test_gpu_basics(self):
        psfdata = self.psfdata

        #- wavelengths outside original range are allowed
        meta = psfdata['PSF'].meta
        nspec = psfdata['PSF']['COEFF'].shape[1]
        nwave = 30
        wavelengths = np.linspace(meta['WAVEMIN']-10, meta['WAVEMAX']+10, nwave)
        psfparams = gpu_evalcoeffs(psfdata, wavelengths)
        self.assertEqual(psfparams['X'].shape, (nspec, nwave))
        psfparams = gpu_evalcoeffs(psfdata, wavelengths, specmin=0, nspec=25)
        self.assertEqual(psfparams['X'].shape, (25, nwave))
        psfparams = gpu_evalcoeffs(psfdata, wavelengths, specmin=25, nspec=5)
        self.assertEqual(psfparams['Y'].shape, (5, nwave))

    @unittest.skipIf(not gpu_available, 'gpu not available')
    def test_compare_gpu(self):
        psfdata = self.psfdata
        meta = psfdata['PSF'].meta
        wavelengths = np.linspace(meta['WAVEMIN'], meta['WAVEMAX'], 30)
        psfparams = evalcoeffs(psfdata, wavelengths)

        psfparams_gpu = gpu_evalcoeffs(psfdata, wavelengths)

        self.assertTrue(np.array_equal(psfparams['X'], cp.asnumpy(psfparams_gpu['X'])))
        self.assertTrue(np.array_equal(psfparams['Y'], cp.asnumpy(psfparams_gpu['Y'])))

        common_keys = set(psfparams.keys() & set(psfparams_gpu.keys()))
        self.assertTrue(len(common_keys) > 0)

        eps_double = np.finfo(np.float64).eps

        for key in common_keys:
            # print(f'Comparing {key}')
            if key == 'GH':
                continue
            ok = np.allclose(psfparams[key], cp.asnumpy(psfparams_gpu[key]), rtol=1e2*eps_double, atol=0)
            self.assertTrue(ok, key)

        for i in range(psfparams['GH'].shape[0]):
            for j in range(psfparams['GH'].shape[1]):
                # print(f'Comparing GH-{i}-{j}')
                ok = np.allclose(psfparams['GH'][i,j], cp.asnumpy(psfparams_gpu['GH'][i,j]), rtol=1e5*eps_double, atol=0)
                self.assertTrue(ok, f'GH-{i}-{j}')


    @unittest.skipIf(not specter_available, 'specter not available')
    def test_compare_specter(self):
        #- gpu_specter version
        psfdata = self.psfdata
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
