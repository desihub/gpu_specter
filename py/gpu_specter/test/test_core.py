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

try:
    from mpi4py import MPI
    mpi_available = True
except:
    mpi_available = False


@unittest.skipIf(not preproc_available, f'{imgfile} not available')
class TestCore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):


        if mpi_available:
            cls.comm = MPI.COMM_WORLD
            cls.rank = cls.comm.Get_rank()
            cls.size = cls.comm.Get_size()
            #- ignore mpi if fewer than 2 ranks by setting comm to None
            if cls.size < 2:
                cls.comm = None
        else:
            cls.comm = None
            cls.rank = 0
            cls.size = 1

        cls.psffile = pkg_resources.resource_filename(
            'gpu_specter', 'test/data/psf-r0-00051060.fits')

        if cls.rank == 0:
            cls.psfdata = read_psf(cls.psffile)
            cls.imgdata = read_img(imgfile)
        else:
            cls.psfdata = None
            cls.imgdata = None

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_extract_frame(self):
        if self.comm is not None:
            self.comm.barrier()

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
            comm=self.comm,
            gpu=None,
            loglevel='WARN',
        )

        if self.rank > 0:
            return

        keys = (
            'wave', 'specflux', 'specivar', 'Rdiags',
            'pixmask_fraction', 'chi2pix', 'modelimage'
        )
        for key in keys:
            self.assertTrue(key in frame.keys(), key)

        self.assertEqual(frame['wave'].shape, (nwave, ))
        self.assertEqual(frame['specflux'].shape, (nspec, nwave))
        self.assertEqual(frame['specivar'].shape, (nspec, nwave))
        # self.assertEqual(frame['Rdiags'].shape, (nspec, ndiag, nwave))
        self.assertEqual(frame['pixmask_fraction'].shape, (nspec, nwave))
        self.assertEqual(frame['chi2pix'].shape, (nspec, nwave))

    @unittest.skipIf(not specter_available, 'specter not available')
    def test_compare_specter(self):
        if self.comm is not None:
            self.comm.barrier()

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
            comm=self.comm,
            gpu=None,
            loglevel='WARN',
        )

        if self.rank > 0:
            return

        self.assertEqual(frame_spex['specflux'].shape[0], nspec)

        psf = specter.psf.load_psf(self.psffile)

        wavelengths = frame_spex['wave']

        frame_specter = specter.extract.ex2d(
            self.imgdata['image'], self.imgdata['ivar'], psf, 
            specmin, nspec, wavelengths, 
            xyrange=None, regularize=0.0, ndecorr=False,
            bundlesize=bundlesize, nsubbundles=nsubbundles,
            wavesize=nwavestep, 
            full_output=True, verbose=True,
            debug=False, psferr=None,
        )

        self.assertEqual(frame_spex['specflux'].shape, frame_specter['flux'].shape)

        diff = frame_spex['specflux'] - frame_specter['flux']
        norm = np.sqrt(1.0/frame_spex['specivar'] + 1.0/frame_specter['ivar'])
        pull = diff/norm
        pull_threshold = 0.01
        pull_fraction = np.average(np.abs(pull).ravel() < pull_threshold)

        self.assertGreaterEqual(pull_fraction, 0.95)

    @unittest.skipIf(not gpu_available, 'gpu not available')
    def test_compare_gpu(self):
        if self.comm is not None:
            self.comm.barrier()

        bundlesize = 10
        wavelength = '5760.0,7620.0,0.8'

        specmin = 0
        nspec = 10
        nwavestep = 50
        nsubbundles = 2

        frame_cpu = extract_frame(
            self.imgdata, self.psfdata, bundlesize,
            specmin, nspec,
            wavelength=wavelength,
            nwavestep=nwavestep, nsubbundles=nsubbundles,
            comm=self.comm,
            gpu=None,
            loglevel='INFO',
        )

        frame_gpu = extract_frame(
            self.imgdata, self.psfdata, bundlesize,
            specmin, nspec,
            wavelength=wavelength,
            nwavestep=nwavestep, nsubbundles=nsubbundles,
            comm=self.comm,
            gpu=True,
            loglevel='INFO',
            batch_subbundle=False,
        )

        if self.rank > 0:
            return

        self.assertEqual(frame_cpu['specflux'].shape, frame_gpu['specflux'].shape)

        diff = frame_cpu['specflux'] - frame_gpu['specflux']
        norm = np.sqrt(1.0/frame_cpu['specivar'] + 1.0/frame_gpu['specivar'])
        pull = diff/norm
        pull_threshold = 5e-4
        self.assertTrue(np.alltrue(np.abs(pull) < pull_threshold))
        # pull_fraction = np.average(np.abs(pull) < pull_threshold)
        # self.assertGreaterEqual(pull_fraction, 0.99)

        eps_double = np.finfo(np.float64).eps
        np.testing.assert_allclose(frame_cpu['specflux'], frame_gpu['specflux'], rtol=1e-3, atol=0)
        np.testing.assert_allclose(frame_cpu['specivar'], frame_gpu['specivar'], rtol=1e-3, atol=0)
        np.testing.assert_allclose(frame_cpu['Rdiags'], frame_gpu['Rdiags'], rtol=1e-5, atol=1e-6)

    @unittest.skipIf(not gpu_available, 'gpu not available')
    def test_gpu_batch_subbundle(self):
        if self.comm is not None:
            self.comm.barrier()

        bundlesize = 10
        wavelength = '5760.0,7620.0,0.8'

        specmin = 0
        nspec = 20
        nwavestep = 50
        nsubbundles = 2

        patch_gpu_frame = extract_frame(
            self.imgdata, self.psfdata, bundlesize,
            specmin, nspec,
            wavelength=wavelength,
            nwavestep=nwavestep, nsubbundles=nsubbundles,
            comm=self.comm,
            gpu=True,
            loglevel='WARN',
            batch_subbundle=False,
        )

        subbundle_gpu_frame = extract_frame(
            self.imgdata, self.psfdata, bundlesize,
            specmin, nspec,
            wavelength=wavelength,
            nwavestep=nwavestep, nsubbundles=nsubbundles,
            comm=self.comm,
            gpu=True,
            loglevel='WARN',
            batch_subbundle=True,
        )

        if self.rank > 0:
            return

        self.assertEqual(patch_gpu_frame['specflux'].shape, subbundle_gpu_frame['specflux'].shape)

        diff = patch_gpu_frame['specflux'] - subbundle_gpu_frame['specflux']
        norm = np.sqrt(1.0/patch_gpu_frame['specivar'] + 1.0/subbundle_gpu_frame['specivar'])
        pull = diff/norm
        pull_threshold = 5e-4
        self.assertTrue(np.alltrue(np.abs(pull) < pull_threshold))
        # pull_fraction = np.average(np.abs(pull) < pull_threshold)
        # self.assertGreaterEqual(pull_fraction, 0.99)

        eps_double = np.finfo(np.float64).eps
        np.testing.assert_allclose(patch_gpu_frame['specflux'], subbundle_gpu_frame['specflux'], rtol=1e-3, atol=0)
        np.testing.assert_allclose(patch_gpu_frame['specivar'], subbundle_gpu_frame['specivar'], rtol=1e-3, atol=0)
        np.testing.assert_allclose(patch_gpu_frame['Rdiags'], subbundle_gpu_frame['Rdiags'], rtol=1e-5, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
