import unittest, os, shutil, uuid
import pkg_resources
from astropy.table import Table
import numpy as np

from gpu_specter.io import read_psf
from gpu_specter.core import Patch
from gpu_specter.extract.cpu import (
    projection_matrix, get_spots, get_resolution_diags,
    ex2d_padded, ex2d_patch
)
from gpu_specter.extract.both import xp_ex2d_patch
from .util import find_test_file

try:
    import specter.psf
    import specter.extract
    specter_available = True
except ImportError:
    specter_available = False

try:
    import cupy as cp
    cupy_available = cp.is_available()
except ImportError:
    cupy_available = False

class TestExtract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.psffile = find_test_file('psf')
        cls.psfdata = read_psf(cls.psffile)

        cls.wavelengths = np.arange(6000, 6050, 1)
        nwave = len(cls.wavelengths)
        nspec = 5

        cls.psferr = cls.psfdata['PSF'].meta['PSFERR']
        cls.spots, cls.corners, psfparams = get_spots(0, nspec, cls.wavelengths, cls.psfdata)
        cls.A4, cls.xyrange = projection_matrix(0, nspec, 0, nwave, cls.spots, cls.corners)

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

        cls.phot = phot

        xmin, xmax, ymin, ymax = cls.xyrange
        ny = ymax - ymin
        nx = xmax - xmin

        A2 = cls.A4.reshape(ny*nx, nspec*nwave)
        cls.img = A2.dot(cls.phot.ravel()).reshape(ny, nx)

        cls.readnoise = 3.0
        # set an arbitrary seed for consistency between test runs
        np.random.seed(9821)
        cls.noisyimg = np.random.normal(loc=0.0, scale=cls.readnoise, size=(ny, nx))
        cls.noisyimg += np.random.poisson(cls.img)
        #- for test, cheat by using noiseless img instead of noisyimg to estimate variance
        cls.imgivar = 1.0/(cls.img + cls.readnoise**2)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basics(self):
        ny, nx, nspec, nwave = self.A4.shape

        img = np.random.randn(ny, nx)
        imgivar = np.ones((ny, nx))

        flux, fluxivar, R, xflux = ex2d_patch(img, imgivar, self.A4)

        self.assertEqual(flux.shape, (nspec, nwave))
        self.assertEqual(fluxivar.shape, (nspec, nwave))
        self.assertEqual(R.shape, (nspec*nwave, nspec*nwave))

    def test_ex2d_padded(self):
        ny, nx, nspec, nwave = self.A4.shape

        ispec = 0
        bundlesize = 5
        wavepad = 5
        nwavepatch = nwave - 2*wavepad
        iwave = wavepad

        from gpu_specter.extract.cpu import get_xyrange

        xmin, xmax, ypadmin, ypadmax = get_xyrange(ispec, nspec, iwave-wavepad, nwave, self.spots, self.corners)
        xlo, xhi, ymin, ymax = get_xyrange(ispec, nspec, iwave, nwavepatch, self.spots, self.corners)

        img = np.zeros((ypadmax, xmax))
        ivar = np.zeros((ypadmax, xmax))

        ny = ymax - ymin

        patchslice = np.s_[ypadmin:ypadmax, xmin:xmax]

        img[ypadmin:ypadmax, xmin:xmax] = self.noisyimg
        ivar[ypadmin:ypadmax, xmin:xmax] = self.imgivar

        patch = Patch(ispec, iwave, 0, bundlesize, nwavepatch, wavepad, nwave, bundlesize, 11)

        result = ex2d_padded(
            img, ivar, patch, self.spots, self.corners, 
            pixpad_frac=0, regularize=1e-8, model=True, psferr=self.psferr
        )

        modelimage = np.zeros_like(img)
        modelimage[result['xyslice']] = result['modelimage']

        # self.assertEqual()

        # img = np.random.randn(ny, nx)
        # imgivar = np.ones((ny, nx))

        # flux, varflux, R = ex2d_patch(img, imgivar, self.A4)

        # self.assertEqual(flux.shape, (nspec, nwave))
        # self.assertEqual(varflux.shape, (nspec, nwave))
        # self.assertEqual(R.shape, (nspec*nwave, nspec*nwave))

    def test_compare_xp_cpu(self):
        # Compare the "signal" decorrelation method
        flux0, ivar0, R0, xflux0 = ex2d_patch(self.noisyimg, self.imgivar, self.A4, decorrelate='signal')
        flux1, ivar1, R1, xflux1 = xp_ex2d_patch(self.noisyimg, self.imgivar, self.A4, decorrelate='signal')

        self.assertTrue(np.allclose(flux0, flux1))
        self.assertTrue(np.allclose(ivar0, ivar1))
        self.assertTrue(np.allclose(R0, R1))
        self.assertTrue(np.allclose(np.abs(flux0 - flux1)/np.sqrt(1./ivar0 + 1./ivar1), np.zeros_like(flux0)))

        # Compare the "noise" decorrelation method
        flux0, ivar0, R0, xflux0 = ex2d_patch(self.noisyimg, self.imgivar, self.A4, decorrelate='noise')
        flux1, ivar1, R1, xflux1 = xp_ex2d_patch(self.noisyimg, self.imgivar, self.A4, decorrelate='noise')

        self.assertTrue(np.allclose(flux0, flux1))
        self.assertTrue(np.allclose(ivar0, ivar1))
        self.assertTrue(np.allclose(R0, R1))
        self.assertTrue(np.allclose(np.abs(flux0 - flux1)/np.sqrt(1./ivar0 + 1./ivar1), np.zeros_like(flux0)))

    @unittest.skipIf(not cupy_available, 'cupy not available')
    def test_compare_icov(self):
        from gpu_specter.extract.cpu import dotdot1, dotdot2, dotdot3

        ny, nx, nspec, nwave = self.A4.shape

        pixel_ivar = self.imgivar.ravel()
        A = self.A4.reshape(ny*nx, nspec*nwave)

        icov0 = A.T.dot(np.diag(pixel_ivar).dot(A))
        icov1 = dotdot1(A, pixel_ivar) # array broadcast
        icov2 = dotdot2(A, pixel_ivar) # scipy sparse
        icov3 = dotdot3(A, pixel_ivar) # numba

        pixel_ivar_gpu = cp.asarray(pixel_ivar)
        A_gpu = cp.asarray(A)
        icov_gpu = (A_gpu.T * pixel_ivar_gpu).dot(A_gpu) # array broadcast

        eps_double = np.finfo(np.float64).eps
        np.testing.assert_allclose(icov0, icov1, rtol=2*eps_double, atol=0)
        np.testing.assert_allclose(icov0, icov2, rtol=10*eps_double, atol=0)
        np.testing.assert_allclose(icov0, icov3, rtol=10*eps_double, atol=0)

        np.testing.assert_allclose(icov0, cp.asnumpy(icov_gpu), rtol=10*eps_double, atol=0)
        np.testing.assert_allclose(icov1, cp.asnumpy(icov_gpu), rtol=10*eps_double, atol=0)
        np.testing.assert_allclose(icov2, cp.asnumpy(icov_gpu), rtol=10*eps_double, atol=0)
        np.testing.assert_allclose(icov3, cp.asnumpy(icov_gpu), rtol=10*eps_double, atol=0)

    @unittest.skipIf(not cupy_available, 'cupy not available')
    def test_dotall(self):
        from gpu_specter.extract.cpu import dotall, dotdot3
        from gpu_specter.extract.both import xp_dotall

        ny, nx, nspec, nwave = self.A4.shape

        pixel_values = self.noisyimg.ravel()
        pixel_ivar = self.imgivar.ravel()
        A = self.A4.reshape(ny*nx, nspec*nwave)

        icov, y, fluxweight = dotall(pixel_values, pixel_ivar, A)
        icov3 = dotdot3(A, pixel_ivar)

        pixel_values_gpu = cp.asarray(pixel_values)
        pixel_ivar_gpu = cp.asarray(pixel_ivar)
        A_gpu = cp.asarray(A)

        icov_gpu, y_gpu, fluxweight_gpu = xp_dotall(pixel_values_gpu, pixel_ivar_gpu, A_gpu)

        eps_double = np.finfo(np.float64).eps
        np.testing.assert_array_equal(icov, icov3)
        where = np.where(~np.isclose(icov, cp.asnumpy(icov_gpu), rtol=1e3*eps_double, atol=0))
        np.testing.assert_allclose(icov, cp.asnumpy(icov_gpu), rtol=1e3*eps_double, atol=0, err_msg=f"where: {where}")
        np.testing.assert_allclose(y, cp.asnumpy(y_gpu), rtol=1e3*eps_double, atol=0)
        np.testing.assert_allclose(fluxweight, cp.asnumpy(fluxweight_gpu), rtol=1e3*eps_double, atol=0)

    @unittest.skipIf(not cupy_available, 'cupy not available')
    def test_compare_solve(self):
        import scipy.linalg

        ny, nx, nspec, nwave = self.A4.shape

        pixel_values = self.noisyimg.ravel()
        pixel_ivar = self.imgivar.ravel()
        A = self.A4.reshape(ny*nx, nspec*nwave)

        icov = (A.T * pixel_ivar).dot(A)
        y = (A.T * pixel_ivar).dot(pixel_values)
        deconvolved_scipy = scipy.linalg.solve(icov, y)
        deconvolved_numpy = np.linalg.solve(icov, y)

        icov_gpu = cp.asarray(icov)
        y_gpu = cp.asarray(y)

        deconvolved_gpu = cp.linalg.solve(icov_gpu, y_gpu)

        eps_double = np.finfo(np.float64).eps
        np.testing.assert_allclose(deconvolved_scipy, deconvolved_numpy, rtol=eps_double, atol=0)
        np.testing.assert_allclose(deconvolved_scipy, cp.asnumpy(deconvolved_gpu), rtol=1e5*eps_double, atol=0)
        np.testing.assert_allclose(deconvolved_numpy, cp.asnumpy(deconvolved_gpu), rtol=1e5*eps_double, atol=0)


    @unittest.skipIf(not cupy_available, 'cupy not available')
    def test_compare_deconvolve(self):

        from gpu_specter.extract.cpu import deconvolve as cpu_deconvolve
        from gpu_specter.extract.both import xp_deconvolve as gpu_deconvolve

        ny, nx, nspec, nwave = self.A4.shape

        pixel_values = self.noisyimg.ravel()
        pixel_ivar = self.imgivar.ravel()
        A = self.A4.reshape(ny*nx, nspec*nwave)

        pixel_values_gpu = cp.asarray(pixel_values)
        pixel_ivar_gpu = cp.asarray(pixel_ivar)
        A_gpu = cp.asarray(A)

        deconvolved0, iCov0 = cpu_deconvolve(pixel_values, pixel_ivar, A)
        deconvolved_gpu, iCov_gpu = gpu_deconvolve(pixel_values_gpu, pixel_ivar_gpu, A_gpu)

        deconvolved1 = cp.asnumpy(deconvolved_gpu)
        iCov1 = cp.asnumpy(iCov_gpu)

        eps_double = np.finfo(np.float64).eps
        np.testing.assert_allclose(deconvolved0, deconvolved1, rtol=1e5*eps_double, atol=0)
        np.testing.assert_allclose(iCov0, iCov1, rtol=1e3*eps_double, atol=0)


    @unittest.skipIf(not cupy_available, 'cupy not available')
    def test_compare_get_Rdiags(self):
        from gpu_specter.extract.gpu import get_resolution_diags as gpu_get_resolution_diags

        nspec, ispec, specmin = 5, 5, 4
        nwave, wavepad, ndiag = 50, 10, 7
        nwavetot = nwave + 2*wavepad
        nspectot = nspec + 2
        n = nwavetot*nspectot
        R = np.arange(n*n).reshape(n, n)

        Rdiags0 = get_resolution_diags(R, ndiag, ispec-specmin, nspec, nwave, wavepad)

        R_gpu = cp.asarray(R)
        s = np.s_[ispec-specmin:ispec-specmin+nspec]
        Rdiags1_gpu = gpu_get_resolution_diags(R_gpu, ndiag, nspectot, nwave, wavepad)[s]

        np.testing.assert_array_equal(Rdiags0, Rdiags1_gpu.get())


    @unittest.skipIf(not cupy_available, 'cupy not available')
    def test_compare_xp_gpu(self):
        noisyimg_gpu = cp.asarray(self.noisyimg)
        imgivar_gpu = cp.asarray(self.imgivar)
        A4_gpu = cp.asarray(self.A4)

        # Compare the "signal" decorrelation method
        flux0, ivar0, R0, xflux0 = ex2d_patch(self.noisyimg, self.imgivar, self.A4, decorrelate='signal')
        flux1_gpu, ivar1_gpu, R1_gpu, xflux1_gpu = xp_ex2d_patch(noisyimg_gpu, imgivar_gpu, A4_gpu, decorrelate='signal')

        flux1 = cp.asnumpy(flux1_gpu)
        ivar1 = cp.asnumpy(ivar1_gpu)
        R1 = cp.asnumpy(R1_gpu)

        eps_double = np.finfo(np.float64).eps

        where = np.where(~np.isclose(flux0, flux1, rtol=1e5*eps_double, atol=0))
        np.testing.assert_allclose(flux0, flux1, rtol=1e5*eps_double, atol=0, err_msg=f"where: {where}")
        self.assertTrue(np.allclose(ivar0, ivar1, rtol=1e3*eps_double, atol=0))
        self.assertTrue(np.allclose(np.diag(R0), np.diag(R1), rtol=1e2*eps_double, atol=1e3*eps_double))
        self.assertTrue(np.allclose(np.abs(flux0 - flux1)/np.sqrt(1./ivar0 + 1./ivar1), np.zeros_like(flux0)))

        # Compare the "noise" decorrelation method
        flux0, ivar0, R0, xflux0 = ex2d_patch(self.noisyimg, self.imgivar, self.A4, decorrelate='noise')
        flux1_gpu, ivar1_gpu, R1_gpu, xflux1_gpu = xp_ex2d_patch(noisyimg_gpu, imgivar_gpu, A4_gpu, decorrelate='noise')

        flux1 = cp.asnumpy(flux1_gpu)
        ivar1 = cp.asnumpy(ivar1_gpu)
        R1 = cp.asnumpy(R1_gpu)

        self.assertTrue(np.allclose(flux0, flux1, rtol=1e5*eps_double, atol=0))
        self.assertTrue(np.allclose(ivar0, ivar1, rtol=1e3*eps_double, atol=0))
        self.assertTrue(np.allclose(np.diag(R0), np.diag(R1), rtol=1e2*eps_double, atol=0))
        self.assertTrue(np.allclose(np.abs(flux0 - flux1)/np.sqrt(1./ivar0 + 1./ivar1), np.zeros_like(flux0)))

    @unittest.skipIf(not cupy_available, 'cupy not available')
    def test_compare_batch_extraction(self):
        from gpu_specter.extract.gpu import _apply_weights, _batch_extraction
        noisyimg_gpu = cp.asarray(self.noisyimg)
        imgivar_gpu = cp.asarray(self.imgivar)
        A4_gpu = cp.asarray(self.A4)

        # Compare the "signal" decorrelation method
        flux0, ivar0, R0, xflux0 = ex2d_patch(self.noisyimg, self.imgivar, self.A4, decorrelate='signal')

        ny, nx, nspec, nwave = self.A4.shape
        icov, y = _apply_weights(noisyimg_gpu.ravel(), imgivar_gpu.ravel(), A4_gpu.reshape(ny*nx, nspec*nwave), regularize=0)
        flux1_gpu, ivar1_gpu, R1_gpu, xflux1_gpu = _batch_extraction(icov, y, nwave)
        # Rdiags = get_resolution_diags(R, ndiag, nspectot, nwave, wavepad)[specslice[0]]

        flux1 = cp.asnumpy(flux1_gpu.reshape(nspec, nwave))
        ivar1 = cp.asnumpy(ivar1_gpu.reshape(nspec, nwave))
        R1 = cp.asnumpy(R1_gpu.reshape(nspec*nwave, nspec*nwave))

        eps_double = np.finfo(np.float64).eps
        eps_single = np.finfo(np.float32).eps

        # require agreement to be a little better (1e-2) than single precision (eps_single).
        np.testing.assert_allclose(flux0, flux1, rtol=1e-2*eps_single, atol=0)
        np.testing.assert_allclose(ivar0, ivar1, rtol=1e-2*eps_single, atol=0)
        np.testing.assert_allclose(np.diag(R0), np.diag(R1), rtol=1e-2*eps_single, atol=0)

        mask = (ivar0 == 0) | (ivar1 == 0)
        var0 = np.reciprocal(~mask*ivar0 + mask)
        var1 = np.reciprocal(~mask*ivar1 + mask)
        ivar = np.reciprocal(~mask*(var0 + var1) + mask)
        dflux = flux0 - flux1
        pull = ~mask*dflux*np.sqrt(ivar)
        np.testing.assert_array_less(np.abs(pull), 1e-2*eps_single)

    @unittest.skipIf(not specter_available, 'specter not available')
    def test_compare_specter(self):
        ny, nx, nspec, nwave = self.A4.shape

        psf = specter.psf.load_psf(self.psffile)
        img = psf.project(self.wavelengths, self.phot, xyrange=self.xyrange)

        # self.assertTrue(np.allclose(self.img, img))

        #- Compare the "signal" decorrelation method
        flux0, ivar0, R0 = specter.extract.ex2d_patch(self.noisyimg, self.imgivar, psf, 0, nspec,
            self.wavelengths, xyrange=self.xyrange, ndecorr=False)
        flux1, ivar1, R1, xflux1 = ex2d_patch(self.noisyimg, self.imgivar, self.A4, decorrelate='signal')

        #- Note that specter is using it's version of the projection matrix
        # A = psf.projection_matrix((0, nspec), self.wavelengths, self.xyrange).toarray()
        # A4 = A.reshape(self.A4.shape)
        # flux1, ivar1, R1 = ex2d_patch(self.noisyimg, self.imgivar, A4, decorrelate='signal')

        self.assertTrue(np.allclose(flux0, flux1))
        self.assertTrue(np.allclose(ivar0, ivar1))
        self.assertTrue(np.allclose(R0, R1))
        #self.assertTrue(np.allclose(np.abs(flux0 - flux1)/np.sqrt(1./ivar0 + 1./ivar1), np.zeros_like(flux0)))

        # Compare the "noise" decorrelation method
        flux0, ivar0, R0 = specter.extract.ex2d_patch(self.noisyimg, self.imgivar, psf, 0, nspec,
            self.wavelengths, xyrange=self.xyrange, ndecorr=True)
        flux1, ivar1, R1, xflux1 = ex2d_patch(self.noisyimg, self.imgivar, self.A4, decorrelate='noise')

        self.assertTrue(np.allclose(flux0, flux1))
        self.assertTrue(np.allclose(ivar0, ivar1))
        self.assertTrue(np.allclose(R0, R1))
        #self.assertTrue(np.allclose(np.abs(flux0 - flux1)/np.sqrt(1./ivar0 + 1./ivar1), np.zeros_like(flux0)))


if __name__ == '__main__':
    unittest.main()
