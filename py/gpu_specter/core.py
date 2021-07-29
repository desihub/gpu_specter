"""
Core scaffolding for divide and conquer extraction algorithm
"""

import sys
import time

import numpy as np

try:
    import numba.cuda
    numba.cuda.is_available()
    import cupy as cp
    import cupy.prof
    cp.is_available()
except ImportError:
    pass

from gpu_specter.util import get_logger
from gpu_specter.util import get_array_module
from gpu_specter.util import Timer
from gpu_specter.util import gather_ndarray

class Patch(object):
    def __init__(self, ispec, iwave, bspecmin, nspectra_per_patch, nwavestep, wavepad, nwave,
        bundlesize, ndiag):
        """Convenience data wrapper for divide and conquer extraction patches

        Args:
            ispec: starting spectrum index
            iwave: starting wavelength index
            bspecmin: starting spectrum index of the bundle that this patch belongs to
            nspectra_per_patch: number of spectra to extract (not including padding)
            nwavestep: number of wavelengths to extract (not including padding)
            wavepad: number of extra wave bins to extract (and discard) on each end
            nwave: number of wavelength bins in for entire bundle
            bundlesize: size of fiber bundles
            ndiag: number of diagonal elements to keep in the resolution matrix

        All args become attributes.

        Additional attributes created:
            specslice: where this patch goes in the bundle result array
            waveslice: where this patch goes in the bundle result array
            keepslice: wavelength slice to keep from padded patch (the final patch in the bundle
                will be narrower when (nwave % nwavestep) != 0)
        """

        self.ispec = ispec
        self.iwave = iwave

        self.nspectra_per_patch = nspectra_per_patch
        self.nwavestep = nwavestep

        #- padding to apply to patch
        self.wavepad = wavepad

        #- where this patch should go
        #- note: spec indexing is relative to subbundle
        self.bspecmin = bspecmin
        self.specslice = np.s_[ispec-bspecmin:ispec-bspecmin+nspectra_per_patch]
        self.waveslice = np.s_[iwave-wavepad:iwave-wavepad+nwavestep]

        #- how much of the patch to keep
        nwavekeep = min(nwavestep, nwave - (iwave-wavepad))
        self.keepslice = np.s_[0:nwavekeep]

        #- to help with reassembly
        self.nwave = nwave
        self.bundlesize = bundlesize
        self.ndiag = ndiag

        #- the image slice covered by this patch
        #- will be set during extaction
        self.xyslice = None


def assemble_bundle_patches(rankresults):
    """
    Assembles bundle patches into output arrays

    Args:
        rankresults: list of lists containing individual patch extraction results

    Returns:
        (spexflux, specivar, Rdiags) tuple
    """

    #- flatten list of lists into single list
    allresults = list()
    for rr in rankresults:
        allresults.extend(rr)

    #- peak at result to get bundle params
    patch = allresults[0][0]
    nwave = patch.nwave
    bundlesize = patch.bundlesize
    ndiag = patch.ndiag

    xp = get_array_module(allresults[0][1]['flux'])

    #- Allocate output arrays to fill
    specflux = xp.zeros((bundlesize, nwave))
    specivar = xp.zeros((bundlesize, nwave))
    Rdiags = xp.zeros((bundlesize, 2*ndiag+1, nwave))
    pixmask_fraction = xp.zeros((bundlesize, nwave))
    chi2pix = xp.zeros((bundlesize, nwave))

    #- Find the global extent of patches in this bundle
    ystart = xstart = float('inf')
    ystop = xstop = -float('inf')
    for patch, result in allresults:
        if patch.xyslice is None:
            continue
        ystart = min(ystart, patch.xyslice[0].start)
        ystop = max(ystop, patch.xyslice[0].stop)
        xstart = min(xstart, patch.xyslice[1].start)
        xstop = max(xstop, patch.xyslice[1].stop)
    ny, nx = ystop - ystart, xstop - xstart
    xyslice = np.s_[ystart:ystop, xstart:xstop]
    modelimage = xp.zeros((ny, nx))

    #- Now put these into the final arrays
    for patch, result in allresults:
        fx = result['flux']
        fxivar = result['ivar']
        xRdiags = result['Rdiags']
        xpixmask_fraction = result['pixmask_fraction']
        xchi2pix = result['chi2pix']

        if patch.xyslice is None:
            # print(f'patch {(patch.ispec, patch.iwave)} is off the edge of the image')
            continue

        #- put the extracted patch into the output arrays
        specflux[patch.specslice, patch.waveslice] = fx[:, patch.keepslice]
        specivar[patch.specslice, patch.waveslice] = fxivar[:, patch.keepslice]
        Rdiags[patch.specslice, :, patch.waveslice] = xRdiags[:, :, patch.keepslice]
        pixmask_fraction[patch.specslice, patch.waveslice] = xpixmask_fraction[:, patch.keepslice]
        chi2pix[patch.specslice, patch.waveslice] = xchi2pix[:, patch.keepslice]

        patchmodel = result['modelimage']
        #- Skip if patchmodel is None, an array of None, or contains any nans
        skip_patchmodel = (
            (patchmodel is None)
            or (not patchmodel.any())
            or (not np.all(np.isfinite(patchmodel)))
        )
        if skip_patchmodel:
            continue
        ymin = patch.xyslice[0].start - ystart
        xmin = patch.xyslice[1].start - xstart
        patchny, patchnx = patchmodel.shape
        modelimage[ymin:ymin+patchny, xmin:xmin+patchnx] += patchmodel

    return specflux, specivar, Rdiags, pixmask_fraction, chi2pix, modelimage, xyslice

# @cupy.prof.TimeRangeDecorator("extract_bundle")
def extract_bundle(image, imageivar, psf, wave, fullwave, bspecmin, bundlesize=25, nsubbundles=1, batch_subbundle=False,
    nwavestep=50, wavepad=10, comm=None, gpu=None, loglevel=None, model=None, regularize=0,
    psferr=None, pixpad_frac=0):
    """
    Extract 1D spectra from a single bundle of a 2D image.

    Args:
        image: full 2D array of image pixels
        imageivar: full 2D array of inverse variance for the image
        psf: dictionary psf object (see gpu_specter.io.read_psf)
        wave: 1D array of wavelengths to extract
        fullwave: Padded 1D array of wavelengths to extract
        bspecmin: index of the first spectrum in the bundle

    Options:
        bundlesize: fixed number of spectra per bundle (25 for DESI)
        nsubbundles: number of spectra per patch
        batch_subbundles: whether or not to use batch subbundle extraction
        nwavestep: number of wavelength bins per patch
        wavepad: number of wavelengths bins to add on each end of patch for extraction
        comm: mpi communicator (no mpi: None)
        gpu: use GPU for extraction (not yet implemented)
        loglevel: log print level
        model: indicate whether or not to compute the image model
        regularize: regularization parameter
        psferr: scale factor to use for psf in chi2
        pixpad_frac: fraction of padded pixels to use in extraction

    Returns:
        bundle: (flux, ivar, resolution, pixmask_fraction, chi2pix, modelimage, xyslice) tuple

    """
    timer = Timer()

    if comm is None:
        rank = 0
        size = 1
    else:
        rank = comm.rank
        size = comm.size

    log = get_logger(loglevel)

    #- Extracting on CPU or GPU?
    if gpu:
        from gpu_specter.extract.gpu import (
            get_spots, ex2d_padded, ex2d_subbundle
        )
    else:
        from gpu_specter.extract.cpu import (
            get_spots, ex2d_padded
        )

    nwave = len(wave)
    ndiag = psf['PSF'].meta['HSIZEY']

    timer.split('init')

    #- Cache PSF spots for all wavelengths for spectra in this bundle
    if gpu:
        cp.cuda.nvtx.RangePush('get_spots')
    spots, corners, psfparams = get_spots(bspecmin, bundlesize, fullwave, psf)
    if gpu:
        cp.cuda.nvtx.RangePop()
    if psferr is None:
        psferr = psf['PSF'].meta['PSFERR']

    timer.split('spots/corners')

    #- Size of the individual spots
    spot_nx, spot_ny = spots.shape[2:4]

    #- Organize what sub-bundle patches to extract
    subbundles = list()
    nspectra_per_patch = bundlesize // nsubbundles
    for ispec in range(bspecmin, bspecmin+bundlesize, nspectra_per_patch):
        patches = list()
        for iwave in range(wavepad, wavepad+nwave, nwavestep):
            patch = Patch(ispec, iwave, bspecmin,
                          nspectra_per_patch, nwavestep, wavepad,
                          nwave, bundlesize, ndiag)
            patches.append(patch)
        subbundles.append(patches)

    timer.split('organize patches')

    #- place to keep extraction patch results before assembling in rank 0
    results = list()

    if gpu and batch_subbundle:
        for subbundle in subbundles[rank::size]:
            result = ex2d_subbundle(
                image, imageivar, subbundle, spots, corners,
                pixpad_frac, regularize, model, psferr
            )
            results += result
    else:
        patches = [patch for subbundle in subbundles for patch in subbundle]
        for patch in patches[rank::size]:
            try:
                result = ex2d_padded(image, imageivar, patch, spots, corners, 
                    pixpad_frac, regularize, model, psferr)
            except RuntimeError:
                if regularize == 0:
                    #- Add a smidgen of regularization and to try to power through...
                    regularize = 1e-4
                    log.warning(f'Error extracting patch ({patch.ispec}, {patch.iwave}) extraction, retrying with regularize={regularize}')
                    result = ex2d_padded(image, imageivar, patch, spots, corners, 
                        pixpad_frac, regularize, model, psferr)
                else:
                    raise
            patch.xyslice = result['xyslice']
            results.append( (patch, result) )

    timer.split('extracted patches')

    bundle = None
    if comm is not None:
        if gpu:
            # If we have gpu and an MPI comm for this bundle, transfer data
            # back to host before assembling the patches
            patches = []
            flux = []
            fluxivar = []
            resolution = []
            pixmask_fraction = []
            chi2pix = []
            modelimage = []
            for patch, result in results:
                patches.append(patch)
                flux.append(result['flux'])
                fluxivar.append(result['ivar'])
                resolution.append(result['Rdiags'])
                pixmask_fraction.append(result['pixmask_fraction'])
                chi2pix.append(result['chi2pix'])
                modelimage.append(cp.asnumpy(result['modelimage']))

            # transfer to host in chunks
            cp.cuda.nvtx.RangePush('copy bundle results to host')
            device_id = cp.cuda.runtime.getDevice()
            log.debug(f'Rank {rank}: Moving bundle {bspecmin} patches to host from device {device_id}')
            flux = cp.asnumpy(cp.array(flux, dtype=cp.float64))
            fluxivar = cp.asnumpy(cp.array(fluxivar, dtype=cp.float64))
            resolution = cp.asnumpy(cp.array(resolution, dtype=cp.float64))
            pixmask_fraction = cp.asnumpy(cp.array(pixmask_fraction, dtype=cp.float64))
            chi2pix = cp.asnumpy(cp.array(chi2pix, dtype=cp.float64))
            cp.cuda.nvtx.RangePop()

            # gather to root MPI rank
            patches = comm.gather(patches, root=0)
            flux = gather_ndarray(flux, comm, root=0)
            fluxivar = gather_ndarray(fluxivar, comm, root=0)
            resolution = gather_ndarray(resolution, comm, root=0)
            pixmask_fraction = gather_ndarray(pixmask_fraction, comm, root=0)
            chi2pix = gather_ndarray(chi2pix, comm, root=0)
            modelimage = comm.gather(modelimage, root=0)

            if rank == 0:
                # unpack patches
                patches = [patch for rankpatches in patches for patch in rankpatches]
                modelimage = [m for _ in modelimage for m in _]

                # repack everything
                rankresults = [
                    zip(patches,
                        map(lambda x: dict(
                                flux=x[0], ivar=x[1], Rdiags=x[2],
                                pixmask_fraction=x[3], chi2pix=x[4], modelimage=x[5]
                            ),
                            zip(
                                flux, fluxivar, resolution, 
                                pixmask_fraction, chi2pix, modelimage
                            )
                        )
                    )
                ]
        else:
            rankresults = comm.gather(results, root=0)
        if rank == 0:
            bundle = assemble_bundle_patches(rankresults)
    else:
        # this is fine for GPU w/out MPI comm
        rankresults = [results,]
        bundle = assemble_bundle_patches(rankresults)
        if gpu:
            bundle = tuple(cp.asnumpy(x) if isinstance(x, cp.ndarray) else x for x in bundle)

    return bundle

# @cupy.prof.TimeRangeDecorator("decompose_comm")
def decompose_comm(comm=None, gpu=False, ranks_per_bundle=None):
    """Decomposes MPI communicator into frame and bundle communicators depending on
    size of communicator, number of GPU devices (if requested), and (optionally) the
    specified number of ranks per bundle.

    Options:
        comm (None, mpi4py.MPI.Intracomm): mpi communicator
        gpu (bool): whether or not to assign ranks to GPUs
        ranks_per_bundle (None, int): number of mpi ranks per bundle

    Returns:
        bundle_comm (None, mpi4py.MPI.Intracomm): bundle-level mpi communicator
        frame_comm (None, mpi4py.MPI.Intracomm): frame-level mpi communicator
        frame_rank (int): the rank with the frame-level communicator
        frame_size (int): the size of root frame-level communicator
    """


    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size

    #- Determine MPI communication strategy based on number of GPU devices and MPI ranks
    if gpu:
        import cupy as cp
        #- Map MPI ranks to GPU devices
        device_count = cp.cuda.runtime.getDeviceCount()
        #- Ignore excess GPU devices
        device_count = min(size, device_count)
        ranks_per_device, remainder = divmod(size, device_count)
        assert remainder == 0, 'Number of MPI ranks must be divisible by number of GPUs'

        device_id = rank // ranks_per_device
        cp.cuda.Device(device_id).use()

        default_ranks_per_bundle = ranks_per_device
    else:
        default_ranks_per_bundle = size

    #- Map ranks to bundles
    if ranks_per_bundle is None:
        ranks_per_bundle = default_ranks_per_bundle

    frame_rank, bundle_rank = divmod(rank, ranks_per_bundle)
    frame_size = (size - 1) // ranks_per_bundle + 1

    if frame_size > 1:
        #- MPI communication needs to happen at frame level
        #- Bundles are processed in parallel
        if ranks_per_bundle > 1:
            #- Also need to communicate at bundle level
            #- Patches/subbundles are processed in parallel
            frame_comm = comm.Split(color=bundle_rank, key=frame_rank)
            bundle_comm = comm.Split(color=frame_rank, key=bundle_rank)
        else:
            #- Only do MPI communication at frame level
            #- Patches/subbundles are processed serially within each MPI rank
            frame_comm = comm
            bundle_comm = None
    else:
        #- MPI communication only happens at bundle level
        #- Bundles are processed serially
        frame_comm = None
        bundle_comm = comm

    return bundle_comm, frame_comm

# @cupy.prof.TimeRangeDecorator("extract_frame")
def extract_frame(img, psf, bundlesize, specmin, nspec, wavelength=None, nwavestep=50, nsubbundles=1,
    model=None, regularize=0, psferr=None, comm=None, gpu=None, loglevel=None, timing=None, 
    wavepad=10, pixpad_frac=0.8, wavepad_frac=0.2, batch_subbundle=True, ranks_per_bundle=None):
    """
    Extract 1D spectra from 2D image.

    Args:
        img: dictionary image object (see gpu_specter.io.read_img)
        psf: dictionary psf object (see gpu_specter.io.read_psf)
        bundlesize: fixed number of spectra per bundle (25 for DESI)
        specmin: index of first spectrum to extract
        nspec: number of spectra to extract

    Options:
        wavelength: wavelength range to extract, formatted as 'wmin,wmax,dw'
        nwavestep: number of wavelength bins per patch
        nsubbundles: number of spectra per patch
        model: indicate whether or not to compute the image model
        regularize: regularization parameter
        psferr: scale factor to use for psf in chi2
        comm: mpi communicator (no mpi: None)
        gpu: use GPU for extraction
        loglevel: log print level
        timing: dictionary to return timing splits
        wavepad: number of wavelength bins to pad extraction with (must be greater than
            spotsize)
        pixpad_frac: fraction of a PSF spotsize to pad in pixels when extracting
        wavepad_frac: fraction of a PSF spotsize to pad in wavelengths when extracting
        batch_subbundle: perform extraction in subbundle batch of patches (GPU-only)
        ranks_per_bundle: number of mpi ranks per bundle comm

    Returns:
        frame: dictionary frame object (see gpu_specter.io.write_frame)
    """

    timer = Timer()
    time_start = time.time()

    log = get_logger(loglevel)

    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size

    #- Disable batch subbundle for CPU extraction
    if not gpu:
        batch_subbundle = False

    #- Batch subbundle extraction constrains the number of MPI ranks per bundle
    if batch_subbundle:
        #- Default to one MPI rank per bundle
        if ranks_per_bundle is None:
            ranks_per_bundle = 1
        assert ranks_per_bundle <= nsubbundles, 'ranks_per_bundle should be <= nsubbundles'
        assert nsubbundles % ranks_per_bundle == 0, 'ranks_per_bundle should evenly divide nsubbundles'

    bundle_comm, frame_comm = decompose_comm(comm, gpu, ranks_per_bundle)

    bundle_rank = 0 if bundle_comm is None else bundle_comm.rank
    bundle_size = 1 if bundle_comm is None else bundle_comm.size
    frame_rank = 0 if frame_comm is None else frame_comm.rank
    frame_size = 1 if frame_comm is None else frame_comm.size

    if rank == 0:
        log.info(f'Using GPU: {gpu}')
        log.info(f'Using batch subbundle extraction: {batch_subbundle}')
        log.info(f'Size of frame MPI comm: {frame_size}')
        log.info(f'Size of bundle MPI comm: {bundle_size}')

    #- MPI rank to bundle/frame comm mapping
    log.debug(f'{rank=} {frame_rank=}/{frame_size=} {bundle_rank=}/{bundle_size=}')

    timer.split('init-mpi-comm')
    time_init_mpi_comm = time.time()

    imgpixels = imgivar = None
    if rank == 0:
        imgpixels = img['image']
        imgivar = img['ivar']

    #- If using MPI, broadcast image, ivar, and psf to all ranks
    if comm is not None:
        # cp.cuda.nvtx.RangePush('mpi bcast')
        if rank == 0:
            log.info('Broadcasting inputs to other MPI ranks')

        if gpu:
            empty = cp.empty
        else:
            empty = np.empty

        # cp.cuda.nvtx.RangePush('shape')
        if rank == 0:
            shape = imgpixels.shape
        else:
            shape = None
        shape = comm.bcast(shape, root=0)
        if rank > 0:
            imgpixels = empty(shape, dtype='f8')
            imgivar = empty(shape, dtype='f8')
        # cp.cuda.nvtx.RangePop() # shape

        # cp.cuda.nvtx.RangePush('imgpixels')
        comm.Bcast(imgpixels, root=0)
        # imgpixels = comm.bcast(imgpixels, root=0)
        # cp.cuda.nvtx.RangePop() # imgpixels

        # cp.cuda.nvtx.RangePush('imgivar')
        comm.Bcast(imgivar, root=0)
        # imgivar = comm.bcast(imgivar, root=0)
        # cp.cuda.nvtx.RangePop() # imgivar

        # cp.cuda.nvtx.RangePush('psf')
        psf = comm.bcast(psf, root=0)
        # cp.cuda.nvtx.RangePop() # psf
        # cp.cuda.nvtx.RangePop() # mpi bcast

    timer.split('mpi-bcast-raw')
    time_mpi_bcast_raw = time.time()

    #- If using GPU, move image and ivar to device
    #- TODO: is there a way for ranks to share a pointer to device memory?
    if gpu:
        cp.cuda.nvtx.RangePush('copy imgpixels, imgivar to device')
        device_id = cp.cuda.runtime.getDevice()
        log.debug(f'Rank {rank}: Moving image data to device {device_id}')
        imgpixels = cp.asarray(imgpixels)
        imgivar = cp.asarray(imgivar)
        cp.cuda.nvtx.RangePop()

        timer.split('host-to-device-raw')
    time_host_to_device_raw = time.time()

    if isinstance(wavelength, np.ndarray):
        wave = wavelength
        wmin, wmax = wave[0], wave[-1]
        dw = np.gradient(wave)[0]
    else:
        if isinstance(wavelength, str):
            wmin, wmax, dw = map(float, wavelength.split(','))
        elif isinstance(wavelength, tuple):
            wmin, wmax, dw = wavelength
        else:
            wmin, wmax = psf['PSF'].meta['WAVEMIN'], psf['PSF'].meta['WAVEMAX']
            dw = 0.8
        wave = np.arange(wmin, wmax + 0.5*dw, dw)

    #- Wavelength range that we want to extract
    if rank == 0:
        log.info(f'Extracting wavelengths {wmin},{wmax},{dw}')

    #- Pad that with buffer wavelengths to extract and discard, including an
    #- extra args.nwavestep bins to allow coverage for a final partial bin

    #- TODO: calculate initial wavepad from psf spotsize instead of using parameter
    wavepad += int(wavepad*wavepad_frac)

    if rank == 0:
        log.info(f'Padding patches with {wavepad} wave bins on both ends')

    wavelo = np.arange(wavepad)*dw
    wavelo -= (np.max(wavelo)+dw)
    wavelo += wmin
    wavehi = wave[-1] + (1.0+np.arange(wavepad+nwavestep))*dw

    fullwave = np.concatenate((wavelo, wave, wavehi))
    assert np.allclose(np.diff(fullwave), dw)
    
    bspecmins = list(range(specmin, specmin+nspec, bundlesize))
    bundles = list()
    for bspecmin in bspecmins[frame_rank::frame_size]:
        # log.info(f'Rank {rank}: Extracting spectra [{bspecmin}:{bspecmin+bundlesize}]')
        # sys.stdout.flush()
        if gpu:
            cp.cuda.nvtx.RangePush('extract_bundle')
        bundle = extract_bundle(
            imgpixels, imgivar, psf,
            wave, fullwave, bspecmin,
            bundlesize=bundlesize, nsubbundles=nsubbundles,
            batch_subbundle=batch_subbundle,
            nwavestep=nwavestep, wavepad=wavepad,
            comm=bundle_comm,
            gpu=gpu,
            loglevel=loglevel,
            model=model,
            regularize=regularize,
            psferr=psferr,
            pixpad_frac=pixpad_frac,
        )
        if gpu:
            cp.cuda.nvtx.RangePop()
        bundles.append((bspecmin, bundle))

        #- for good measure, have other ranks wait for rank 0
        if bundle_comm is not None:
            bundle_comm.barrier()

    timer.split('extracted-bundles')
    time_extracted_bundles = time.time()

    # cp.cuda.nvtx.RangePush('mpi gather')
    if frame_comm is not None:
        # gather results from multiple mpi groups
        if bundle_comm is None or bundle_comm.rank == 0:
            bspecmins, bundles = zip(*bundles)
            flux, ivar, resolution, pixmask_fraction, chi2pix, modelimage, xyslice = zip(*bundles)
            bspecmins = frame_comm.gather(bspecmins, root=0)
            xyslice = frame_comm.gather(xyslice, root=0)
            flux = gather_ndarray(flux, frame_comm)
            ivar = gather_ndarray(ivar, frame_comm)
            resolution = gather_ndarray(resolution, frame_comm)
            pixmask_fraction = gather_ndarray(pixmask_fraction, frame_comm)
            chi2pix = gather_ndarray(chi2pix, frame_comm)
            modelimage = frame_comm.gather(modelimage, root=0)
            if rank == 0:
                bspecmin = [bspecmin for rankbspecmins in bspecmins for bspecmin in rankbspecmins]
                modelimage = [m for _ in modelimage for m in _]
                mxy = [xy for rankxyslice in xyslice for xy in rankxyslice]
                rankbundles = [list(zip(bspecmin, zip(flux, ivar, resolution, pixmask_fraction, chi2pix, modelimage, mxy))), ]
    else:
        # no mpi or single group with all ranks
        rankbundles = [bundles,]
    # cp.cuda.nvtx.RangePop() # mpi gather

    timer.split('staged-bundles')
    time_staged_bundles = time.time()

    #- Finalize and write output
    frame = None
    # cp.cuda.nvtx.RangePush('finalize output')
    if rank == 0:

        #- flatten list of lists into single list
        allbundles = list()
        for rb in rankbundles:
            allbundles.extend(rb)

        allbundles.sort(key=lambda x: x[0])

        specflux = np.vstack([b[1][0] for b in allbundles])
        specivar = np.vstack([b[1][1] for b in allbundles])
        Rdiags = np.vstack([b[1][2] for b in allbundles])
        pixmask_fraction = np.vstack([b[1][3] for b in allbundles])
        chi2pix = np.vstack([b[1][4] for b in allbundles])

        if model:
            modelimage = np.zeros(imgpixels.shape)
            for b in allbundles:
                bundleimage = b[1][5]
                xyslice = b[1][6]
                modelimage[xyslice] += bundleimage
        else:
            modelimage = None

        timer.split(f'merged-bundles')
        time_merged_bundles = time.time()

        #- Convert flux to photons/A instead of photons/bin
        dwave = np.gradient(wave)
        specflux /= dwave
        specivar *= dwave**2

        #- TODO: specmask and chi2pix
        # mask = np.zeros(flux.shape, dtype=np.uint32)
        # mask[results['pixmask_fraction']>0.5] |= specmask.SOMEBADPIX
        # mask[results['pixmask_fraction']==1.0] |= specmask.ALLBADPIX
        # mask[chi2pix>100.0] |= specmask.BAD2DFIT
        specmask = (specivar == 0).astype(int)

        frame = dict(
            specflux = specflux,
            specivar = specivar,
            specmask = specmask,
            wave = wave,
            Rdiags = Rdiags,
            pixmask_fraction = pixmask_fraction,
            chi2pix = chi2pix,
            modelimage = modelimage,
        )

        timer.split(f'assembled-frame')
        time_assembled_frame = time.time()
        timer.log_splits(log)
    else:
        time_merged_bundles = time.time()
        time_assembled_frame = time.time()
    # cp.cuda.nvtx.RangePop() # finalize output

    if isinstance(timing, dict):
        timing['init-mpi-comm'] = time_init_mpi_comm
        timing['init-mpi-bcast'] = time_mpi_bcast_raw
        timing['host-to-device'] = time_host_to_device_raw
        timing['extracted-bundles'] = time_extracted_bundles
        timing['staged-bundles'] = time_staged_bundles
        timing['merged-bundles'] = time_merged_bundles
        timing['assembled-frame'] = time_assembled_frame

    return frame
