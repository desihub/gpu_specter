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
        if patchmodel is None or ~np.all(np.isfinite(patchmodel)):
            continue
        ymin = patch.xyslice[0].start - ystart
        xmin = patch.xyslice[1].start - xstart
        patchny, patchnx = patchmodel.shape
        modelimage[ymin:ymin+patchny, xmin:xmin+patchnx] += patchmodel

    return specflux, specivar, Rdiags, pixmask_fraction, chi2pix, modelimage, xyslice


def extract_bundle(image, imageivar, psf, wave, fullwave, bspecmin, bundlesize=25, nsubbundles=1,
    nwavestep=50, wavepad=10, comm=None, gpu=None, loglevel=None, model=None, regularize=0,
    psferr=None):
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
        nwavestep: number of wavelength bins per patch
        wavepad: number of wavelengths bins to add on each end of patch for extraction
        comm: mpi communicator (no mpi: None)
        rank: integer process identifier (no mpi: 0)
        size: number of mpi processes (no mpi: 1)
        gpu: use GPU for extraction (not yet implemented)
        loglevel: log print level

    Returns:
        bundle: (flux, ivar, R) tuple

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
        from gpu_specter.extract.gpu import \
                get_spots, ex2d_padded
    else:
        from gpu_specter.extract.cpu import \
                get_spots, ex2d_padded

    nwave = len(wave)
    ndiag = psf['PSF'].meta['HSIZEY']

    timer.split('init')

    #- Cache PSF spots for all wavelengths for spectra in this bundle
    if gpu:
        cp.cuda.nvtx.RangePush('get_spots')
    spots, corners = get_spots(bspecmin, bundlesize, fullwave, psf)
    if gpu:
        cp.cuda.nvtx.RangePop()
    if psferr is None:
        psferr = psf['PSF'].meta['PSFERR']

    timer.split('spots/corners')

    #- Size of the individual spots
    spot_nx, spot_ny = spots.shape[2:4]

    #- Organize what sub-bundle patches to extract
    patches = list()
    nspectra_per_patch = bundlesize // nsubbundles
    for ispec in range(bspecmin, bspecmin+bundlesize, nspectra_per_patch):
        for iwave in range(wavepad, wavepad+nwave, nwavestep):
            patch = Patch(ispec, iwave, bspecmin,
                          nspectra_per_patch, nwavestep, wavepad,
                          nwave, bundlesize, ndiag)
            patches.append(patch)

    if rank == 0:
        log.debug(f'Dividing {len(patches)} patches between {size} ranks')

    timer.split('organize patches')

    #- place to keep extraction patch results before assembling in rank 0
    results = list()
    for patch in patches[rank::size]:

        log.debug(f'rank={rank}, ispec={patch.ispec}, iwave={patch.iwave}')

        #- Always extract the same patch size (more efficient for GPU
        #- memory transfer) then decide post-facto whether to keep it all

        if gpu:
            cp.cuda.nvtx.RangePush('ex2d_padded')

        result = ex2d_padded(image, imageivar,
                             patch.ispec-bspecmin, patch.nspectra_per_patch,
                             patch.iwave, patch.nwavestep,
                             spots, corners, psferr,
                             wavepad=patch.wavepad,
                             bundlesize=bundlesize,
                             model=model,
                             regularize=regularize,
                             )
        patch.xyslice = result['xyslice']
        if gpu:
            cp.cuda.nvtx.RangePop()

        results.append( (patch, result) )

    timer.split('extracted patches')

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
            for patch, results in results:
                patches.append(patch)
                flux.append(results['flux'])
                fluxivar.append(results['ivar'])
                resolution.append(results['Rdiags'])
                pixmask_fraction.append(results['pixmask_fraction'])
                chi2pix.append(results['chi2pix'])
                modelimage.append(cp.asnumpy(results['modelimage']))

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
                                pixmask_fraction=x[3], chi2pix=x[4], modelimage=x[3]
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
    else:
        # this is fine for GPU w/out MPI comm
        rankresults = [results,]

    timer.split('gathered patches')

    bundle = None
    if rank == 0:
        if gpu:
            cp.cuda.nvtx.RangePush('assemble patches on device')
            device_id = cp.cuda.runtime.getDevice()
            log.debug(f'Rank {rank}: Assembling bundle {bspecmin} patches on device {device_id}')
        bundle = assemble_bundle_patches(rankresults)
        if gpu:
            cp.cuda.nvtx.RangePop()
            if comm is None:
                cp.cuda.nvtx.RangePush('copy bundle results to host')
                device_id = cp.cuda.runtime.getDevice()
                log.debug(f'Rank {rank}: Moving bundle {bspecmin} to host from device {device_id}')
                specflux, specivar, Rdiags, pixmask_fraction, chi2pix, modelimage, xyslice = bundle
                bundle = (
                    cp.asnumpy(specflux),
                    cp.asnumpy(specivar),
                    cp.asnumpy(Rdiags),
                    cp.asnumpy(pixmask_fraction),
                    cp.asnumpy(chi2pix),
                    cp.asnumpy(modelimage),
                    xyslice
                )
                # bundle = tuple(cp.asnumpy(x) for x in bundle)
                cp.cuda.nvtx.RangePop()
        timer.split('assembled patches')
        # timer.log_splits(log)
    return bundle


def extract_frame(img, psf, bundlesize, specmin, nspec, wavelength=None, nwavestep=50, nsubbundles=1,
    model=None, regularize=0, psferr=None, comm=None, rank=0, size=1, gpu=None, loglevel=None, timing=None):
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
        comm: mpi communicator (no mpi: None)
        rank: integer process identifier (no mpi: 0)
        size: number of mpi processes (no mpi: 1)
        gpu: use GPU for extraction (not yet implemented)
        loglevel: log print level

    Returns:
        frame: dictionary frame object (see gpu_specter.io.write_frame)
    """

    timer = Timer()
    time_start = time.time()

    log = get_logger(loglevel)

    #- Determine MPI communication strategy based on number of gpu devices and MPI ranks
    if gpu:
        import cupy as cp
        #- TODO: specify number of gpus to use?
        device_count = cp.cuda.runtime.getDeviceCount()
        device_count = min(size, device_count)
        assert size % device_count == 0, 'Number of MPI ranks must be divisible by number of GPUs'
        device_id = rank % device_count
        cp.cuda.Device(device_id).use()

        #- Divide mpi ranks evenly among gpus
        device_size = size // device_count
        bundle_rank = rank // device_count

        if device_count > 1:
            #- Multi gpu, MPI communication needs to happen at frame level
            frame_comm = comm.Split(color=bundle_rank, key=device_id)
            if device_size > 1:
                #- If multiple ranks per gpu, also need to communicate at bundle level
                bundle_comm = comm.Split(color=device_id, key=bundle_rank)
            else:
                #- If only one rank per gpu, don't need bundle level communication
                bundle_comm = None
        else:
            #- Single gpu, only do MPI communication at bundle level
            frame_comm = None
            bundle_comm = comm
    else:
        #- No gpu, do MPI communication at bundle level
        frame_comm = None
        bundle_comm = comm

    timer.split('init-mpi-comm')
    time_init_mpi_comm = time.time()

    imgpixels = imgivar = None
    if rank == 0:
        imgpixels = img['image']
        imgivar = img['ivar']

    #- If using MPI, broadcast image, ivar, and psf to all ranks
    if comm is not None:
        if rank == 0:
            log.info('Broadcasting inputs to other MPI ranks')
        imgpixels = comm.bcast(imgpixels, root=0)
        imgivar = comm.bcast(imgivar, root=0)
        psf = comm.bcast(psf, root=0)

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

    if wavelength is not None:
        wmin, wmax, dw = map(float, wavelength.split(','))
    else:
        wmin, wmax = psf['PSF'].meta['WAVEMIN'], psf['PSF'].meta['WAVEMAX']
        dw = 0.8

    if rank == 0:
        log.info(f'Extracting wavelengths {wmin},{wmax},{dw}')
    
    #- TODO: calculate this instead of hardcoding it
    wavepad = 10

    #- Wavelength range that we want to extract
    wave = np.arange(wmin, wmax + 0.5*dw, dw)
    nwave = len(wave)
    
    #- Pad that with buffer wavelengths to extract and discard, including an
    #- extra args.nwavestep bins to allow coverage for a final partial bin
    wavelo = np.arange(wavepad)*dw
    wavelo -= (np.max(wavelo)+dw)
    wavelo += wmin
    wavehi = wave[-1] + (1.0+np.arange(wavepad+nwavestep))*dw
    
    fullwave = np.concatenate((wavelo, wave, wavehi))
    assert np.allclose(np.diff(fullwave), dw)
    
    #- TODO: barycentric wavelength corrections

    #- Work bundle by bundle
    if frame_comm is None:
        bundle_start = 0
        bundle_step = 1
    else:
        bundle_start = device_id
        bundle_step = device_count
    bspecmins = list(range(specmin, specmin+nspec, bundlesize))
    bundles = list()
    for bspecmin in bspecmins[bundle_start::bundle_step]:
        # log.info(f'Rank {rank}: Extracting spectra [{bspecmin}:{bspecmin+bundlesize}]')
        # sys.stdout.flush()
        if gpu:
            cp.cuda.nvtx.RangePush('extract_bundle')
        bundle = extract_bundle(
            imgpixels, imgivar, psf,
            wave, fullwave, bspecmin,
            bundlesize=bundlesize, nsubbundles=nsubbundles,
            nwavestep=nwavestep, wavepad=wavepad,
            comm=bundle_comm,
            gpu=gpu,
            loglevel=loglevel,
            model=model,
            regularize=regularize,
            psferr=psferr,
        )
        if gpu:
            cp.cuda.nvtx.RangePop()
        bundles.append((bspecmin, bundle))

        #- for good measure, have other ranks wait for rank 0
        if bundle_comm is not None:
            bundle_comm.barrier()

    timer.split('extracted-bundles')
    time_extracted_bundles = time.time()

    if frame_comm is not None:
        # gather results from multiple mpi groups
        if bundle_rank == 0:
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

    timer.split('staged-bundles')
    time_staged_bundles = time.time()

    #- Finalize and write output
    frame = None
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
        specmask = (specivar == 0).astype(np.int)

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

    if isinstance(timing, dict):
        timing['init-mpi-comm'] = time_init_mpi_comm
        timing['init-mpi-bcast'] = time_mpi_bcast_raw
        timing['host-to-device'] = time_host_to_device_raw
        timing['extracted-bundles'] = time_extracted_bundles
        timing['staged-bundles'] = time_staged_bundles
        timing['merged-bundles'] = time_merged_bundles
        timing['assembled-frame'] = time_assembled_frame

    return frame
