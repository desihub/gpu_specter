"""
Core scaffolding for divide and conquer extraction algorithm
"""

import sys

import numpy as np

from gpu_specter.util import get_logger

class Patch(object):
    def __init__(self, ispec, iwave, bspecmin, nspec, nwavestep, wavepad, nwave):
        self.ispec = ispec
        self.iwave = iwave

        self.nspec = nspec
        self.nwavestep = nwavestep

        #- padding to apply to patch
        self.wavepad = wavepad

        #- where this patch should go
        #- note: spec indexing is relative to subbundle
        self.bspecmin = bspecmin
        self.specslice = np.s_[ispec-bspecmin:ispec-bspecmin+nspec]
        self.waveslice = np.s_[iwave-wavepad:iwave-wavepad+nwavestep]

        #- how much of the patch to keep
        nwavekeep = min(nwavestep, nwave - (iwave-wavepad))
        self.keepslice = np.s_[0:nwavekeep]


def assemble_bundle_patches(rankresults, bundlesize, nwave, ndiag):

    #- flatten list of lists into single list
    allresults = list()
    for rr in rankresults:
        allresults.extend(rr)

    #- Allocate output ar`rays to fill
    specflux = np.zeros((bundlesize, nwave))
    specivar = np.zeros((bundlesize, nwave))
    Rdiags = np.zeros((bundlesize, 2*ndiag+1, nwave))

    #- Now put these into the final arrays
    for patch, result in allresults:
        fx = result['flux']
        fxivar = result['ivar']
        xRdiags = result['Rdiags']

        #- put the extracted patch into the output arrays
        specflux[patch.specslice, patch.waveslice] = fx[:, patch.keepslice]
        specivar[patch.specslice, patch.waveslice] = fxivar[:, patch.keepslice]
        Rdiags[patch.specslice, :, patch.waveslice] = xRdiags[:, :, patch.keepslice]

    return specflux, specivar, Rdiags


def extract_bundle(image, imageivar, psf, bspecmin, bundlesize, nsubbundles,
    wavepad, nwavestep, wave, fullwave, comm, rank, size, gpu=None, loglevel=None):

    log = get_logger(loglevel)

    #- Extracting on CPU or GPU?
    if gpu:
        from gpu_specter.extract.gpu import \
                get_spots, projection_matrix, ex2d_padded
    else:
        from gpu_specter.extract.cpu import \
                get_spots, projection_matrix, ex2d_padded

    nwave = len(wave)
    ndiag = psf['PSF'].meta['HSIZEY']

    #- Cache PSF spots for all wavelengths for spectra in this bundle
    spots = corners = None
    if rank == 0:
        spots, corners = get_spots(bspecmin, bundlesize, fullwave, psf)
    
    #- TODO: it might be faster for all ranks to calculate instead of bcast
    if comm is not None:
        spots = comm.bcast(spots, root=0)
        corners = comm.bcast(corners, root=0)

    #- Size of the individual spots
    spot_nx, spot_ny = spots.shape[2:4]

    #- Organize what sub-bundle patches to extract
    patches = list()
    nspec = bundlesize // nsubbundles
    for ispec in range(bspecmin, bspecmin+bundlesize, nspec):
        for iwave in range(wavepad, wavepad+nwave, nwavestep):
            patch = Patch(ispec, iwave, bspecmin, nspec, nwavestep, wavepad, nwave)
            patches.append(patch)

    #- place to keep extraction patch results before assembling in rank 0
    results = list()
    for patch in patches[rank::size]:

        log.debug(f'rank={rank}, ispec={patch.ispec}, iwave={patch.iwave}')

        #- Always extract the same patch size (more efficient for GPU
        #- memory transfer) then decide post-facto whether to keep it all

        result = ex2d_padded(image, imageivar,
                             patch.ispec-bspecmin, patch.nspec,
                             patch.iwave, patch.nwavestep,
                             spots, corners,
                             wavepad=patch.wavepad,
                             bundlesize=bundlesize)
        results.append( (patch, result) )

    if comm is not None:
        rankresults = comm.gather(results, root=0)
    else:
        rankresults = [results,]

    bundle = None
    if rank == 0:
        bundle = assemble_bundle_patches(rankresults, bundlesize, nwave, ndiag)

    return bundle


def extract_frame(img, psf, bundlesize, specmin, nspec, wavelength, nwavestep, nsubbundles=1, 
    comm=None, rank=0, size=1, gpu=None, loglevel=None):

    log = get_logger(loglevel)

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

    #- Allocate output arrays to fill
    #- TODO: with multiprocessing, use shared memory?
    ndiag = psf['PSF'].meta['HSIZEY']
    if rank == 0:
        specflux = np.zeros((nspec, nwave))
        specivar = np.zeros((nspec, nwave))
        Rdiags = np.zeros((nspec, 2*ndiag+1, nwave))

    #- Work bundle by bundle
    for bspecmin in range(specmin, specmin+nspec, bundlesize):
        if rank == 0:
            log.info(f'Extracting spectra [{bspecmin}:{bspecmin+bundlesize}]')
            sys.stdout.flush()

        bundle = extract_bundle(
            img['image'], img['ivar'], psf, 
            bspecmin, bundlesize, nsubbundles, 
            wavepad, nwavestep, 
            wave, fullwave,
            comm, rank, size, gpu
        )

        #- for good measure, have other ranks wait for rank 0
        if comm is not None:
            comm.barrier()

        if rank == 0:
            #- TODO: use vstack instead of preallocation and slicing?
            specslice = np.s_[bspecmin:bspecmin+bundlesize]
            flux, ivar, R = bundle
            specflux[specslice, :] = flux
            specivar[specslice, :] = ivar
            Rdiags[specslice, :, :] = R

    #- Finalize and write output
    frame = None
    if rank == 0:

        #- Convert flux to photons/A instead of photons/bin
        dwave = np.gradient(wave)
        specflux /= dwave
        specivar *= dwave**2

         #- TODO: specmask and chi2pix
        specmask = (specivar > 0).astype(np.int)
        chi2pix = np.ones(specflux.shape)

        frame = dict(
            imagehdr = img['imagehdr'],
            specflux = specflux,
            specivar = specivar,
            specmask = specmask,
            wave = wave,
            Rdiags = Rdiags,
            fibermap = img['fibermap'],
            fibermaphdr = img['fibermaphdr'],
            chi2pix = np.ones(specflux.shape),
        )

    return frame

