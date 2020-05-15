"""
Core scaffolding for divide and conquer extraction algorithm
"""

import sys

import numpy as np

from gpu_specter.util import get_logger


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

    #- Allocate output arrays to fill
    if rank == 0:
        specflux = np.zeros((bundlesize, nwave))
        specivar = np.zeros((bundlesize, nwave))
        #- TODO: refactor ndiag calculation to PSF object?
        ndiag = psf['PSF'].meta['HSIZEY']
        Rdiags = np.zeros((bundlesize, 2*ndiag+1, nwave))

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
            patches.append((ispec, iwave))

    #- place to keep extraction patch results before assembling in rank 0
    results = list()

    for ispec, iwave in patches[rank::size]:
        log.debug(f'rank={rank}, ispec={ispec}, iwave={iwave}')

        #- Always extract the same patch size (more efficient for GPU
        #- memory transfer) then decide post-facto whether to keep it all

        result = ex2d_padded(image, imageivar,
                             ispec-bspecmin, nspec,
                             iwave, nwavestep,
                             spots, corners,
                             wavepad=wavepad,
                             bundlesize=bundlesize)
        results.append( (ispec, iwave, result) )

    if comm is not None:
        rankresults = comm.gather(results, root=0)
    else:
        rankresults = [results,]

    if rank == 0:
        #- flatten list of lists into single list
        allresults = list()
        for rr in rankresults:
            allresults.extend(rr)

        #- Now put these into the final arrays
        for ispec, iwave, result in allresults:
            fx = result['flux']
            fxivar = result['ivar']
            xRdiags = result['Rdiags']

            assert fx.shape == (nspec, nwavestep)

            #- put the extracted patch into the output arrays
            specslice = np.s_[ispec-bspecmin:ispec-bspecmin+nspec]
            waveslice = np.s_[iwave-wavepad:iwave-wavepad+nwavestep]

            nwavekeep = min(nwavestep, nwave - (iwave-wavepad))
            
            keepslice = np.s_[0:nwavekeep]

            specflux[specslice, waveslice] = fx[:, keepslice]
            specivar[specslice, waveslice] = fxivar[:, keepslice]
            Rdiags[specslice, :, waveslice] = xRdiags[:, :, keepslice]

    #- for good measure, have other ranks wait for rank 0
    if comm is not None:
        comm.barrier()

    bundle_results = None

    if rank == 0:
        bundle_results = dict(
            flux=specflux,
            ivar=specivar,
            R=Rdiags,
        )

    return bundle_results


def extract_frame(img, psf, bundlesize, specmin, nspec, 
    wmin, wmax, dw, nwavestep, nsubbundles, comm, rank, size, gpu=None, loglevel=None):

    log = get_logger(loglevel)

    if rank == 0:
        log.info(f'Extracting wavelengths {wmin},{wmax},{dw}')
    
    #- Wavelength range that we want to extract
    wave = np.arange(wmin, wmax + 0.5*dw, dw)
    nwave = len(wave)

    #- TODO: calculate this instead of hardcoding it
    wavepad = 10
    
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
    if rank == 0:
        specflux = np.zeros((nspec, nwave))
        specivar = np.zeros((nspec, nwave))
        Rdiags = np.zeros((nspec, 2*psf['PSF'].meta['HSIZEY']+1, nwave))

    #- Work bundle by bundle
    for bspecmin in range(specmin, specmin+nspec, bundlesize):
        if rank == 0:
            log.info(f'Extracting spectra [{bspecmin}:{bspecmin+bundlesize}]')
            sys.stdout.flush()

        bundle_results = extract_bundle(
            img['image'], img['ivar'], psf, 
            bspecmin, bundlesize, nsubbundles, 
            wavepad, nwavestep, 
            wave, fullwave,
            comm, rank, size, gpu)

        #- for good measure, have other ranks wait for rank 0
        if comm is not None:
            comm.barrier()

        if rank == 0:
            specslice = np.s_[bspecmin:bspecmin+bundlesize]
            specflux[specslice, :] = bundle_results['flux']
            specivar[specslice, :] = bundle_results['ivar']
            Rdiags[specslice, :, :] = bundle_results['R']

    #- Finalize and write output
    extract = None
    if rank == 0:

        #- Convert flux to photons/A instead of photons/bin
        dwave = np.gradient(wave)
        specflux /= dwave
        specivar *= dwave**2

         #- TODO: specmask and chi2pix
        specmask = (specivar > 0).astype(np.int)
        chi2pix = np.ones(specflux.shape)

        extract = dict(
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

    return extract

