#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 18:47:08 2020

@author: stephey
"""

from __future__ import absolute_import, division, print_function

import sys
import traceback
import os
import re
import os.path
import time
import argparse
import numpy as np

#ideally we could do this whole thing without touching specter
#at least we aren't using it in the extraction part anyway
import specter #need this for io stuff
from astropy.io import fits
from astropy.table import Table

#we should get rid of desispec too
#we'll worry about that later
from desispec import io
from desiutil.log import get_logger
from desispec.frame import Frame
from desispec.maskbits import specmask

import desispec.scripts.mergebundles as mergebundles
from desispec.specscore import compute_and_append_frame_scores
from desispec.heliocentric import heliocentric_velocity_multiplicative_corr

#import our hackathon stuff
#from cpu_extract import ex2d #for cpu
from gpu_extract import ex2d #for gpu
#from both_extract import ex2d #for debugging

import cupy as cp

def parse(options=None):
    parser = argparse.ArgumentParser(description="Extract spectra from pre-processed raw data.")
    #parser.add_argument("-i", "--input", type=str, required=True,
    #                    help="input image") #hardcode for hackathon
    #parser.add_argument("-f", "--fibermap", type=str, required=False,
    #                    help="input fibermap file") #hardcode for hackathon
    parser.add_argument("-p", "--psf", type=str, required=False,
                        help="input psf file") #hardcode for hackathon
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output extracted spectra file")
    parser.add_argument("-m", "--model", type=str, required=False,
                        help="output 2D pixel model file")
    parser.add_argument("-w", "--wavelength", type=str, required=False,
                        help="wavemin,wavemax,dw")
    parser.add_argument("-s", "--specmin", type=int, required=False, default=0,
                        help="first spectrum to extract")
    parser.add_argument("-n", "--nspec", type=int, required=False, default=500,
                        help="number of spectra to extract")
    parser.add_argument("-r", "--regularize", type=float, required=False, default=0.0,
                        help="regularization amount (default %(default)s)")
    parser.add_argument("--bundlesize", type=int, required=False, default=25,
                        help="number of spectra per bundle")
    parser.add_argument("--nsubbundles", type=int, required=False, default=6,
                        help="number of extraction sub-bundles")
    parser.add_argument("--nwavestep", type=int, required=False, default=50,
                        help="number of wavelength steps per divide-and-conquer extraction step")
    parser.add_argument("-v", "--verbose", action="store_true", help="print more stuff")
    #parser.add_argument("--mpi", action="store_true", help="Use MPI for parallelism") #hardcode for hackathon
    parser.add_argument("--decorrelate-fibers", action="store_true", help="Not recommended")
    parser.add_argument("--no-scores", action="store_true", help="Do not compute scores")
    parser.add_argument("--psferr", type=float, default=None, required=False,
                        help="fractional PSF model error used to compute chi2 and mask pixels (default = value saved in psf file)")
    # parser.add_argument("--fibermap-index", type=int, default=None, required=False,
    #                     help="start at this index in the fibermap table instead of using the spectro id from the camera")
    #parser.add_argument("--heliocentric-correction", action="store_true", help="apply heliocentric correction to wavelength")
    parser.add_argument("-t", "--test", action="store_true", help="flag to compare results to reference results")


    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

#- Util function to trim path to something that fits in a fits file (!)
def _trim(filepath, maxchar=40):
    if len(filepath) > maxchar:
        return '...{}'.format(filepath[-maxchar:])


def main(args, timing=None):

    mark_start = time.time()

    log = get_logger()

    #also we need an input file? or can we cheat the same way stephen does in his notebook?

    #try doing all our gpu preprocessing
    #lets try it
    psf_file = 'psf.fits'
    psfdata = Table.read(psf_file)

    #hack!
    wavelengths = np.arange(psfdata['WAVEMIN'][0]+100, psfdata['WAVEMAX'][0]-100, 0.8)
 
    print("len(wavelengths)", len(wavelengths))

    #right now cache_spots happens once per bundle to make bookkeeping less of a nighmare
    #and also not to blow memory

    #these parameters are interpreted as the *global* spec range,
    # to be divided among processes.
    specmin = 0 #hardcode for hackathon
    #use the nspec we get out evalcoeffs, not sure if it's the right one

    #just read an existing image file? try it out
    input_file = '/global/cscratch1/sd/stephey/desitest/data/pix-r0-00003578.fits'
    img = io.read_image(input_file)
    
    mark_read_input = time.time()

    # get spectral range

    #let's skip providing a fibermap
    #if args.fibermap is not None:
    #    fibermap = io.read_fibermap(args.fibermap)
    #else:
    #    try:
    #        fibermap = io.read_fibermap(args.input)
    #    except (AttributeError, IOError, KeyError):
    #        fibermap = None

    #if fibermap is not None:
    #    fibermap = fibermap[specmin:specmin+nspec]
    #    if nspec > len(fibermap):
    #        log.warning("nspec {} > len(fibermap) {}; reducing nspec to {}".format(
    #            nspec, len(fibermap), len(fibermap)))
    #        nspec = len(fibermap)
    #    fibers = fibermap['FIBER']
    #else:

    nspec = args.nspec
    fibers = np.arange(specmin, specmin+nspec)

    specmax = specmin + nspec

    #if args.wavelength is not None:
    #    wstart, wstop, dw = [float(tmp) for tmp in args.wavelength.split(',')]
    #else:
    #    wstart = np.ceil(psfdata['wmin_all'])
    #    wstop = np.floor(psfdata['wmax_all'])
    #    dw = 0.7

    wavemin = psfdata['WAVEMIN'][0]
    wavemax = psfdata['WAVEMAX'][0]

    wstart = wavemin
    wstop = wavemax
    dw = 0.7

    #if args.heliocentric_correction :
    #    heliocentric_correction_factor = heliocentric_correction_multiplicative_factor(img.meta)        
    #    wstart /= heliocentric_correction_factor
    #    wstop  /= heliocentric_correction_factor
    #    dw     /= heliocentric_correction_factor
    #else :
    #    heliocentric_correction_factor = 1.

    wave = np.arange(wstart, wstop+dw/2.0, dw)
    nwave = len(wave)

    #- Confirm that this PSF covers these wavelengths for these spectra
    #skip this for the hackathon

    #psf_wavemin = np.max(psfdata['wavelength'](list(range(specmin, specmax)), y=-0.5))
    #psf_wavemax = np.min(psfdata['wavelength'](list(range(specmin, specmax)), y=psfdata['npix_y']-0.5))
    #let's take a shortcut for now, although this is definitely not right...
    #psf_wavemin = psfdata['wmin_all']
    #psf_wavemax = psfdata['wmax_all']
    #if psf_wavemin-5 > wstart:
    #    raise ValueError('Start wavelength {:.2f} < min wavelength {:.2f} for these fibers'.format(wstart, psf_wavemin))
    #if psf_wavemax+5 < wstop:
    #    raise ValueError('Stop wavelength {:.2f} > max wavelength {:.2f} for these fibers'.format(wstop, psf_wavemax))

    # Now we divide our spectra into bundles

    bundlesize = args.bundlesize
    checkbundles = set()
    checkbundles.update(np.floor_divide(np.arange(specmin, specmax), bundlesize*np.ones(nspec)).astype(int))
    bundles = sorted(checkbundles)
    nbundle = len(bundles)

    bspecmin = {}
    bnspec = {}
    for b in bundles:
        if specmin > b * bundlesize:
            bspecmin[b] = specmin
        else:
            bspecmin[b] = b * bundlesize
        if (b+1) * bundlesize > specmax:
            bnspec[b] = specmax - bspecmin[b]
        else:
            bnspec[b] = bundlesize

    # Now we assign bundles to processes

    nproc = 1
    rank = 0
    ###if comm is not None:
    ###    nproc = comm.size
    ###    rank = comm.rank

    mynbundle = int(nbundle // nproc)
    myfirstbundle = 0
    leftover = nbundle % nproc
    if rank < leftover:
        mynbundle += 1
        myfirstbundle = rank * mynbundle
    else:
        myfirstbundle = ((mynbundle + 1) * leftover) + (mynbundle * (rank - leftover))

    #- Print parameters
    log.info("extract:  input = {}".format(input_file))
    log.info("extract:  psf = {}".format(psf_file))
    log.info("extract:  specmin = {}".format(specmin))
    log.info("extract:  nspec = {}".format(nspec))
    log.info("extract:  wavelength = {},{},{}".format(wstart, wstop, dw))
    log.info("extract:  nwavestep = {}".format(args.nwavestep))
    log.info("extract:  regularize = {}".format(args.regularize))

    # get the root output file

    outpat = re.compile(r'(.*)\.fits')
    outmat = outpat.match(args.output)
    if outmat is None:
        raise RuntimeError("extraction output file should have .fits extension")
    outroot = outmat.group(1)

    outdir = os.path.normpath(os.path.dirname(outroot))
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    mark_preparation = time.time()

    time_total_extraction = 0.0
    time_total_write_output = 0.0

    failcount = 0

    ####fill some reference arrays for answer checking
    ####still the "wrong" answer compared to real specter but hopefully a consistently wrong answer
    flux_all = np.zeros((nspec,nwave))
    ivar_all = np.zeros((nspec,nwave))
    Rdata_all = np.zeros((nspec,17,nwave)) #not sure why 17 but ok
    chi2pix_all = np.zeros((nspec,nwave))

    #for now we'll do one bundle at a time
    #remember patch size is a hyperparameter

    for b in range(myfirstbundle, myfirstbundle+mynbundle):
        mark_iteration_start = time.time()

        #try streams
        stream_num = 'stream_' + str(b)
        #print("stream started", stream_num)

        #stream_num = cp.cuda.stream.Stream() #starts a new stream that is no longer default

        #entering bundle stream
        #i think we need a function here to make this easier for parallel streaming
        flux, ivar, Rdata, chi2pix, log = extract_bundle(outroot, b, rank, input_file, psf_file, bspecmin, specmin, bnspec, nspec, fibers, 
                img, psfdata, wave, nwave, wstart, wstop, dw, args, bundlesize, log, failcount)

        mark_extraction = time.time()
        mark_write_output = time.time()

        time_total_extraction += mark_extraction - mark_iteration_start
        time_total_write_output += mark_write_output - mark_extraction        
        
        #cp.cuda.stream.use(stream_num)

        ####append results from every bundle
        ###bstart = bspecmin[b]
        ###bstop = bspecmin[b] + bnspec[b]
        ###flux_all[bstart:bstop,:] = flux
        ###ivar_all[bstart:bstop,:] = ivar
        ###Rdata_all[bstart:bstop,:,:] = Rdata
        ###chi2pix_all[bstart:bstop,:] = chi2pix

    #lets compare the merged outputs for all bundles
    if args.test == True:
        flux_cpu_ref = np.load('/global/cfs/cdirs/m1759/desi/ref_files/flux_cpu_ref.npy')
        ivar_cpu_ref = np.load('/global/cfs/cdirs/m1759/desi/ref_files/ivar_cpu_ref.npy')
        Rdata_cpu_ref = np.load('/global/cfs/cdirs/m1759/desi/ref_files/Rdata_cpu_ref.npy')
        chi2pix_cpu_ref = np.load('/global/cfs/cdirs/m1759/desi/ref_files/chi2pix_cpu_ref.npy')
        #and now test
        assert np.allclose(flux_all, flux_cpu_ref)
        assert np.allclose(ivar_all, ivar_cpu_ref)
        assert np.allclose(Rdata_all, Rdata_cpu_ref)
        assert np.allclose(chi2pix_all, chi2pix_cpu_ref)
        print("reference bundle tests passed")

    #failcount = comm.allreduce(failcount)

    if failcount > 0:
        # all processes throw
        raise RuntimeError("some extraction bundles failed")

    time_merge = None
    ###if rank == 0:
    mark_merge_start = time.time()
    ###    mergeopts = [
    ###        '--output', args.output,
    ###        '--force',
    ###        '--delete'
    ###    ]
    ###    mergeopts.extend([ "{}_{:02d}.fits".format(outroot, b) for b in bundles ])
    ###    mergeargs = mergebundles.parse(mergeopts)
    ###    mergebundles.main(mergeargs)

    ###    if args.model is not None:
    ###        model = None
    ###        for b in bundles:
    ###            outmodel = "{}_model_{:02d}.fits".format(outroot, b)
    ###            if model is None:
    ###                model = fits.getdata(outmodel)
    ###            else:
    ###                #- TODO: test and warn if models overlap for pixels with
    ###                #- non-zero values
    ###                model += fits.getdata(outmodel)

    ###            os.remove(outmodel)

    ###        fits.writeto(args.model, model)
    mark_merge_end = time.time()
    time_merge = mark_merge_end - mark_merge_start

    # Resolve difference timer data

    if type(timing) is dict:
        timing["read_input"] = mark_read_input - mark_start
        timing["preparation"] = mark_preparation - mark_read_input
        timing["total_extraction"] = time_total_extraction
        timing["total_write_output"] = time_total_write_output
        timing["merge"] = time_merge

def extract_bundle(outroot, b, rank, input_file, psf_file, bspecmin, specmin, bnspec, nspec, fibers,
                   img, psfdata, wave, nwave, wstart, wstop, dw, args, bundlesize, log, failcount):

    outbundle = "{}_{:02d}.fits".format(outroot, b)
    outmodel = "{}_model_{:02d}.fits".format(outroot, b)

    log.info('extract:  Rank {} extracting {} spectra {}:{} at {}'.format(
        rank, os.path.basename(input_file),
        bspecmin[b], bspecmin[b]+bnspec[b], time.asctime(),
        ) )
    sys.stdout.flush()
    
    #get ready to pin memory based on usual patch size
    #final patch will be padded with zeros to make this work
    #this will need to be fixed at some point for production

        #flux.shape (25, 50)
        #ivar.shape (25, 50)
        #R.shape (1250, 1250)
        #xflux.shape (25, 50)
        #A.shape (16720, 1250)
        #iCov.shape (1250, 1250)

    #preallocate first (these will only work for 2 bundles, need to make more general)
    flux_out = np.empty((25,50))
    ivar_out = np.empty((25,50))
    Rdata_out = np.empty((1250,1250)) #hardcode for now
    xflux_out = np.empty((25,50))
    A_out = np.empty((20000,1250))
    iCov_out = np.empty((1250,1250))

    #then pin
    flux_pinned = _pin_memory(flux_out)
    ivar_pinned = _pin_memory(ivar_out)
    Rdata_pinned = _pin_memory(Rdata_out)
    xflux_pinned = _pin_memory(xflux_out)
    A_pinned = _pin_memory(A_out)
    iCov_pinned = _pin_memory(iCov_out)

    #append all pointers into a list
    pinned_list = np.array([flux_pinned, ivar_pinned, Rdata_pinned, xflux_pinned, A_pinned, iCov_pinned])

    #- The actual extraction
    try:
        results = ex2d(img.pix, img.ivar*(img.mask==0), psfdata, pinned_list, bspecmin[b],
            bnspec[b], wave, regularize=args.regularize, ndecorr=args.decorrelate_fibers,
            bundlesize=bundlesize, wavesize=args.nwavestep, verbose=args.verbose,
            full_output=True, nsubbundles=args.nsubbundles)

        #print(results.keys())

        flux = results['flux']
        ivar = results['ivar']
        Rdata = results['resolution_data']
        chi2pix = results['chi2pix']

        ##try flux_out (pinned memory) instead
        #mask = np.zeros(flux_out.shape, dtype=np.uint32)
        #mask[results['pixmask_fraction']>0.5] |= specmask.SOMEBADPIX
        #mask[results['pixmask_fraction']==1.0] |= specmask.ALLBADPIX
        #mask[chi2pix>100.0] |= specmask.BAD2DFIT

        #if heliocentric_correction_factor != 1 :
        #    #- Apply heliocentric correction factor to the wavelength
        #    #- without touching the spectra, that is the whole point
        #    wave   *= heliocentric_correction_factor
        #    wstart *= heliocentric_correction_factor
        #    wstop  *= heliocentric_correction_factor
        #    dw     *= heliocentric_correction_factor
        #    img.meta['HELIOCOR']   = heliocentric_correction_factor

        #- Augment input image header for output
        img.meta['NSPEC']   = (nspec, 'Number of spectra')
        img.meta['WAVEMIN'] = (wstart, 'First wavelength [Angstroms]')
        img.meta['WAVEMAX'] = (wstop, 'Last wavelength [Angstroms]')
        img.meta['WAVESTEP']= (dw, 'Wavelength step size [Angstroms]')
        img.meta['SPECTER'] = (specter.__version__, 'https://github.com/desihub/specter')
        img.meta['IN_PSF']  = (_trim(psf_file), 'Input spectral PSF')
        img.meta['IN_IMG']  = (_trim(input_file), 'Input image')

        #if fibermap is not None:
        #    bfibermap = fibermap[bspecmin[b]-specmin:bspecmin[b]+bnspec[b]-specmin]
        #else:
        #lets stick with fibermap default
        bfibermap = None

        bfibers = fibers[bspecmin[b]-specmin:bspecmin[b]+bnspec[b]-specmin]
        ###frame = Frame(wave, flux, ivar, mask=mask, resolution_data=Rdata,
        ###            fibers=bfibers, meta=img.meta, fibermap=bfibermap,
        ###            chi2pix=chi2pix)

        ####- Add unit
        ####   In specter.extract.ex2d one has flux /= dwave
        ####   to convert the measured total number of electrons per
        ####   wavelength node to an electron 'density'
        ###frame.meta['BUNIT'] = 'count/Angstrom'

        ####- Add scores to frame
        ####compute_and_append_frame_scores(frame,suffix="RAW")

        ####- Write output
        ###io.write_frame(outbundle, frame)

        ###if args.model is not None:
        ###    from astropy.io import fits
        ###    fits.writeto(outmodel, results['modelimage'], header=frame.meta)

        log.info('extract:  Done {} spectra {}:{} at {}'.format(os.path.basename(input_file),
            bspecmin[b], bspecmin[b]+bnspec[b], time.asctime()))
        sys.stdout.flush()

    except:
        # Log the error and increment the number of failures
        log.error("extract:  FAILED bundle {}, spectrum range {}:{}".format(b, bspecmin[b], bspecmin[b]+bnspec[b]))
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        log.error(''.join(lines))
        failcount += 1
        sys.stdout.flush()

    return flux, ivar, Rdata, chi2pix, log

def _pin_memory(array):
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret

if __name__ == '__main__':
    args = parse()
    main(args) 

    
