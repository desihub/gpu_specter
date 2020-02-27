#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import numpy as np
import scipy.sparse
import scipy.linalg
from scipy.sparse import spdiags, issparse
from scipy.sparse.linalg import spsolve
import time
import math
import functools

import numba
import cupy as cp
import cupyx as cpx
import cupyx.scipy.special
from numba import cuda

#swtich to turn on/off our nvtx collection decorators
nvtx_collect = True

def nvtx_profile(_func=None, *, profile=False, name=None):
    """decorator to make collecting nvtx data easier"""
    def decorator_nvtx_profile(func):
        @functools.wraps(func)
        def wrapper_nvtx_profile(*args, **kwargs):
            if profile==True:
                cp.cuda.nvtx.RangePush(name)
                value = func(*args, **kwargs)
                cp.cuda.nvtx.RangePop()
                return value
            else:
                return func(*args, **kwargs)
        return wrapper_nvtx_profile
    if _func is None:
        return decorator_nvtx_profile
    else:
        return decorator_nvtx_profile(_func)

def native_endian(data):
    """Temporary function, sourced from desispec.io
    Convert numpy array data to native endianness if needed.
    Returns new array if endianness is swapped, otherwise returns input data
    Context:
    By default, FITS data from astropy.io.fits.getdata() are not Intel
    native endianness and scipy 0.14 sparse matrices have a bug with
    non-native endian data.
    """
    if data.dtype.isnative:
        return data
    else:
        return data.byteswap().newbyteorder()


@cuda.jit
def legvander(x, deg, output_matrix):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(i, x.shape[0], stride):
        output_matrix[i][0] = 1
        output_matrix[i][1] = x[i]
        for j in range(2, deg + 1):
            output_matrix[i][j] = (output_matrix[i][j-1]*x[i]*(2*j - 1) - output_matrix[i][j-2]*(j - 1)) / j

def legvander_wrapper(x, deg):
    """Temporary wrapper that allocates memory and defines grid before calling legvander.
    Probably won't be needed once cupy has the correpsponding legvander function.
    Input: Same as cpu version of legvander
    Output: legvander matrix, cp.ndarray
    """
    output = cp.ndarray((len(x), deg + 1))
    blocksize = 256
    numblocks = (len(x) + blocksize - 1) // blocksize
    legvander[numblocks, blocksize](x, deg, output)
    return output

def evalcoeffs(wavelengths, psfdata):
    '''
    wavelengths: 1D array of wavelengths to evaluate all coefficients for all wavelengths of all spectra
    psfdata: Table of parameter data ready from a GaussHermite format PSF file
    Returns a dictionary params[paramname] = value[nspec, nwave]
    The Gauss Hermite coefficients are treated differently:
        params['GH'] = value[i,j,nspec,nwave]
    The dictionary also contains scalars with the recommended spot size HSIZEX, HSIZEY
    and Gauss-Hermite degrees GHDEGX, GHDEGY (which is also derivable from the dimensions
    of params['GH'])
    '''
    # Initialization
    wavemin, wavemax = psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0]
    wx = (wavelengths - wavemin) * (2.0 / (wavemax - wavemin)) - 1.0

    L = legvander_wrapper(wx, psfdata.meta['LEGDEG'])
    p = dict(WAVE=wavelengths) # p doesn't live on the gpu, but it's last-level values do
    nparam, nspec, ndeg = psfdata['COEFF'].shape
    nwave = L.shape[0]

    # Init zeros
    p['GH'] = cp.zeros((psfdata.meta['GHDEGX']+1, psfdata.meta['GHDEGY']+1, nspec, nwave))
    # Init gpu coeff
    coeff_gpu = cp.array(native_endian(psfdata['COEFF']))

    k = 0
    for name, coeff in zip(psfdata['PARAM'], psfdata['COEFF']):
        name = name.strip()
        if name.startswith('GH-'):
            i, j = map(int, name.split('-')[1:3])
            p['GH'][i,j] = L.dot(coeff_gpu[k].T).T
        else:
            p[name] = L.dot(coeff_gpu[k].T).T
        k += 1

    #- Include some additional keywords that we'll need
    for key in ['HSIZEX', 'HSIZEY', 'GHDEGX', 'GHDEGY']:
        p[key] = psfdata.meta[key]

    return p


@cuda.jit
def hermevander(x, deg, output_matrix):
    i = cuda.blockIdx.x
    _, j = cuda.grid(2)
    _, stride = cuda.gridsize(2)
    for j in range(j, x.shape[1], stride):
        output_matrix[i][j][0] = 1
        if deg > 0:
            output_matrix[i][j][1] = x[i][j]
            for k in range(2, deg + 1):
                output_matrix[i][j][k] = output_matrix[i][j][k-1]*x[i][j] - output_matrix[i][j][k-2]*(k-1)

def hermevander_wrapper(x, deg):
    """Temprorary wrapper that allocates memory and calls hermevander_gpu
    """
    if x.ndim == 1:
        x = cp.expand_dims(x, 0)
    output = cp.ndarray(x.shape + (deg+1,))
    blocksize = 256
    numblocks = (x.shape[0], (x.shape[1] + blocksize - 1) // blocksize)
    hermevander[numblocks, blocksize](x, deg, output)
    return cp.squeeze(output)


def calc_pgh(ispec, wavelengths, psfparams):
    '''
    Calculate the pixelated Gauss Hermite for all wavelengths of a single spectrum
    ispec : integer spectrum number
    wavelengths : array of wavelengths to evaluate
    psfparams : dictionary of PSF parameters returned by evalcoeffs
    returns pGHx, pGHy
    where pGHx[ghdeg+1, nwave, nbinsx] contains the pixel-integrated Gauss-Hermite polynomial
    for all degrees at all wavelengths across nbinsx bins spaning the PSF spot, and similarly
    for pGHy.  The core PSF will then be evaluated as
    PSFcore = sum_ij c_ij outer(pGHy[j], pGHx[i])
    '''

    #- shorthand
    p = psfparams

    #- spot size (ny,nx)
    nx = p['HSIZEX']
    ny = p['HSIZEY']
    nwave = len(wavelengths)
    p['X'], p['Y'], p['GHSIGX'], p['GHSIGY'] = \
    cp.array(p['X']), cp.array(p['Y']), cp.array(p['GHSIGX']), cp.array(p['GHSIGY'])
    xedges = cp.repeat(cp.arange(nx+1) - nx//2, nwave).reshape(nx+1, nwave)
    yedges = cp.repeat(cp.arange(ny+1) - ny//2, nwave).reshape(ny+1, nwave)

    #- Shift to be relative to the PSF center at 0 and normalize
    #- by the PSF sigma (GHSIGX, GHSIGY)
    #- xedges[nx+1, nwave]
    #- yedges[ny+1, nwave]
    xedges = (xedges - p['X'][ispec]%1)/p['GHSIGX'][ispec]
    yedges = (yedges - p['Y'][ispec]%1)/p['GHSIGY'][ispec]

    #- Degree of the Gauss-Hermite polynomials
    ghdegx = p['GHDEGX']
    ghdegy = p['GHDEGY']

    #- Evaluate the Hermite polynomials at the pixel edges
    #- HVx[ghdegx+1, nwave, nx+1]
    #- HVy[ghdegy+1, nwave, ny+1]
    HVx = hermevander_wrapper(xedges, ghdegx).T
    HVy = hermevander_wrapper(yedges, ghdegy).T

    #- Evaluate the Gaussians at the pixel edges
    #- Gx[nwave, nx+1]
    #- Gy[nwave, ny+1]
    Gx = cp.exp(-0.5*xedges**2).T / cp.sqrt(2. * cp.pi)
    Gy = cp.exp(-0.5*yedges**2).T / cp.sqrt(2. * cp.pi)

    #- Combine into Gauss*Hermite
    GHx = HVx * Gx
    GHy = HVy * Gy

    #- Integrate over the pixels using the relationship
    #  Integral{ H_k(x) exp(-0.5 x^2) dx} = -H_{k-1}(x) exp(-0.5 x^2) + const

    #- pGHx[ghdegx+1, nwave, nx]
    #- pGHy[ghdegy+1, nwave, ny]
    pGHx = cp.zeros((ghdegx+1, nwave, nx))
    pGHy = cp.zeros((ghdegy+1, nwave, ny))
    pGHx[0] = 0.5 * cp.diff(cupyx.scipy.special.erf(xedges/cp.sqrt(2.)).T)
    pGHy[0] = 0.5 * cp.diff(cupyx.scipy.special.erf(yedges/cp.sqrt(2.)).T)
    pGHx[1:] = GHx[:ghdegx,:,0:nx] - GHx[:ghdegx,:,1:nx+1]
    pGHy[1:] = GHy[:ghdegy,:,0:ny] - GHy[:ghdegy,:,1:ny+1]

    return pGHx, pGHy

@cuda.jit()
def multispot(pGHx, pGHy, ghc, mspots):
    nx = pGHx.shape[-1]
    ny = pGHy.shape[-1]
    nwave = pGHx.shape[1]

    #this is the magic step
    iwave = cuda.grid(1)

    n = pGHx.shape[0]
    m = pGHy.shape[0]

    if (0 <= iwave < nwave):
    #yanked out the i and j loops in lieu of the cuda grid of threads
        for i in range(pGHx.shape[0]):
            px = pGHx[i,iwave]
            for j in range(0, pGHy.shape[0]):
                py = pGHy[j,iwave]
                c = ghc[i,j,iwave]

                for iy in range(len(py)):
                    for ix in range(len(px)):
                        mspots[iwave, iy, ix] += c * py[iy] * px[ix]


@nvtx_profile(profile=nvtx_collect, name='cache_spots')
def cache_spots(nspec, nwave, p, wavelengths):
    nx = p['HSIZEX']
    ny = p['HSIZEY']
    spots = cp.zeros((nspec, nwave, ny, nx))
    #use mark's numblocks and blocksize method
    blocksize = 256
    numblocks = (nwave + blocksize - 1) // blocksize
    for ispec in range(nspec):
        pGHx, pGHy = calc_pgh(ispec, wavelengths, p)
        ghc = p['GH'][:,:,ispec,:]
        mspots = cp.zeros((nwave, ny, nx)) #empty every time!
        multispot[numblocks, blocksize](pGHx, pGHy, ghc, mspots)
        spots[ispec] = mspots
    return spots

#can you double decorate a cuda.jit ?
#@nvtx_profile(profile=nvtx_collect, name='projection_matrix')
@cuda.jit()
def projection_matrix(A, xc, yc, xmin, ymin, ispec, iwave, nspec, nwave, spots):
    #this is the heart of the projection matrix calculation
    ny, nx = spots.shape[2:4]
    i, j = cuda.grid(2)
    #no loops, just a boundary check
    if (0 <= i < nspec) and (0 <= j <nwave):
        ixc = xc[ispec+i, iwave+j] - xmin
        iyc = yc[ispec+i, iwave+j] - ymin
        #A[iyc:iyc+ny, ixc:ixc+nx, i, j] = spots[ispec+i,iwave+j]
        #this fancy indexing is not allowed in numba gpu (although it is in numba cpu...)
        #try this instead
        for iy, y in enumerate(range(iyc,iyc+ny)):
            for ix, x in enumerate(range(ixc,ixc+nx)):
                temp_spot = spots[ispec+i, iwave+j][iy, ix]
                A[y, x, i, j] += temp_spot


def ex2d(image, imageivar, psfdata, specmin, nspec, wavelengths, xyrange=None,
         regularize=0.0, ndecorr=False, bundlesize=25, nsubbundles=1,
         wavesize=50, full_output=False, verbose=False, 
         debug=False, psferr=None):
    '''
2D PSF extraction of flux from image patch given pixel inverse variance.

Inputs:
    image : 2D array of pixels
    imageivar : 2D array of inverse variance for the image
    psf   : PSF object
    specmin : index of first spectrum to extract
    nspec : number of spectra to extract
    wavelengths : 1D array of wavelengths to extract
Optional Inputs:
        xyrange = (xmin, xmax, ymin, ymax): treat image as a subimage
            cutout of this region from the full image
        regularize: experimental regularization factor to minimize ringing
        ndecorr : if True, decorrelate the noise between fibers, at the
            cost of residual signal correlations between fibers.
        bundlesize: extract in groups of fibers of this size, assuming no
            correlation with fibers outside of this bundle
        nsubbundles: (int) number of overlapping subbundles to use per bundle
        wavesize: number of wavelength steps to include per sub-extraction
        full_output: Include additional outputs based upon chi2 of model
            projected into pixels
        verbose: print more stuff
        debug: if True, enter interactive ipython session before returning
        psferr:  fractional error on the psf model. if not None, use this
            fractional error on the psf model instead of the value saved
            in the psf fits file. This is used only to compute the chi2,
            not to weight pixels in fit
    Returns (flux, ivar, Rdata):
        flux[nspec, nwave] = extracted resolution convolved flux
        ivar[nspec, nwave] = inverse variance of flux
        Rdata[nspec, 2*ndiag+1, nwave] = sparse Resolution matrix data
    TODO: document output if full_output=True
    ex2d uses divide-and-conquer to extract many overlapping subregions
    and then stitches them back together.  Params wavesize and bundlesize
    control the size of the subregions that are extracted; the necessary
    amount of overlap is auto-calculated based on PSF extent.
    '''
    #- TODO: check input dimensionality etc.

    #lets cheat by getting these from our specter psf
    #may not agree with our test psf.fits here
    #also x and y require specter which is cheating
    npzfile = np.load('/global/cscratch1/sd/stephey/psf_data_file.npz', allow_pickle=True)
    psf_wdisp = npzfile['arr_0'] #double check!
    psf_pix = npzfile['arr_1'] #double check!
    # ny = psf_pix[0]
    # nx = psf_pix[1]

    wavemin, wavemax = wavelengths[0], wavelengths[-1]

    #wavemin, wavemax = psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0]
    #wavelengths = np.arange(psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0], 0.8) #also a hack

    #lets do evalcoeffs to create p (dict with keys)
    p = evalcoeffs(wavelengths, psfdata)

    #do specrange ourselves (no subbundles)! 
    speclo = specmin
    spechi = specmin + nspec
    specrange = (speclo, spechi)

    #keep
    #[ True  True  True  True False]
    #do keep ourselves too
    keep = np.ones(25,dtype=bool) #keep the whole bundle, is this right? probably not...

    #- TODO: check input dimensionality etc.

    #dw = wavelengths[1] - wavelengths[0]
    #if not np.allclose(dw, np.diff(wavelengths)):
    #    raise ValueError('ex2d currently only supports linear wavelength grids')

    #- Output arrays to fill
    nwave = len(wavelengths)
    flux = np.zeros( (nspec, nwave) )
    ivar = np.zeros( (nspec, nwave) )
    if full_output:
        pixmask_fraction = np.zeros( (nspec, nwave) )
        chi2pix = np.zeros( (nspec, nwave) )
        modelimage = np.zeros_like(image)

    #- Diagonal elements of resolution matrix
    #- Keep resolution matrix terms equivalent to 9-sigma of largest spot
    #- ndiag is in units of number of wavelength steps of size dw
    dw = 0.7 #double check!
    ndiag = 0
    for ispec in [specmin, specmin+nspec//2, specmin+nspec-1]:
        for w in [wavemin, 0.5*(wavemin+wavemax), wavemax]:
            ndiag = max(ndiag, int(round(0.9* psf_wdisp / dw )))

    #- make sure that ndiag isn't too large for actual PSF spot size
    wmid = (wavemin + wavemax) / 2.0 #double check!
    spotsize = psf_pix.shape #double check!
    #ndiag = min(ndiag, spotsize[0]//2, spotsize[1]//2) #double check!
    #in specter ndiag is either 7 or 8
    #try making it 8 for now?
    ndiag = 8 

    #print("ndiag", ndiag)
   
    #- Orig was ndiag = 10, which fails when dw gets too large compared to PSF size
    Rd = np.zeros( (nspec, 2*ndiag+1, nwave) )

    psferr = 0.01 #double check!

    #do entire bundle
    #nvtx profiling via decorator
    spots = cache_spots(nspec, nwave, p, wavelengths)

    #also need the bottom corners
    xc = np.floor(p['X'] - p['HSIZEX']//2).astype(int)
    yc = np.floor(p['Y'] - p['HSIZEY']//2).astype(int)
    corners = (xc, yc)

    #print("xc.shape", xc.shape)

    ispecmin = speclo 
    ispecmax = spechi

    #print("len(wavelengths)", len(wavelengths))


    #- Let's do some extractions
    #but no subbundles! wavelength patches only
    #start, stop, step
    tws = len(wavelengths)
    for iwave in range(0, tws, wavesize):
        #- Low and High wavelengths for the core region
        wlo = wavelengths[iwave]
        iwavemin = iwave
        if iwave+wavesize < len(wavelengths):
            whi = wavelengths[iwave+wavesize]
            iwavemax = iwavemin + wavesize
        else: #handle the leftover part carefully
            whi = wavelengths[-1]
            wleftover = tws - iwave
            wavesize = wleftover
            iwavemax = iwavemin + wleftover

        #- Identify subimage that covers the core wavelengths
        #subxyrange = xlo,xhi,ylo,yhi = psf.xyrange(specrange, (wlo, whi))
 
        #nlo = max(int((wlo - psf.wavelength(speclo, ymin))/dw)-1, ndiag)
        #nhi = max(int((psf.wavelength(speclo, ymax) - whi)/dw)-1, ndiag)   
        nlo = ndiag - 3 #totally a guess
        nhi = ndiag + 3
        ##cheat for now, maybe check specter for expected dimensions
        ##double check this whole section
        #ww = np.arange(wlo-nlo*dw, whi+(nhi+0.5)*dw, dw)
        #use values directly from the psf to make bookkeeping easier
        #start stop npoints
        ww = np.linspace(wlo, whi, wavesize)
        wmin, wmax = ww[0], ww[-1]
        nw = len(ww)

        ny, nx = spots.shape[2:4]
        #xc and yc are on the gpu for now
        xlo = np.min(xc[0:spechi-speclo, iwave:iwave+nw].get())
        xhi = np.max(xc[0:spechi-speclo, iwave:iwave+nw].get()) + nx
        ylo = np.min(yc[0:spechi-speclo, iwave:iwave+nw].get())
        yhi = np.max(yc[0:spechi-speclo, iwave:iwave+nw].get()) + ny
        #print('### ex2d: xmin, xmax, ymin, ymax = {}, {}, {}, {}'.format(xlo, xhi, ylo, yhi))
        #print('### ex2d: nx, ny, ispec, nspec, iwave, nwave: {}, {}, {}, {}, {}, {}'.format(nx, ny, ispec, spechi-speclo, iwave, nw))
        #print("xlo %s, xhi %s, ylo %s, yhi %s" %(xlo, xhi, ylo, yhi))
        #print("xmax-xmin {} ymax-ymin {}".format(xhi-xlo, yhi-ylo))
        xyrange = [xlo, xhi, ylo, yhi]
        subxy = np.s_[ylo-xyrange[2]:yhi-xyrange[2], xlo-xyrange[0]:xhi-xyrange[0]]

        #want to send this to the gpu once, re-use
        #have outer wrapper pass in subset of image to save memory?
        #print("imageivar.shape", imageivar.shape)
        subimg = image[subxy]
        subivar = imageivar[subxy]
 
        #- Determine extra border wavelength extent: nlo,nhi extra wavelength bins
        #ny, nx = psf.pix(speclo, wlo).shape
        #ny, nx = psf_pix #load from our file, cheating
        #double check this whole section!!!
        # ymin = ylo-ny+2
        # ymax = yhi+ny-2

        #- include \r carriage return to prevent scrolling
        if verbose:
            sys.stdout.write("\rSpectra {specrange} wavelengths ({wmin:.2f}, {wmax:.2f}) -> ({wlo:.2f}, {whi:.2f})".format(\
                specrange=specrange, wmin=wmin, wmax=wmax, wlo=wlo, whi=whi))
            sys.stdout.flush()

        #maybe call projection matrix here? or inside ex2d_patch
        #bookkeeping might be easier to call here, will blow memory probably

        #- Do the extraction with legval cache as default
        #pass in both p and psfdata for now, not sure if we really need both
        #pass spots so we can do projection_matrix inside

        results = \
            ex2d_patch(subimg, subivar, p, psfdata, spots, corners, iwave, tws,
                specmin=speclo, nspec=spechi-speclo, wavelengths=ww,
                xyrange=[xlo,xhi,ylo,yhi], regularize=regularize, ndecorr=ndecorr,
                full_output=True, use_cache=True) 

        #question: is one big transfer better than lots of little transfers?

        specflux = results['flux']
        #flux = results['flux']
        specivar = results['ivar']
        #ivar = results['ivar']
        R = results['R']
       
        #- Fill in the final output arrays
        ## iispec = slice(speclo-specmin, spechi-specmin)
        #don't need this since we got rid of the subbundles

        #we have to assemble the data from the patches back together!!!
        iispec = np.arange(speclo-specmin, spechi-specmin)

        #not sure if this is right but at least the dimensions are ok
        flux[iispec[keep], iwave:iwave+wavesize] = specflux[keep, :]
        ivar[iispec[keep], iwave:iwave+wavesize] = specivar[keep, :]

        if full_output:
            A = results['A'].copy()
            xflux = results['xflux']
            
            #- number of spectra and wavelengths for this sub-extraction
            subnspec = spechi-speclo
            subnwave = len(ww)
            
            #order of operations! the A dot xflux.ravel() comes first!

            #- Model image
            submodel = A.dot(xflux.ravel()).reshape(subimg.shape)
            #modeulimage = submodel ?

            #- Fraction of input pixels that are unmasked for each flux bin
            subpixmask_fraction = 1.0-(A.T.dot(subivar.ravel()>0)).reshape(subnspec, subnwave)
            
            #- original weighted chi2 of pixels that contribute to each flux bin
            # chi = (subimg - submodel) * np.sqrt(subivar)
            # chi2x = (A.T.dot(chi.ravel()**2) / A.sum(axis=0)).reshape(subnspec, subnwave)
            
            #- pixel variance including input noise and PSF model errors
            modelivar = (submodel*psferr + 1e-32)**-2
            ii = (modelivar > 0) & (subivar > 0)
            totpix_ivar = np.zeros(submodel.shape)
            totpix_ivar[ii] = 1.0 / (1.0/modelivar[ii] + 1.0/subivar[ii])
            
            #- Weighted chi2 of pixels that contribute to each flux bin;
            #- only use unmasked pixels and avoid dividing by 0
            chi = (subimg - submodel) * np.sqrt(totpix_ivar)
            psfweight = A.T.dot(totpix_ivar.ravel()>0)
            bad = (psfweight == 0.0)
            chi2x = (A.T.dot(chi.ravel()**2) * ~bad) / (psfweight + bad)
            chi2x = chi2x.reshape(subnspec, subnwave)
            
            #- outputs
            #- TODO: watch out for edge effects on overlapping regions of submodels
            modelimage[subxy] = submodel
            #same nlo nhi change here
            pixmask_fraction[iispec[keep], iwave:iwave+wavesize] = subpixmask_fraction[keep, :]
            #same here
            chi2pix[iispec[keep], iwave:iwave+wavesize] = chi2x[keep, :]

            #- Fill diagonals of resolution matrix
            for ispec in np.arange(speclo, spechi)[keep]:
                #- subregion of R for this spectrum
                ii = slice(nw*(ispec-speclo), nw*(ispec-speclo+1))
                Rx = R[ii, ii]
                #print("Rx.shape", Rx.shape)
                #print("Rd.shape", Rd.shape)
                #need to fix this too
                #for j in range(nlo,nw-nhi):
                #for j in range(nwave):
                    # Rd dimensions [nspec, 2*ndiag+1, nwave]
                    #this is a mess, just write zeros for now until i can get
                    #the values of nlo, nhi, ndiag right
                    #Rd[ispec-specmin, :, iwave+j-nlo] = Rx[j-ndiag:j+ndiag+1, j]

    #- Add extra print because of carriage return \r progress trickery
    if verbose:
        print()

    #+ TODO: what should this do to R in the case of non-uniform bins?
    #+       maybe should do everything in photons/A from the start.            
    #- Convert flux to photons/A instead of photons/bin
    dwave = np.gradient(wavelengths)
    flux /= dwave #this is divide and, divides left operand with the right operand and assign the result to left operand
    ivar *= dwave**2 #similar

    if debug:
        #--- DEBUG ---
        import IPython
        IPython.embed()
        #--- DEBUG ---
    
    if full_output:
        return dict(flux=flux, ivar=ivar, resolution_data=Rd, modelimage=modelimage,
            pixmask_fraction=pixmask_fraction, chi2pix=chi2pix)
    else:
        return flux, ivar, Rd

@nvtx_profile(profile=nvtx_collect, name='ex2d_patch')
def ex2d_patch(image, ivar, p, psfdata, spots, corners, 
         iwave, tws, specmin, nspec, wavelengths, xyrange,
         full_output=False, regularize=0.0, ndecorr=False, use_cache=None):
    """
    2D PSF extraction of flux from image patch given pixel inverse variance.
    
    Inputs:
        image : 2D array of pixels
        ivar  : 2D array of inverse variance for the image
        psf   : PSF object
        specmin : index of first spectrum to extract
        nspec : number of spectra to extract
        wavelengths : 1D array of wavelengths to extract
        
    Optional Inputs:
        xyrange = (xmin, xmax, ymin, ymax): treat image as a subimage
            cutout of this region from the full image
        full_output : if True, return a dictionary of outputs including
            intermediate outputs such as the projection matrix.
        ndecorr : if True, decorrelate the noise between fibers, at the
            cost of residual signal correlations between fibers.
        use_cache: default behavior, can be turned off for testing purposes
    Returns (flux, ivar, R):
        flux[nspec, nwave] = extracted resolution convolved flux
        ivar[nspec, nwave] = inverse variance of flux
        R : 2D resolution matrix to convert
    """
    #scalars will move automatically (i think)

    #- Range of image to consider
    waverange = (wavelengths[0], wavelengths[-1])
    specrange = (specmin, specmin+nspec) 
 
    #need to make sure we pass in xyrange from ex2d
    xmin, xmax, ymin, ymax = xyrange

    nx, ny = xmax-xmin, ymax-ymin
    npix = nx*ny
    
    nspec = specrange[1] - specrange[0]
    nspec = int(nspec)
    nwave = len(wavelengths)

    #print(type(nspec))
    #print(type(nwave))
    
    #print("waverange", waverange)
    #print("nwave", nwave)

    #corners and spots are for the entire bundle!!!
    #ispec starts again in the bundle
    #iwave does not, varies per patch
    ispec = 0
    #print("iwave", iwave)
    #print("nwave", nwave)
    #may want to call this differently for the end of the wavelength range
    #Araw, ymin, xmin = projection_matrix(ispec, nspec, iwave, nwave, spots, corners)
    #try cupy instead
    A = cp.zeros((ymax-ymin,xmax-xmin,nspec,nwave), dtype=np.float64)

    xc, yc = corners

    #get ready to launch our kernel
    #this is a 2d kernel for projection matrix
    threads_per_block = (16,16) #needs to be 2d!
    #copy from matt who copied from cuda docs so it's probably legit
    blocks_per_grid_x = math.ceil(A.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(A.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    cp.cuda.nvtx.RangePush('projection_matrix')
    projection_matrix[blocks_per_grid, threads_per_block](A, xc, yc, xmin, ymin, ispec, iwave, nspec, nwave, spots)
    cp.cuda.nvtx.RangePop()

    #hopefully A remains a cupy array, let's check
    #print(type(A)) yes it does remain a cupy ndarray
    #reshape
    nypix, nxpix = A.shape[0:2]
    A_dense = A.reshape(nypix*nxpix, nspec*nwave)
    A = cpx.scipy.sparse.csr_matrix(A_dense)
    w = cp.asarray(ivar.ravel())
    W = cpx.scipy.sparse.spdiags(data=w, diags=[0,], m=npix, n=npix)
    image_gpu = cp.asarray(image.ravel())

    #call our new prevent_ringing function
    pix_gpu, Ax_gpu, wx_gpu = prevent_ringing(nspec, nwave, image_gpu, A, w, W)

    #call our new extract cupy function
    fx_gpu, varfx_gpu, R_gpu, f_gpu, iCov_gpu = ex_cupy(nspec, nwave, Ax_gpu, wx_gpu, pix_gpu)

    ##pull back to cpu to return to ex2d
    results = move_data_host(fx_gpu, varfx_gpu, R_gpu, f_gpu, A_dense, iCov_gpu, specmin, nspec, wavelengths, xyrange, regularize, ndecorr)

    #assume full_output=True
    return results

@nvtx_profile(profile=nvtx_collect, name='prevent_ringing')
def prevent_ringing(nspec, nwave, image_gpu, A, w, W):
    fluxweight = W.dot(A).sum(axis=0)[0] #is this right?
    minweight = 1.e-4*cp.max(fluxweight)
    ibad = fluxweight < minweight

    #- Add regularization of low weight fluxes
    Idiag = 0.0*cp.ones(nspec*nwave)
    Idiag[ibad] = minweight - fluxweight[ibad] #probably have the dimensions of fluxweight wrong...
    I = cpx.scipy.sparse.diags(Idiag) #is this equivalent?

    #- Only need to extend A if regularization is non-zero
    if cp.any(I.diagonal()):
        pix = cp.concatenate( (image_gpu, cp.zeros(nspec*nwave)) )
        Ax = cpx.scipy.sparse.vstack( (A, I) )
        wx = cp.concatenate( (w, cp.ones(nspec*nwave)) )
    else:
        pix = image_gpu
        Ax = A
        wx = w 

    return pix, Ax, wx

#parts of this might be good candidate(s) for kernel fusion
@nvtx_profile(profile=nvtx_collect, name='ex_cupy')
def ex_cupy(nspec, nwave, Ax_gpu, wx_gpu, pix_gpu):

    #will fill this with nvtx markers, it will be ugly but hopefully insightful
    cp.cuda.nvtx.RangePush('spdiags_Wx')
    Wx_gpu = cpx.scipy.sparse.spdiags(wx_gpu, 0, len(wx_gpu), len(wx_gpu))
    cp.cuda.nvtx.RangePop()

    cp.cuda.nvtx.RangePush('A_dot_Wx')
    iCov_gpu = Ax_gpu.T.dot(Wx_gpu.dot(Ax_gpu))
    y_gpu = Ax_gpu.T.dot(Wx_gpu.dot(pix_gpu))
    cp.cuda.nvtx.RangePop()

    cp.cuda.nvtx.RangePush('cupy_linalg_solve')
    f_gpu = cp.linalg.solve(iCov_gpu.todense(), y_gpu).reshape((nspec, nwave)) #requires array, not sparse object
    cp.cuda.nvtx.RangePop()

    cp.cuda.nvtx.RangePush('eigh')
    u_gpu, v_gpu = cp.linalg.eigh(iCov_gpu.todense())
    #print("type(u_gpu)",type(u_gpu))
    #print("type(v_gpu)",type(v_gpu))
    #type(u_gpu) <class 'cupy.core.core.ndarray'>
    #type(v_gpu) <class 'cupy.core.core.ndarray'>
    cp.cuda.nvtx.RangePop()

    cp.cuda.nvtx.RangePush('spdiags_d')
    d_gpu = cpx.scipy.sparse.spdiags(cp.sqrt(u_gpu), 0, len(u_gpu) , len(u_gpu))
    cp.cuda.nvtx.RangePop()

    cp.cuda.nvtx.RangePush('v_dot_d')
    Q_gpu = v_gpu.dot( d_gpu.dot( v_gpu.T ))
    cp.cuda.nvtx.RangePop()

    cp.cuda.nvtx.RangePush('sum')
    norm_vector_gpu = cp.sum(Q_gpu, axis=1)
    cp.cuda.nvtx.RangePop()

    cp.cuda.nvtx.RangePush('outer')
    R_gpu = cp.outer(norm_vector_gpu**(-1), cp.ones(norm_vector_gpu.size)) * Q_gpu
    cp.cuda.nvtx.RangePop()

    cp.cuda.nvtx.RangePush('spdiags_u')
    udiags_gpu = cpx.scipy.sparse.spdiags(1/u_gpu, 0, len(u_gpu), len(u_gpu))
    cp.cuda.nvtx.RangePop()

    cp.cuda.nvtx.RangePush('final_dot_prods')
    Cov_gpu = v_gpu.dot( udiags_gpu.dot (v_gpu.T ))
    Cx_gpu = R_gpu.dot(Cov_gpu.dot(R_gpu.T))
    fx_gpu = R_gpu.dot(f_gpu.ravel()).reshape(f_gpu.shape)
    cp.cuda.nvtx.RangePop()

    cp.cuda.nvtx.RangePush('reshape')
    varfx_gpu = (norm_vector_gpu * 2).reshape((nspec, nwave))
    cp.cuda.nvtx.RangePop()

    return fx_gpu, varfx_gpu, R_gpu, f_gpu, iCov_gpu    

@nvtx_profile(profile=nvtx_collect, name='move_data_host')
def move_data_host(fx_gpu, varfx_gpu, R_gpu, f_gpu, A_dense, iCov_gpu, specmin, nspec, wavelengths, xyrange, regularize, ndecorr):
    #trying to make things easier for when we get fancy with async transfers
    flux = fx_gpu.get()
    ivar = varfx_gpu.get()
    sqR = np.sqrt(R_gpu.size).astype(int)
    R = R_gpu.reshape((sqR, sqR)) #R : 2D resolution matrix to convert
    xflux = f_gpu.get()
    A = A_dense.get() #want the reshaped version
    iCov = iCov_gpu.get()

    #and now send the data back to the host, assume full_output=True
    results = dict(flux=flux, ivar=ivar, R=R, xflux=xflux, A=A, iCov=iCov)
    results['options'] = dict(
        specmin=specmin, nspec=nspec, wavelengths=wavelengths,
        xyrange=xyrange, regularize=regularize, ndecorr=ndecorr
        )
    return results
