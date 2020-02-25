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

import numba
import cupy as cp
import cupyx as cpx
import cupyx.scipy.special
from numba import cuda

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


def cache_spots(nx, ny, nspec, nwave, p, wavelengths):
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

    #need nx and ny for cache_spots
    nx = p['HSIZEX']
    ny = p['HSIZEY']

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
    cp.cuda.nvtx.RangePush('cache_spots')
    spots = cache_spots(nx, ny, nspec, nwave, p, wavelengths)
    cp.cuda.nvtx.RangePop()

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

        #print("wlo", wlo)
        #print("whi", whi)

        ##for easier bookkeeping put cache_spots here, but eventually move higher    
        #spots = cache_spots(nspec, nwave, p, wavelengths)

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

        #print("len(ww)", len(ww))
        #print("iwave", iwave)

        cp.cuda.nvtx.RangePush('ex2d_patch')
        results = \
            ex2d_patch(subimg, subivar, p, psfdata, spots, corners, iwave, tws,
                specmin=speclo, nspec=spechi-speclo, wavelengths=ww,
                xyrange=[xlo,xhi,ylo,yhi], regularize=regularize, ndecorr=ndecorr,
                full_output=True, use_cache=True) 
        cp.cuda.nvtx.RangePop()  

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

        #print("wavesize", wavesize)
        #print("specflux.shape", specflux.shape)
        #print("nw", nw)

        #flux[iispec[keep], iwave:iwave+wavesize+1] = specflux[keep, nlo:-nhi]
        #ivar[iispec[keep], iwave:iwave+wavesize+1] = specivar[keep, nlo:-nhi]
        #wavediff = nw - wavesize
        #nl = wavediff//2
        #nh = wavediff//2 + wavesize + 1
        #print("wavediff", wavediff)
        #print("nl", nl)
        #print("nh", nh)
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
    #A_cpu = scipy.sparse.csr_matrix(A_dense)
    A = cpx.scipy.sparse.csr_matrix(A_dense)

    #- Pixel weights matrix
    #w = ivar.ravel()
    w = cp.asarray(ivar.ravel())

    #- Set up the equation to solve (B&S eq 4)
    #get the cpu values too
    #noisyimg_cpu = noisyimg_gpu.get()
    #imgweights_cpu = imgweights_gpu.get()
    #A_cpu = A_gpu.get()

    #print("ivar.shape", ivar.shape)
    #W = scipy.sparse.spdiags(data=ivar.ravel(), diags=[0,], m=npix, n=npix) #scipy sparse object
    W = cpx.scipy.sparse.spdiags(data=w, diags=[0,], m=npix, n=npix)
    #yank gpu back to cpu so we can compare
    #W_yank = W_gpu.get()
    #assert np.allclose(W_cpu.todense(), W_yank.todense()) #todense bc this is a sparse object
    #passes

    ####################################################################################
    #patch specter cleanup in here 
    #this is all on the cpu for right now, needs to be converted to gpu!

    #-----
    #- Extend A with an optional regularization term to limit ringing.
    #- If any flux bins don't contribute to these pixels,
    #- also use this term to constrain those flux bins to 0.
    
    #- Original: exclude flux bins with 0 pixels contributing
    # ibad = (A.sum(axis=0).A == 0)[0]
    
    #- Identify fluxes with very low weights of pixels contributing  
    
    #fluxweight = W.dot(A).sum(axis=0).A[0]
    #this syntax does not work in cupy
    #what is this line of code even doing
    #print(type(A))
    #print(type(W))

    fluxweight = W.dot(A).sum(axis=0)[0] #is this right?

    # The following minweight is a regularization term needed to avoid ringing due to 
    # a flux bias on the edge flux bins in the
    # divide and conquer approach when the PSF is not perfect
    # (the edge flux bins are constrained only by a few CCD pixels and the wings of the PSF).
    # The drawback is that this is biasing at the high flux limit because bright pixels
    # have a relatively low weight due to the Poisson noise.
    # we set this weight to a value of 1-e4 = ratio of readnoise**2 to Poisson variance for 1e5 electrons 
    # 1e5 electrons/pixel is the CCD full well, and 10 is about the read noise variance.
    # This was verified on the DESI first spectrograph data.
    #minweight = 1.e-4*np.max(fluxweight)
    minweight = 1.e-4*cp.max(fluxweight) 
    ibad = fluxweight < minweight
    
    #- Original version; doesn't work on older versions of scipy
    # I = regularize*scipy.sparse.identity(nspec*nwave)
    # I.data[0,ibad] = minweight - fluxweight[ibad]
    
    #- Add regularization of low weight fluxes
    #Idiag = regularize*np.ones(nspec*nwave)
    Idiag = 0.0*cp.ones(nspec*nwave)
    Idiag[ibad] = minweight - fluxweight[ibad] #probably have the dimensions of fluxweight wrong...
    #I = scipy.sparse.identity(nspec*nwave)
    #I = cpx.scipy.sparse.identity(nspec*nwave)
    #I.setdiag(Idiag) doesn't exist in cupy
    I = cpx.scipy.sparse.diags(Idiag) #is this equivalent?

###    #- Only need to extend A if regularization is non-zero
###    if np.any(I.diagonal()):
###        pix = np.concatenate( (image.ravel(), np.zeros(nspec*nwave)) )
###        Ax = scipy.sparse.vstack( (A, I) )
###        wx = np.concatenate( (w, np.ones(nspec*nwave)) )
###    else:
###        pix = image.ravel()
###        Ax = A
###        wx = w

    #- Only need to extend A if regularization is non-zero
    if cp.any(I.diagonal()):
        pix = cp.concatenate( (image.ravel(), cp.zeros(nspec*nwave)) )
        Ax = cpx.scipy.sparse.vstack( (A, I) )
        wx = cp.concatenate( (w, cp.ones(nspec*nwave)) )
    else:
        pix = image.ravel()
        Ax = A
        wx = w

    ####################################################################################
    #we now return to our regularly scheduled gpu extraction

    #for now move Ax to the gpu while projection_matrix is still on the cpu
    #and also wx and pix
    Ax_gpu = cpx.scipy.sparse.csr_matrix(Ax)
    wx_gpu = cp.asarray(wx) #better, asarray does not copy
    pix_gpu = cp.asarray(pix)

    #make our new and improved wx using specter cleanup
    Wx_gpu = cpx.scipy.sparse.spdiags(wx_gpu, 0, len(wx_gpu), len(wx_gpu))
    #Wx_cpu = scipy.sparse.spdiags(wx, 0, len(wx), len(wx))
    #Wx = Wx_cpu

    iCov_gpu = Ax_gpu.T.dot(Wx_gpu.dot(Ax_gpu))
    #iCov_cpu = Ax.T.dot(Wx.dot(Ax))
    #yank gpu back to cpu so we can compare
    #iCov_yank = iCov_gpu.get()
    #assert np.allclose(iCov_cpu.todense(), iCov_yank.todense()) #todense bc this is sparse
    #passes

    y_gpu = Ax_gpu.T.dot(Wx_gpu.dot(pix_gpu))
    #y_cpu = Ax.T.dot(Wx.dot(pix))
    #yank gpu back and compare
    #y_yank = y_gpu.get()
    #assert np.allclose(y_cpu, y_yank)
    #passes

    #using instead of spsolve (not currently on the gpu)
    #try again with np.solve and cp.solve
    f_gpu = cp.linalg.solve(iCov_gpu.todense(), y_gpu).reshape((nspec, nwave)) #requires array, not sparse object
    #f_cpu = np.linalg.solve(iCov_cpu.todense(), y_cpu).reshape((nspec, nwave)) #requires array, not sparse object
    #yank back and compare
    #f_yank = f_gpu.get()
    #assert np.allclose(f_cpu, f_yank)
    #passes

    #numpy and scipy don't agree!
    #assert np.allclose(f_cpu, f_cpu_sp)

    #- Eigen-decompose iCov to assist in upcoming steps
    u_gpu, v_gpu = cp.linalg.eigh(iCov_gpu.todense())
    #u, v = np.linalg.eigh(iCov_cpu.todense())
    #u_cpu = np.asarray(u)
    #v_cpu = np.asarray(v)
    #yank back and compare
    #u_yank = u_gpu.get()
    #v_yank = v_gpu.get()
    #assert np.allclose(u_cpu, u_yank)
    #assert np.allclose(v_cpu, v_yank)
    #passes

    #- Calculate C^-1 = QQ (B&S eq 10)
    d_gpu = cpx.scipy.sparse.spdiags(cp.sqrt(u_gpu), 0, len(u_gpu) , len(u_gpu))
    #d_cpu = scipy.sparse.spdiags(np.sqrt(u_cpu), 0, len(u_cpu), len(u_cpu))
    #yank back and compare
    #d_yank = d_gpu.get()
    #assert np.allclose(d_cpu.todense(), d_yank.todense())
    #passes

    Q_gpu = v_gpu.dot( d_gpu.dot( v_gpu.T ))
    #Q_cpu = v_cpu.dot( d_cpu.dot( v_cpu.T ))
    #yank back and compare
    #Q_yank = Q_gpu.get()
    #assert np.allclose(Q_cpu, Q_yank)
    #passes

    #- normalization vector (B&S eq 11)
    norm_vector_gpu = cp.sum(Q_gpu, axis=1)
    #norm_vector_cpu = np.sum(Q_cpu, axis=1)
    #yank back and compare
    #norm_vector_yank = norm_vector_gpu.get()
    #assert np.allclose(norm_vector_cpu, norm_vector_yank)
    #passes

    #- Resolution matrix (B&S eq 12)
    R_gpu = cp.outer(norm_vector_gpu**(-1), cp.ones(norm_vector_gpu.size)) * Q_gpu
    #R_cpu = np.outer(norm_vector_cpu**(-1), np.ones(norm_vector_cpu.size)) * Q_cpu
    #yank back and compare
    #R_yank = R_gpu.get()
    #assert np.allclose(R_cpu, R_yank)
    #passes

    #- Decorrelated covariance matrix (B&S eq 13-15)
    udiags_gpu = cpx.scipy.sparse.spdiags(1/u_gpu, 0, len(u_gpu), len(u_gpu))
    #udiags_cpu = scipy.sparse.spdiags(1/u_cpu, 0, len(u_cpu), len(u_cpu))
    #yank back and compare
    #udiags_yank = udiags_gpu.get()
    #assert np.allclose(udiags_cpu.todense(),udiags_yank.todense()) #sparse objects
    #passes

    Cov_gpu = v_gpu.dot( udiags_gpu.dot (v_gpu.T ))
    #Cov_cpu = v_cpu.dot( udiags_cpu.dot( v_cpu.T ))
    #yank back and compare
    #Cov_yank = Cov_gpu.get()
    #assert np.allclose(Cov_cpu, Cov_yank)
    #passes

    #OOM here when cusparse tries to allocate memory
    Cx_gpu = R_gpu.dot(Cov_gpu.dot(R_gpu.T))
    #Cx_cpu = R_cpu.dot(Cov_cpu.dot(R_cpu.T))
    #yank back and compare
    #Cx_yank = Cx_gpu.get()
    #assert np.allclose(Cx_cpu, Cx_yank)
    #passes

    #- Decorrelated flux (B&S eq 16)
    fx_gpu = R_gpu.dot(f_gpu.ravel()).reshape(f_gpu.shape)
    #fx_cpu = R_cpu.dot(f_cpu.ravel()).reshape(f_cpu.shape)
    #yank back and compare
    #fx_yank = fx_gpu.get()
    #assert np.allclose(fx_cpu, fx_yank)
    #passes

    #- Variance on f (B&S eq 13)
    #in specter fluxivar = norm_vector**2 ??
    varfx_gpu = (norm_vector_gpu * 2).reshape((nspec, nwave))
    #varfx_cpu = (norm_vector_cpu * 2).reshape((nspec, nwave)) #let's try it 
    #yank back and compare
    #varfx_yank = varfx_gpu.get()
    #assert np.allclose(varfx_cpu, varfx_yank)
    #passes

    #####pull back to cpu to return to ex2d
    ###flux = fx_cpu
    ###ivar = varfx_cpu
    ###sqR = np.sqrt(R_cpu.size).astype(int)
    ###R = R_cpu.reshape((sqR, sqR)) #R : 2D resolution matrix to convert
    ###xflux = f_cpu
    ####A is on the cpu for now
    ###iCov = iCov_cpu

    ##pull back to cpu to return to ex2d
    flux = fx_gpu.get()
    ivar = varfx_gpu.get()
    sqR = np.sqrt(R_gpu.size).astype(int)
    R = R_gpu.reshape((sqR, sqR)) #R : 2D resolution matrix to convert
    xflux = f_gpu.get()
    A = A_dense.get() #want the reshaped version
    iCov = iCov_gpu.get()

    if full_output:
        results = dict(flux=flux, ivar=ivar, R=R, xflux=xflux, A=A, iCov=iCov)
        results['options'] = dict(
            specmin=specmin, nspec=nspec, wavelengths=wavelengths,
            xyrange=xyrange, regularize=regularize, ndecorr=ndecorr
            )
        return results
    else:
        return flux, ivar, R
