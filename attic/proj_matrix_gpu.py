#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:17:17 2019

@author: stephey
"""
import os, sys, time
import numpy as np
from astropy.table import Table
import numba
import scipy.special
from numpy.polynomial import hermite_e as He

import numba
import cupy as cp
import cupyx as cpx
from numba import cuda

#import matplotlib.pyplot as plt


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
    wavemin, wavemax = psfdata['WAVEMIN'][0], psfdata['WAVEMAX'][0]
    wx = (wavelengths - wavemin) * (2.0 / (wavemax - wavemin)) - 1.0
    L = np.polynomial.legendre.legvander(wx, psfdata.meta['LEGDEG'])
    
    p = dict(WAVE=wavelengths)
    nparam, nspec, ndeg = psfdata['COEFF'].shape
    nwave = L.shape[0]
    p['GH'] = np.zeros((psfdata.meta['GHDEGX']+1, psfdata.meta['GHDEGY']+1, nspec, nwave))
    for name, coeff in zip(psfdata['PARAM'], psfdata['COEFF']):
        name = name.strip()
        if name.startswith('GH-'):
            i, j = map(int, name.split('-')[1:3])
            p['GH'][i,j] = L.dot(coeff.T).T
        else:
            p[name] = L.dot(coeff.T).T
    
    #- Include some additional keywords that we'll need
    for key in ['HSIZEX', 'HSIZEY', 'GHDEGX', 'GHDEGY']:
        p[key] = psfdata.meta[key]
    
    return p


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
    # print('Spot size (ny,nx) = {},{}'.format(ny, nx))
    # print('nwave = {}'.format(nwave))

    #- x and y edges of bins that span the center of the PSF spot
    xedges = np.repeat(np.arange(nx+1) - nx//2, nwave).reshape(nx+1, nwave)
    yedges = np.repeat(np.arange(ny+1) - ny//2, nwave).reshape(ny+1, nwave)
    
    #- Shift to be relative to the PSF center at 0 and normalize
    #- by the PSF sigma (GHSIGX, GHSIGY)
    #- xedges[nx+1, nwave]
    #- yedges[ny+1, nwave]
    xedges = ((xedges - p['X'][ispec]%1)/p['GHSIGX'][ispec])
    yedges = ((yedges - p['Y'][ispec]%1)/p['GHSIGY'][ispec])    
#     print('xedges.shape = {}'.format(xedges.shape))
#     print('yedges.shape = {}'.format(yedges.shape))

    #- Degree of the Gauss-Hermite polynomials
    ghdegx = p['GHDEGX']
    ghdegy = p['GHDEGY']

    #- Evaluate the Hermite polynomials at the pixel edges
    #- HVx[ghdegx+1, nwave, nx+1]
    #- HVy[ghdegy+1, nwave, ny+1]
    HVx = He.hermevander(xedges, ghdegx).T
    HVy = He.hermevander(yedges, ghdegy).T
    # print('HVx.shape = {}'.format(HVx.shape))
    # print('HVy.shape = {}'.format(HVy.shape))

    #- Evaluate the Gaussians at the pixel edges
    #- Gx[nwave, nx+1]
    #- Gy[nwave, ny+1]
    Gx = np.exp(-0.5*xedges**2).T / np.sqrt(2. * np.pi)   # (nwave, nedges)
    Gy = np.exp(-0.5*yedges**2).T / np.sqrt(2. * np.pi)
    # print('Gx.shape = {}'.format(Gx.shape))
    # print('Gy.shape = {}'.format(Gy.shape))

    #- Combine into Gauss*Hermite
    GHx = HVx * Gx
    GHy = HVy * Gy

    #- Integrate over the pixels using the relationship
    #  Integral{ H_k(x) exp(-0.5 x^2) dx} = -H_{k-1}(x) exp(-0.5 x^2) + const

    #- pGHx[ghdegx+1, nwave, nx]
    #- pGHy[ghdegy+1, nwave, ny]
    pGHx = np.zeros((ghdegx+1, nwave, nx))
    pGHy = np.zeros((ghdegy+1, nwave, ny))
    pGHx[0] = 0.5 * np.diff(scipy.special.erf(xedges/np.sqrt(2.)).T)
    pGHy[0] = 0.5 * np.diff(scipy.special.erf(yedges/np.sqrt(2.)).T)
    pGHx[1:] = GHx[:ghdegx,:,0:nx] - GHx[:ghdegx,:,1:nx+1]
    pGHy[1:] = GHy[:ghdegy,:,0:ny] - GHy[:ghdegy,:,1:ny+1]
    # print('pGHx.shape = {}'.format(pGHx.shape))
    # print('pGHy.shape = {}'.format(pGHy.shape))
    
    return pGHx, pGHy

#have to preallocate spots
@cuda.jit()
def multispot(pGHx, pGHy, ghc, mspots):
    '''
    TODO: Document
    '''
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


@numba.jit(nopython=True)
def projection_matrix(ispec, nspec, iwave, nwave, spots, corners):
    '''
    Create the projection matrix A for p = Af
    
    Args:
        ispec: starting spectrum index
        nspec: number of spectra
        iwave: starting wavelength index
        nwave: number of wavelengths
        spots: 4D array[ispec, iwave, ny, nx] of PSF spots
        corners: (xc,yc) where each is 2D array[ispec,iwave] lower left corner of spot
    
    Returns 4D A[iy, ix, ispec, iwave] projection matrix
    
    Cast to 2D for using with linear algebra:
    
        nypix, nxpix, nspec, nwave = A.shape
        A2D = A.reshape((nypix*nxpix, nspec*nwave))
        pix1D = A2D.dot(flux1D)
    '''
    ny, nx = spots.shape[2:4]
    xc, yc = corners
    xmin = np.min(xc[ispec:ispec+nspec, iwave:iwave+nwave])
    xmax = np.max(xc[ispec:ispec+nspec, iwave:iwave+nwave]) + nx
    ymin = np.min(yc[ispec:ispec+nspec, iwave:iwave+nwave])
    ymax = np.max(yc[ispec:ispec+nspec, iwave:iwave+nwave]) + ny
    # print('ymin, ymax = {}, {}'.format(ymin, ymax))
    A = np.zeros((ymax-ymin,xmax-xmin,nspec,nwave))
    # print('A.shape = {}'.format(A.shape))
    for i in range(nspec):
        for j in range(nwave):
            ixc = xc[ispec+i, iwave+j] - xmin
            iyc = yc[ispec+i, iwave+j] - ymin
            A[iyc:iyc+ny, ixc:ixc+nx, i, j] = spots[ispec+i,iwave+j]
    
    return A, ymin, xmin


#- Read the PSF parameters from a PSF file without using specter
psfdata = Table.read('psf.fits')

#- Generate some fake input data
wavemin, wavemax = 6000., 6050.
wavelengths = np.arange(wavemin, wavemax)
nwave = len(wavelengths)
nspec = 5
influx = np.zeros((nspec, nwave))
for i in range(nspec):
    influx[i, 5*(i+1)] = 100*(i+1)

#first function, contains legvander
p = evalcoeffs(wavelengths, psfdata)

nx = p['HSIZEX']
ny = p['HSIZEY']

xc = np.floor(p['X'] - p['HSIZEX']//2).astype(int)
yc = np.floor(p['Y'] - p['HSIZEY']//2).astype(int)
corners = (xc, yc)

#preallocate
spots = np.zeros((nspec, nwave, ny, nx))
mspots = np.zeros((nwave, ny, nx)) 

#gpu stuff (for v100, total number of threads per multiprocessor = 2048)
#max threads per block is 1024

threads_per_block = 2
blocks_per_grid = 1024

ntrials = 500

for n in range(ntrials):
    for ispec in range(nspec):
        #second function, contains hermvander
        pGHx, pGHy = calc_pgh(ispec, wavelengths,p)
        #solve numba continguous array error
        ghc = p['GH'][:,:,ispec,:]
        ghc_contig = np.ascontiguousarray(ghc)
        #try to handle the HtD and DtH ourselves with CuPy
        pGHx_gpu = cp.asarray(pGHx)
        pGHy_gpu = cp.asarray(pGHy)
        ghc_contig_gpu = cp.asarray(ghc_contig)
        mspots_gpu = cp.asarray(mspots)
        #launch the GPU kernel
        multispot[blocks_per_grid, threads_per_block](pGHx_gpu, pGHy_gpu, ghc_contig_gpu, mspots_gpu)
        #spots[ispec] = multispot(pGHx, pGHy, ghc)
        #convert mspots back to a cpu function
        mspots_cpu = mspots_gpu.get()
        spots[ispec] = mspots_cpu
        #and do it again

#def projection_matrix(ispec, nspec, iwave, nwave, spots, corners):
#last function, parent function to all others
#resides inside of ex2d_patch for now
A, ymin, xmin = projection_matrix(0, nspec, 0, nwave, spots, corners)

nypix, nxpix = A.shape[0:2]
Ax = A.reshape(nypix*nxpix, nspec*nwave)
image = Ax.dot(influx.ravel()).reshape(nypix, nxpix)
#plt.imshow(image)
#save image so we can check if we got the right answer....
np.save('gpu_image.npy', image)

#also save spots
np.save('gpu_spots.npy', spots)



