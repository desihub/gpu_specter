#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:53:36 2019
minial reproducer for possible cupy bug
@author: stephey
"""

import numpy as np
import specter.psf
import scipy.sparse
import cupy as cp
import cupyx as cpx

#set a random seed to save ourselves future pain!
np.random.seed(1)
cp.random.seed(1) #cupy and numpy random initialization don't seem to agree...

#- Load a PSF model (included here)
psf = specter.psf.load_psf('psf.fits')

#- Generate some fake input data
wavemin, wavemax = 6000., 6050.
wavelengths = np.arange(wavemin, wavemax)
nwave = len(wavelengths)
nspec = 5
influx_cpu = np.zeros((nspec, nwave))
for i in range(nspec):
    influx_cpu[i, 5*(i+1)] = 100*(i+1)
influx_gpu = cp.asarray(influx_cpu)    

#use specter for now, GPUize this later (other notebook)
specrange = (0,nspec)
xyrange = xmin, xmax, ymin, ymax = psf.xyrange(specrange, wavelengths)
#also need this
ny = ymax-ymin
nx = xmax-xmin
npix = ny*nx
readnoise = 3.0

A_cpu = psf.projection_matrix(specrange, wavelengths, xyrange) #scipy sparse object
#gpuizing starts here ------------------------
A_gpu = cpx.scipy.sparse.csr_matrix(A_cpu)

#this comes after we generate A
img_gpu = A_gpu.dot(influx_gpu.ravel()).reshape((ny, nx))

noisyimg_gpu = cp.random.poisson(img_gpu) + cp.random.normal(scale=readnoise, size=(ny,nx))
#random cupy and numpy don't agree ?!?!?!

imgweights_gpu = 1/(readnoise**2 + noisyimg_gpu.clip(0, 1e6))

#########################
def extract(noisyimg_gpu, imgweights_gpu, A_gpu):
    #- Set up the equation to solve (B&S eq 4)
    W_gpu = cpx.scipy.sparse.spdiags(data=imgweights_gpu.ravel(), diags=[0,], m=npix, n=npix)

    iCov_gpu = A_gpu.T.dot(W_gpu.dot(A_gpu))

    y_gpu = A_gpu.T.dot(W_gpu.dot(noisyimg_gpu.ravel()))

    ##- Solve f (B&S eq 4)
    f_gpu = cp.linalg.solve(iCov_gpu.todense(), y_gpu) #requires array, not sparse object

    #- Eigen-decompose iCov to assist in upcoming steps
    u_gpu, v_gpu = cp.linalg.eigh(iCov_gpu.todense())

    #- Calculate C^-1 = QQ (B&S eq 10)
    d_gpu = cpx.scipy.sparse.spdiags(cp.sqrt(u_gpu), 0, len(u_gpu) , len(u_gpu))

    Q_gpu = v_gpu.dot( d_gpu.dot( v_gpu.T ))

    #- normalization vector (B&S eq 11)
    norm_vector_gpu = cp.sum(Q_gpu, axis=1)

    #- Resolution matrix (B&S eq 12)
    R_gpu = cp.outer(norm_vector_gpu**(-1), cp.ones(norm_vector_gpu.size)) * Q_gpu

    #- Decorrelated covariance matrix (B&S eq 13-15)
    udiags_gpu = cpx.scipy.sparse.spdiags(1/u_gpu, 0, len(u_gpu), len(u_gpu))

    Cov_gpu = v_gpu.dot( udiags_gpu.dot (v_gpu.T ))

    Cx_gpu = R_gpu.dot(Cov_gpu.dot(R_gpu.T))

    #- Decorrelated flux (B&S eq 16)
    fx_gpu = R_gpu.dot(f_gpu.ravel()).reshape(f_gpu.shape)

    #- Variance on f (B&S eq 13)
    varfx_gpu = cp.diagonal(Cx_gpu)

    return fx_gpu, varfx_gpu, R_gpu

fx_gpu, varfx_gpu, R_gpu = extract(noisyimg_gpu, imgweights_gpu, A_gpu)


#be careful
fx_cpu = fx_gpu.get()
varfx_cpu = varfx_gpu.get()
R_cpu = R_gpu.get()

np.savez('extract_gpu_out',fx_gpu=fx_cpu, varfx_gpu=varfx_cpu, R_gpu=R_cpu)


