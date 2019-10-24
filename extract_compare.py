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
    #get the cpu values too
    noisyimg_cpu = noisyimg_gpu.get()
    imgweights_cpu = imgweights_gpu.get()
    A_cpu = A_gpu.get()

    W_cpu = scipy.sparse.spdiags(data=imgweights_cpu.ravel(), diags=[0,], m=npix, n=npix) #scipy sparse object
    W_gpu = cpx.scipy.sparse.spdiags(data=imgweights_gpu.ravel(), diags=[0,], m=npix, n=npix)
    #yank gpu back to cpu so we can compare
    W_yank = W_gpu.get()
    assert np.allclose(W_cpu.todense(), W_yank.todense()) #todense bc this is a sparse object
    #passes

    iCov_gpu = A_gpu.T.dot(W_gpu.dot(A_gpu))
    iCov_cpu = A_cpu.T.dot(W_cpu.dot(A_cpu))
    #yank gpu back to cpu so we can compare
    iCov_yank = iCov_gpu.get()
    assert np.allclose(iCov_cpu.todense(), iCov_yank.todense()) #todense bc this is sparse
    #passes

    y_gpu = A_gpu.T.dot(W_gpu.dot(noisyimg_gpu.ravel()))
    y_cpu = A_cpu.T.dot(W_cpu.dot(noisyimg_cpu.ravel()))
    #yank gpu back and compare
    y_yank = y_gpu.get()
    assert np.allclose(y_cpu, y_yank)
    #passes

    ##- Solve f (B&S eq 4)
    #f_gpu_tup = cpx.scipy.sparse.linalg.lsqr(iCov_gpu, y_gpu)
    ##returns f_gpu as a tuple... need to reshape?
    #f_gpu_0 = f_gpu_tup[0] #take only zeroth element of tuple, rest are None for some reason
    #f_gpu = cp.asarray(f_gpu_0).reshape(nspec, nwave) #the tuple thing is dumb bc i think it goes back to the cpu, have to manually bring it back as a cupy array
    #f_cpu_0 = scipy.sparse.linalg.lsqr(iCov_cpu, y_cpu)[0] #need to take 0th element of tuple
    #f_cpu = np.asarray(f_cpu_0).reshape(nspec, nwave) #and then reshape, make less confusing in separate step
    ##yank back and compare
    #f_yank = f_gpu.get()
    #assert np.allclose(f_cpu, f_yank)
    ##fails! #maybe lsqr is the problem
    ##what was the other one?
    ##i think instead we want numpy solve and cupy solve
    ##that one at least passed our tests...

    #try again with np.solve and cp.solve
    #cp.linalg.solve
    f_gpu = cp.linalg.solve(iCov_gpu.todense(), y_gpu) #requires array, not sparse object
    f_cpu = np.linalg.solve(iCov_cpu.todense(), y_cpu) #requires array, not sparse object
    #yank back and compare
    f_yank = f_gpu.get()
    assert np.allclose(f_cpu, f_yank)
    #passes

    #- Eigen-decompose iCov to assist in upcoming steps
    u_gpu, v_gpu = cp.linalg.eigh(iCov_gpu.todense())
    u, v = np.linalg.eigh(iCov_cpu.todense())
    u_cpu = np.asarray(u)
    v_cpu = np.asarray(v)
    #yank back and compare
    u_yank = u_gpu.get()
    v_yank = v_gpu.get()
    assert np.allclose(u_cpu, u_yank)
    assert np.allclose(v_cpu, v_yank)
    #passes

    #- Calculate C^-1 = QQ (B&S eq 10)
    d_gpu = cpx.scipy.sparse.spdiags(cp.sqrt(u_gpu), 0, len(u_gpu) , len(u_gpu))
    d_cpu = scipy.sparse.spdiags(np.sqrt(u_cpu), 0, len(u_cpu), len(u_cpu))
    #yank back and compare
    d_yank = d_gpu.get()
    assert np.allclose(d_cpu.todense(), d_yank.todense())
    #passes

    Q_gpu = v_gpu.dot( d_gpu.dot( v_gpu.T ))
    Q_cpu = v_cpu.dot( d_cpu.dot( v_cpu.T ))
    #yank back and compare
    Q_yank = Q_gpu.get()
    assert np.allclose(Q_cpu, Q_yank)
    #passes

    #- normalization vector (B&S eq 11)
    norm_vector_gpu = cp.sum(Q_gpu, axis=1)
    norm_vector_cpu = np.sum(Q_cpu, axis=1)
    #yank back and compare
    norm_vector_yank = norm_vector_gpu.get()
    assert np.allclose(norm_vector_cpu, norm_vector_yank)
    #passes

    #- Resolution matrix (B&S eq 12)
    R_gpu = cp.outer(norm_vector_gpu**(-1), cp.ones(norm_vector_gpu.size)) * Q_gpu
    R_cpu = np.outer(norm_vector_cpu**(-1), np.ones(norm_vector_cpu.size)) * Q_cpu
    #yank back and compare
    R_yank = R_gpu.get()
    assert np.allclose(R_cpu, R_yank)
    #passes

    #- Decorrelated covariance matrix (B&S eq 13-15)
    udiags_gpu = cpx.scipy.sparse.spdiags(1/u_gpu, 0, len(u_gpu), len(u_gpu))
    udiags_cpu = scipy.sparse.spdiags(1/u_cpu, 0, len(u_cpu), len(u_cpu))
    #yank back and compare
    udiags_yank = udiags_gpu.get()
    assert np.allclose(udiags_cpu.todense(),udiags_yank.todense()) #sparse objects
    #passes

    Cov_gpu = v_gpu.dot( udiags_gpu.dot (v_gpu.T ))
    Cov_cpu = v_cpu.dot( udiags_cpu.dot( v_cpu.T ))
    #yank back and compare
    Cov_yank = Cov_gpu.get()
    assert np.allclose(Cov_cpu, Cov_yank)
    #passes

    Cx_gpu = R_gpu.dot(Cov_gpu.dot(R_gpu.T))
    Cx_cpu = R_cpu.dot(Cov_cpu.dot(R_cpu.T))
    #yank back and compare
    Cx_yank = Cx_gpu.get()
    assert np.allclose(Cx_cpu, Cx_yank)
    #passes

    #- Decorrelated flux (B&S eq 16)
    fx_gpu = R_gpu.dot(f_gpu.ravel()).reshape(f_gpu.shape)
    fx_cpu = R_cpu.dot(f_cpu.ravel()).reshape(f_cpu.shape)
    #yank back and compare
    fx_yank = fx_gpu.get()
    assert np.allclose(fx_cpu, fx_yank)
    #passes

    #- Variance on f (B&S eq 13)
    varfx_gpu = cp.diagonal(Cx_gpu)
    varfx_cpu = np.diagonal(Cx_cpu)
    #yank back and compare
    varfx_yank = varfx_gpu.get()
    assert np.allclose(varfx_cpu, varfx_yank)
    #passes

    return fx_gpu, fx_cpu, varfx_gpu, varfx_cpu, R_gpu, R_cpu

fx_gpu, fx_cpu, varfx_gpu, varfx_cpu, R_gpu, R_cpu = extract(noisyimg_gpu, imgweights_gpu, A_gpu)

#first save the cpu data
np.savez('extract_cpu_out',fx_cpu=fx_cpu, varfx_cpu=varfx_cpu, R_cpu=R_cpu)

#be careful
fx_cpu = fx_gpu.get()
varfx_cpu = varfx_gpu.get()
R_cpu = R_gpu.get()

np.savez('extract_gpu_out',fx_gpu=fx_cpu, varfx_gpu=varfx_cpu, R_gpu=R_cpu)


