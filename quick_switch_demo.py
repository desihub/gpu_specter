#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

###################
# Approach 1
###################

USE_GPU = True # Or from settings import USE_GPU
if USE_GPU:
    import cupy as np
    from desi_kernels_gpu import *
else:
    from desi_kernels_cpu import *
    import numpy as np
    from numpy.polynomial.hermite_e import hermevander
    from numpy.polynomial.legendre import legvander

# desi_kernels_gpu would contain gpu versions of functions like hermevander and multispots and their wrappers.
# We'll probably rename hermevander to hermevander_k and hermevander_wrapper to hermevander.
# Since we can't override name spaces like np, we will also need to change the way we call such functions
# i.e. instead of calling np.polynomial.hermite_e.hermevander after just importing np, we should first
# import the function and then just call hermevander.

# desi_kernels_cpu would contain cpu reference code for kernels that doens't have numpy counterparts

# Then we can write normal np code! (following the one exception on imports detailed above)

# Challenges:
# 1. This scheme assumes that functions inside desi_kernels won't have to call functions defined in this file.
# Otherwise, we get circular dependencies. One solution is to move import statements in desi_kernels to
# the functions that's requiring this module.

def kernel1():
    from quick_switch_demo import something

# Notes:
# 1. This scheme assumes that we will matain a cpu reference version of all custom kernels that's separate from the gpu kernels
# instead of trying to use decorator magic to turn a numba kernel into a function that works both for the cpu and the gpu

##################
# Approach 1b
##################

# At the start of the pipeline, we explicitly define our inputs as cupy arrays
# Then at each function we write, we define `xp = cupy.get_array_module(input)`, and then use xp everywhere
# This seems to be the way recommended by the cupy docs but it has the con that in every function we write we'll need to add this statement
# in the begining. Furthermore, we'll still need a global USE_GPU variable to determine which version of our custom kernels to import.

