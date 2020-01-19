import numpy as np
import cupy as cp

# Create example arrays
A_cpu = np.arange(9.0).reshape(3,3)
x_cpu_native = np.arange(3.0)
x_cpu_nonnative = x_cpu_native.byteswap().newbyteorder() # create a copy of x that has non-native endianness
A_gpu = cp.array(A_cpu)
x_gpu_nonnative = cp.array(x_cpu_nonnative)
x_gpu_native = cp.array(x_cpu_native)

# Simply transferring a non-native array to the gpu and back seems to preserver endianness information
# As we'll later see, this is not the case when properforming a dot product on the gpu
assert np.allclose(x_cpu_native, x_gpu_nonnative.get())

# Chenck endianness
# CPU
assert not x_cpu_nonnative.dtype.isnative
assert x_cpu_native.dtype.isnative
# GPU
assert not x_gpu_nonnative.dtype.isnative
assert x_gpu_native.dtype.isnative
# Fetching from GPU to CPU
assert not x_gpu_nonnative.get().dtype.isnative
assert x_gpu_native.get().dtype.isnative

# Dot product
# CPU
Ax_cpu_nonnative = A_cpu.dot(x_cpu_nonnative)
Ax_cpu_native = A_cpu.dot(x_cpu_native)
# GPU
Ax_gpu_nonnative = A_gpu.dot(x_gpu_nonnative)
Ax_gpu_native = A_gpu.dot(x_gpu_native)

# Print variables
print('A_cpu:')
print(A_cpu)
print('x_cpu_native: ', x_cpu_native)
print('Ax_cpu_native: ', Ax_cpu_native)
print('Ax_cpu_nonnative: ', Ax_cpu_nonnative)
print('Ax_gpu_native: ', Ax_gpu_native)
print('Ax_gpu_nonnative: ', Ax_gpu_nonnative)

# Compare
# After byte-swapping, the dot products on the cpu remain the same
assert np.allclose(Ax_cpu_nonnative, Ax_cpu_native)
# The dot product on the cpu and the dot product of the native version on the gpu is also the same
assert np.allclose(Ax_cpu_nonnative, Ax_gpu_native.get())
# However, when the nonnative version is used on the gpu, dot product returns the wrong answer
assert np.allclose(Ax_cpu_nonnative, Ax_gpu_nonnative.get())

