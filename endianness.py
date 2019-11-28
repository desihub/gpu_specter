import numpy as np
import cupy as cp

# Load arrays, coeff_cpu is not in native endianness
L_cpu = np.load('L.npy')
coeff_cpu = np.load('coeff.npy')
coeff_cpu_byteswapped = coeff_cpu.byteswap().newbyteorder() # Swap endianness
L_gpu = cp.array(L_cpu)
coeff_gpu = cp.array(coeff_cpu)
coeff_gpu_byteswapped = cp.array(coeff_cpu_byteswapped)

# Print endianness
print('coeff_cpu.dtype.isnative:', coeff_cpu.dtype.isnative) # False
print('coeff_cpu_byteswapped.dtype.isnative', coeff_cpu_byteswapped.dtype.isnative) # True

# Dot product
# CPU
result_cpu = L_cpu.dot(coeff_cpu)
result_cpu_byteswapped = L_cpu.dot(coeff_cpu_byteswapped)
# GPU
result_gpu = L_gpu.dot(coeff_gpu)
result_gpu_byteswapped = L_gpu.dot(coeff_gpu_byteswapped)

# Compare
# After byte-swapping, the dot products on the cpu remain the same
assert np.allclose(result_cpu, result_cpu_byteswapped)
# The dot product on the cpu and the dot product of the byteswapped version on the gpu is also the same
assert np.allclose(result_cpu, result_gpu_byteswapped.get())
# Without byteswapping, the gpu result disagrees with the cpu
assert np.allclose(result_cpu, result_gpu.get())

