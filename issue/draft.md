### ndarray.dot produces wrong values

Some numpy arrays have non-native endianess. Numpy uses dtype.isnative to denote the endianness of an object and handles both non-native and native endianness gracefully. However, cupy.ndarray.dot seems to ignore this flag. This leads to incorrect values being calculated on the gpu.

### Conditions
 - CuPy Version          : 6.4.0
 - CUDA Root             : /usr/common/software/cuda/10.1.168
 - CUDA Build Version    : 10010
 - CUDA Driver Version   : 10010
 - CUDA Runtime Version  : 10010
 - cuDNN Build Version   : 7603
 - cuDNN Version         : 7603
 - NCCL Build Version    : 2402
 - NCCL Runtime Version  : 2402

### Code to reproduce
Download these numpy array files and place them in the same folder with the script below: [coeff.npy](https://github.com/ziyaointl/gpu_specter/blob/cupy_legvander/issue/coeff.npy?raw=true) [L.npy](https://github.com/ziyaointl/gpu_specter/blob/cupy_legvander/issue/L.npy?raw=true)

```python
import numpy as np
import cupy as cp

# Load arrays, coeff_cpu is not in native endianness
L_cpu = np.load('L.npy')
coeff_cpu = np.load('coeff.npy')
coeff_cpu_byteswapped = coeff_cpu.byteswap().newbyteorder() # Swap endianness
L_gpu = cp.array(L_cpu)
coeff_gpu = cp.array(coeff_cpu)
coeff_gpu_byteswapped = cp.array(coeff_cpu_byteswapped)

# Simply transferring the array to the gpu and back seems to produce the expected result
# As we'll later see, this is not the case when properforming a dot product on the gpu
assert np.allclose(coeff_cpu, coeff_gpu.get())

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
```

### Error messages, stack traces, or logs
```
coeff_cpu.dtype.isnative: False
coeff_cpu_byteswapped.dtype.isnative True
Traceback (most recent call last):
  File "endianness.py", line 34, in <module>
    assert np.allclose(result_cpu, result_gpu.get())
AssertionError
```

