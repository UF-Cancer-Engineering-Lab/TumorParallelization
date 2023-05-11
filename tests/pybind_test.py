import cuda_kernels
import numpy as np

# print(dir(cuda_kernels))
# cuda_kernels.haroon_print()
arr = cuda_kernels.walk_particles_gpu(np.array([1,2,3], dtype=np.int32),np.array([1,2,3], dtype=np.int32),0,np.float32(0.0), 6)
print(arr)