import cuda_kernels
import numpy as np
import math
import random

# print(dir(cuda_kernels))
# cuda_kernels.haroon_print()


# Walk particles will expect a flattened array (each particle is 3 int). This is contiguous and easier to work with on cuda side. Memcpy is faster too.
arr = cuda_kernels.walk_particles_gpu(
    np.array([[1, 2, 3], [4, 5, 6], [-1, -1, 1]], dtype=np.int32),
    np.array([], dtype=np.int32),
    5,
    np.float32(0.0),
    6,
)
print(arr)
