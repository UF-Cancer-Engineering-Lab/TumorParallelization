from scene import Scene
from test_util import NODE_SIZE
import cuda_kernels
from config import *
import numpy as np


def walkParticlesGPU(
    initialSphere,
    scene: Scene,
    maxTries=maxTries,
    n=n,
    sphereRadius=sphereRadius,
):
    boundRange = (np.float32)((n + 2 + sphereRadius) * 2)
    immovable_particles_list = scene.get_boundaries_numpy()

    NODE_COUNT = np.int32(max_vram_allocation_gb * 1e9 / NODE_SIZE / 4)

    return cuda_kernels.walk_particles_gpu(
        initialSphere,
        immovable_particles_list,
        n,
        boundRange,
        maxTries,
        tree_buffer_size_nodes=NODE_COUNT,
        recording_interval=recording_interval,
    )
