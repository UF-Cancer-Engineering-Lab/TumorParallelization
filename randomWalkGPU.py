from postProcessingGPU import *
from octTreeGPU import *
from scene import Scene
import time


def walkParticlesGPU(
    initialSphere,
    scene:Scene,
    maxTries=maxTries,
    n=n,
    sphereRadius=sphereRadius,
):
    boundRange = (np.float32)((n + 2 + sphereRadius) * 2)

    # Create buffer
    GPUBufferSizeNodes = (int)(max_vram_allocation_gb * 1e9 // NODE_SIZE // 4)
    # GB/(num of bytes per node)
    treeBuffer = makeGPUTreeBuffer(GPUBufferSizeNodes)
    treeBufferSize = cuda.device_array(1, dtype=np.int32)

    nthreadsX = 32
    # Build the static tree data. Will be copied back over into treebuffer each frame
    # immovable_particles_list = scene.get_boundaries_numpy()
    # immovable_particles_list_gpu = cuda.to_device(immovable_particles_list)
    # particle_count = np.shape(immovable_particles_list)[0]
    # nblocksXClear = (GPUBufferSizeNodes // nthreadsX) + 1
    # nblocksXBuild = (particle_count // nthreadsX) + 1
    # nblocksXRead = nblocksXClear
    # clearTree[nblocksXClear, nthreadsX](treeBuffer, treeBufferSize, NODE_SIZE)
    # buildTree[nblocksXBuild, nthreadsX](
    #     treeBuffer,
    #     treeBufferSize,
    #     immovable_particles_list_gpu,
    #     0,  # Immovable Cancer Cell
    #     boundRange,
    #     maxTries,
    #     False,
    #     create_xoroshiro128p_states(
    #         nblocksXBuild * nthreadsX, seed=time.time_ns()
    #     ),  # Unused
    # )
    # Copy this over to another cuda buffer, will be re-used each frame
    # Create gpu buffer to hold static data, then copy current treeBuffer into it
    static_tree_data = cuda.device_array(
        [treeBufferSize[0], 1], dtype=np.int32
    )  # Potentially move to constant memory
    static_tree_data_size = treeBufferSize[0]
    static_tree_data.copy_to_device(treeBuffer[0:static_tree_data_size])
    # TODO: Deallocate immovable_particles_list_gpu

    # Walk Particles
    particles = [initialSphere]
    latestParticlesGPU = cuda.to_device(initialSphere)
    particle_count = np.shape(initialSphere)[0]
    nblocksXClear = (GPUBufferSizeNodes // nthreadsX) + 1
    nblocksXBuild = (particle_count // nthreadsX) + 1
    nblocksXRead = nblocksXClear
    for i in range(1, n + 1):
        i % 100 == 0 and print(i)
        clearTree[nblocksXClear, nthreadsX](
            treeBuffer, treeBufferSize
        )
        # treeBuffer[0:static_tree_data_size].copy_to_device(
        #     static_tree_data[0:static_tree_data_size]
        # )
        buildTree[nblocksXBuild, nthreadsX](
            treeBuffer,
            treeBufferSize,
            latestParticlesGPU,
            1,  # Movable Cancer Cell
            boundRange,
            maxTries,
            True,
            create_xoroshiro128p_states(nblocksXBuild * nthreadsX, seed=time.time_ns()),
        )
        readTree[nblocksXRead, nthreadsX](treeBuffer, latestParticlesGPU)

        latestParticlesData = getBufferFromGPU(latestParticlesGPU)
        particles.append(latestParticlesData)
    return particles
