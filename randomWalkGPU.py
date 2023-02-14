from postProcessingGPU import *
from octTreeGPU import *
import time


def walkParticlesGPU(
    initialSphere,
    maxTries=maxTries,
    n=n,
    capillaryRadius=capillaryRadius,
    sphereRadius=sphereRadius,
):
    boundRange = (np.float32)((n + 2 + sphereRadius) * 2)
    squaredRadius = sphereRadius**2
    squaredCapillaryRadius = capillaryRadius**2

    numParticles = np.shape(initialSphere)[0]
    # TODO: Estimate tree size accurately
    GPUBufferSizeNodes = 1000000  # estimateTreeSizeFromLeafCount(numParticles)
    treeBuffer = makeGPUTreeBuffer(GPUBufferSizeNodes)
    treeBufferSize = cuda.device_array(1, dtype=np.int32)
    initialSphereGPU = cuda.to_device(initialSphere)
    latestParticlesGPU = cuda.to_device(initialSphere)
    MLD_Buffer = cuda.device_array(n, dtype=np.float32)

    nthreadsX = 32
    nblocksXClear = (GPUBufferSizeNodes // nthreadsX) + 1
    nblocksXBuild = (numParticles // nthreadsX) + 1
    nblocksXRead = nblocksXClear
    nblocksXInitMLD = nblocksXDivide = (n // nthreadsX) + 1
    nblocksXSumMLD = (numParticles // nthreadsX) + 1

    particles = [initialSphere]

    INIT_MLD_BUFFER[nblocksXInitMLD, nthreadsX](MLD_Buffer)
    for i in range(1, n + 1):
        i % 100 == 0 and print(i)
        clearTree[nblocksXClear, nthreadsX](treeBuffer, treeBufferSize)
        buildTree[nblocksXBuild, nthreadsX](
            treeBuffer,
            treeBufferSize,
            latestParticlesGPU,
            boundRange,
            maxTries,
            True,
            create_xoroshiro128p_states(nblocksXBuild * nthreadsX, seed=time.time_ns()),
            squaredRadius,
            squaredCapillaryRadius,
        )
        readTree[nblocksXRead, nthreadsX](treeBuffer, latestParticlesGPU)
        # MLD_CUDA_SUM[nblocksXSumMLD, nthreadsX](
        #     MLD_Buffer, initialSphereGPU, latestParticlesGPU, i
        # )
        # print(MLD_Buffer.copy_to_host())
        latestParticlesData = getBufferFromGPU(latestParticlesGPU)
        particles.append(latestParticlesData)
    # MLD_CUDA_DIVIDE_ALL[nblocksXDivide, nthreadsX](MLD_Buffer, numParticles, n)
    MLD = MLD_Buffer.copy_to_host().tolist()
    return (particles, MLD)
