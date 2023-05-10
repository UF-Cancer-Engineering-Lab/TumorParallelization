import numpy as np
from octTreeCPU import TreeNode, buildTreeCPU
from octTreeGPU import clearTree, buildTree, getBufferFromGPU, makeGPUTreeBuffer
from numba import cuda

def buildTreeGPU(particles):
    GPUBufferSizeNodes = 10000000

    buffer = makeGPUTreeBuffer(GPUBufferSizeNodes)
    bufferSize = cuda.device_array(1, dtype=np.int32)
    latestParticlesGPU = cuda.to_device(particles)
    boundRange = np.float32((max(particles.abs().max()) + 1) * 2)

    nthreadsX = 32
    nblocksXClear = (GPUBufferSizeNodes // nthreadsX) + 1
    nblocksXBuild = (len(particles) // nthreadsX) + 1

    clearTree[nblocksXClear, nthreadsX](buffer, bufferSize)
    buildTree[nblocksXBuild, nthreadsX](
        buffer,
        bufferSize,
        latestParticlesGPU,
        boundRange,
        maxTries,
        False,
        None,
        sphereRadius**2,
        capillaryRadius**2,
    )
    bufferData = getBufferFromGPU(buffer)

    return bufferData

def testAgreementCPU_GPU_Helper(gpuBuffer, currNode: TreeNode, currIndex):
    # data at buffer position
    # particleID = gpuBuffer[currIndex]
    # particlePos = gpuBuffer[currIndex + 1 : currIndex + 4]
    # childIndex = gpuBuffer[currIndex + 4]
    # lock = gpuBuffer[currIndex + 5]

    # isLeaf = len(currNode.children) == 0

    # # Validate both nodes agree whether they are a leaf or not
    # if (isLeaf and lock == -2) or (not isLeaf and lock == -1) or (lock >= 0):
    #     print("Lock state invalid or does not match CPU version.")
    #     return False

    # # If no particle, return True
    # if isLeaf and currNode.particlePosition is None:
    #     return childIndex == -1

    # # If there is a particle here and leaf. Ensure positions match
    # if isLeaf and (not currNode.particlePosition is None):
    #     particlesAreEqual = (
    #         particlePos[0] == currNode.particlePosition[0]
    #         and particlePos[1] == currNode.particlePosition[1]
    #         and particlePos[2] == currNode.particlePosition[2]
    #     )

    #     if not particlesAreEqual:
    #         print("Particles between cpu and gpu do not share values!")

    #     return particlesAreEqual and childIndex == -2

    # # If this is a non-leaf. Validate children
    # if not isLeaf and childIndex >= 0:
    #     isValid = True
    #     for i in range(8):
    #         # i represents the octant according to https://commons.wikimedia.org/wiki/Category:Octant_%28geometry%29
    #         nextNode = currNode.children[i]
    #         nextIndex = childIndex + 6 * i
    #         isValid = isValid and self.validateGPUResultHelper(
    #             gpuBuffer, nextNode, nextIndex
    #         )
    #     return isValid

    # print("Invalid Algorithm State Detected.")
    return False

def testAgreementCPU_GPU(inputParticles, boundRange):
    boundRange = np.float32((max(inputParticles.abs().max()) + 1) * 2)
    root = buildTreeCPU(inputParticles, boundRange)
    gpuBuffer = buildTreeGPU(inputParticles)
    return testAgreementCPU_GPU_Helper(gpuBuffer, root, 0)