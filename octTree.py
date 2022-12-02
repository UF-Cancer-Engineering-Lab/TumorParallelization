from numba import cuda
import numpy as np
import pandas as pd
from randomWalk import getInitialSphere
from config import *

print(cuda.gpus)  # Will error out if there is no compatible gpu

# The buffer is organized as follows:
#   Each "node" is 6 ints
#       int1 is the ID of the particle
#       int2 is the x of the particle
#       int3 is the y of the particle
#       int4 is the z of the particle
#       int5 is the childNode of this particle
#           -1 if there is no child yet (when lock == -2 && childNode == -1 this is a leaf)
#       int6 is the lock state of the node
#           lock will be -1 if this is an open node to insert at
#           lock will be -2 if this is a non-leaf node (okay to continue traversal for insertion)
#           lock == particleID if the node is locked


def estimateTreeSizeFromLeafCount(leafCount):
    treeSize = leafCount
    while leafCount > 0:
        treeSize += leafCount / 8
        leafCount /= 8
    return 2 * round(treeSize)  # *2 for good measure lol


def makeGPUTreeBuffer(numberOfNodes):
    return cuda.device_array(numberOfNodes * 6, dtype=np.int32)


def getTreeBufferFromGPU(cudaBuffer):
    return cudaBuffer.copy_to_host()


@cuda.jit
def clearTree(buffer):
    x = cuda.grid(1)
    nodeCount = len(buffer) / 6
    if x < nodeCount:
        buffer[x * 6 + 5] = -1  # reset the locks to -1 to indicate tree is empty


@cuda.jit
def buildTree(buffer, latestParticles):

    numberOfParticles = len(latestParticles)

    x = cuda.grid(1)
    particleID = x
    if x < numberOfParticles:
        bufferPos = particleID * 6
        buffer[bufferPos] = particleID
        buffer[bufferPos + 1] = latestParticles[particleID][0]
        buffer[bufferPos + 2] = latestParticles[particleID][1]
        buffer[bufferPos + 3] = latestParticles[particleID][2]
        buffer[bufferPos + 4] = -1


@cuda.jit
def readTree(buffer, latestParticles):

    x = cuda.grid(1)
    bufferPos = x * 6
    if bufferPos < len(buffer):
        lock = buffer[bufferPos + 5]
        child = buffer[bufferPos + 4]
        if lock == -2 and child == -1:
            particleID = buffer[bufferPos]
            particleX = buffer[bufferPos + 1]
            particleY = buffer[bufferPos + 2]
            particleZ = buffer[bufferPos + 3]
            latestParticles[particleID][0] = particleX
            latestParticles[particleID][1] = particleY
            latestParticles[particleID][2] = particleZ


def walkParticlesGPU(particles):
    numParticles = len(particles[0].index)
    GPUBufferSizeNodes = estimateTreeSizeFromLeafCount(numParticles)
    buffer = makeGPUTreeBuffer(GPUBufferSizeNodes)
    latestParticles = particles[len(particles) - 1].to_numpy()

    nthreadsX = 32
    nblocksXClear = (GPUBufferSizeNodes // 32) + 1
    nblocksXBuild = (numParticles // nthreadsX) + 1
    nblocksXRead = nblocksXClear

    for i in range(1, n + 1):
        clearTree[nblocksXClear, nthreadsX](buffer)
        buildTree[nblocksXBuild, nthreadsX](buffer, latestParticles)
        readTree[nblocksXRead, nthreadsX](buffer, latestParticles)
        particles.append(
            pd.DataFrame(
                {
                    "x": latestParticles[:, 0],
                    "y": latestParticles[:, 1],
                    "z": latestParticles[:, 2],
                }
            )
        )

    # TESTING AREA
    bufferData = getTreeBufferFromGPU(buffer)
    print("ORIGINAL DATA")
    print(latestParticles)
    print("Setting data on GPU:")
    for i in range(3):
        print("Particle: ", bufferData[i * 6])
        print(
            "Pos: ", bufferData[i * 6 + 1], bufferData[i * 6 + 2], bufferData[i * 6 + 3]
        )
        print("Lock: ", bufferData[i * 6 + 5])


# Now lets actually run the code
initialSphere = getInitialSphere(particlesNumber, porosityFraction, sphereRadius)
particles = [initialSphere]
walkParticlesGPU(particles)
