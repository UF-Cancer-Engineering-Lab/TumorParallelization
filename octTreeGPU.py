import numba
from numba import cuda
import numpy as np
import pandas as pd
from config import *
from numba.cuda import float32x3

# The buffer is organized as follows:
#   Each "node" is 6 ints
#       int1 is the ID of the particle
#       int2 is the x of the particle
#       int3 is the y of the particle
#       int4 is the z of the particle
#       int5 is the childNode of this particle
#           -1 if there is no child yet and no particle
#           -2 if there is a particle but no children
#           #  pointing to child index in buffer if a non-leaf node of tree
#       int6 is the lock state of the node
#           lock will be -1 if unlocked
#           lock will be -2 if non-leaf node
#           lock == particleID if the node is locked


def estimateTreeSizeFromLeafCount(leafCount):
    treeSize = leafCount
    while leafCount > 0:
        treeSize += leafCount / 8
        leafCount /= 8
    return 9 + 2 * round(treeSize)


def makeGPUTreeBuffer(numberOfNodes):
    return cuda.device_array(numberOfNodes * 6, dtype=np.int32)


def getBufferFromGPU(cudaBuffer):
    return cudaBuffer.copy_to_host()


@cuda.jit
def clearTree(buffer, bufferSize):
    x = cuda.grid(1)
    nodeCount = len(buffer) // 6

    if x == 0:
        bufferSize[0] = 6  # accounts for root node not being "subdivided"

    if x < nodeCount:
        buffer[
            x * 6 + 4
        ] = -1  # indicate node is free to insert in (no children, no current particle)
        buffer[x * 6 + 5] = -1  # reset the locks to -1 to indicate tree is empty


@cuda.jit(device=True)
def isInBounds(boundartStart, boundRange, particlePos):
    boundMaxX = boundartStart.x + boundRange
    boundMaxY = boundartStart.y + boundRange
    boundMaxZ = boundartStart.z + boundRange
    inBounds = (
        particlePos[0] >= boundartStart.x
        and particlePos[1] >= boundartStart.y
        and particlePos[2] >= boundartStart.z
        and particlePos[0] < boundMaxX
        and particlePos[1] < boundMaxY
        and particlePos[2] < boundMaxZ
    )

    return inBounds


@cuda.jit(device=True)
def getNextOctant(boundStart, boundRange, particlePos):
    # Determine if the particles position is in O1,O1...O8
    centerX = np.float32(boundStart.x + (boundRange / 2.0))
    centerY = np.float32(boundStart.y + (boundRange / 2.0))
    centerZ = np.float32(boundStart.z + (boundRange / 2.0))

    particleX = np.float32(particlePos[0])
    particleY = np.float32(particlePos[1])
    particleZ = np.float32(particlePos[2])

    octant = 0
    # Convention followed found here: https://commons.wikimedia.org/wiki/Category:Octant_%28geometry%29
    if particleX >= centerX:
        if particleY >= centerY:
            if particleZ >= centerZ:
                octant = 0
            else:
                octant = 4
        else:
            if particleZ >= centerZ:
                octant = 3
            else:
                octant = 7
    else:
        if particleY >= centerY:
            if particleZ >= centerZ:
                octant = 1
            else:
                octant = 5
        else:
            if particleZ >= centerZ:
                octant = 2
            else:
                octant = 6

    return octant


@cuda.jit(device=True)
def getNewBoundStart(boundStart, boundRange, offset):
    centerX = np.float32(boundStart.x + (boundRange / 2.0))
    centerY = np.float32(boundStart.y + (boundRange / 2.0))
    centerZ = np.float32(boundStart.z + (boundRange / 2.0))

    # Convention followed found here: https://commons.wikimedia.org/wiki/Category:Octant_%28geometry%29
    if offset == 0:
        return float32x3(centerX, centerY, centerZ)
    elif offset == 1:
        return float32x3(boundStart.x, centerY, centerZ)
    elif offset == 2:
        return float32x3(boundStart.x, boundStart.y, centerZ)
    elif offset == 3:
        return float32x3(centerX, boundStart.y, centerZ)
    elif offset == 4:
        return float32x3(centerX, centerY, boundStart.z)
    elif offset == 5:
        return float32x3(boundStart.x, centerY, boundStart.z)
    elif offset == 6:
        return float32x3(boundStart.x, boundStart.y, boundStart.z)
    elif offset == 7:
        return float32x3(centerX, boundStart.y, boundStart.z)


@cuda.jit(device=True)
def particlesSharePosition(particle1, particle2):
    return (
        particle1[0] == particle2[0]
        and particle1[1] == particle2[1]
        and particle1[2] == particle2[2]
    )


@cuda.jit
def buildTree(
    buffer, bufferSize, latestParticles, boundRange, maxTries, shouldRandomWalk=True
):

    numberOfParticles = len(latestParticles)

    # print("======KERNEL DEBUG INFO======\n\n\n")

    x = cuda.grid(1)
    particleID = x
    if particleID < numberOfParticles:

        # will be used to keep track of position as we traverse down the tree
        insertedNode = False
        currentNodePos = 0
        boundStart = float32x3(
            -np.float32(boundRange) / 2.0,
            -np.float32(boundRange) / 2.0,
            -np.float32(boundRange) / 2.0,
        )
        # boundStart = cuda.local.array(shape=3, dtype=np.float32)
        # boundStart[0] = -np.float32(boundRange) / 2.0
        # boundStart[1] = -np.float32(boundRange) / 2.0
        # boundStart[2] = -np.float32(boundRange) / 2.0
        currentBoundRange = np.float32(boundRange)
        particlePos = latestParticles[particleID]
        numberOfFailedAttempts = 0

        # We will travel down the tree and find where we can insert the particle
        while (
            insertedNode == False
            and currentNodePos < len(buffer)
            and numberOfFailedAttempts < maxTries
        ):

            currNode_LOCK_ary = buffer[currentNodePos + 5 : currentNodePos + 6]
            nextOctant = getNextOctant(boundStart, currentBoundRange, particlePos)

            # If current node is non-leaf then traverse to child in correct octant
            if currNode_LOCK_ary[0] == -2:
                currNode_CHILD = buffer[currentNodePos + 4]
                currentNodePos = currNode_CHILD + nextOctant * 6
                # Also update the boundRange and boundStart for next iteration
                boundStart = getNewBoundStart(boundStart, currentBoundRange, nextOctant)
                currentBoundRange /= 2

            # Attempt to acquire a lock to insert at this node of the tree
            elif -1 == cuda.atomic.compare_and_swap(currNode_LOCK_ary, -1, particleID):
                cuda.threadfence()

                currNode_CHILD = buffer[currentNodePos + 4]

                if currNode_CHILD == -1:
                    # Free to insert here
                    buffer[currentNodePos] = particleID
                    buffer[currentNodePos + 1] = latestParticles[particleID][0]
                    buffer[currentNodePos + 2] = latestParticles[particleID][1]
                    buffer[currentNodePos + 3] = latestParticles[particleID][2]
                    buffer[
                        currentNodePos + 4
                    ] = -2  # Indicate there is a particle here now
                    insertedNode = True
                    buffer[currentNodePos + 5] = -1  # Release lock

                else:
                    # We need to move this node and the node we wanna insert down the tree
                    # This represents a subdivision of this octant
                    # It is guaranteed both will be moved into the same level of the tree
                    existingParticlePos = latestParticles[buffer[currentNodePos]]

                    # Try again if there is a conflict
                    if particlesSharePosition(particlePos, existingParticlePos):
                        numberOfFailedAttempts += 1
                        currentBoundRange = np.float32(boundRange)
                        boundStart = float32x3(
                            -np.float32(boundRange) / 2.0,
                            -np.float32(boundRange) / 2.0,
                            -np.float32(boundRange) / 2.0,
                        )
                        currentNodePos = 0  # LEAVES ROOM FOR OPTIMIZATION!!!
                        # print("Particles share position. Retrying...", particleID)
                        buffer[currentNodePos + 5] = -1  # Release lock

                    # Move both particles down the tree
                    else:
                        offsetExisting = -1
                        offsetNew = -1
                        subtreeIndex = currentNodePos

                        keepSubdividing = True
                        while keepSubdividing and subtreeIndex < len(buffer):
                            # Calculate offsets for existing and new offset
                            offsetNew = getNextOctant(
                                boundStart, currentBoundRange, particlePos
                            )
                            offsetExisting = getNextOctant(
                                boundStart, currentBoundRange, existingParticlePos
                            )

                            # get the next avaialble index to add nodes in the tree
                            # + 6 because the atomic instruction returns previous value before add
                            childrenSize = 8 * 6
                            nextAvailableIndex = cuda.atomic.add(
                                bufferSize, 0, childrenSize
                            )

                            # Set existing nodes child index to the next level
                            buffer[subtreeIndex + 4] = nextAvailableIndex
                            if (
                                currentNodePos != subtreeIndex
                            ):  # Set as non-leaf. Be careful not to release original lock. This will be done at end of function.
                                buffer[subtreeIndex + 5] = -2

                            # Subdivide the bounds if another iteration is needed
                            boundStart = getNewBoundStart(
                                boundStart, currentBoundRange, offsetNew
                            )
                            currentBoundRange /= 2

                            # Determine if we should subdivide again based on offsets
                            keepSubdividing = offsetExisting == offsetNew
                            subtreeIndex = nextAvailableIndex
                            if keepSubdividing:
                                subtreeIndex += 6 * offsetNew

                        # Invalid buffer position. End thread work.
                        if subtreeIndex >= len(buffer):
                            # reset child value as non-leaf
                            insertedNode = True
                            buffer[currentNodePos + 4] = -2
                            buffer[currentNodePos + 5] = -1  # Release lock

                        else:
                            # Move the current and existing into their offset positions
                            # existing
                            offsetExisting *= 6
                            buffer[subtreeIndex + offsetExisting] = buffer[
                                currentNodePos
                            ]
                            buffer[
                                subtreeIndex + offsetExisting + 1
                            ] = existingParticlePos[0]
                            buffer[
                                subtreeIndex + offsetExisting + 2
                            ] = existingParticlePos[1]
                            buffer[
                                subtreeIndex + offsetExisting + 3
                            ] = existingParticlePos[2]
                            buffer[subtreeIndex + offsetExisting + 4] = -2

                            # current
                            offsetNew *= 6
                            buffer[subtreeIndex + offsetNew] = particleID
                            buffer[subtreeIndex + offsetNew + 1] = latestParticles[
                                particleID
                            ][0]
                            buffer[subtreeIndex + offsetNew + 2] = latestParticles[
                                particleID
                            ][1]
                            buffer[subtreeIndex + offsetNew + 3] = latestParticles[
                                particleID
                            ][2]
                            buffer[subtreeIndex + offsetNew + 4] = -2

                            insertedNode = True
                            buffer[currentNodePos + 5] = -2  # Release lock

                cuda.threadfence()


# Updates latest particles with new data
@cuda.jit
def readTree(buffer, latestParticles):

    x = cuda.grid(1)
    bufferPos = x * 6
    if bufferPos < len(buffer):
        child = buffer[bufferPos + 4]
        if child == -2:
            particleID = buffer[bufferPos]
            particleX = buffer[bufferPos + 1]
            particleY = buffer[bufferPos + 2]
            particleZ = buffer[bufferPos + 3]
            latestParticles[particleID][0] = particleX
            latestParticles[particleID][1] = particleY
            latestParticles[particleID][2] = particleZ


def walkParticlesGPU(
    particles, boundRange=(np.float32)((n + 2 + sphereRadius) * 2), maxTries=maxTries
):
    numParticles = len(particles[0].index)
    GPUBufferSizeNodes = estimateTreeSizeFromLeafCount(numParticles)
    buffer = makeGPUTreeBuffer(GPUBufferSizeNodes)
    bufferSize = cuda.device_array(1, dtype=np.int32)
    latestParticlesGPU = cuda.to_device(
        particles[len(particles) - 1].to_numpy(dtype=np.int32)
    )

    nthreadsX = 32
    nblocksXClear = (GPUBufferSizeNodes // 32) + 1
    nblocksXBuild = (numParticles // nthreadsX) + 1
    nblocksXRead = nblocksXClear

    for i in range(1, n + 1):
        clearTree[nblocksXClear, nthreadsX](buffer, bufferSize)
        buildTree[nblocksXBuild, nthreadsX](
            buffer, bufferSize, latestParticlesGPU, boundRange, maxTries
        )
        readTree[nblocksXRead, nthreadsX](buffer, latestParticlesGPU)
        latestParticlesData = getBufferFromGPU(latestParticlesGPU)
        particles.append(
            pd.DataFrame(
                {
                    "x": latestParticlesData[:, 0],
                    "y": latestParticlesData[:, 1],
                    "z": latestParticlesData[:, 2],
                }
            )
        )
