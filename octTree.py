import numba
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
#           -1 if there is no child yet and no particle
#           -2 if this a leaf node (there is a particle but no children)
#           #  pointing to child index in buffer if a non-leaf node of tree
#       int6 is the lock state of the node
#           lock will be -1 if unlocked
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
def clearTree(buffer, bufferSize):
    x = cuda.grid(1)
    nodeCount = len(buffer) / 6

    if x == 0:
        bufferSize[0] = 0

    if x < nodeCount:
        buffer[
            x * 6 + 4
        ] = -1  # indicate node is free to insert in (no children, no current particle)
        buffer[x * 6 + 5] = -1  # reset the locks to -1 to indicate tree is empty


@cuda.jit(device=True)
def getOctantInNextLevel(boundStart, boundRange, particlePos):
    # Determine if the particles position is in O1,O1...O8
    centerX = boundStart[0] + boundRange / 2
    centerY = boundStart[1] + boundRange / 2
    centerZ = boundStart[2] + boundRange / 2

    particleX = particlePos[0]
    particleY = particlePos[1]
    particleZ = particlePos[2]

    # Convention followed found here: https://commons.wikimedia.org/wiki/Category:Octant_%28geometry%29
    if particleX >= centerX:
        if particleY >= centerY:
            if particleZ >= centerZ:
                return 0
            else:
                return 4
        else:
            if particleZ >= centerZ:
                return 3
            else:
                return 7
    else:
        if particleY >= centerY:
            if particleZ >= centerZ:
                return 1
            else:
                return 5
        else:
            if particleZ >= centerZ:
                return 2
            else:
                return 6


@cuda.jit(device=True)
def updateBoundStartFromOffset(boundStart, boundRange, offset):

    centerX = boundStart[0] + boundRange / 2
    centerY = boundStart[1] + boundRange / 2
    centerZ = boundStart[2] + boundRange / 2

    # Convention followed found here: https://commons.wikimedia.org/wiki/Category:Octant_%28geometry%29
    if offset == 0:
        boundStart[0] = centerX
        boundStart[1] = centerY
        boundStart[2] = centerZ
    elif offset == 1:
        boundStart[0] = boundStart[0]
        boundStart[1] = centerY
        boundStart[2] = centerZ
    elif offset == 2:
        boundStart[0] = boundStart[0]
        boundStart[1] = boundStart[1]
        boundStart[2] = centerZ
    elif offset == 3:
        boundStart[0] = centerX
        boundStart[1] = boundStart[1]
        boundStart[2] = centerZ
    elif offset == 4:
        boundStart[0] = centerX
        boundStart[1] = centerY
        boundStart[2] = boundStart[2]
    elif offset == 5:
        boundStart[0] = boundStart[0]
        boundStart[1] = centerY
        boundStart[2] = boundStart[2]
    elif offset == 6:
        boundStart[0] = boundStart[0]
        boundStart[1] = boundStart[1]
        boundStart[2] = boundStart[2]
    elif offset == 7:
        boundStart[0] = centerX
        boundStart[1] = boundStart[1]
        boundStart[2] = boundStart[2]


@cuda.jit(device=True)
def particlesSharePosition(particle1, particle2):
    return (
        particle1[0] == particle2[0]
        and particle1[1] == particle2[1]
        and particle1[2] == particle2[2]
    )


@cuda.jit
def buildTree(buffer, bufferSize, latestParticles, boundRange, maxTries):

    numberOfParticles = len(latestParticles)

    x = cuda.grid(1)
    particleID = x
    if particleID < numberOfParticles:

        # will be used to keep track of position as we traverse down the tree
        insertedNode = False
        currentNodePos = 0
        boundStart = cuda.local.array(shape=3, dtype=numba.int64)
        boundStart[0] = -boundRange / 2
        boundStart[1] = -boundRange / 2
        boundStart[2] = -boundRange / 2
        currentBoundRange = boundRange
        particlePos = latestParticles[particleID]

        numberOfFailedAttempts = 0

        # We will travel down the tree and find where we can insert the particle
        while (
            insertedNode == False
            and currentNodePos < len(buffer)
            and numberOfFailedAttempts < maxTries
        ):

            currNode_CHILD = buffer[currentNodePos + 4]
            currNode_LOCK_ary = buffer[currentNodePos + 5 : currentNodePos + 6]
            nextOctant = getOctantInNextLevel(
                boundStart, currentBoundRange, particlePos
            )

            # If current node is non-leaf then traverse to child in correct octant
            if currNode_CHILD >= 0:
                currentNodePos = currNode_CHILD + nextOctant * 6
                # Also update the boundRange and boundStart for next iteration
                updateBoundStartFromOffset(boundStart, currentBoundRange, nextOctant)
                currentBoundRange /= 2

            # Attempt to acquire a lock to insert at this node of the tree
            elif particleID == cuda.atomic.compare_and_swap(
                currNode_LOCK_ary, -1, particleID
            ):
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

                else:
                    # We need to move this node and the node we wanna insert down the tree
                    # This represents a subdivision of this octant
                    # It is guaranteed both will be moved into the same level of the tree
                    existingParticlePos = latestParticles[buffer[currentNodePos]]

                    # Try again if there is a conflict
                    if particlesSharePosition(particlePos, existingParticlePos):
                        numberOfFailedAttempts += 1
                        boundStart[0] = -boundRange / 2
                        boundStart[1] = -boundRange / 2
                        boundStart[2] = -boundRange / 2
                        currentBoundRange = boundRange
                        currentNodePos = 0  # LEAVES ROOM FOR OPTIMIZATION!!!

                    # Move both particles down the tree
                    else:
                        offsetExisting = -1
                        offsetNew = -1
                        subtreeIndex = currentNodePos

                        keepSubdividing = True
                        while keepSubdividing and subtreeIndex < len(buffer):
                            # Calculate offsets for existing and new offset
                            offsetNew = getOctantInNextLevel(
                                boundStart, currentBoundRange, particlePos
                            )
                            offsetExisting = getOctantInNextLevel(
                                boundStart, currentBoundRange, existingParticlePos
                            )

                            # get the next avaialble index to add nodes in the tree
                            # + 6 because the atomic instruction returns previous value before add
                            nextAvailableIndex = cuda.atomic.add(bufferSize, 0, 6) + 6

                            # Set existing nodes child index to the next level
                            buffer[subtreeIndex + 4] = nextAvailableIndex
                            subtreeIndex = nextAvailableIndex

                            # Subdivide the bounds if another iteration is needed
                            updateBoundStartFromOffset(
                                boundStart, currentBoundRange, nextOctant
                            )
                            currentBoundRange /= 2

                            # Determine if we should subdivide again based on offsets
                            keepSubdividing = offsetExisting == offsetNew

                        # Invalid buffer position. End thread work.
                        if subtreeIndex >= len(buffer):
                            # reset child value as non-leaf
                            buffer[currentNodePos + 4] = -2
                            insertedNode = True
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
                            buffer[
                                subtreeIndex + offsetExisting + 4
                            ] = -2  # Indicate there is a particle here now

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
                            buffer[
                                subtreeIndex + offsetNew + 4
                            ] = -2  # Indicate there is a particle here now

                            insertedNode = True

                buffer[currentNodePos + 5] = -1  # Release lock

            insertedNode = True


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


def walkParticlesGPU(particles):
    numParticles = len(particles[0].index)
    GPUBufferSizeNodes = estimateTreeSizeFromLeafCount(numParticles)
    buffer = makeGPUTreeBuffer(GPUBufferSizeNodes)
    bufferSize = cuda.device_array(1, dtype=np.int32)
    latestParticles = particles[len(particles) - 1].to_numpy()

    boundRange = (np.int64)((n + 2 + sphereRadius) * 2)

    nthreadsX = 32
    nblocksXClear = (GPUBufferSizeNodes // 32) + 1
    nblocksXBuild = (numParticles // nthreadsX) + 1
    nblocksXRead = nblocksXClear

    for i in range(1, n + 1):
        clearTree[nblocksXClear, nthreadsX](buffer, bufferSize)
        buildTree[nblocksXBuild, nthreadsX](
            buffer, bufferSize, latestParticles, boundRange, maxTries
        )
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
