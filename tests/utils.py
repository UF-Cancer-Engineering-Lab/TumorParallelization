import numpy as np
import sys
import os
import cuda_kernels

parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_folder)
from octTreeCPU import TreeNode, buildTreeCPU


def testAgreementCPU_GPU(inputParticles):
    boundRange = (
        np.float32(np.max(np.absolute(inputParticles)) + 1) * 2.0
        if len(inputParticles)
        else np.float32(1.0)
    )
    root = buildTreeCPU(inputParticles, boundRange)
    gpuBuffer = buildTreeGPU(inputParticles, boundRange)
    gpuBuffer.shape[0] <= 32 * 8 and printGPUBuffer(gpuBuffer)
    return isValidOctTree(
        gpuBuffer, 0, [-boundRange / 2, -boundRange / 2, -boundRange / 2], boundRange
    ) and testAgreementCPU_GPU_Helper(gpuBuffer, root, 0)


def buildTreeGPU(inputParticles, boundRange):
    return cuda_kernels.walk_particles_gpu(
        inputParticles,
        [],
        1,
        boundRange,
        random_walk=False,
        return_gpu_tree_buffer=True,
        # tree_buffer_size_nodes=32,
    )


def isValidOctTree(gpuBuffer, currIndex, boundPos, boundRange):
    # The buffer is organized as follows:
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
    #           lock == particleID if the node is locked
    #       int7 is for cell type
    #           0 is immovable cell
    #           1 is movable cancer cell
    #       int8 is reserved for now
    NODE_SIZE = 8

    # Grab data at buffer position
    particleID = gpuBuffer[currIndex]
    particlePos = gpuBuffer[currIndex + 1 : currIndex + 4]
    childIndex = gpuBuffer[currIndex + 4]
    lock = gpuBuffer[currIndex + 5]
    cell_type = gpuBuffer[currIndex + 6]

    # Validate lock state
    if lock >= 0:
        print("Found locked node! Check concurrency!")
        return False

    # If there is no particle here, return true
    if lock == -1 and childIndex == -1:
        return True

    # If there is a particle here, verify it is within bounds
    if lock == -1 and childIndex == -2:
        return particleInBounds(particlePos, boundPos, boundRange)

    # Go through children and verify them
    isValid = True
    for i in range(8):
        newBoundPos = getBoundStartFromOffset(i, boundPos, boundRange)
        childPos = childIndex + NODE_SIZE * i
        isValid = isValid and isValidOctTree(
            gpuBuffer, childPos, newBoundPos, boundRange / 2
        )
    return isValid


def testAgreementCPU_GPU_Helper(gpuBuffer, currNode: TreeNode, currIndex):
    # The buffer is organized as follows:
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
    #           lock == particleID if the node is locked
    #       int7 is for cell type
    #           0 is immovable cell
    #           1 is movable cancer cell
    #       int8 is reserved for now
    NODE_SIZE = 8

    # Grab data at buffer position
    particleID = gpuBuffer[currIndex]
    particlePos = gpuBuffer[currIndex + 1 : currIndex + 4]
    childIndex = gpuBuffer[currIndex + 4]
    lock = gpuBuffer[currIndex + 5]
    cell_type = gpuBuffer[currIndex + 6]

    isLeaf = len(currNode.children) == 0

    # Check leaf state matches
    if (isLeaf and childIndex >= 0) or (not isLeaf and childIndex < 0):
        print(
            "Leaf states do not match!\n Expected: "
            + str(isLeaf)
            + " Found: "
            + str(childIndex)
        )
        return False

    # Leaf but no particle
    if isLeaf and currNode.particlePosition is None:
        return childIndex == -1

    # Leaf and particle
    if isLeaf and currNode.particlePosition is not None:
        particlesAreEqual = (
            particlePos[0] == currNode.particlePosition[0]
            and particlePos[1] == currNode.particlePosition[1]
            and particlePos[2] == currNode.particlePosition[2]
        )

        if not particlesAreEqual:
            print(
                "Particles between cpu and gpu do not share values!\nExpected: "
                + str(currNode.particlePosition)
                + " Found: "
                + str(particlePos)
            )

        return particlesAreEqual and childIndex == -2

    # If this is a non-leaf. Validate children
    if not isLeaf and childIndex >= 0:
        isValid = True
        for i in range(8):
            # i represents the octant according to https://commons.wikimedia.org/wiki/Category:Octant_%28geometry%29
            nextNode = currNode.children[i]
            nextIndex = childIndex + NODE_SIZE * i
            isValid = isValid and testAgreementCPU_GPU_Helper(
                gpuBuffer, nextNode, nextIndex
            )
        return isValid

    print("Tree Incorrect. Cause Unkown. Double check tree is in valid state.")
    return False


def particleInBounds(particlePos, boundPos, boundRange):
    boundMaxX = boundPos[0] + boundRange
    boundMaxY = boundPos[1] + boundRange
    boundMaxZ = boundPos[2] + boundRange

    inBounds = (
        particlePos[0] >= boundPos[0]
        and particlePos[1] >= boundPos[1]
        and particlePos[2] >= boundPos[2]
        and boundPos[0] < boundMaxX
        and boundPos[1] < boundMaxY
        and boundPos[2] < boundMaxZ
    )

    if not inBounds:
        print(
            "Particle with position: ",
            particlePos,
            "is not contained within bounds",
            boundPos,
            boundRange,
        )

    return inBounds


def getBoundStartFromOffset(offset, boundPos, boundRange):
    centerX = boundPos[0] + boundRange / 2
    centerY = boundPos[1] + boundRange / 2
    centerZ = boundPos[2] + boundRange / 2

    newBound = [0, 0, 0]  # will be udpated down below

    # Convention followed found here: https://commons.wikimedia.org/wiki/Category:Octant_%28geometry%29
    if offset == 0:
        newBound[0] = centerX
        newBound[1] = centerY
        newBound[2] = centerZ
    elif offset == 1:
        newBound[0] = boundPos[0]
        newBound[1] = centerY
        newBound[2] = centerZ
    elif offset == 2:
        newBound[0] = boundPos[0]
        newBound[1] = boundPos[1]
        newBound[2] = centerZ
    elif offset == 3:
        newBound[0] = centerX
        newBound[1] = boundPos[1]
        newBound[2] = centerZ
    elif offset == 4:
        newBound[0] = centerX
        newBound[1] = centerY
        newBound[2] = boundPos[2]
    elif offset == 5:
        newBound[0] = boundPos[0]
        newBound[1] = centerY
        newBound[2] = boundPos[2]
    elif offset == 6:
        newBound[0] = boundPos[0]
        newBound[1] = boundPos[1]
        newBound[2] = boundPos[2]
    elif offset == 7:
        newBound[0] = centerX
        newBound[1] = boundPos[1]
        newBound[2] = boundPos[2]
    return newBound


def printGPUBuffer(buffer):
    print()
    for i in range(len(buffer) // 8):
        for j in range(8):
            print(buffer[i * 8 + j], end=" ")
        print()
