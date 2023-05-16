import numpy as np
import sys
import os
import cuda_kernels

parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_folder)
from octTreeCPU import TreeNode, buildTreeCPU


##################
# Agreements Utils
##################
def buildTreeGPU(inputParticles, boundRange):
    return cuda_kernels.walk_particles_gpu(
        inputParticles, [], 1, boundRange, return_gpu_tree_buffer=True
    )


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

    # Validate all locks are unlocked
    if lock != -1:
        print("Found a locked node! Check for race conditions...")
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

    print("Invalid State Detected.")
    return False


def testAgreementCPU_GPU(inputParticles):
    boundRange = (
        np.float32(np.max(np.absolute(inputParticles)) + 1) * 2.0
        if len(inputParticles)
        else np.float32(1.0)
    )
    root = buildTreeCPU(inputParticles, boundRange)
    gpuBuffer = buildTreeGPU(inputParticles, boundRange)
    return testAgreementCPU_GPU_Helper(gpuBuffer, root, 0)


##################
# Validation Utils
##################
