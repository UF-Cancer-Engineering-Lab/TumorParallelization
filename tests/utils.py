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


def testAgreementCPU_GPU(inputParticles):
    boundRange = np.float32(np.max(np.absolute(inputParticles)) + 1) * 2.0
    root = buildTreeCPU(inputParticles, boundRange)
    gpuBuffer = buildTreeGPU(inputParticles, boundRange)
    return testAgreementCPU_GPU_Helper(gpuBuffer, root, 0)


##################
# Validation Utils
##################
