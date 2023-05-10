import time
import unittest
from octTreeGPU import *
from octTreeCPU import *
from randomWalk import getInitialSphereNumpy
from time import sleep

# silence NumbaPerformanceWarning for tests
# import warnings
# from numba.core.errors import NumbaPerformanceWarning


class TestOctTree(unittest.TestCase):
    def setUp(self):
        # Suppress performance warnings for the small test cases
        print("Performing Setup...")
        # warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

    def buildTreeGPU(self, particles):
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

    def testAgreementCPU_GPU_Helper(self, gpuBuffer, currNode: TreeNode, currIndex):
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

    def testAgreementCPU_GPU(self, inputParticles, boundRange):
        boundRange = np.float32((max(inputParticles.abs().max()) + 1) * 2)
        root = buildTreeCPU(inputParticles, boundRange)
        gpuBuffer = self.buildTreeGPU(inputParticles)
        return self.testAgreementCPU_GPU_Helper(gpuBuffer, root, 0)

    def test_01_clearBuffer(self):
        for GPUBufferSizeNodes in range(100):
            buffer = makeGPUTreeBuffer(GPUBufferSizeNodes)
            bufferSize = cuda.device_array(1, dtype=np.int32)

            nthreadsX = 32
            nblocksXClear = (GPUBufferSizeNodes // 32) + 1

            clearTree[nblocksXClear, nthreadsX](buffer, bufferSize)

            # Root node means the first 6 buffer positions are allocated
            self.assertEqual(getBufferFromGPU(bufferSize)[0], 6)
            # Require that all locks and children are reset to -1
            bufferData = getBufferFromGPU(buffer)
            for i in range(GPUBufferSizeNodes):
                # data at buffer position
                childIndex = bufferData[i * 6 + 4]
                lock = bufferData[i * 6 + 5]

                # Assertions
                self.assertEqual(childIndex, -1)
                self.assertEqual(lock, -1)

    def test_02_noInsertion(self):
        particles = np.empty((0, 3)) 
        self.assertTrue(self.testAgreementCPU_GPU(particles))

    def test_03_oneInsertion(self):
        for i in range(10):
            particles = np.array([[1, 1, 1]])  
            self.assertTrue(self.testAgreementCPU_GPU(particles))

    def test_04_twoInsertion(self):
        for i in range(10):
            particles = np.array([[1, 1, 1], [-1, -1, -1]])  
            self.assertTrue(self.testAgreementCPU_GPU(particles))

    def test_05_eightParticlesDifferentQuadrants(self):
        for i in range(10):
            particles = np.array(
                [
                    [1, 1, 1],
                    [-1, 1, 1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, -1],
                    [1, -1, -1],
                ]
            )  
            self.assertTrue(self.testAgreementCPU_GPU(particles))

    def test_06_duplicateInsertion(self):
        for i in range(10):
            particles = np.array([[1, 1, 1], [1, 1, 1]])  
            self.assertTrue(self.testAgreementCPU_GPU(particles))

    def test_07_nestingInTree(self):
        for i in range(10):
            particles = np.array([[1, 1, 1], [2, 2, 2]])  
            self.assertTrue(self.testAgreementCPU_GPU(particles))

    def test_08_variedNesting(self):
        for i in range(10):
            particles = np.array(
                [[1000, 1000, 1000], [1, 1, 1], [2, 2, 2]]
            )  
            self.assertTrue(self.testAgreementCPU_GPU(particles))


if __name__ == "__main__":
    unittest.main()
