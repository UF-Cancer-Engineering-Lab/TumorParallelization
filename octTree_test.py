import unittest
from octTreeGPU import *
from octTreeCPU import *

# silence NumbaPerformanceWarning for tests
import warnings
from numba.core.errors import NumbaPerformanceWarning


class TestOctTree(unittest.TestCase):
    def setUp(self):
        # Suppress performance warnings for the small test cases
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

    def clearBufferTest(self, GPUBufferSizeNodes):

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
            particleID = bufferData[i * 6]
            particlePos = bufferData[i * 6 + 1 : i * 6 + 4]
            childIndex = bufferData[i * 6 + 4]
            lock = bufferData[i * 6 + 5]

            # Assertions
            self.assertEqual(childIndex, -1)
            self.assertEqual(lock, -1)

    def test_1_clearBuffer(self):
        for i in range(1000):
            self.clearBufferTest(i)

    def insertParticlesIntoTreeTest(self, particles, GPUBufferSizeNodes=-1):
        if GPUBufferSizeNodes == -1:
            GPUBufferSizeNodes = estimateTreeSizeFromLeafCount(len(particles.index))

        buffer = makeGPUTreeBuffer(GPUBufferSizeNodes)
        bufferSize = cuda.device_array(1, dtype=np.int32)
        latestParticlesGPU = cuda.to_device(particles.to_numpy(dtype=np.int32))
        boundRange = max(particles.max()) + 1

        nthreadsX = 32
        nblocksXClear = (GPUBufferSizeNodes // nthreadsX) + 1
        nblocksXBuild = (len(particles) // nthreadsX) + 1
        nblocksXRead = nblocksXClear

        clearTree[nblocksXClear, nthreadsX](buffer, bufferSize)
        buildTree[nblocksXBuild, nthreadsX](
            buffer, bufferSize, latestParticlesGPU, boundRange, maxTries, False
        )
        bufferData = getBufferFromGPU(latestParticlesGPU)

        # Now compare the read in particle data with cpu version
        particlesArr = particles.to_numpy(dtype=np.int32)
        root = buildTreeCPU(particlesArr, boundRange)
        print("sdg")

    def test_2_noInsertion(self):
        particles = pd.DataFrame(data={"x": [], "y": [], "z": []})
        self.insertParticlesIntoTreeTest(particles)

    def test_3_oneInsertion(self):
        particles = pd.DataFrame(data={"x": [1], "y": [1], "z": [1]})
        self.insertParticlesIntoTreeTest(particles)

    def test_4_twoInsertion(self):
        particles = pd.DataFrame(data={"x": [1, -1], "y": [1, -1], "z": [1, -1]})
        self.insertParticlesIntoTreeTest(particles)

    def test_5_eightParticlesDifferentQuadrants(self):
        particles = pd.DataFrame(
            data={
                "x": [1, -1, -1, 1, 1, -1, -1, 1],
                "y": [1, 1, -1, -1, 1, 1, -1, -1],
                "z": [1, 1, 1, 1, -1, -1, -1, -1],
            }
        )
        self.insertParticlesIntoTreeTest(particles)


if __name__ == "__main__":
    unittest.main()

# allOctants = [
#     pd.DataFrame(
#         data={
#             "x": [1, -1, -1, 1, 1, -1, -1, 1],
#             "y": [1, 1, -1, -1, 1, 1, -1, -1],
#             "z": [1, 1, 1, 1, -1, -1, -1, -1],
#         }
#     )
# ]
