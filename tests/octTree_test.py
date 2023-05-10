import time
import unittest

import numpy as np
from octTreeGPU import clearTree, getBufferFromGPU, makeGPUTreeBuffer
from tests.util.agreements_util import testAgreementCPU_GPU

# silence NumbaPerformanceWarning for tests
# import warnings
# from numba.core.errors import NumbaPerformanceWarning


class TestOctTree(unittest.TestCase):
    def setUp(self):
        # Suppress performance warnings for the small test cases
        print("Performing Setup...")
        # warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

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
        self.assertTrue(testAgreementCPU_GPU(particles))

    def test_03_oneInsertion(self):
        for i in range(10):
            particles = np.array([[1, 1, 1]])  
            self.assertTrue(testAgreementCPU_GPU(particles))

    def test_04_twoInsertion(self):
        for i in range(10):
            particles = np.array([[1, 1, 1], [-1, -1, -1]])  
            self.assertTrue(testAgreementCPU_GPU(particles))

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
            self.assertTrue(testAgreementCPU_GPU(particles))

    def test_06_duplicateInsertion(self):
        for i in range(10):
            particles = np.array([[1, 1, 1], [1, 1, 1]])  
            self.assertTrue(testAgreementCPU_GPU(particles))

    def test_07_nestingInTree(self):
        for i in range(10):
            particles = np.array([[1, 1, 1], [2, 2, 2]])  
            self.assertTrue(testAgreementCPU_GPU(particles))

    def test_08_variedNesting(self):
        for i in range(10):
            particles = np.array(
                [[1000, 1000, 1000], [1, 1, 1], [2, 2, 2]]
            )  
            self.assertTrue(testAgreementCPU_GPU(particles))


if __name__ == "__main__":
    unittest.main()
