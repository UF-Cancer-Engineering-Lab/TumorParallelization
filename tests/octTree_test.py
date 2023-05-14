import unittest
import numpy as np
import cuda_kernels

from utils import testAgreementCPU_GPU

NODE_SIZE = 8


class TestOctTree(unittest.TestCase):
    def test_01_clearBuffer(self):
        for GPUBufferSizeNodes in range(100):
            bufferData = cuda_kernels.walk_particles_gpu(
                [],
                [],
                1,
                1000,
                run_build_kernel=False,
                return_gpu_tree_buffer=True,
                buffer_size_nodes=GPUBufferSizeNodes,
            )
            for i in range(GPUBufferSizeNodes):
                # data at buffer position
                childIndex = bufferData[i * NODE_SIZE + 4]
                lock = bufferData[i * NODE_SIZE + 5]

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
            particles = np.array([[1000, 1000, 1000], [1, 1, 1], [2, 2, 2]])
            self.assertTrue(testAgreementCPU_GPU(particles))


if __name__ == "__main__":
    unittest.main(verbosity=2)
