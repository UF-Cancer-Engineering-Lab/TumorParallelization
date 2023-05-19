import unittest
import numpy as np
import cuda_kernels

from utils import testAgreementCPU_GPU
from utils import printGPUBuffer
from utils import NODE_SIZE


class TestOctTree(unittest.TestCase):
    def test_01_clearBuffer(self):
        for GPUBufferSizeNodes in range(100):
            bufferData = cuda_kernels.walk_particles_gpu(
                [],
                [],
                1,
                1000,
                random_walk=False,
                return_gpu_tree_buffer=True,
                tree_buffer_size_nodes=GPUBufferSizeNodes,
            )
            for i in range(GPUBufferSizeNodes):
                # data at buffer position
                childIndex = bufferData[i * NODE_SIZE + 4]
                lock = bufferData[i * NODE_SIZE + 5]

                # Assertions
                self.assertEqual(childIndex, -1)
                self.assertEqual(lock, -1)

    def test_02_readTree(self):
        for data_size in range(100):
            original_data = np.random.randint(
                0, data_size, size=(data_size, 3), dtype=np.int32
            )
            processed_data = cuda_kernels.walk_particles_gpu(
                original_data,
                [],
                1,
                data_size * 2,
                random_walk=False,
            )
            # print("\nFound:")
            # print(processed_data)
            # print("Expected:")
            # print(original_data)
            # print("Buffer:")
            # printGPUBuffer(buffer_data)
            self.assertTrue((original_data == processed_data[0]).all())

    def test_03_noInsertion(self):
        # for i in range(100):
        particles = np.empty((0, 3))
        self.assertTrue(testAgreementCPU_GPU(particles))

    def test_04_oneInsertion(self):
        # for i in range(100):
        particles = np.array([[1, 1, 1]])
        self.assertTrue(testAgreementCPU_GPU(particles))

    def test_05_twoInsertion(self):
        # for i in range(100):
        particles = np.array([[1, 1, 1], [-1, -1, -1]])
        self.assertTrue(testAgreementCPU_GPU(particles))

    def test_06_eightParticlesDifferentQuadrants(self):
        # for i in range(100):
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

    def test_07_duplicateInsertion(self):
        # for i in range(100):
        particles = np.array([[1, 1, 1], [1, 1, 1]])
        self.assertTrue(testAgreementCPU_GPU(particles))

    def test_08_nestingInTree(self):
        # for i in range(100):
        particles = np.array([[1, 1, 1], [2, 2, 2]])
        self.assertTrue(testAgreementCPU_GPU(particles))

    def test_09_variedNesting(self):
        # for i in range(100):
        particles = np.array([[1000, 1000, 1000], [1, 1, 1], [2, 2, 2]])
        self.assertTrue(testAgreementCPU_GPU(particles))


if __name__ == "__main__":
    unittest.main(verbosity=2)
