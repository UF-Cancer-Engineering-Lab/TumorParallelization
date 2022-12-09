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

    def compareCPUOctTreeLeaves(self, root1: TreeNode, root2: TreeNode):

        # If leaf, compare
        isLeaf1 = len(root1.children) == 0
        isLeaf2 = len(root2.children) == 0

        if isLeaf1 != isLeaf2:
            return False

        if isLeaf1:
            if root1.particlePosition is None:
                return root2.particlePosition is None
            isSamePosition = (root1.particlePosition == root2.particlePosition).all()
            return isSamePosition

        # Go through children and compare
        isValid = True
        for i in range(8):
            isValid = isValid and self.compareCPUOctTreeLeaves(
                root1.children[i], root2.children[i]
            )
            if not isValid:
                print("WTH")
        return isValid

    def test_00_isOctTreeDeterministic(self):
        for i in range(1, 10):
            particles = getInitialSphere(sphereRadius=i, particlesNumber=i * i * i)
            particlesArr = particles.to_numpy(dtype=np.int32)
            shuffled = particlesArr.copy()
            np.random.shuffle(shuffled)
            boundRange = (max(particles.abs().max()) + 1) * 2
            root1 = buildTreeCPU(particlesArr, boundRange)
            root2 = buildTreeCPU(particlesArr, boundRange)
            validationResult = self.compareCPUOctTreeLeaves(root1, root2)
            self.assertTrue(validationResult)

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

    def test_01_clearBuffer(self):
        for i in range(100):
            self.clearBufferTest(i)

    def validateGPUResultHelper(self, gpuResult, currNode: TreeNode, currIndex):

        # data at buffer position
        particleID = gpuResult[currIndex]
        particlePos = gpuResult[currIndex + 1 : currIndex + 4]
        childIndex = gpuResult[currIndex + 4]
        lock = gpuResult[currIndex + 5]

        isLeaf = len(currNode.children) == 0

        # Validate both nodes agree whether they are a leaf or not
        if (isLeaf and lock == -2) or (not isLeaf and lock == -1) or (lock >= 0):
            print("Lock state invalid or does not match CPU version.")
            return False

        # If no particle, return True
        if isLeaf and currNode.particlePosition is None:
            return childIndex == -1

        # If there is a particle here and leaf. Ensure positions match
        if isLeaf and (not currNode.particlePosition is None):
            particlesAreEqual = (
                particlePos[0] == currNode.particlePosition[0]
                and particlePos[1] == currNode.particlePosition[1]
                and particlePos[2] == currNode.particlePosition[2]
            )

            if not particlesAreEqual:
                print("Particles between cpu and gpu do not share values!")

            return particlesAreEqual and childIndex == -2

        # If this is a non-leaf. Validate children
        if not isLeaf and childIndex >= 0:
            isValid = True
            for i in range(8):
                # i represents the octant according to https://commons.wikimedia.org/wiki/Category:Octant_%28geometry%29
                nextNode = currNode.children[i]
                nextIndex = childIndex + 6 * i
                isValid = isValid and self.validateGPUResultHelper(
                    gpuResult, nextNode, nextIndex
                )
            return isValid

        print("Invalid Algorithm State Detected.")
        return False

    def validateGPUResult(self, gpuResult, inputParticles, boundRange):
        root = buildTreeCPU(inputParticles, boundRange)
        return self.validateGPUResultHelper(gpuResult, root, 0)

    def insertParticlesIntoTreeTest(self, particles, GPUBufferSizeNodes=-1):
        if GPUBufferSizeNodes == -1:
            GPUBufferSizeNodes = 10000000  # estimateTreeSizeFromLeafCount(len(particles.index)) staying on the safe side with large buffer

        buffer = makeGPUTreeBuffer(GPUBufferSizeNodes)
        bufferSize = cuda.device_array(1, dtype=np.int32)
        latestParticlesGPU = cuda.to_device(particles.to_numpy(dtype=np.int32))
        boundRange = np.float32((max(particles.abs().max()) + 1) * 2)

        nthreadsX = 32
        nblocksXClear = (GPUBufferSizeNodes // nthreadsX) + 1
        nblocksXBuild = (len(particles) // nthreadsX) + 1
        nblocksXRead = nblocksXClear

        clearTree[nblocksXClear, nthreadsX](buffer, bufferSize)
        buildTree[nblocksXBuild, nthreadsX](
            buffer, bufferSize, latestParticlesGPU, boundRange, maxTries, False
        )
        bufferData = getBufferFromGPU(buffer)

        # Now compare the read in particle data with cpu version
        particlesArr = particles.to_numpy(dtype=np.int32)
        return self.validateGPUResult(bufferData, particlesArr, boundRange)

    def test_02_noInsertion(self):
        particles = pd.DataFrame(data={"x": [], "y": [], "z": []})
        validationResult = self.insertParticlesIntoTreeTest(particles)
        self.assertTrue(validationResult)

    def test_03_oneInsertion(self):
        for i in range(10):
            particles = pd.DataFrame(data={"x": [1], "y": [1], "z": [1]})
            validationResult = self.insertParticlesIntoTreeTest(particles)
            self.assertTrue(validationResult)

    def test_04_twoInsertion(self):
        for i in range(10):
            particles = pd.DataFrame(data={"x": [1, -1], "y": [1, -1], "z": [1, -1]})
            validationResult = self.insertParticlesIntoTreeTest(particles)
            self.assertTrue(validationResult)

    def test_05_eightParticlesDifferentQuadrants(self):
        for i in range(10):
            particles = pd.DataFrame(
                data={
                    "x": [1, -1, -1, 1, 1, -1, -1, 1],
                    "y": [1, 1, -1, -1, 1, 1, -1, -1],
                    "z": [1, 1, 1, 1, -1, -1, -1, -1],
                }
            )
            validationResult = self.insertParticlesIntoTreeTest(particles)
            self.assertTrue(validationResult)

    def test_06_duplicateInsertion(self):
        for i in range(10):
            particles = pd.DataFrame(data={"x": [1, 1], "y": [1, 1], "z": [1, 1]})
            validationResult = self.insertParticlesIntoTreeTest(particles)
            self.assertTrue(validationResult)

    def test_07_nestingInTree(self):
        for i in range(10):
            particles = pd.DataFrame(data={"x": [1, 2], "y": [1, 2], "z": [1, 2]})
            validationResult = self.insertParticlesIntoTreeTest(particles)
            self.assertTrue(validationResult)

    def test_08_deepNesting(self):
        for i in range(10):
            particles = pd.DataFrame(
                data={"x": [1000, 1, 2], "y": [1000, 1, 2], "z": [1000, 1, 2]}
            )
            validationResult = self.insertParticlesIntoTreeTest(particles)
            self.assertTrue(validationResult)

    def particleInBounds(self, particlePos, boundPos, boundRange):
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

    def getBoundStartFromOffset(self, offset, boundPos, boundRange):
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

    def isValidGPUOctTreeHelper(self, bufferData, currIndex, boundPos, boundRange):

        # data at buffer position
        particleID = bufferData[currIndex]
        particlePos = bufferData[currIndex + 1 : currIndex + 4]
        childIndex = bufferData[currIndex + 4]
        lock = bufferData[currIndex + 5]

        # Validate lock state
        if lock >= 0:
            return False

        # Validate child state
        if lock == -2 and childIndex < 0:
            return False

        # If there is no particle here, return true
        if lock == -1 and childIndex == -1:
            return True

        # If there is a particle here, verify it is within bounds
        if lock == -1 and childIndex == -2:
            return self.particleInBounds(particlePos, boundPos, boundRange)

        # Go through children and verify them
        isValid = True
        for i in range(8):
            newBoundPos = self.getBoundStartFromOffset(i, boundPos, boundRange)
            childPos = childIndex + 6 * i
            isValid = isValid and self.isValidGPUOctTreeHelper(
                bufferData, childPos, newBoundPos, boundRange / 2
            )
        return isValid

    def isValidGPUOctTree(self, particles):
        GPUBufferSizeNodes = 10000000  # estimateTreeSizeFromLeafCount(len(particles.index)) staying on the safe side with large buffer

        buffer = makeGPUTreeBuffer(GPUBufferSizeNodes)
        bufferSize = cuda.device_array(1, dtype=np.int32)
        latestParticlesGPU = cuda.to_device(particles.to_numpy(dtype=np.int32))
        boundRange = np.float32((max(particles.abs().max()) + 1) * 2)

        nthreadsX = 32
        nblocksXClear = (GPUBufferSizeNodes // nthreadsX) + 1
        nblocksXBuild = (len(particles) // nthreadsX) + 1
        nblocksXRead = nblocksXClear

        clearTree[nblocksXClear, nthreadsX](buffer, bufferSize)
        buildTree[nblocksXBuild, nthreadsX](
            buffer, bufferSize, latestParticlesGPU, boundRange, maxTries, False
        )
        bufferData = getBufferFromGPU(buffer)

        leafCount = 0
        for i in range(GPUBufferSizeNodes):
            # Count the number of leaves
            if bufferData[i * 6 + 5] == -1 and bufferData[i * 6 + 4] == -2:
                leafCount += 1

        if leafCount != len(particles):
            print(
                "Leaf count is not equal to number of particles!",
                leafCount,
                len(particles),
            )
            return False

        return self.isValidGPUOctTreeHelper(
            bufferData,
            0,
            [-boundRange / 2, -boundRange / 2, -boundRange / 2],
            boundRange,
        )

    def test_09_smallInitialSphere(self):
        for i in range(10):
            particles = getInitialSphere(sphereRadius=3, particlesNumber=27)
            isValidOctTree = self.isValidGPUOctTree(particles)
            self.assertTrue(isValidOctTree)
            matchesCPUData = self.insertParticlesIntoTreeTest(particles)
            self.assertTrue(matchesCPUData)
            cuda.synchronize()

    def test_10_mediumInitialSphere(self):
        for i in range(10):
            particles = getInitialSphere(sphereRadius=5, particlesNumber=100)
            isValidOctTree = self.isValidGPUOctTree(particles)
            print(len(particles.index))
            self.assertTrue(isValidOctTree)
            # validationResult = self.insertParticlesIntoTreeTest(particles)
            # self.assertTrue(validationResult)
            cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
