import numpy as np
from config import *


class TreeNode:
    def __init__(self, boundPosition, boundRange):
        self.particlePosition = None
        self.boundPosition = boundPosition
        self.boundRange = boundRange
        self.particleCount = 0
        self.children = []

    def particleInBounds(self, position):
        posX = position[0]
        posY = position[1]
        posZ = position[2]
        boundX = self.boundPosition[0]
        boundY = self.boundPosition[1]
        boundZ = self.boundPosition[2]
        boundXMax = boundX + self.boundRange
        boundYMax = boundY + self.boundRange
        boundZMax = boundZ + self.boundRange
        return (
            posX >= boundX
            and posY >= boundY
            and posZ >= boundZ
            and posX < boundXMax
            and posY < boundYMax
            and posZ < boundZMax
        )

    def subdivide(self):
        curBoundX = self.boundPosition[0]
        curBoundY = self.boundPosition[1]
        curBoundZ = self.boundPosition[2]
        centerX = curBoundX + (self.boundRange / 2)
        centerY = curBoundY + (self.boundRange / 2)
        centerZ = curBoundZ + (self.boundRange / 2)
        newBoundRange = self.boundRange / 2

        # Convention followed found here: https://commons.wikimedia.org/wiki/Category:Octant_%28geometry%29
        self.children.append(TreeNode([centerX, centerY, centerZ], newBoundRange))
        self.children.append(TreeNode([curBoundX, centerY, centerZ], newBoundRange))
        self.children.append(TreeNode([curBoundX, curBoundY, centerZ], newBoundRange))
        self.children.append(TreeNode([centerX, curBoundY, centerZ], newBoundRange))
        self.children.append(TreeNode([centerX, centerY, curBoundZ], newBoundRange))
        self.children.append(TreeNode([curBoundX, centerY, curBoundZ], newBoundRange))
        self.children.append(TreeNode([curBoundX, curBoundY, curBoundZ], newBoundRange))
        self.children.append(TreeNode([centerX, curBoundY, curBoundZ], newBoundRange))

    def insert(self, position):
        if not self.particleInBounds(position):
            return False

        # Don't add duplicates to tree
        if (position == self.particlePosition).all():
            return False

        # No subdivison, free to insert here. EMPTY LEAF
        if self.particlePosition is None:
            self.particlePosition = position
            self.particleCount += 1
            return True

        else:
            # Handle if we have to subdivide. Also insert current particle deeper down tree.
            # NON-EMPTY LEAF
            if len(self.children) == 0:
                self.subdivide()
                for child in self.children:
                    if child.insert(self.particlePosition) == True:
                        break
            # Insert the particle into the children
            for child in self.children:
                if child.insert(position) == True:
                    self.particleCount += 1
                    return True
            return False  # Duplicate recieved from child node


# particles is expected to be a DF
def buildTreeCPU(
    particleArr=np.array([], dtype=np.int32),
    boundRange=np.float32((1 + sphereRadius + n) * 2),
):
    boundStart = [-boundRange / 2, -boundRange / 2, -boundRange / 2]
    root = TreeNode(boundStart, boundRange)
    for particlePos in particleArr:
        root.insert(particlePos)
    return root
