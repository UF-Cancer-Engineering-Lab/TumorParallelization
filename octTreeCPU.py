import numpy as np
import pandas as pd
from randomWalk import getInitialSphere
from randomWalk import plotCellData
from config import *
from typing import Union


class TreeNode:
    def __init__(self, boundPosition: list[int], boundRange: int):
        self.particlePosition = None
        self.boundPosition = boundPosition
        self.boundRange = boundRange
        self.particleCount = 0
        self.children: list[TreeNode] = []

    def particleInBounds(self, position: list[int]):
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
            and posX <= boundXMax
            and posY <= boundYMax
            and posZ <= boundZMax
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

        # No subdivison, free to insert here
        if self.particlePosition is None:
            self.particlePosition = position
            self.particleCount += 1
            return True

        else:
            # Handle if we have to subdivide. Also insert current particle deeper down tree.
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
            print("Failed to insert particle with position: ", position)
            return False


# particles is expected to be a DF
def buildTreeCPU(particles, boundRange):
    particleArr = particles.to_numpy(dtype=np.int32)
    boundStart = [-boundRange / 2, -boundRange / 2, -boundRange / 2]
    root = TreeNode(boundStart, boundRange)
    print("Length of particle arr", len(particleArr))
    for particlePos in particleArr:
        root.insert(particlePos)
    return root
