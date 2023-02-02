# Python code for 2D random walk.
# from cmath import sqrt
from cmath import sqrt

# from logging import _Level
import math
import numpy as np
import pandas
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from config import *
from numba import jit
from octTreeCPU import *

# -----------------------------------------plotting stuff: --------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plotCellData(particlesDF):

    num_frames = len(particlesDF)

    fig = go.Figure(
        data=go.Scatter3d(
            x=[],
            y=[],
            z=[],
            marker=go.scatter3d.Marker(size=3, colorscale="Viridis", opacity=0.8),
            opacity=0.8,
            mode="markers",
        ),
    )

    frames = [
        go.Frame(
            data=[
                go.Scatter3d(
                    x=particlesDF[frame]["x"],
                    y=particlesDF[frame]["y"],
                    z=particlesDF[frame]["z"],
                )
            ],
            traces=[0],
            name=f"frame{frame}",
        )
        for frame in range(len(particlesDF))
    ]
    fig.update(frames=frames)

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title="Cancer Simulation Animation",
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    # {
                    #     "args": [[None], frame_args(0)],
                    #     "label": "&#9724;",  # pause symbol
                    #     "method": "animate",
                    # },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    fig.update_scenes(aspectmode="data")

    fig.show()


# ----------------------------------------- Calculate LSD --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plotMLD(MLD):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(0, len(MLD), 1), MLD, "k-")  # black lines, semitransparent alpha=0.1
    plt.show()


def saveMLD(MLD, particles):
    dMSD = pandas.DataFrame(MLD, columns=["MSD (u.u)"])
    dMSD.to_csv("MSD.csv")

    # Save simulation results
    for particleN in range(0, len(particles)):
        particles[particleN].to_csv(str(particleN) + ".csv")

    print("Calculation complete. Printing PNGs")
    maxFrame = len(particles) - 1
    xmin = min(particles[maxFrame]["x"])
    xmax = max(particles[maxFrame]["x"])
    ymin = min(particles[maxFrame]["y"])
    ymax = max(particles[maxFrame]["y"])
    for frameN in range(0, maxFrame):
        maxIP = pandas.DataFrame(columns=["x", "y", "z"])
        for x in range(xmin, xmax, 1):
            for y in range(ymin, ymax, 1):
                maxIP = maxIP.append(
                    {
                        "x": x,
                        "y": y,
                        "z": (
                            (particles[frameN]["x"] == x)
                            & (particles[frameN]["y"] == y)
                        ).sum(),
                    },
                    ignore_index=True,
                )
        z = maxIP.pivot(columns="x", index="y", values="z")
        x = z.columns
        y = z.index
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.contourf(x, y, z, 16)  # , cmap='viridis');
        plt.savefig(str(frameN) + ".png")

    print("PNGs printing complete")
    plt.close("all")


def calculateLinearDistance(particles):
    initialSphere = particles[0]
    MLD = [0]
    sumMSD = 0.0
    for frame in particles:
        for particleN in frame.index:
            sumMSD = (
                sumMSD
                + (frame.iloc[particleN]["x"] - initialSphere.iloc[particleN]["x"]) ** 2
                + (frame.iloc[particleN]["y"] - initialSphere.iloc[particleN]["y"]) ** 2
                + (frame.iloc[particleN]["z"] - initialSphere.iloc[particleN]["z"]) ** 2
            )
        MLD.append(np.int32(math.sqrt(sumMSD)) / len(frame.index))

    return MLD


def calculateLinearDistanceNumpy(particles):
    initialSphere = particles[0]
    MLD = [0]
    sumMSD = 0.0
    for frame in particles:
        for particleN in range(0, frame.shape[0]):
            sumMSD = (
                sumMSD
                + (frame[particleN][0] - initialSphere[particleN][0]) ** 2
                + (frame[particleN][1] - initialSphere[particleN][1]) ** 2
                + (frame[particleN][2] - initialSphere[particleN][2]) ** 2
            )
        MLD.append(np.int32(math.sqrt(sumMSD)) / frame.shape[0])
    return MLD
