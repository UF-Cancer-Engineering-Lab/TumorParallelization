import json
import numpy as np
from typing import List, TypedDict


class Location(TypedDict):
    x: int
    y: int
    z: int


class Scene_JSON_Data(TypedDict):
    boundaries: List[Location]


class Scene:
    def __init__(self, json_data):
        self.json_data: Scene_JSON_Data = json_data

    def get_boundaries_numpy(self):
        return np.array(
            [
                np.array(
                    [
                        np.int32(location["x"]),
                        np.int32(location["y"]),
                        np.int32(location["z"]),
                    ]
                )
                for location in self.json_data["boundaries"]
            ],
            dtype=np.int32,
        )


def load_scene(scene_path: str) -> Scene:
    # Remove .json extension if present
    scene_path = "./scenes/" + scene_path.replace(".json", "") + ".json"
    json_data = json.load(open(scene_path))
    return Scene(json_data)


def write_particles_to_scene(filename, boundary):
    scene_path = "./scenes/" + filename.replace(".json", "") + ".json"
    dict = {
        "boundaries": [
            {"x": int(particle[0]), "y": int(particle[1]), "z": int(particle[2])}
            for particle in boundary
        ]
    }

    json_obj = json.dumps(dict)
    with open(scene_path, "w") as outfile:
        outfile.write(json_obj)
