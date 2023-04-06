import json
import numpy as np
from typing import List, TypedDict


# TODO: Add initialsphere to the scene (no need having it separate)
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
                np.array([np.int32(location["x"]), np.int32(location["y"]), np.int32(location["z"])])
                for location in self.json_data["boundaries"]
            ]
        , dtype=np.int32)


def load_scene(scene_path: str) -> Scene:
    # Remove .json extension if present
    scene_path = "./scenes/" + scene_path.replace(".json", "") + ".json"
    json_data = json.load(open(scene_path))
    return Scene(json_data)