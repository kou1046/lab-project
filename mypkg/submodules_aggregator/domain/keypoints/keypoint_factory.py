from __future__ import annotations
from abc import abstractclassmethod, ABCMeta
import json
import numpy as np 

from .keypoint import KeyPoint
from .probabilistic_point import ProbabilisticPoint

class IKeypointFactory(metaclass=ABCMeta):
    @abstractclassmethod
    def create_instances(self, openpose_json_path: str): ...

class KeypointFactory(IKeypointFactory):
    def _read_json(self, path: str):
        with open(path, "r") as f:
            item = json.load(f)
        return np.array([np.array(person["pose_keypoints_2d"]).reshape(-1, 3) for person in item["people"]])

    def create_instances(self, openpose_json_path: str) -> list[KeyPoint] | None:
        value = self._read_json(openpose_json_path)
        if not len(value):
            return None  
        keypoints: list[KeyPoint] = []
        for row in value:
            points = [ProbabilisticPoint(cell[0], cell[1], cell[2]) for cell in row]
            keypoints.append(KeyPoint(*points))
        return keypoints