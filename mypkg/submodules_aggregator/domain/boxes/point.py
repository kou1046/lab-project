from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Point:
    x: float | int
    y: float | int

    def __post_init__(self):
        if self.x < 0:
            object.__setattr__(self, "x", 0)
        if self.y < 0:
            object.__setattr__(self, "y", 0)

    def distance_to(self, other: Point):
        return np.linalg.norm(np.array([self.x - other.x, self.y - other.y]))

    def angle(self, a_point: Point, c_point: Point):
        vec_1 = np.array([a_point.x - self.x, c_point.y - c_point.y])
        vec_2 = np.array([c_point.x - self.x, c_point.y - c_point.y])
        return np.arccos((np.dot(vec_1, vec_2)) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))) * 180 / np.pi


@dataclass(frozen=True)
class ProbabilisticPoint(Point):
    p: float | int = 100.0
