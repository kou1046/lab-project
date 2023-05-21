from __future__ import annotations

from dataclasses import dataclass

import cv2
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


@dataclass(frozen=True)
class BoundingBox:
    id: int
    min: Point
    max: Point

    def center(self) -> Point:
        center_x = (self.max.x - self.min.x) / 2.0 + float(self.min.x)
        center_y = (self.max.y - self.min.y) / 2.0 + float(self.min.y)
        return Point(center_x, center_y)

    def contains(self, point: Point) -> bool:
        return point.x >= self.min.x and point.x <= self.max.x and point.y >= self.min.y and point.y <= self.max.y


POINT_NAMES = [
    "nose",
    "neck",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "midhip",
    "r_hip",
    "r_knee",
    "r_ankle",
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_eye",
    "l_eye",
    "r_ear",
    "l_ear",
    "l_bigtoe",
    "l_smalltoe",
    "l_heel",
    "r_bigtoe",
    "r_smalltoe",
    "r_hell",
]


@dataclass(frozen=True)
class KeyPoint:
    nose: ProbabilisticPoint
    neck: ProbabilisticPoint
    r_shoulder: ProbabilisticPoint
    r_elbow: ProbabilisticPoint
    r_wrist: ProbabilisticPoint
    l_shoulder: ProbabilisticPoint
    l_elbow: ProbabilisticPoint
    l_wrist: ProbabilisticPoint
    midhip: ProbabilisticPoint
    r_hip: ProbabilisticPoint
    r_knee: ProbabilisticPoint
    r_ankle: ProbabilisticPoint
    l_hip: ProbabilisticPoint
    l_knee: ProbabilisticPoint
    l_ankle: ProbabilisticPoint
    r_eye: ProbabilisticPoint
    l_eye: ProbabilisticPoint
    r_ear: ProbabilisticPoint
    l_ear: ProbabilisticPoint
    l_bigtoe: ProbabilisticPoint
    l_smalltoe: ProbabilisticPoint
    l_heel: ProbabilisticPoint
    r_bigtoe: ProbabilisticPoint
    r_smalltoe: ProbabilisticPoint
    r_hell: ProbabilisticPoint

    def get_all_points(self) -> list[ProbabilisticPoint]:
        return [getattr(self, point_name) for point_name in POINT_NAMES]

    @property
    def face_angle(self) -> float | None:
        return self.neck.angle(self.r_eye, self.l_eye)

    @property
    def neck_angle(self) -> float | None:
        return self.neck.angle(self.nose, self.midhip)

    @property
    def r_elbow_angle(self) -> float | None:
        return self.r_elbow.angle(self.r_shoulder, self.r_wrist)

    @property
    def l_elbow_angle(self) -> float | None:
        return self.l_elbow.angle(self.l_shoulder, self.l_wrist)

    @property
    def r_shoulder_angle(self) -> float | None:
        return self.r_shoulder.angle(self.neck, self.r_elbow)

    @property
    def l_shoulder_angle(self) -> float | None:
        return self.l_shoulder.angle(self.neck, self.l_elbow)

    @property
    def shoulder_dis(self) -> float | None:
        return self.r_shoulder.distance_to(self.l_shoulder)

    @property
    def eye_dis(self) -> float | None:
        return self.r_eye.distance_to(self.l_eye)

    @property
    def elbow_dis(self) -> float | None:
        return self.r_elbow.distance_to(self.l_elbow)


@dataclass(frozen=True)
class Person:
    keypoint: KeyPoint
    box: BoundingBox

    @property
    def id(self):
        return self.box.id


@dataclass(frozen=True)
class Group:
    name: str


@dataclass(frozen=True)
class CombinedFrame:
    group: Group
    people: list[Person]
    img_path: str
    number: int

    @property
    def img(self):
        return cv2.imread(self.img_path)

    def contains_person(self):
        return any(self.people)

    def person_ids(self) -> set[int]:
        return {person.id for person in self.people}

    def visualize(
        self,
        draw_keypoints: bool = False,
        color: tuple[int, int, int] = (0, 0, 0),
        point_radius: int = 5,
        thickness: int = 3,
    ) -> np.ndarray:
        img = self.img().copy()
        for person in self.people:
            cv2.putText(
                img,
                f"ID:{person.box.id}",
                (person.box.min.x, person.box.min.y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                thickness,
            )
            cv2.rectangle(
                img,
                (person.box.min.x, person.box.min.y),
                (person.box.max.x, person.box.max.y),
                color,
                thickness,
            )
            if draw_keypoints:
                for point in person.keypoints.get_all_points():
                    cv2.circle(
                        img,
                        (int(point.x), int(point.y)),
                        point_radius,
                        color,
                        thickness,
                    )
        return img
