from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, ClassVar
from .probabilistic_point import ProbabilisticPoint


KeyPointAttr = Literal[
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
    NAMES: ClassVar[list[str]] = [
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
        return [getattr(self, point_name) for point_name in self.NAMES]

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
