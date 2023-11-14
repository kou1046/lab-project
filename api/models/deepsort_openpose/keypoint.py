from __future__ import annotations

from django.db import models

from ..base.uuid_model import UUIDModel
from .probabilistic_point import ProbabilisticPoint

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


class KeyPoint(UUIDModel):
    class Meta:
        db_table = "keypoint"
        default_related_name = "+"

    nose = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    neck = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_shoulder = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_elbow = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_wrist = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_shoulder = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_elbow = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_wrist = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    midhip = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_hip = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_knee = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_ankle = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_hip = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_knee = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_ankle = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_eye = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_eye = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_ear = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_ear = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_bigtoe = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_smalltoe = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_heel = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_bigtoe = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_smalltoe = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_hell = models.OneToOneField(ProbabilisticPoint, models.CASCADE)

    def get_all_points(self) -> list[ProbabilisticPoint]:
        return [getattr(self, name) for name in POINT_NAMES]

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
