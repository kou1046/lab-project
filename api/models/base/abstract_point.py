from __future__ import annotations

import numpy as np
from django.db import models

from .nonnegative_field import NonNegativeFloatField
from .uuid_model import UUIDModel
from ..deepsort_openpose.group import Group


class AbstractPoint(UUIDModel):
    class Meta:
        abstract = True

    x = NonNegativeFloatField()
    y = NonNegativeFloatField()
    group = models.ForeignKey(Group, models.CASCADE, related_name="+")  # Groupを消去したときに全て関連するデータを消去したいので必要

    def distance_to(self, other: AbstractPoint):
        return np.linalg.norm(np.array([self.x - other.x, self.y - other.y]))

    def angle(self, a_point: AbstractPoint, c_point: AbstractPoint):
        vec_1 = np.array([a_point.x - self.x, c_point.y - c_point.y])
        vec_2 = np.array([c_point.x - self.x, c_point.y - c_point.y])
        return np.arccos((np.dot(vec_1, vec_2)) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))) * 180 / np.pi
