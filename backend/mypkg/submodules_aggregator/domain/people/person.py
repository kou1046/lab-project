from dataclasses import dataclass
from ..keypoints.keypoint import KeyPoint
from ..boxes import BoundingBox


@dataclass(frozen=True)
class Person:
    keypoint: KeyPoint
    box: BoundingBox

    @property
    def id(self):
        return self.box.id
