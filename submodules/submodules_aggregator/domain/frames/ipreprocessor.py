from __future__ import annotations

from abc import ABCMeta
import numpy as np

from ..boxes import BoundingBox
from ..keypoints import KeyPoint, KeyPointAttr
from ..frames import Frame


class IPreprocessor(metaclass=ABCMeta):
    def __init__(self, base_point: KeyPointAttr):
        self.base_point: KeyPointAttr = base_point

    def preprocess(
        self,
        keypoints: list[KeyPoint],
        boxes: list[BoundingBox],
        img: np.ndarray,
        frame_num: int,
        prev_frame: Frame | None,
    ) -> tuple[list[KeyPoint], list[BoundingBox]]:
        keypoints = [self.preprocess_keypoint(keypoint, img, frame_num, prev_frame) for keypoint in keypoints]
        boxes = [self.preprocess_box(box, img, frame_num, prev_frame) for box in boxes]

        filterd_keypoints = [
            keypoint for keypoint in keypoints if self.filter_keypoint(keypoint, img, frame_num, prev_frame)
        ]
        filterd_boxes = [box for box in boxes if self.filter_box(box, img, frame_num, prev_frame)]

        return filterd_keypoints, filterd_boxes

    def preprocess_keypoint(
        self,
        keypoint: KeyPoint,
        img: np.ndarray,
        frame_num: int,
        prev_frame: Frame | None,
    ) -> KeyPoint:
        return keypoint

    def preprocess_box(
        self,
        box: BoundingBox,
        img: np.ndarray,
        frame_num: int,
        prev_frame: Frame | None,
    ) -> BoundingBox:
        return box

    def filter_keypoint(
        self,
        keypoint: KeyPoint,
        img: np.ndarray,
        frame_num: int,
        prev_frame: Frame | None,
    ) -> bool:
        return True

    def filter_box(
        self,
        box: BoundingBox,
        img: np.ndarray,
        frame_num: int,
        prev_frame: Frame | None,
    ) -> bool:
        return True
