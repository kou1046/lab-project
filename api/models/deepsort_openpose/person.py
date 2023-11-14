from __future__ import annotations

import base64

import cv2
import numpy as np
from django.db import models

from ..base.uuid_model import UUIDModel
from .bounding_box import BoundingBox
from .frame import Frame
from .group import Group
from .keypoint import KeyPoint


class Person(UUIDModel):
    class Meta:
        db_table = "person"

    keypoint = models.OneToOneField(KeyPoint, models.CASCADE, related_name="person")
    box = models.OneToOneField(BoundingBox, models.CASCADE, related_name="person")
    frame = models.ForeignKey(Frame, models.CASCADE, related_name="people")
    group = models.ForeignKey(Group, models.CASCADE, related_name="people")

    @property
    def img(self) -> np.ndarray:
        frame_img = self.frame.img
        screen_height, screen_width, _ = frame_img.shape
        img = frame_img[
            int(self.box.min.y) : (int(self.box.max.y) if self.box.max.y <= screen_height else screen_height),
            int(self.box.min.x) : (int(self.box.max.x) if self.box.max.x <= screen_width else screen_width),
        ]
        return img

    @property
    def img_base64(self):
        ret, dst_data = cv2.imencode(".jpg", self.img)
        return base64.b64encode(dst_data)

    def visualize_screen_img(
        self,
        draw_keypoints: bool = False,
        color: tuple[int, int, int] = (0, 0, 0),
        point_radius: int = 5,
        thickness: int = 3,
        isbase64=False,
    ) -> np.ndarray:
        img_copy = self.frame.img
        cv2.rectangle(
            img_copy,
            (self.box.min.x, self.box.min.y),
            (self.box.max.x, self.box.max.y),
            color,
            thickness,
        )
        if draw_keypoints:
            for point in self.keypoint.get_all_points():
                cv2.circle(
                    img_copy,
                    (int(point.x), int(point.y)),
                    point_radius,
                    color,
                    thickness,
                )
        if isbase64:
            ret, dst_data = cv2.imencode(".jpg", img_copy)
            return base64.b64encode(dst_data)
        return img_copy

    def visualize(
        self,
        color: tuple[int, int, int] = (0, 0, 0),
        point_radius: int = 5,
        thickness: int = 3,
        isbase64=False,
    ) -> np.ndarray:
        img_copy = self.img.copy()
        for point in self.keypoint.get_all_points():
            cv2.circle(
                img_copy,
                (
                    int(point.x) - int(self.box.min.x),
                    int(point.y) - int(self.box.min.y),
                ),
                point_radius,
                color,
                thickness,
            )
        if isbase64:
            ret, dst_data = cv2.imencode(".jpg", img_copy)
            return base64.b64encode(dst_data)
        return img_copy
