import base64

import cv2
import numpy as np
from django.db import models

from ..base.uuid_model import UUIDModel
from .group import Group


class Frame(UUIDModel):
    number = models.IntegerField()
    img_path = models.CharField(max_length=100)
    group = models.ForeignKey(Group, models.CASCADE, related_name="frames")
    # people (逆参照)

    class Meta:
        db_table = "frame"
        constraints = [models.UniqueConstraint(fields=["number", "group"], name="group_frame_num_unique")]
        ordering = ("number",)

    @property
    def img(self) -> np.ndarray:
        return cv2.imread(self.img_path)

    @property
    def img_base64(self) -> str:
        ret, dst_data = cv2.imencode(".jpg", self.img)
        return base64.b64encode(dst_data)

    def visualize(
        self,
        draw_keypoint: bool = False,
        color: tuple[int, int, int] = (0, 0, 0),
        point_radius: int = 5,
        thickness: int = 3,
    ) -> np.ndarray:
        img = self.img.copy()
        for person in self.people.all().select_related("keypoint"):
            cv2.putText(
                img,
                f"ID:{person.box.id}",
                (int(person.box.min.x), int(person.box.min.y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                thickness,
            )
            cv2.rectangle(
                img,
                (int(person.box.min.x), int(person.box.min.y)),
                (int(person.box.max.x), int(person.box.max.y)),
                color,
                thickness,
            )
            if draw_keypoint:
                for point in person.keypoint.get_all_points():
                    cv2.circle(
                        img,
                        (int(point.x), int(point.y)),
                        point_radius,
                        color,
                        thickness,
                    )
        return img
