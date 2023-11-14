import base64

import cv2
import numpy as np
from django.db import models

from ..base.uuid_model import UUIDModel
from ..deepsort_openpose.frame import Frame
from ..deepsort_openpose.group import Group
from .mouse_click import MouseClick
from .mouse_pos import MousePos
from .mouse_release import MouseRelease


class Device(UUIDModel):
    class Meta:
        db_table = "device"

    screenshot_path = models.CharField(max_length=100)
    frame = models.OneToOneField(Frame, models.CASCADE)
    group = models.ForeignKey(Group, models.CASCADE, "devices")
    mouse_pos = models.ForeignKey(MousePos, models.PROTECT)
    mouse_click = models.OneToOneField(MouseClick, models.PROTECT, null=True)
    mouse_release = models.OneToOneField(MouseRelease, models.PROTECT, null=True)

    @property
    def screenshot(self) -> np.ndarray:
        return cv2.imread(self.screenshot_path)

    @property
    def screenshot_base64(self):
        ret, dst_data = cv2.imencode(".jpg", self.screenshot)
        return base64.b64encode(dst_data)

    @property
    def drawn_screenshot(self) -> np.ndarray:
        ss = self.screenshot
        color = (0, 0, 0) if self.mouse_click is None and self.mouse_release is None else (0, 0, 255)
        cv2.circle(ss, (self.mouse_pos.x, self.mouse_pos.y), 5, color, 10)
        return ss

    @property
    def drawn_screenshot_base64(self) -> np.ndarray:
        ret, dst_data = cv2.imencode(".jpg", self.drawn_screenshot)
        return base64.b64encode(dst_data)
