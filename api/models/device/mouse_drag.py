import datetime

import numpy as np
from django.db import models

from ..base.uuid_model import UUIDModel
from ..deepsort_openpose.group import Group
from ..deepsort_openpose.person import Person
from .mouse_click import MouseClick
from .mouse_release import MouseRelease


class MouseDrag(UUIDModel):
    class Meta:
        db_table = "drag"
        constraints = [models.UniqueConstraint(fields=["click", "group", "release"], name="unique_drag")]

    click = models.OneToOneField(MouseClick, on_delete=models.CASCADE)
    release = models.OneToOneField(MouseRelease, on_delete=models.CASCADE)
    group = models.ForeignKey(Group, on_delete=models.CASCADE, related_name="drags")
    person = models.OneToOneField(Person, models.CASCADE, null=True)

    @property
    def distance(self) -> float:
        return np.linalg.norm(np.array([self.click.x - self.release.x, self.click.y - self.release.y]))

    @property
    def time(self) -> float:
        d = datetime.datetime.now().date()
        td = datetime.datetime.combine(d, self.release.time) - datetime.datetime.combine(d, self.click.time)
        return td.total_seconds()
