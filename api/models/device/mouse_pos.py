from django.db import models

from ..base.uuid_model import UUIDModel
from ..deepsort_openpose.group import Group


class MousePos(UUIDModel):
    class Meta:
        db_table = "mouse_position"
        constraints = [models.UniqueConstraint(fields=["group", "time"], name="unique_mouse_position")]

    group = models.ForeignKey(Group, models.CASCADE, related_query_name="mouse_postions")
    time = models.TimeField()
    x = models.IntegerField()
    y = models.IntegerField()
