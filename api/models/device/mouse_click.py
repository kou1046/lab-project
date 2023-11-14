from django.db import models

from ..base.uuid_model import UUIDModel
from ..deepsort_openpose.frame import Frame


class MouseClick(UUIDModel):
    class Meta:
        db_table = "click"

    time = models.TimeField()
    x = models.IntegerField()
    y = models.IntegerField()
    frame = models.OneToOneField(Frame, models.CASCADE, null=True)
