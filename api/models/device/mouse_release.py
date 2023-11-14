from django.db import models

from ..base.uuid_model import UUIDModel
from ..deepsort_openpose.frame import Frame


class MouseRelease(UUIDModel):
    class Meta:
        db_table = "release"

    time = models.TimeField()
    x = models.IntegerField()
    y = models.IntegerField()
    frame = models.OneToOneField(Frame, models.CASCADE, null=True)
