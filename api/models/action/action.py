from django.db import models

from ..deepsort_openpose.person import Person


class Action(models.Model):
    class Meta:
        db_table = "action"

    person = models.OneToOneField(Person, models.CASCADE, related_name="action")
    is_programming = models.BooleanField()
    is_having_pen = models.BooleanField()
    is_watching_display = models.BooleanField()
