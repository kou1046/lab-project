from django.db import models

from ..deepsort_openpose.person import Person


class Action(models.Model):
    class Meta:
        db_table = "action"

    person = models.OneToOneField(Person, models.CASCADE, related_name="action")
    programming = models.BooleanField()
    using_computer = models.BooleanField()
    watching_display = models.BooleanField()
