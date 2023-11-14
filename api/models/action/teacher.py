from django.db import models

from ..deepsort_openpose.person import Person
from .inference_model import InferenceModel


class Teacher(models.Model):
    class Meta:
        db_table = "teacher"
        constraints = [models.UniqueConstraint(fields=["person", "model"], name="unique_teacher")]

    person = models.ForeignKey(Person, models.CASCADE, related_name="teachers")
    label = models.IntegerField(default=0)
    model = models.ForeignKey(InferenceModel, on_delete=models.CASCADE, related_name="teachers")
