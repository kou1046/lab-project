from django.db import models

from ..base.uuid_model import UUIDModel


class InferenceModel(UUIDModel):
    class Meta:
        db_table = "inference_model"

    name = models.CharField(max_length=50, unique=True)
    label_description = models.CharField(max_length=100)
    model_path = models.CharField(max_length=50, null=True)

    def __str__(self):
        return self.name
