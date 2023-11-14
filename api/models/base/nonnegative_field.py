from django.db import models


class NonNegativeFloatField(models.FloatField):
    def to_python(self, value):
        value = super().to_python(value)
        if value is not None and value < 0:
            return 0
        return value
