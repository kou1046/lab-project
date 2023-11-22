from django.db import models


class Group(models.Model):
    class Meta:
        db_table = "group"

    name = models.CharField(primary_key=True, max_length=50)
    start_date = models.DateTimeField(null=True)
    end_date = models.DateTimeField(null=True)
