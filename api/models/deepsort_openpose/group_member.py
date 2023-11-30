import uuid

from .group import Group
from django.db import models


class GroupMember(models.Model):
    pk_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    id = models.IntegerField()
    group = models.ForeignKey(Group, models.CASCADE, related_name="members")

    class Meta:
        db_table = "group_member"
