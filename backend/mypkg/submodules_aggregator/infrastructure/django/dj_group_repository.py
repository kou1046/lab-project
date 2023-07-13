from __future__ import annotations
from dataclasses import asdict
from dacite import from_dict

from api import models
from api.serializers import FrameListSerializer
from ...domain.groups import IGroupRepository
from ... import domain


class DjGroupRepository(IGroupRepository):
    def save(self, domain_group: domain.Group):
        serializer = FrameListSerializer(
            data=[{**asdict(frame), **{"group": {"name": domain_group.name}}} for frame in domain_group.frames]
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()

    def find(self, group_name: str):
        # group = models.Group.objects.get(name=group_name)
        # serializer = FrameListSerializer(group.frames.all())
        # frames = [
        #     [kwarg.pop("group"), from_dict(domain.Frame, kwarg)][1] for kwarg in serializer.data
        # ]  # "groupキーを消した後の辞書を引数に与えている"
        # return domain.Group(group_name, frames)

        return super().find(group_name)
