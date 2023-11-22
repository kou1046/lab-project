from django.db import models

from submodules.uiflow_dom_logger.pyparser import blocks
import datetime

from ..deepsort_openpose.group import Group
from ..base.uuid_model import UUIDModel
from ..deepsort_openpose.frame import Frame

block_factory = blocks.AssignableChildrenBlockFactory()


class UIFlowBlockDOM(UUIDModel):
    timestamp = models.BigIntegerField()  # microsecond
    html = models.TextField()
    frame = models.OneToOneField(Frame, models.CASCADE, related_name="dom")
    group = models.ForeignKey(Group, models.CASCADE, related_name="doms")

    class Meta:
        db_table = "uiflow_block_dom"

    def to_datamodel(self) -> blocks.UiflowBlockDOM:
        return blocks.UiflowBlockDOM(self.timestamp, self.html)

    def blocks(self) -> blocks.Block:
        return block_factory.create_instances(self.html)

    @property
    def date(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.timestamp / 1000, tz=datetime.timezone(datetime.timedelta(hours=9)))
