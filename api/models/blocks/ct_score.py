from django.db import models

from submodules.uiflow_dom_logger.pyparser import blocks
import datetime

from ..blocks.uiflow_block_dom import UIFlowBlockDOM
from ..deepsort_openpose.group import Group
from ..base.uuid_model import UUIDModel
from ..deepsort_openpose.frame import Frame

score_name = ("loop", "logic", "data", "data_representation", "module")


class CTScore(UUIDModel):
    loop = models.IntegerField()
    logic = models.IntegerField()
    data = models.IntegerField()
    data_representation = models.IntegerField()
    module = models.IntegerField()

    frame = models.OneToOneField(Frame, models.CASCADE, related_name="ct_score")
    group = models.ForeignKey(Group, models.CASCADE, related_name="ct_scores")

    class Meta:
        db_table = "ct_score"

    def named_score(self) -> list[tuple[str, int]]:
        return [(name, getattr(self, name)) for name in score_name]

    def score(self) -> list[int]:
        return [score for name, score in self.named_score()]
