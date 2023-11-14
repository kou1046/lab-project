from ..base.abstract_point import AbstractPoint
from ..base.nonnegative_field import NonNegativeFloatField


class ProbabilisticPoint(AbstractPoint):
    class Meta:
        db_table = "probabilistic_point"

    p = NonNegativeFloatField(default=100.0)
