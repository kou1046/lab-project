from ..base.abstract_point import AbstractPoint


class Point(AbstractPoint):
    class Meta:
        db_table = "point"
