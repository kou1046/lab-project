import uuid

from django.db import models

from .point import Point


class BoundingBox(models.Model):
    class Meta:
        db_table = "boundingbox"

    pk_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    id = models.IntegerField()
    min = models.OneToOneField(Point, models.CASCADE, related_name="+")
    max = models.OneToOneField(Point, models.CASCADE, related_name="+")

    def center(self) -> Point:
        center_x = (self.max.x - self.min.x) / 2.0 + float(self.min.x)
        center_y = (self.max.y - self.min.y) / 2.0 + float(self.min.y)
        return Point(x=center_x, y=center_y)

    def contains(self, point: Point) -> bool:
        return point.x >= self.min.x and point.x <= self.max.x and point.y >= self.min.y and point.y <= self.max.y
