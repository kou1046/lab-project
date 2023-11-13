from dataclasses import dataclass
from .point import Point


@dataclass(frozen=True)
class BoundingBox:
    id: int
    min: Point
    max: Point

    def center(self) -> Point:
        center_x = (self.max.x - self.min.x) / 2.0 + float(self.min.x)
        center_y = (self.max.y - self.min.y) / 2.0 + float(self.min.y)
        return Point(center_x, center_y)

    def contains(self, point: Point) -> bool:
        return point.x >= self.min.x and point.x <= self.max.x and point.y >= self.min.y and point.y <= self.max.y
