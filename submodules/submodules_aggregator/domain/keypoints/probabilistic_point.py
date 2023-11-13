from __future__ import annotations
from dataclasses import dataclass
from ..boxes import Point


@dataclass(frozen=True)
class ProbabilisticPoint(Point):
    x: float | int
    y: float | int
    p: float | int = 100.0
