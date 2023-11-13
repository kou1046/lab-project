from __future__ import annotations
from abc import abstractmethod, ABCMeta
import numpy as np 
import pandas as pd 
from pandas.errors import EmptyDataError

from .bounding_box import BoundingBox
from .point import Point


class IBoundingBoxFactory(metaclass=ABCMeta):
    @abstractmethod
    def create_instances(self, deepsort_csv_path: str) -> list[BoundingBox] | None: ...
    

class BoundingBoxFactory(IBoundingBoxFactory):

    def _read_csv(self, path: str) -> np.ndarray | None:
        try:
            table = pd.read_csv(path).values
        except EmptyDataError:
            return None
        return table

    def create_instances(self, deepsort_csv_path: str) -> list[BoundingBox] | None:
        data = self._read_csv(deepsort_csv_path)
        if data is None:
            return None
        return [BoundingBox(row[0], Point(row[1], row[2]), Point(row[3], row[4])) for row in data.tolist()]