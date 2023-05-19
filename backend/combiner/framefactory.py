from __future__ import annotations

import json
from functools import reduce

import numpy as np
import pandas as pd
from datatypes import *
from pandas.errors import EmptyDataError


class LoadedDataFromPath:
    extension: str = ""

    def __init__(self, path: str):
        self.path = path
        assert self._check_extension(), "拡張子が間違っています"

    def _check_extension(self):
        if not self.extension:
            raise NotImplementedError("クラス変数extensionが定義されていません")
        return self.path.split(".")[-1] == self.extension


class OpenPoseJsonData(LoadedDataFromPath):
    extension = "json"

    def __init__(self, path: str):
        super().__init__(path)
        self.value: np.ndarray = self._read_json(path)

    def _read_json(self, path: str):
        with open(path, "r") as f:
            item = json.load(f)
        return np.array([np.array(person["pose_keypoints_2d"]).reshape(-1, 3) for person in item["people"]])

    def is_empty(self):
        return not self.value.any()

    def generate_keypoints(self) -> list[KeyPoint]:
        assert self.is_empty() == False, "人が見つかりません"
        value = self.value.tolist()
        keypoints: list[KeyPoint] = []
        for row in value:
            points = [ProbabilisticPoint(cell[0], cell[1], cell[2]) for cell in row]
            keypoints.append(KeyPoint(points))
        return keypoints


class DeepSortJpgData(LoadedDataFromPath):
    extension = "jpg"


class DeepSortCsvData(LoadedDataFromPath):
    extension = "csv"

    def __init__(self, path: str):
        super().__init__(path)
        self.value: np.ndarray | None = self._read_csv(path)

    def _read_csv(self, path: str) -> np.ndarray | None:
        try:
            table = pd.read_csv(path).values
        except EmptyDataError:
            return None
        return table

    def is_empty(self):
        return self.value is None

    def generate_boxes(self) -> list[BoundingBox]:
        assert self.is_empty() is not None, "人が見つかりません"
        return [BoundingBox(row[0], Point(row[1], row[2]), Point(row[3], row[4])) for row in self.value.tolist()]


class CombinedFrameFactory:
    def __init__(self, base_point: str = "midhip"):
        """_summary_
          中間処理が必要になった場合, このクラスを継承又は包含し, サブクラスでpreprcessメソッドをオーバーライドする
        Args:
            base_point (str, optional): _description_. Defaults to "midhip".

        """
        self.frame_number: int = 0
        self.base_point: str = base_point

    def _preprocess_keypoints(self, keypoints: list[KeyPoint]):
        return [keypoint for keypoint in keypoints if getattr(keypoint, self.base_point).p != 0]

    def _preprocess_boxes(self, boxes: list[BoundingBox]):
        return boxes

    def create(
        self,
        op_data: OpenPoseJsonData,
        ds_csv_data: DeepSortCsvData,
        ds_jpg_data: DeepSortJpgData,
    ) -> CombinedFrame:
        self.frame_number += 1
        if ds_csv_data.is_empty():
            return CombinedFrame([], ds_jpg_data.path, self.frame_number)
        keypoints = sorted(
            op_data.generate_keypoints(),
            key=lambda arg: getattr(arg, self.base_point).x,
        )
        boxes = sorted(ds_csv_data.generate_boxes(), key=lambda arg: arg.center().x)

        preprocessed_keypoints = self._preprocess_keypoints(keypoints)
        preprocessed_boxes = self._preprocess_boxes(boxes)

        chosen_items: dict[BoundingBox, KeyPoint] = {}
        for keypoint in preprocessed_keypoints:
            base_point: Point = getattr(keypoint, self.base_point)
            isin_boxes = [box.contains(base_point) for box in boxes]
            isin_count = isin_boxes.count(True)
            if isin_count == 0:
                continue
            elif isin_count == 1:
                chosen_box = [box for box, isinbox in zip(preprocessed_boxes, isin_boxes) if isinbox][0]
            else:  # 2つ以上のBoxに関節点が入っている場合，Boxの中心と関節点の距離が小さいBoxを採用する
                in_boxes = [preprocessed_boxes[i] for i, isin in enumerate(isin_boxes) if isin]
                chosen_box = reduce(
                    lambda box1, box2: box1
                    if base_point.distance_to(box1.center()) < base_point.distance_to(box2.center())
                    else box2,
                    in_boxes,
                )

            if not chosen_box in chosen_items:
                chosen_items[chosen_box] = keypoint
                continue

            else:  # 既に，採用されたボックスがchosen_boxesに入っている（1つのボックスに関節基準点が複数入っていたとき）
                """
                複数人の関節基準点とボックスの中心点における距離をそれぞれ算出し，一番距離が小さかった人を採用する．
                """

                processed_keypoint = chosen_items[chosen_box]
                processed_base_point: Point = getattr(processed_keypoint, self.base_point)
                chosen_items[chosen_box] = (
                    keypoint
                    if base_point.distance_to(chosen_box.center())
                    < processed_base_point.distance_to(chosen_box.center())
                    else processed_keypoint
                )

        subject_people = [Person(keypoint, box) for box, keypoint in chosen_items.items()]
        return CombinedFrame(subject_people, ds_jpg_data.path, self.frame_number)


class ComplementCombinedFrameFactory(CombinedFrameFactory):
    def __init__(self, base_point: str = "midhip"):
        super().__init__(base_point=base_point)

    def _preprocess_boxes(self, boxes: list[BoundingBox]):
        return boxes
