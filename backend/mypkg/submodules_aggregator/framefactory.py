from __future__ import annotations

import json
from functools import reduce

import numpy as np
import pandas as pd
from .intermediate_model import *
from .preprocessor import Preprocessor
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
            keypoints.append(KeyPoint(*points))
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
    def __init__(
        self,
        group_name: str,
        base_point: str = "midhip",
        group_cls=Group,
        preprocessor: Preprocessor | None = None,
    ):
        self.frame_number: int = 0
        self.base_point: str = base_point
        self.group = group_cls(group_name)
        self.preprocessor = preprocessor
        self.prev_frame: CombinedFrame | None = None

    def create(
        self,
        op_data: OpenPoseJsonData,
        ds_csv_data: DeepSortCsvData,
        ds_jpg_data: DeepSortJpgData,
    ) -> CombinedFrame:
        self.frame_number += 1
        if ds_csv_data.is_empty() or op_data.is_empty():
            return CombinedFrame(self.group, [], ds_jpg_data.path, self.frame_number)

        keypoints = op_data.generate_keypoints()
        boxes = ds_csv_data.generate_boxes()

        if self.preprocessor is not None:
            keypoints, boxes = self.preprocessor.preprocess(
                keypoints,
                boxes,
                cv2.imread(ds_jpg_data.path),
                self.frame_number,
                self.prev_frame,
            )

        sorted_keypoints = sorted(
            keypoints,
            key=lambda arg: getattr(arg, self.base_point).x,
        )
        sorted_boxes = sorted(boxes, key=lambda arg: arg.center().x)

        chosen_items: dict[BoundingBox, KeyPoint] = {}
        for keypoint in sorted_keypoints:
            base_point: Point = getattr(keypoint, self.base_point)
            isin_boxes = [box.contains(base_point) for box in sorted_boxes]
            isin_count = isin_boxes.count(True)
            if isin_count == 0:
                continue
            elif isin_count == 1:
                chosen_box = [box for box, isinbox in zip(sorted_boxes, isin_boxes) if isinbox][0]
            else:  # 2つ以上のBoxに関節点が入っている場合，Boxの中心と関節点の距離が小さいBoxを採用する
                in_boxes = [sorted_boxes[i] for i, isin in enumerate(isin_boxes) if isin]
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
        combined_frame = CombinedFrame(self.group, subject_people, ds_jpg_data.path, self.frame_number)
        self.prev_frame = combined_frame
        return combined_frame
