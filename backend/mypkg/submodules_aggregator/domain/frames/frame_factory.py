from __future__ import annotations

import cv2
from functools import reduce

from ..boxes import BoundingBox, Point, BoundingBoxFactory
from ..keypoints import KeyPoint, KeypointFactory
from ..people import Person
from .frame import Frame
from .frame_element_directory import FrameElementDirectory
from .ipreprocessor import IPreprocessor


class FrameFactory:
    def __init__(
        self, 
        frame_element_directory: FrameElementDirectory,
        base_point: str = "midhip",
        preprocessor: IPreprocessor | None = None,
    ):
        self.frame_element_directory = frame_element_directory
        self.keypoint_factory = KeypointFactory()
        self.box_factory = BoundingBoxFactory()
        self.frame_number: int = 0
        self.base_point: str = base_point
        self.preprocessor = preprocessor
        self.prev_frame: Frame | None = None

    def create(
        self,
    ) -> Frame:
        self.frame_number += 1
        
        op_json_path, ds_csv_path, ds_jpg_path = next(self.frame_element_directory)
        keypoints = self.keypoint_factory.create_instances(op_json_path)
        boxes = self.box_factory.create_instances(ds_csv_path)
        if keypoints is None or boxes is None:
            return Frame([], ds_jpg_path, self.frame_number)

        if self.preprocessor is not None:
            keypoints, boxes = self.preprocessor.preprocess(
                keypoints,
                boxes,
                cv2.imread(ds_jpg_path),
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
        combined_frame = Frame(subject_people, ds_jpg_path, self.frame_number)
        self.prev_frame = combined_frame
        return combined_frame
