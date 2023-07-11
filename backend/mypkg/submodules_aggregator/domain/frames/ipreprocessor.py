from __future__ import annotations

from abc import ABCMeta
import numpy as np

from boxes import BoundingBox
from keypoints import KeyPoint, KeyPointAttr
from frames import Frame

from ....complementidcreator import (
    CreateIds,
    MonitorIds,
    ReplaceIds,
    StopIds,
    TrackingBoundingBox,
    deserialize_complement_ids,
)


class IPreprocessor(metaclass=ABCMeta):
    def __init__(self, base_point: KeyPointAttr):
        self.base_point: KeyPointAttr = base_point

    def preprocess(
        self,
        keypoints: list[KeyPoint],
        boxes: list[BoundingBox],
        img: np.ndarray,
        frame_num: int,
        prev_frame: Frame | None,
    ) -> tuple[list[KeyPoint], list[BoundingBox]]:
        keypoints = [self.preprocess_keypoint(keypoint, img, frame_num, prev_frame) for keypoint in keypoints]
        boxes = [self.preprocess_box(box, img, frame_num, prev_frame) for box in boxes]

        filterd_keypoints = [
            keypoint for keypoint in keypoints if self.filter_keypoint(keypoint, img, frame_num, prev_frame)
        ]
        filterd_boxes = [box for box in boxes if self.filter_box(box, img, frame_num, prev_frame)]

        return filterd_keypoints, filterd_boxes

    def preprocess_keypoint(
        self,
        keypoint: KeyPoint,
        img: np.ndarray,
        frame_num: int,
        prev_frame: Frame | None,
    ) -> KeyPoint:
        return keypoint

    def preprocess_box(
        self,
        box: BoundingBox,
        img: np.ndarray,
        frame_num: int,
        prev_frame: Frame | None,
    ) -> BoundingBox:
        return box

    def filter_keypoint(
        self,
        keypoint: KeyPoint,
        img: np.ndarray,
        frame_num: int,
        prev_frame: Frame | None,
    ) -> bool:
        return True

    def filter_box(
        self,
        box: BoundingBox,
        img: np.ndarray,
        frame_num: int,
        prev_frame: Frame | None,
    ) -> bool:
        return True


class Complementer(IPreprocessor):
    def __init__(
        self,
        monitor_json_path: str,
        replace_json_path: str,
        stop_json_path: str,
        create_json_path: str,
    ):
        ids_data = deserialize_complement_ids(monitor_json_path, replace_json_path, stop_json_path, create_json_path)
        self.monitor_ids: MonitorIds = ids_data[0]
        self.replace_ids: ReplaceIds = ids_data[1]
        self.stop_ids: StopIds = ids_data[2]
        self.create_ids: CreateIds = ids_data[3]
        self.replace_id_memo: dict[
            int, int
        ] = {}  # {補間されるid:補完先のmonitor_id, ...} replace_idの累積．置き換え以降も続けてidの置き換えを続けるために必要な変数．

    def preprocess(
        self,
        keypoints: list[KeyPoint],
        boxes: list[BoundingBox],
        img: np.ndarray,
        frame_num: int,
        prev_frame: Frame | None,
    ) -> tuple[list[KeyPoint], list[BoundingBox]]:
        if frame_num in self.replace_ids:  # 毎回キーとバリューを逆転して追加することでキーもバリューも最新の状態のみになるよう更新する
            new_invert_replace_dict = {do: done for done, do in self.replace_ids[frame_num].items()}
            invert_replace_dict = {do: done for done, do in self.replace_id_memo.items()}
            update_replace_dict = {**invert_replace_dict, **new_invert_replace_dict}
            self.replace_id_memo = {done: do for do, done in update_replace_dict.items()}

        """
        ↑　逆転しない場合，例えば replace_id_memo = {2:1, 3:1} (2を1, 3を1に置き換える)という辞書を得てしまう場合がある．(1はmonitor_id)
        この場合, 1つのフレームに2と3のidが出現したとき, 1が2体いると認識されてしまうため, キーだけでなくバリューも一意にする必要がある. 
        先の例の辞書を反転すると，{1:2, 1:3} →　(キーは複数存在できないので新しい方に上書き) → {1:3} → 最後に反転 → {3:1}
            
        """

        added_boxes: list[TrackingBoundingBox] = [box for box in self.create_ids if box.id in self.replace_id_memo]
        for added_box in added_boxes:
            if added_box.start_frame <= frame_num:
                if added_box.istracking and prev_frame is not None:
                    added_box.update_tracking_range_from_img(prev_frame.img, img)
                    id_, min, max = [
                        getattr(added_box, attr)
                        for attr in (
                            "id",
                            "tracking_min",
                            "tracking_max",
                        )
                    ]
                else:
                    id_, min, max = [getattr(added_box, attr) for attr in ("id", "min", "max")]
                boxes.append(BoundingBox(id_, min, max))

        # ↓ 要調整 どれかのidをmonitor_idに置き換えた後に真のmonitor_idが出現するとバグる.　monitor_idのうち，一回でも置き換えたことがあるものならそのmonitor_idは真っ先に省いておく
        replacing_people_ids = set(self.replace_id_memo.values())

        boxes = [box for box in boxes if box.id not in replacing_people_ids]

        return super().preprocess(keypoints, boxes, img, frame_num, prev_frame)

    def filter_box(
        self,
        box: BoundingBox,
        img: np.ndarray,
        frame_num: int,
        prev_frame: Frame | None,
    ) -> bool:
        if box.id not in self.monitor_ids:
            return False

        stopped_people_ids = set(self.stop_ids)
        if box.id in stopped_people_ids:  # stopの処理はreplaceした後に処理
            if self.stop_ids[box.id].needsstop(frame_num):  # resumeなのかstopなのか判断
                return False  # stop中なら排除

        return True

    def preprocess_box(
        self,
        box: BoundingBox,
        img: np.ndarray,
        frame_num: int,
        prev_frame: Frame | None,
    ) -> BoundingBox:
        if box.id in self.replace_id_memo:
            replaced_id = self.replace_id_memo[box.id]
            replaced_box = BoundingBox(replaced_id, box.min, box.max)
            return replaced_box
        return box
