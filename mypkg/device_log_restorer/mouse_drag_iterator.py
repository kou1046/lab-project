from __future__ import annotations

from typing import Literal, Sequence
import datetime

import pandas as pd

from .utils import frame_num_to_time
from ..device_logger.mouse import Drag, Click, Release
from ..submodules_aggregator import domain


class MouseDragIterator:
    """
    イテラブルオブジェクト. イテレーション毎にマウスが今クリック中かどうかのステータスと，マウスをリリースした時点において, Dragオブジェクトを返す. （それ以外はNone)
    frames : CombinedFrame の配列． または，フレーム数の配列．
    click_log_path : click_log.csvのパス.

    Example:
    base_dir = askdirectory()
    files = glob.glob(os.path.join(base_dir, '*.pickle'))[5:8]
    frames:list[domain.Frame] = []
    converter = combiner.BinaryConverter()
    imgs = []
    for file in files:
        frames += converter.deserialize(file)
    obs = MousePosObserver(frames, os.path.join('..', 'log', 'G4', 'mouse_pos_log.csv'), 15)
    drag_obs = idm.MouseDragObserver(frames, os.path.join('..', 'log', 'G4', 'click_log.csv'))
    for data, (ismousemoving, pos), (status, drag) in zip(frames, obs, drag_obs):
        img = data.get_visualized_image(color=(0, 0, 255))
        if ismousemoving:
            cv2.putText(img, 'mouse is moving', (0, 40), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)
        print(drag.time)
        cv2.imshow('img', img)
        cv2.waitKey(3)
    """

    def __init__(
        self,
        click_log_path: str,
    ):
        self.click_events, self.release_events = mouse_drag_read_csv(click_log_path)
        self.click_queue: None | Click = None
        self.status: Literal["click", "release"] = "release"
        self.index: int = 0

    def _toggle_status(self):
        self.status = "click" if self.status == "release" else "release"

    def __getitem__(self, index: int | datetime.time) -> tuple[Literal["click", "release"], Drag | None]:
        if isinstance(index, int):
            index = frame_num_to_time(index)

        about_time = index.replace(microsecond=index.microsecond // 10**5)
        self.index += 1

        target_events = self.click_events if self.status == "release" else self.release_events
        if about_time not in target_events:
            return self.status, None
        if not target_events[about_time]:
            return self.status, None

        event = target_events[about_time].pop(0)
        self._toggle_status()

        if self.status == "click":
            self.click_queue = event
            return self.status, None
        else:
            release = event
            drag = Drag(self.click_queue, release)
            self.click_queue = None
            return self.status, drag


def mouse_drag_read_csv(
    csv_path: str,
) -> tuple[dict[datetime.time, Click], dict[datetime.time, list[Release]]]:
    def checkduplicate(old_name):  # 滅多にないが，もしclickやreleaseの行が二回連続で続いた場合を検出するためのジェネレータ
        while True:
            cur_name = yield
            if cur_name == old_name:
                yield True
            old_name = cur_name

    data = pd.read_csv(csv_path).values
    click_values: dict[datetime.time, list[Click]] = {}
    release_values: dict[datetime.time, list[Release]] = {}
    isduplicate = checkduplicate("release")
    next(isduplicate)
    for row in data:
        str_time, x, y, side, type_ = row
        if side == "right" or side == "middle":  # 右クリックやミドルクリックのデータは扱わない
            continue
        time = datetime.datetime.strptime(str_time, "%Hh%Mm%Ss%fms")
        key_date_time = time.replace(microsecond=time.microsecond // 10**5)  # microsecondの一番上の位より下は切り捨て
        key_time = datetime.time(
            key_date_time.hour,
            key_date_time.minute,
            key_date_time.second,
            key_date_time.microsecond,
        )
        time = datetime.time(time.hour, time.minute, time.second, time.microsecond)
        if type_ == "click":
            if isduplicate.send(type_):
                next(isduplicate)
                continue
            click = Click(time, x, y)
            if key_time in click_values:
                continue
            else:
                click_values[key_time] = [click]

        if type_ == "release":
            if isduplicate.send(type_):
                next(isduplicate)
                continue
            release = Release(time, x, y)
            if key_time in release_values:
                if key_time in click_values:
                    del click_values[key_time]
                continue
            else:
                release_values[key_time] = [release]
    return click_values, release_values
