from __future__ import annotations

import datetime
from typing import Iterator, Sequence

import pandas as pd

from api import models
from .utils import time_to_frame_num, frame_num_to_time, str_to_time
from ..mylab_py_utils.common_function import central_idxes_gen
from ..device_logger.mouse import Drag, MousePos
from ..submodules_aggregator import domain


class MousePosCollection:
    def __init__(
        self,
        mouse_pos_path: str,
        interval_num: int = 10,
    ):
        """
        __getitem__も定義しており, フレーム番号あるいはtimeオブジェクトから, マウスを動かしているかどうかのブール値と，現在のフレームのマウス座標を返す．
        mouse_pos_csv : mouse_pos_log.csvのパス
        interval_num : 動かしているかどうかを判断するには前のデータのマウス座標が必要になる.その前をどれくらいとるかの値.単位はフレーム
        """

        self.posses: dict[datetime.time, MousePos] = mouse_pos_read_csv(mouse_pos_path)
        self.interval_num: int = interval_num

    def __getitem__(self, index: int) -> tuple[bool, MousePos]:
        batch_frame_numbers = get_prev_indices(index, self.interval_num)
        batch_posses: list[MousePos] = [self._get_pos(frame_num) for frame_num in batch_frame_numbers]
        xs = [pos.x for pos in batch_posses]
        ys = [pos.y for pos in batch_posses]
        dxs = [xs[i] - xs[i - 1] for i, _ in enumerate(xs[:-1], 1)]
        dys = [ys[i] - ys[i - 1] for i, _ in enumerate(ys[:-1], 1)]
        pos = self._get_pos(index)
        return any(dxs + dys), pos

    def _get_pos(self, time: datetime.time | int) -> MousePos:
        if isinstance(time, int):
            time = frame_num_to_time(time)

        time = time.replace(microsecond=time.microsecond // 10**5)
        if time in self.posses:
            pos = self.posses[time]
        else:  # もし真ん中の要素がない場合 過去のものを採用(マウス座標はサンプリング間隔が長いので隙間がある場合がある)
            while True:
                try:
                    time = time.replace(microsecond=time.microsecond - 1)
                except ValueError:  # ('microsecond must be in 0..999999')
                    try:
                        time = time.replace(second=time.second - 1, microsecond=9)
                    except ValueError:  # (second must be in 0..59)
                        time = time.replace(minute=time.minute - 1, second=59, microsecond=9)
                if time in self.posses:
                    break
            pos = self.posses[time]
        return pos

    def get_click_to_release_pos(self, drag: Drag) -> dict[int, list[MousePos]]:
        dt = datetime.datetime.combine(datetime.date.today(), drag.release.time) - datetime.datetime.combine(
            datetime.date.today(), drag.click.time
        )
        dt_sec_e_1 = int(dt.total_seconds() * 10)
        frame_pos: dict[int, list[MousePos]] = {}
        for ms in range(0, int(dt_sec_e_1) + 1):
            time = (
                datetime.datetime.combine(datetime.date.today(), drag.click.time)
                + datetime.timedelta(microseconds=ms * 10**5)
            ).time()
            if time_to_frame_num(time) in frame_pos:
                frame_pos[time_to_frame_num(time)].append(self[time])
            else:
                frame_pos[time_to_frame_num(time)] = [self[time]]
        return frame_pos


# 後方のインデックス群を取得する
def get_prev_indices(index: int, prev_num: int):
    indices: list[int] = []
    min_ = index - prev_num
    for i in range(min_, index + 1):
        if i < 0:
            indices.append(0)
        else:
            indices.append(i)
    return indices


def mouse_pos_read_csv(csv_path: str) -> dict[datetime.time, MousePos]:
    data = pd.read_csv(csv_path).values
    return_values: dict[datetime.time, MousePos] = {}
    for row in data:
        time = datetime.datetime.strptime(row[0], "%Hh%Mm%Ss%fms")
        time = time.replace(microsecond=time.microsecond // 10**5)  # microsecondの一番上の位より下は切り捨て
        key = datetime.time(time.hour, time.minute, time.second, time.microsecond)
        row[0] = str_to_time(row[0])
        return_values[key] = MousePos(*row)
    return return_values
