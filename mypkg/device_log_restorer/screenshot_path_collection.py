from __future__ import annotations

import datetime
import os
from typing import Sequence

import numpy as np


from .utils import time_to_frame_num, frame_num_to_time
from ..device_logger.mouse import Drag


class ScreenShotPathCollection:
    """
    辞書ライクなオブジェクト. timeオブジェクト又はフレーム番号からフレーム画像を取得できる. 渡された時間に一番近いスクリーンショットの画像を返す．
    dataset : CombinedFrame の配列． または，フレーム数の配列．
    screenshot_paths : スクリーンショットパスの配列．
    """

    def __init__(self, screenshot_paths: Sequence[str]):
        self.time_paths: dict[datetime.time, str] = read_screenshots(screenshot_paths)

    def __getitem__(self, time: datetime.time | int) -> np.ndarray:
        if isinstance(time, int):
            time = frame_num_to_time(time)
        if time.microsecond != 0:
            time = time.replace(microsecond=0)
        if time not in self.time_paths:
            while True:
                time = (datetime.datetime.combine(datetime.date.today(), time) - datetime.timedelta(seconds=1)).time()
                if time in self.time_paths:
                    break
        return self.time_paths[time]

    def get_click_to_release_ss(self, drag: Drag) -> dict[int, np.ndarray]:
        dt = datetime.datetime.combine(datetime.date.today(), drag.release.time) - datetime.datetime.combine(
            datetime.date.today(), drag.click.time
        )
        dt_sec_e_1 = int(dt.total_seconds())
        frame_img: dict[int, np.ndarray] = {}
        for s in range(0, int(dt_sec_e_1) + 1):
            time = (
                datetime.datetime.combine(datetime.date.today(), drag.click.time) + datetime.timedelta(seconds=s)
            ).time()
            frame_img[time_to_frame_num(time)] = self[time]
        return frame_img


def read_screenshots(screenshot_paths: Sequence[str]) -> dict[datetime.time, str]:
    return_values: dict[datetime.time, str] = {}
    for path in screenshot_paths:
        file_name, _ = os.path.splitext(os.path.basename(path))
        shot_time = datetime.datetime.strptime(file_name, "%Hh%Mm%Ss")
        key = datetime.time(shot_time.hour, shot_time.minute, shot_time.second)
        return_values[key] = path
    return return_values
