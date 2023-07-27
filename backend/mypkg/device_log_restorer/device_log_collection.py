from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Sequence
import glob

from .mouse_drag_iterator import MouseDragIterator
from .mouse_pos_collection import MousePosCollection
from .screenshot_path_collection import ScreenShotPathCollection
from ..device_logger.mouse import Drag, MousePos
from ..submodules_aggregator import domain


@dataclass(frozen=True)
class DeviceDirectory:
    dir_path: str

    def __post_init__(self):
        click_log_path = os.path.join(self.dir_path, "click_log.csv")
        screenshot_dir_path = os.path.join(self.dir_path, "screenshot")
        mouse_pos_path = os.path.join(self.dir_path, "mouse_pos_log.csv")

        if not os.path.exists(click_log_path):
            raise ValueError(f"{click_log_path}が見つかりません")

        if not os.path.exists(screenshot_dir_path):
            raise ValueError(f"{screenshot_dir_path}が見つかりません")

        if not os.path.exists(mouse_pos_path):
            raise ValueError(f"{mouse_pos_path}が見つかりません")

        object.__setattr__(self, "click_log_path", click_log_path)
        object.__setattr__(self, "screenshot_dir_path", screenshot_dir_path)
        object.__setattr__(self, "mouse_pos_path", mouse_pos_path)


class DeviceLogCollection:
    def __init__(self, device_log_dir_path: str):
        """
        カギ括弧とフレーム数でアクセス．(マウスが動いているかどうか，マウス座標, "click" or "release", (リリース時)Dragオブジェクト), ｽｸﾘｰﾝｼｮｯﾄのパスを返す．
        """
        device_directory = DeviceDirectory(device_log_dir_path)
        self.drag_collection = MouseDragIterator(device_directory.click_log_path)
        self.pos_collection = MousePosCollection(device_directory.mouse_pos_path)
        self.screenshot_path_collection = ScreenShotPathCollection(
            glob.glob(os.path.join(device_directory.screenshot_dir_path, "*.jpg"))
        )

    def __getitem__(self, frame_number: int):
        is_mouse_move, mouse_pos = self.pos_collection[frame_number]
        mouse_status, drag = self.drag_collection[frame_number]
        screenshot_path = self.screenshot_path_collection[frame_number]
        return (is_mouse_move, mouse_pos, mouse_status, drag), screenshot_path
