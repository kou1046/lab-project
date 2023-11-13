from __future__ import annotations

import datetime

FPS = 25
START_TIME = datetime.time(8, 30, 0)


def str_to_time(str_time: str) -> datetime.time:
    """
    ##h##m##s######msという文字列表記をtimeオブジェクトに変換する.
    """
    units = ("h", "m", "s", "ms")
    hours = int(str_time[: str_time.index("h")])
    minutes, seconds, microseconds = [
        int(str_time[str_time.index(units[i]) + 1 : str_time.index(units[i + 1])]) for i, _ in enumerate(units[:-1])
    ]
    return datetime.time(hours, minutes, seconds, microseconds)


def time_to_str(time: datetime.time) -> str:
    """
    timeオブジェクトを##h##m##s######msという文字列表記に変換する.
    """
    h = time.hour
    m = time.minute
    s = time.second
    ms = time.microsecond
    return f"{h}h{m}m{s}s{ms}ms"


def frame_num_to_time(frame_num: int) -> datetime.time:
    """
    フレーム番号からtimeオブジェクトに変換する.
    """
    abs_time = datetime.datetime.combine(datetime.date.today(), START_TIME) + datetime.timedelta(
        seconds=frame_num / FPS
    )
    hour, minute, second, microsecond = [
        getattr(abs_time, attr) for attr in ("hour", "minute", "second", "microsecond")
    ]
    return datetime.time(hour, minute, second, microsecond)


def time_to_frame_num(time: datetime.time) -> int:
    """
    timeオブジェクトからフレーム番号に変換する.
    """
    dt = datetime.datetime.combine(datetime.date.today(), time) - datetime.datetime.combine(
        datetime.date.today(), START_TIME
    )
    secs_frame_num = dt.seconds * FPS
    msecs_frame_num = dt.microseconds // (1 / FPS * 10**6)
    return int(secs_frame_num + msecs_frame_num)
