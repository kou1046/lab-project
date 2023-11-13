from dataclasses import dataclass

from submodules.deepsort_openpose.api.domain import Frame


@dataclass(frozen=True)
class Group:
    name: str
    frames: set[Frame]
