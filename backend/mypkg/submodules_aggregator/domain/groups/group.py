from dataclasses import dataclass
from ..frames import Frame


@dataclass(frozen=True)
class Group:
    name: str
    frames: set[Frame]
