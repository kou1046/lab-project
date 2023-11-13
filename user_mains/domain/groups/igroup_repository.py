from __future__ import annotations
from abc import abstractmethod, ABCMeta

from .group import Group


class IGroupRepository(metaclass=ABCMeta):
    @abstractmethod
    def save(self, group: Group) -> None:
        ...

    @abstractmethod
    def find(self, group_name: str) -> Group | None:
        ...
