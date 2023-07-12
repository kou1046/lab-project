from abc import abstractmethod, ABCMeta

from .group import Group


class IGroupRepository(metaclass=ABCMeta):
    @abstractmethod
    def save(self, group: Group):
        ...

    @abstractmethod
    def find(self, group_name: str):
        ...
