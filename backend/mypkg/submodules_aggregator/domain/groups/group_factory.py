from ..frames import Frame
from .igroup_repository import IGroupRepository, Group


class GroupFactory:
    def __init__(self, group_repository: IGroupRepository):
        self.group_repository = group_repository

    def create(self, group_name: str, frames: list[Frame]):
        if self.group_repository.find(group_name) is not None:
            raise KeyError(f"{group_name}は既に存在しています")
        return Group(group_name, frames)
