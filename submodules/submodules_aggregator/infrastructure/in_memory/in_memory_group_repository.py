from ...domain.groups import Group, IGroupRepository


class InMemoryGroupRepository(IGroupRepository):
    def __init__(self):
        self.store: dict[str, Group] = {}

    def save(self, group: Group):
        if group.name not in self.store:
            self.store[group.name] = group
        else:
            self.store[group.name].frames.union(group.frames)

    def find(self, group_name: str):
        if group_name not in self.store:
            return None
        return self.store[group_name]
