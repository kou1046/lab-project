from ...domain.groups import Group, IGroupRepository

class InMemoryGroupRepository(IGroupRepository):
    def __init__(self):
        self.store: dict[str, Group] = {}
    def save(self, group: Group):
        self.store[group.name] = Group
    def find(self, group_name: str):
        if group_name not in self.store:
            raise KeyError(group_name)
        return self.store[group_name]
        