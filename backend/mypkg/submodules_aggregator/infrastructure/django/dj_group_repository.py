from ...domain.groups import IGroupRepository, Group

class DjGroupRepository(IGroupRepository):
    def save(group: Group):
        return super().save()
    def find(group_name: str):
        return super().find()