from user_mains.infrastructure import DjGroupRepository
from user_mains.utils.group_register_cli_application import GroupRegisterCLIApplication

repository = DjGroupRepository()
app = GroupRegisterCLIApplication(repository)
app.register()
