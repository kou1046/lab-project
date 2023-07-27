from mypkg.submodules_aggregator import *

repository = InMemoryGroupRepository()
app = GroupRegisterCLIApplication(repository)
app.register()
