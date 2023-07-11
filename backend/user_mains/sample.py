from mypkg.submodules_aggregator.application import application
import pickle

data_generator = application.group_data_generator(group_name="2022_ube_g2", base_point="neck")
for data in data_generator:
    application.save_group_to_db(data)
