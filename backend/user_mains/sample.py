from mypkg.submodules_aggregator import utils
import pickle

data_generator = utils.group_data_generator(group_name="2022_ube_g2", base_point="neck")
for data in data_generator:
    utils.save_group_to_db(data)
