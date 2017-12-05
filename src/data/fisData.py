from src.data.data import Data
from dataset.dataset1 import DATASET1_PATH
import src.model.utils.utils as utils
from src.configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST
import csv
import numpy as np


class FISData(Data):
    standard_key_list = utils.load_json(CONFIG_STANDARD_KEY_LIST + '/fisDataKeyList.json')

    def __init__(self, config):
        super().__init__(config)
        self.state_list = []
        self.output_list = []

    def load_data(self, *args, **kwargs):
        # super().load_data(*args, **kwargs)
        with open(self.config.config_dict['FILE_PATH']) as f:
            reader = csv.DictReader(f)
            for row in reader:
                state_data_sample = []
                output_data_sample = []
                for name in self.config.config_dict['STATE_NAME_LIST']:
                    state_data_sample.append(float(row[name]))
                for name in self.config.config_dict['OUTPUT_NAME_LIST']:
                    output_data_sample.append(float(row[name]))
                self.state_list.append(state_data_sample)
                self.output_list.append(output_data_sample)
                self.data_list.append({"STATE": state_data_sample, "OUTPUT": output_data_sample})
        pass

    def shuffle_data(self):
        temp_data = np.array(self.data_list)
        np.random.shuffle(temp_data)
        self.data_list = temp_data
        self.state_list = []
        self.output_list = []
        for sample in self.data_list:
            self.state_list.append(sample['STATE'])
            self.output_list.append(sample['OUTPUT'])
        pass
        self.state_list = np.array(self.state_list)
        self.output_list = np.array(self.output_list)

    def return_batch_data(self, index, size):
        return self.state_list[index * size:index * size + size], self.output_list[index * size:index * size + size]
        pass


if __name__ == '__main__':
    from src.config.config import Config
    from src.configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST
    from src.configuration import CONFIG_PATH
    conf = Config(standard_key_list=FISData.standard_key_list, config_dict=None)
    config = utils.load_json(file_path=CONFIG_PATH + '/testFisAttackDataConfig.json')
    config['FILE_PATH'] = DATASET1_PATH + '/Attack.csv'
    conf.load_config(path=None, json_dict=config)
    data = FISData(config=conf)
    data.load_data()
    data.shuffle_data()




