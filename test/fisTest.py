from src.configuration import CONFIG_PATH
from dataset.dataset1 import DATASET1_PATH
from src.configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST
from src.config.config import Config
import src.model.utils.utils as utils
from src.data.fisData import FISData
from src.model.denseModel import DenseModel
import tensorlayer as tl
import tensorflow as tf
import numpy as np


def create_data():
    conf = Config(standard_key_list=FISData.standard_key_list, config_dict=None)
    config = utils.load_json(file_path=CONFIG_PATH + '/testFisTaskDataConfig.json')
    config['FILE_PATH'] = DATASET1_PATH + '/Task.csv'
    conf.load_config(path=None, json_dict=config)
    data = FISData(config=conf)
    data.load_data()
    return data


def train(test_config):

    sess = tf.InteractiveSession()
    data = create_data()
    data.shuffle_data()
    a = Config(config_dict=None, standard_key_list=DenseModel.standard_key_list)
    a.load_config(path=CONFIG_PATH + '/testDenseTaskConfig.json')
    model = DenseModel(config=a)
    # data_generator = tl.iterate.minibatches(inputs=np.array(data.state_list, dtype=np.float),
    #                                         targets=np.array(data.output_list, dtype=np.float),
    #                                         batch_size=test_config.config_dict['BATCH_SIZE'],
    #                                         shuffle=False)

    tl.layers.initialize_global_variables(sess)
    model.net.print_layers()
    model.net.print_params()

    test_config.config_dict['BATCH_COUNT'] = len(data.data_list) // test_config.config_dict['BATCH_SIZE']

    for i in range(test_config.config_dict['EPOCH']):
        aver_loss = 0.0
        count = 0
        state = []
        label = []
        for j in range(test_config.config_dict['BATCH_COUNT']):
            state, label = data.return_batch_data(j, test_config.config_dict['BATCH_SIZE'])
            if j == 0:
                res = sess.run(fetches=[model.net.outputs],
                               feed_dict={
                                   model.state('STATE'): state,
                                   model.label: label
                               })
                print("res = ", res)
                print("label = ", label)

            loss, _ = sess.run(fetches=[model.loss, model.optimize_loss],
                               feed_dict={
                                   model.state('STATE'): state,
                                   model.label: label
                               })
            count = count + 1
            aver_loss = (aver_loss * (count - 1) + loss) / count

        print("epoch = ", i, " loss = ", aver_loss)

    # tl.utils.fit(sess=sess,
    #              network=model.net,
    #              cost=model.loss,
    #              x=model.state('STATE'),
    #              y_=model.label,
    #              train_op=model.optimize_loss,
    #              X_train=data.state_list,
    #              y_train=data.output_list,
    #              batch_size=10,
    #              n_epoch=500,
    #              print_freq=1,
    #              tensorboard=True,
    #              tensorboard_epoch_freq=1
    #              )


if __name__ == '__main__':
    test_config_standard_key_list = utils.load_json(file_path=CONFIG_STANDARD_KEY_LIST + '/fisTestKeyList.json')
    test_config = Config(config_dict=None, standard_key_list=test_config_standard_key_list)

    test_config_dict = utils.load_json(file_path=CONFIG_PATH + '/testFisTestConfig.json')
    test_config.load_config(json_dict=test_config_dict, path=None)

    train(test_config)
