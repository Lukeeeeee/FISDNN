from src.configuration import CONFIG_PATH
from dataset.dataset1 import DATASET1_PATH
from src.configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST
from test.logs import TEST_LOG_PATH
from src.config.config import Config
import src.model.utils.utils as utils
from src.data.fisData import FISData
from src.model.denseModel import DenseModel
import tensorlayer as tl
import tensorflow as tf
from src.model.twoLayerDenseModel import TwoLayerDenseModel
import datetime
import os


def create_data():
    conf = Config(standard_key_list=FISData.standard_key_list, config_dict=None)
    config = utils.load_json(file_path=CONFIG_PATH + '/data/testFisAttackDataConfig.json')
    config['FILE_PATH'] = DATASET1_PATH + '/Attack.csv'
    conf.load_config(path=None, json_dict=config)
    data = FISData(config=conf)
    data.load_data()
    return data


def create_dense_model():
    a = Config(config_dict=None, standard_key_list=DenseModel.standard_key_list)
    a.load_config(path=CONFIG_PATH + '/model/testDenseAttackConfig.json')
    model = DenseModel(config=a)
    return model


def create_two_layer_dense_model():
    a = Config(config_dict=None, standard_key_list=TwoLayerDenseModel.standard_key_list)
    a.load_config(path=CONFIG_PATH + '/model/testTwoLayerDenseAttackConfig.json')
    model = TwoLayerDenseModel(config=a)
    return model


def train(test_config, model):

    sess = tf.InteractiveSession()
    data = create_data()
    data.shuffle_data()

    # data_generator = tl.iterate.minibatches(inputs=np.array(data.state_list, dtype=np.float),
    #                                         targets=np.array(data.output_list, dtype=np.float),
    #                                         batch_size=test_config.config_dict['BATCH_SIZE'],
    #                                         shuffle=False)

    tl.layers.initialize_global_variables(sess)
    model.net.print_layers()
    model.net.print_params()

    utils.variable_summaries(model.loss, 'loss')

    merged = tf.summary.merge_all()

    ti = datetime.datetime.now()
    log_dir = TEST_LOG_PATH + '/' + str(ti.month) + '-' + str(ti.day) + '-' + str(ti.hour) + '-' + \
              str(ti.minute) + '-' + str(ti.second) + '/'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_writer = tf.summary.FileWriter(logdir=log_dir)

    test_config.config_dict['BATCH_COUNT'] = len(data.data_list) // test_config.config_dict['BATCH_SIZE']

    for i in range(test_config.config_dict['EPOCH']):
        aver_loss = 0.0
        count = 0
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
            aver_loss = (float(aver_loss) * (count - 1) + loss) / float(count)
        state, label = data.return_batch_data(0, data.sample_count)
        summary = sess.run(fetches=[merged],
                           feed_dict={
                               model.state('STATE'): state,
                               model.label: label
                           })
        train_writer.add_summary(summary[0], global_step=i)
        print("epoch = ", i, " loss = ", aver_loss)


if __name__ == '__main__':
    test_config_standard_key_list = utils.load_json(file_path=CONFIG_STANDARD_KEY_LIST + '/fisTestKeyList.json')
    test_config = Config(config_dict=None, standard_key_list=test_config_standard_key_list)

    test_config_dict = utils.load_json(file_path=CONFIG_PATH + '/testFisTestConfig.json')
    test_config.load_config(json_dict=test_config_dict, path=None)

    model = create_two_layer_dense_model()

    train(test_config, model)
