import tensorlayer as tl
import tensorflow as tf
from src.model.denseModel import DenseModel
from src.configuration.standard_key_list import CONFIG_STANDARD_KEY_LIST
import src.model.utils.utils as utils
from src.model.inputs.inputs import Inputs


class TwoLayerDenseModel(DenseModel):
    standard_key_list = utils.load_json(file_path=CONFIG_STANDARD_KEY_LIST + '/twoLayerDenseKeyList.json')

    def __init__(self, config, sess_flag=False, data=None):
        super(DenseModel, self).__init__(config, sess_flag, data)
        self.state = Inputs(config=self.config.config_dict['STATE'])
        self.label = tf.placeholder(tf.float32, shape=[None, self.config.config_dict['OUTPUT_DIM']])
        self.net = self.create_model(self.state('STATE'), 'DENSE_')
        self.loss, self.optimizer = self.create_training_method()
        self.optimize_loss = self.optimizer.minimize(loss=self.loss, var_list=self.net.all_params)

    def create_model(self, state, name_prefix):
        net = tl.layers.InputLayer(inputs=state,
                                   name=name_prefix + 'INPUT_LAYER')
        net = tl.layers.DenseLayer(layer=net,
                                   n_units=self.config.config_dict['DENSE_LAYER_1_UNIT'],
                                   act=tf.nn.leaky_relu,
                                   name=name_prefix + 'DENSE_LAYER_1')
        # net = tl.layers.DenseLayer(layer=net,
        #                            n_units=self.config.config_dict['DENSE_LAYER_2_UNIT'],
        #                            act=tf.nn.leaky_relu,
        #                            name=name_prefix + 'DENSE_LAYER_2')

        net = tl.layers.DenseLayer(layer=net,
                                   n_units=self.config.config_dict['OUTPUT_DIM'],
                                   act=tf.nn.softmax,
                                   name=name_prefix + 'OUTPUT_LAYER')

        return net

    def create_training_method(self):
        # weight_decay = tf.add_n([self.config.config_dict['L2'] * tf.nn.l2_loss(var) for var in self.var_list])
        # loss = tf.reduce_mean(tf.square(self.label - self.net.outputs)) + weight_decay
        loss = tf.reduce_mean(tf.square(self.label - self.net.outputs))
        optimizer = tf.train.AdamOptimizer(self.config.config_dict['LEARNING_RATE'])
        return loss, optimizer


if __name__ == '__main__':
    from src.config.config import Config
    from src.configuration import CONFIG_PATH

    a = Config(config_dict=None, standard_key_list=TwoLayerDenseModel.standard_key_list)
    a.load_config(path=CONFIG_PATH + '/model/testTwoLayerDenseAttackConfig.json')
    actor = TwoLayerDenseModel(config=a)
    pass
