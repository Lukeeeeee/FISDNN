

class Data(object):
    standard_key_list = None

    def __init__(self, config=None):
        self.config = config
        self.data_list = []

    def load_data(self, *args, **kwargs):
        pass

    def return_batch_data(self, *args, **kwargs):
        pass

