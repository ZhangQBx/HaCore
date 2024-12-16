import numpy as np
import pandas as pd
from utils import timer
from .basic_dataset import BasicDataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time


class Mushrooms(BasicDataset):
    """
    Generate Mushrooms dataset for vfl
    """

    def __init__(self, cfg, client_rank, train=True, is_label_owner=False):
        super(Mushrooms, self).__init__(cfg, client_rank, train,
                                        is_label_owner)
        key = ''
        if cfg.trainer == 'mlp':
            key = 'mlp_conf'
        elif cfg.trainer == 'lr':
            key = 'lr_conf'
        self.csv_path = cfg[key].data_path

        self._get_train_and_test_tensor_data()
        # print(self.train_data_tensor.shape)
        # print(self.train_label_tensor.shape)
        # print(self.test_data_tensor.shape)
        # print(self.test_label_tensor.shape)

    # @timer
    def _load_data_from_csv(self):
        data = pd.read_csv(self.csv_path)
        for feature in data.columns:
            data[feature] = LabelEncoder().fit_transform(data[feature])

        x = data.drop('class', axis=1)
        y = data['class']


        if self.cfg.trainer == 'mlp':
            enc = OneHotEncoder()
            y = enc.fit_transform(y.values.reshape(-1, 1)).toarray()
            y = y.astype(int)

        return x, y
