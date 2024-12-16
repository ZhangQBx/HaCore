import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import timer
from .basic_dataset import BasicDataset


class BodyPerformance(BasicDataset):
    """
    Generate body performance dataset for vfl
    """

    def __init__(self, cfg, client_rank, train=True, is_label_owner=False):
        super(BodyPerformance, self).__init__(cfg, client_rank, train,
                                              is_label_owner)
        key = ''
        if cfg.trainer == 'mlp':
            key = 'mlp_conf'
            self.one_hot_label = True
        elif cfg.trainer == 'knn':
            key = 'knn_conf'
            self.one_hot_label = False
        self.csv_path = cfg[key].data_path

        self._get_train_and_test_tensor_data()
        # print(self.train_data_tensor.shape)
        # print(self.train_label_tensor.shape)
        # print(self.test_data_tensor.shape)
        # print(self.test_label_tensor.shape)

    # @timer
    def _load_data_from_csv(self):
        data = pd.read_csv(self.csv_path)
        data['gender'] = LabelEncoder().fit_transform(data['gender'])
        for item in zip(['A', 'B', 'C', 'D'], [4, 3, 2, 1]):
            data = data.replace({'class': item[0]}, item[1])

        y = data.iloc[:, -1]
        x = data.iloc[:, :-1]

        return x, y
