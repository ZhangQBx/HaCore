import numpy as np
import pandas as pd
from utils import timer
from .basic_dataset import BasicDataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time


class Higgs(BasicDataset):
    """
    Generate Higgs-Bosen dataset for vfl
    """

    def __init__(self, cfg, client_rank, train=True, is_label_owner=False):
        super(Higgs, self).__init__(cfg, client_rank, train,
                                    is_label_owner)
        key = ''
        if cfg.trainer == 'mlp':
            key = 'mlp_conf'
        elif cfg.trainer == 'lr':
            key = 'lr_conf'
        elif cfg.trainer == 'knn':
            key = 'knn_conf'
        self.csv_path = cfg[key].data_path
        self.shuffle_data = False
        self._get_train_and_test_tensor_data()
        # print(self.train_data_tensor.shape)
        # print(self.train_label_tensor.shape)
        # print(self.test_data_tensor.shape)

        # print(self.test_label_tensor.shape)

    # @timer
    # def _load_data_from_csv(self):
    #     data = pd.read_csv(self.csv_path)
    #     data['label'] = LabelEncoder().fit_transform(data['label'])
    #     # print(data['Label'])
    #     data = data.sample(100000)
    #     x = data.drop(['label', 'EventId'], axis=1, inplace=False)
    #     y = data['label']
    #
    #     if self.cfg.trainer == 'mlp':
    #         enc = OneHotEncoder()
    #         y = enc.fit_transform(y.values.reshape(-1, 1)).toarray()
    #         y = y.astype(np.int)
    #
    #     return x, y

    def _load_data_from_csv(self):
        data = pd.read_csv(self.csv_path, header=None)
        data = data.sample(1000000)

        y = data.iloc[:, -1]
        x = data.iloc[:, :-1]

        # One hot label
        if self.cfg.trainer == 'mlp':
            enc = OneHotEncoder()
            y = enc.fit_transform(y.values.reshape(-1, 1)).toarray()
            y = y.astype(int)

        return x, y