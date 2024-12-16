import numpy as np
import time

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from .basic_dataset import BasicDataset
from utils import timer


class Bank(BasicDataset):
    """
    generate bank data for vfl...
    """

    def __init__(self, cfg, client_rank, train=True, is_label_owner=False):
        super(Bank, self).__init__(cfg, client_rank, train,
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

    # @timer
    def _load_data_from_csv(self):
        data = pd.read_csv(self.csv_path)
        data = data.drop(['customer_id'], axis=1)
        data['country'] = LabelEncoder().fit_transform(data['country'])
        data['gender'] = LabelEncoder().fit_transform(data['gender'])

        y = data.iloc[:, -1]
        x = data.iloc[:, :-1]
        if self.cfg.trainer == 'mlp':
            enc = OneHotEncoder()
            y = enc.fit_transform(y.values.reshape(-1, 1)).toarray()
            y = y.astype(int)

        return x, y
