import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from utils import timer
from .basic_dataset import BasicDataset


class SUSY(BasicDataset):
    def __init__(self, cfg, rank, train=True, label_owner=False):
        super(SUSY, self).__init__(cfg, rank, train,
                                   label_owner)
        key = ''
        if cfg.trainer == 'mlp':
            key = 'mlp_conf'
        elif cfg.trainer == 'lr':
            key = 'lr_conf'
        self.csv_path = self.cfg[key].data_path
        # self.num_train_data = 4500000
        self._get_train_and_test_tensor_data()

    def _load_data_from_csv(self):
        data = pd.read_csv(self.csv_path)
        data = data.sample(1000000)
        # data = data.drop('id', axis=1)

        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # One hot label
        if self.cfg.trainer == 'mlp':
            enc = OneHotEncoder()
            y = enc.fit_transform(y.values.reshape(-1, 1)).toarray()
            y = y.astype(int)

        return x, y
