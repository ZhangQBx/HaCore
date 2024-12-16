# -*- coding: utf-8 -*-
"""
@Auth ： ZQB
@Time ： 2023/9/30 18:50
@File ：covtype.py
"""
import time

import pandas as pd
from utils import timer
from .basic_dataset import BasicDataset
from sklearn.preprocessing import OneHotEncoder


class Covtype(BasicDataset):
    def __init__(self, cfg, rank, train=True, label_owner=False):
        super(Covtype, self).__init__(cfg, rank, train,
                                      label_owner)
        key = ''
        if cfg.trainer == 'mlp':
            key = 'mlp_conf'
        elif cfg.trainer == 'lr':
            key = 'lr_conf'
        self.csv_path = self.cfg[key].data_path
        self._get_train_and_test_tensor_data()

    def _load_data_from_csv(self):
        data = pd.read_csv(self.csv_path, header=None)

        y = data.iloc[:, -1]
        x = data.iloc[:, :-1]

        # One hot label
        enc = OneHotEncoder()
        y = enc.fit_transform(y.values.reshape(-1, 1)).toarray()
        y = y.astype(int)

        return x, y
