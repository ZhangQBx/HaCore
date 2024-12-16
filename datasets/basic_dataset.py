
import random
import time

import numpy as np
import torch
import torch.utils.data as tud
from utils import timer
from sklearn.preprocessing import StandardScaler, LabelBinarizer


def label_binarizer(y):
    label_as_one_hot = LabelBinarizer()
    y = label_as_one_hot.fit_transform(y)

    return y


def shuffle_data(x, y):
    """
    shuffle data
    :param x: x
    :param y: y
    :return:
    """
    index = [i for i in range(len(x))]
    random.shuffle(index)
    # print(f"index: {index[:20]}")

    x = x[index]
    y = y[index]

    return x, y


class BasicDataset(tud.Dataset):
    """
    basic dataset class
    """

    def __init__(self, cfg, client_rank, train=True, is_label_owner=False):
        seed = cfg.defs.seed
        random.seed(seed)
        np.random.seed(seed)
        super(BasicDataset, self).__init__()
        self.cfg = cfg
        self.client_rank = client_rank
        self.train = train
        self.is_label_owner = is_label_owner
        self.split_rate = cfg.defs.vertical_fl.train_test_split_rate
        self.num_clients = cfg.defs.num_clients
        self.num_features = None

        self.shuffle_data = True
        self.num_train_data = None
        self.one_hot_label = False

        self.origin_data = None
        self.train_data_numpy = None
        self.train_label_numpy = None
        self.train_data_tensor = None
        self.train_label_tensor = None

        self.test_data_numpy = None
        self.test_label_numpy = None
        self.test_data_tensor = None
        self.test_label_tensor = None

    def __len__(self):
        if self.train:
            if self.is_label_owner:
                return len(self.train_label_tensor)
            return len(self.train_data_tensor)
        else:
            if self.is_label_owner:
                return len(self.test_label_tensor)
            return len(self.test_data_tensor)

    def __getitem__(self, index):
        if self.train:
            if self.is_label_owner:
                return self.train_label_tensor[index]
            return self.train_data_tensor[index]
        else:
            if self.is_label_owner:
                return (self.test_data_tensor[index], self.test_label_tensor[index])
            return self.test_data_tensor[index]

    def update_dataset_via_indexes(self, indexes):
        # print(self.train_data_numpy)
        self.train_data_numpy = np.asarray([self.train_data_numpy[i].tolist() for i in indexes])
        self.train_label_numpy = np.asarray([self.train_label_numpy[i].tolist() for i in indexes])
        print(f"New length: {len(self.train_label_numpy)}")

        self.train_data_tensor = torch.from_numpy(self.train_data_numpy).float()
        self.train_label_tensor = torch.from_numpy(self.train_label_numpy).float()

        return self

    # @timer
    def _get_train_and_test_tensor_data(self):
        x, y, origin_data = self.__get_all_numpy_data()
        if not self.is_label_owner:
            x, origin_data = self.__generate_vertical_numpy_data(x, origin_data)

        self.origin_data = origin_data[:self.num_train_data]
        self.train_data_numpy = x[:self.num_train_data]
        self.train_label_numpy = y[:self.num_train_data]

        self.test_data_numpy = x[self.num_train_data:]
        self.test_label_numpy = y[self.num_train_data:]

        self.train_data_tensor = torch.from_numpy(x[:self.num_train_data]).float()
        self.train_label_tensor = torch.from_numpy(y[:self.num_train_data]).float()

        self.test_data_tensor = torch.from_numpy(x[self.num_train_data:]).float()
        self.test_label_tensor = torch.from_numpy(y[self.num_train_data:]).float()

        # print(len(self.train_data_tensor))

    def __generate_vertical_numpy_data(self, data, origin_data):
        """

        :param data: data need to be split
        :return:
        """
        num_data_features = data.shape[1]
        num_split_features = num_data_features // self.num_clients
        data_feature_indexes = [index for index in range(num_data_features)]
        random.shuffle(data_feature_indexes)

        data_feature_index_begin = self.client_rank * num_split_features

        index = data_feature_indexes[data_feature_index_begin:data_feature_index_begin + num_split_features]
        if self.client_rank == self.num_clients - 1:
            index = data_feature_indexes[data_feature_index_begin:]
            client_data = self.__generate_client_specific_numpy_data(index, data)
            origin_data = self.__generate_client_origin_specific_numpy_data(index, origin_data)
        else:
            client_data = self.__generate_client_specific_numpy_data(index, data)
            origin_data = self.__generate_client_origin_specific_numpy_data(index, origin_data)
        # print(index)
        self.num_features = len(index)

        return client_data, origin_data

    # @timer
    def __generate_client_specific_numpy_data(self, feature_indexes, data):
        """

        :param feature_indexes: chosen feature index
        :param data: preprocess data
        :return: client specific data :type:ndarray
        """
        client_data = []
        for data_item in data:
            client_data_item = [data_item[feature_index] for feature_index in feature_indexes]
            client_data.append(client_data_item)

        return np.asarray(client_data)

    def __generate_client_origin_specific_numpy_data(self, feature_indexes, data):
        """

        :param feature_indexes: chosen feature index
        :param data: preprocess data
        :return: client specific data :type:ndarray
        """
        client_data = []
        for data_item in data:
            client_data_item = [data_item[feature_index] for feature_index in feature_indexes]
            client_data.append(client_data_item)

        return np.asarray(client_data)

    def __get_all_numpy_data(self):
        x, y = self._load_data_from_csv()
        origin_data = x.to_numpy()
        scalar = StandardScaler()
        x = scalar.fit_transform(x)
        x = np.asarray(x)
        y = np.asarray(y)

        if self.one_hot_label:
            y = label_binarizer(y)
        if self.shuffle_data:
            x, y = shuffle_data(x, y)
        if self.num_train_data is None:
            self.num_train_data = int(len(x) * self.split_rate)
        # print("num_train_data:",  self.num_train_data)

        return x, y, origin_data

    def _load_data_from_csv(self):
        raise NotImplementedError
