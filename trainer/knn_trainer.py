import numpy as np
from rpc.grpc_file import vfl_server_service_pb2
from utils import timer
from .basic_trainer import BasicTrainer
from tqdm import tqdm
import time
import tenseal as ts


class KNNTrainer(BasicTrainer):
    def __init__(self, rank, logger, cfg, he_file, device, ip_address, type_,
                 is_label_owner=False, is_regression=False):
        super(KNNTrainer, self).__init__(rank, logger, cfg, he_file, device,
                                         ip_address, type_, is_label_owner, is_regression)

        self.type = 'knn'
        self.batch_size_normal = self.cfg.knn_conf.batch_size_normal
        self.name = self.cfg.knn_conf.dataset_name
        self.top_k = self.cfg.knn_conf.top_k
        self.n_classes = self.cfg.knn_conf.n_classes
        self.k = self.cfg.knn_conf.kmeans.k_per_clients

        self._generate_origin_dataloader()

    @timer
    def train_test_vertical_model(self):
        if self.cfg.defs.vertical_fl.train_type == 'cc':
            self._preprocess_local_data()
            time.sleep(0.5)
        else:
            self._rsa_psi([i for i in range(len(self.origin_train_dataset))],
                          self.ip_address, self.rank, 1)
        if not self.is_label_owner:
            self.__find_top_k()

    @timer
    def __find_top_k(self):
        if self.rank == 0:
            d = {'Clients': self.origin_train_dataset.num_clients}
            for index, test_data in tqdm(enumerate(self.test_dataset.test_data_numpy), desc="Training Process",
                                         postfix=d):
                # print(len(self.test_dataset.test_data_numpy))
                dist = self.__square_euclidean_np(test_data)
                continue_iter = self.__send_knn_msg_to_server(index, dist)

                if not continue_iter:
                    raise RuntimeError
        else:
            for index, test_data in enumerate(self.test_dataset.test_data_numpy):
                dist = self.__square_euclidean_np(test_data)
                continue_iter = self.__send_knn_msg_to_server(index, dist)

                if not continue_iter:
                    raise RuntimeError

    def __send_knn_msg_to_server(self, index, dist):
        vfl_server_stub = self._get_vfl_server_rpc_stub()

        # index = ts.ckks_vector(self.he_key, index)
        # dist = ts.ckks_vector(self.he_key, dist)
        request = vfl_server_service_pb2.knn_distance_request(
            cid=self.rank,
            index=index,
            dist=dist
        )

        response = vfl_server_stub.gather_knn_distance(request)

        return response.continue_iter

    def __square_euclidean_np(self, test_data):
        dist = np.sum((self.train_dataset.train_data_numpy - test_data) ** 2, axis=1)

        return dist

    def _train_client_iteration(self, epoch):
        pass

    def _test_client_iteration(self):
        pass

    def _adjust_learning_rate(self, epoch):
        pass

    def _get_align_item_label(self, index):
        # label_class = None
        if self.name == 'bp':
            label_class = int(self.origin_train_dataset[index].item()) - 1
        else:
            label_class = int(self.origin_train_dataset[index].item())

        # return label_class, self.n_classes
        return label_class