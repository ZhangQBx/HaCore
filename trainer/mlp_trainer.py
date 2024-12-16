import time
import torch
import torch.optim as optim
from model import MLPTopModel, MLPBottomModel
from utils import timer
import rpc.grpc_file.vfl_server_service_pb2 as vfl_server_service_pb2
from .basic_trainer import BasicTrainer
from tqdm import tqdm
import tenseal as ts


class MLPTrainer(BasicTrainer):
    def __init__(self, rank, logger, cfg, device, ip_address, type_,
                 is_label_owner=False, is_regression=False):
        super(MLPTrainer, self).__init__(rank, logger, cfg, device,
                                         ip_address, type_, is_label_owner, is_regression)

        self.type = 'mlp'
        self.epochs = self.cfg.mlp_conf.epochs
        self.batch_size_normal = self.cfg.mlp_conf.batch_size_normal
        self.batch_size_lsh = self.cfg.mlp_conf.batch_size_lsh
        self.n_bottom_out = self.cfg.mlp_conf.n_bottom_out
        self.n_top_out = self.cfg.mlp_conf.n_top_out

        self.lr = self.cfg.mlp_conf.lr
        self.lr_gamma = self.cfg.mlp_conf.lr_gamma
        self.lr_step = self.cfg.mlp_conf.lr_step
        self.bottom_model = MLPBottomModel(self.origin_train_dataset.train_data_tensor.shape[1],
                                           self.n_bottom_out)
        self.bottom_model.to(self.device)
        self.bottom_optimizer = optim.Adam(self.bottom_model.parameters(), lr=self.lr)
        self.k = self.cfg.mlp_conf.kmeans.k_per_clients
        self.bucket_width = self.cfg.mlp_conf.bucket_width
        self.total_num_hash_func = self.cfg.mlp_conf.num_hash_funs

        # self.top_model = MLPTopModel(self.n_bottom_out * self.origin_train_dataset.num_clients,
        #                              self.n_top_out)
        # self.top_model.to(self.device)
        # self.top_optimizer = optim.Adam(self.top_model.parameters(), lr=self.lr)

        self._generate_origin_dataloader()

    def _get_align_item_label(self, index):
        label_list = list(self.origin_train_dataset[index].numpy())
        single_label = label_list.index(1.0)

        return single_label

    def _train_client_iteration(self, epoch):
        self._adjust_learning_rate(epoch)
        for batch_index, data in enumerate(self.train_loader):
            self.bottom_optimizer.zero_grad()
            data = data.to(self.device)
            bottom_f = self.bottom_model(data)
            bottom_f_numpy = bottom_f.detach().cpu().numpy()
            batch_grad, early_stop = self.__send_train_mlp_bottom_msg_to_server(epoch, batch_index, bottom_f_numpy)
            if early_stop:
                self.early_stop = early_stop
                print(">>>Early stop.")
                return
            batch_grad = torch.tensor(batch_grad)
            batch_grad = batch_grad.to(self.device)
            bottom_f.backward(batch_grad)
            self.bottom_optimizer.step()

    def _test_client_iteration(self):
        for batch_index, data in enumerate(self.test_loader):
            data = data.to(self.device)
            bottom_f = self.bottom_model(data)
            bottom_f_numpy = bottom_f.detach().cpu().numpy()
            continue_iter = self.__send_test_mlp_bottom_msg_to_server(batch_index, bottom_f_numpy)

            if not continue_iter:
                raise RuntimeError

    def _adjust_learning_rate(self, epoch):
        if epoch in self.lr_step:
            self.lr *= self.lr_gamma
            for param_group in self.bottom_optimizer.param_groups:
                param_group['lr'] = self.lr
            if self.rank == 0:
                tqdm.write(">>>Learning rate decay.")
                tqdm.write(f">>>Learning rate: {self.lr}")
            self.logger.warning(f"Learning rate decay: {self.lr}")

    def __send_train_mlp_bottom_msg_to_server(self, epoch, batch_index, msg):
        vfl_server_stub = self._get_vfl_server_rpc_stub()
        bottom_forward = self.__generate_bottom_forward_rpc_msg(msg)

        request = vfl_server_service_pb2.mlp_train_bottom_forward_request(
            cid=self.rank,
            batch_index=batch_index,
            epoch=epoch,
            # bottom_forward = enc_bottom_forward.serialize()
            bottom_forward=bottom_forward
        )

        response = vfl_server_stub.gather_mlp_train_bottom_forward(request)
        assert response.cid == self.rank
        batch_grad = []
        for item in response.batch_gradient:
            batch_grad.append(item.grad)

        early_stop = response.early_stop
        # print(f"{self.rank}, {batch_grad}")
        return batch_grad, early_stop

    def __send_test_mlp_bottom_msg_to_server(self, batch_index, msg):
        vfl_server_stub = self._get_vfl_server_rpc_stub()
        bottom_forward = self.__generate_bottom_forward_rpc_msg(msg)
        # enc_bottom_forward = ts.ckks_vector(self.he_key, bottom_forward)
        request = vfl_server_service_pb2.mlp_test_bottom_forward_request(
            cid=self.rank,
            batch_index=batch_index,
            # bottom_forward=enc_bottom_forward.serialize()
            bottom_forward=bottom_forward
        )

        response = vfl_server_stub.gather_mlp_test_bottom_forward(request)
        assert response.cid == self.rank

        return response.continue_iter

    def __generate_bottom_forward_rpc_msg(self, msg):
        bottom_forward = []
        for item in msg:
            single_msg = vfl_server_service_pb2.internal_bottom_forward(
                forward=item
            )
            bottom_forward.append(single_msg)

        return bottom_forward
