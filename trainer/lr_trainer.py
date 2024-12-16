import torch
import tenseal as ts
import torch.optim as optim
from rpc.grpc_file import vfl_server_service_pb2
from model import Linear
from utils import timer
from .basic_trainer import BasicTrainer
from tqdm import tqdm


class LRTrainer(BasicTrainer):
    def __init__(self, rank, logger, cfg, device, ip_address, type_,
                 is_label_owner=False, is_regression=False):
        super(LRTrainer, self).__init__(rank, logger, cfg, device,
                                        ip_address, type_, is_label_owner, is_regression)

        self.type = 'lr'
        self.epochs = self.cfg.lr_conf.epochs
        self.batch_size_normal = self.cfg.lr_conf.batch_size_normal
        self.batch_size_lsh = self.cfg.lr_conf.batch_size_lsh
        self.model = Linear(self.origin_train_dataset.train_data_tensor.shape[1])
        self.model.to(self.device)
        self.lr = self.cfg.lr_conf.lr
        self.lr_gamma = self.cfg.lr_conf.lr_gamma
        self.lr_step = self.cfg.lr_conf.lr_step
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.k = self.cfg.lr_conf.kmeans.k_per_clients
        self.current_batch_index = -1
        self.batch_gradient = None
        self.is_batch_gradient_update = False
        self.is_generate_response = False
        self.bucket_width = self.cfg.lr_conf.bucket_width
        self.total_num_hash_func = self.cfg.lr_conf.num_hash_funs

        self._generate_origin_dataloader()

    def _get_align_item_label(self, index):
        if self.is_regression:
            return int(self.origin_train_dataset[index].item())#, 89
        else:
            return int(self.origin_train_dataset[index].item())#, 2

    def _train_client_iteration(self, epoch):
        # self._adjust_learning_rate(epoch)
        for batch_index, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            # data = data.to(self.device)
            y_pred = self.model(data)

            # y_pred_numpy = y_pred.detach().cpu().numpy()
            y_pred_numpy = y_pred.detach().numpy()
            batch_grad, early_stop = self.__send_train_lr_msg_to_server(batch_index, y_pred_numpy)
            if early_stop:
                self.early_stop = early_stop
                print(">>>Early stop.")
                return
            grad = torch.tensor(batch_grad).reshape(-1, 1)
            y_pred.backward(grad)
            self.optimizer.step()

    def _test_client_iteration(self):
        for batch_index, data in enumerate(self.test_loader):
            # data = data.to(self.device)
            y_pred = self.model(data)

            y_pred_numpy = y_pred.detach().numpy()
            continue_iter = self.__send_test_lr_msg_to_server(batch_index, y_pred_numpy)

            if not continue_iter:
                raise RuntimeError

    def _adjust_learning_rate(self, epoch):

        """
        lr version
        :return:
        """
        assert self.optimizer is not None
        if epoch in self.lr_step:
            self.lr *= self.lr_gamma

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

            if self.rank == 0:
                tqdm.write(">>>Learning rate decay.")
                tqdm.write(f">>>Learning rate: {self.lr}")
            self.logger.warning(f"Learning rate decay: {self.lr}")

    def __send_train_lr_msg_to_server(self, batch_index, msg):
        """
        use grpc to deliver message
        :param msg: forward result
        :return:
        """
        # print(msg.shape)
        msg = msg.reshape(-1)
        # print(msg)
        # print(msg.shape)
        vfl_server_stub = self._get_vfl_server_rpc_stub()

        # enc_msg = ts.ckks_vector(self.he_key, msg)
        request = vfl_server_service_pb2.lr_train_forward_request(
            cid=self.rank,
            batch_index=batch_index,
            # forward_result=enc_msg.serialize()
            forward_result=msg
        )
        # print(">>>Send lr forward result to server")

        response = vfl_server_stub.gather_lr_train_forward(request)
        batch_grad = response.batch_gradient
        early_stop = response.early_stop
        # print(f"{self.rank}: {receive_flag}")

        return batch_grad, early_stop

    def __send_test_lr_msg_to_server(self, batch_index, msg):
        msg = msg.reshape(-1)
        lr_server_stub = self._get_vfl_server_rpc_stub()

        # enc_msg = ts.ckks_vector(self.he_key, msg)
        request = vfl_server_service_pb2.lr_test_forward_request(
            cid=self.rank,
            batch_index=batch_index,
            # test_forward=enc_msg.serialize()
            test_forward=msg
        )

        response = lr_server_stub.gather_lr_test_forward(request)
        continue_iter = response.continue_iter

        return continue_iter
