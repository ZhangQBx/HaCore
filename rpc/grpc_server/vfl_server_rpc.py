import numpy as np
from tqdm import tqdm
import rpc.grpc_file.vfl_server_service_pb2_grpc as vfl_server_service_pb2_grpc
import rpc.grpc_file.vfl_server_service_pb2 as vfl_server_service_pb2
import rpc.grpc_file.vfl_label_owner_service_pb2_grpc as vfl_label_owner_service_pb2_grpc
import rpc.grpc_file.vfl_label_owner_service_pb2 as vfl_label_owner_service_pb2
import rpc.grpc_file.vfl_client_service_pb2_grpc as vfl_client_service_pb2_grpc
import rpc.grpc_file.vfl_client_service_pb2 as vfl_client_service_pb2

import time
import grpc
import torch
from array import array
from collections import deque
from model import MLPTopModel
import torch.optim as optim
from bitarray import bitarray


class VFLServer(vfl_server_service_pb2_grpc.VFLServerServiceServicer):
    def __init__(self, num_clients, options, device, cfg, trainer='lr'):
        self.cfg = cfg
        self.num_clients = num_clients
        self.sleep = 0.001
        self.options = options
        self.device = device
        self.client_id = []
        torch.manual_seed(self.cfg.defs.seed)
        self.num_hash_funs = self.cfg.lr_conf.num_hash_funs

        # mlp
        if trainer == 'mlp':
            self.mlp_client_id = []
            self.num_mlp_request = 0
            self.num_mlp_response = 0
            self.epochs = self.cfg.mlp_conf.epochs
            self.current_epoch = 0
            self.lr = self.cfg.mlp_conf.lr
            self.lr_gamma = self.cfg.mlp_conf.lr_gamma
            self.lr_step = self.cfg.mlp_conf.lr_step
            self.n_bottom_out = self.cfg.mlp_conf.n_bottom_out
            self.n_top_out = self.cfg.mlp_conf.n_top_out
            self.top_model = MLPTopModel(self.n_bottom_out * self.num_clients,
                                         self.n_top_out)
            self.top_model.to(self.device)
            self.top_optimizer = optim.Adam(self.top_model.parameters(), lr=self.lr)
            # self.mlp_add_info_queue = deque()
            self.concat_list = [0 for _ in range(self.num_clients)]
            self.bottom_grad_list = None
            self.num_hash_funs = self.cfg.mlp_conf.num_hash_funs

        # train lr
        self.lr_client_id = []
        self.lr_forward_res = []
        self.receive_flag = False
        self.num_lr_request = 0
        self.num_lr_response = 0
        self.current_batch_index = None
        self.sum_lr_forward = None
        self.is_summed = False
        self.batch_grad = None
        self.early_stop = False

        # test lr
        self.is_calculate_accuracy = False
        self.continue_test_iter_flag = False


        # For align index
        self.num_lsh_request = 0
        self.num_lsh_response = 0
        self.lsh_add_info_queue = deque()
        self.align_cluster_label = []
        self.align_distance = []
        self.align_info = []
        self.selected_index = None
        self.hash_table = []
        self.num_samples = 0
        self.hamming_distance = []
        self.weight_list = []
        self.num_features = 0
        self.weight_add_info_queue = deque()
        self.allocated_hash_funs = []
        self.num_weight_response = 0
        self.hash_bits = []


    def gather_lr_train_forward(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        cid = request.cid
        first_request_cid = None
        if not len(self.lr_client_id):
            first_request_cid = cid
            self.current_batch_index = request.batch_index
        while request.batch_index != self.current_batch_index:
            time.sleep(self.sleep)

        forward_result = request.forward_result

        self.lr_client_id.append(cid)
        self.lr_forward_res.append(torch.tensor(forward_result))
        self.num_lr_request += 1

        while self.num_lr_request % self.num_clients != 0:
            time.sleep(self.sleep)
        if cid == first_request_cid:
            self.__sum_lr_forward()
            self.__get_lr_train_batch_gradient_from_label_owner()

        while not self.batch_grad:
            time.sleep(self.sleep)

        response = vfl_server_service_pb2.lr_train_forward_response(
            cid=cid,
            batch_gradient=self.batch_grad,
            early_stop=self.early_stop
        )
        self.num_lr_response += 1

        while self.num_lr_response % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            self.__reset_lr_train_status()
        time.sleep(self.sleep)

        return response

    def gather_lr_test_forward(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        # print(self.is_train)
        cid = request.cid
        first_request_cid = None
        if not len(self.lr_client_id):
            first_request_cid = cid
            self.current_batch_index = request.batch_index
        while request.batch_index != self.current_batch_index:
            time.sleep(self.sleep)


        test_forward = request.test_forward

        self.lr_client_id.append(cid)
        self.lr_forward_res.append(torch.tensor(test_forward))
        self.num_lr_request += 1

        while self.num_lr_request % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            self.__sum_lr_forward()
            response = self.__send_lr_test_forward_result_to_label_owner()
            self.continue_test_iter_flag = response.receive_flag
            self.is_calculate_accuracy = True

        while not self.is_calculate_accuracy:
            time.sleep(self.sleep)

        response = vfl_server_service_pb2.lr_test_forward_response(
            cid=cid,
            continue_iter=self.continue_test_iter_flag
        )
        self.num_lr_response += 1

        while self.num_lr_response % self.num_clients != 0:
            time.sleep(self.sleep)
        if cid == first_request_cid:
            self.__reset_lr_test_status()
        time.sleep(self.sleep)

        return response

    def gather_mlp_train_bottom_forward(self, request, context):
        """

        :param request:
        :param context:
        :return:
        """
        cid = request.cid
        batch_index = request.batch_index
        epoch = request.epoch
        # bottom_forward = request.bottom_forward
        bottom_forward = self.__get_mlp_bottom_forward_rpc_msg(request)

        first_request_cid = None
        if not len(self.mlp_client_id):
            first_request_cid = cid

        self.mlp_client_id.append(cid)
        self.concat_list[cid] = torch.tensor(bottom_forward)
        self.num_mlp_request += 1

        while self.num_mlp_request % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            concat_bottom_forward = torch.cat(self.concat_list, dim=1)
            concat_bottom_forward.requires_grad = True
            concat_bottom_forward.retain_grad()
            top_forward, grad = self.__get_mlp_train_batch_gradient_from_label_owner(concat_bottom_forward,
                                                                                     batch_index)
            # grad = []
            # for item in batch_grad_msg:
            #     grad.append(item.grad)

            self.__adjust_mlp_learning_rate(epoch)
            self.top_optimizer.zero_grad()
            # grad = torch.tensor(grad)
            # grad = grad.to(self.device)
            top_forward.backward(grad)
            self.top_optimizer.step()
            # print(concat_bottom_forward.grad)
            self.__split_bottom_grad(concat_bottom_forward.grad)

        while not self.bottom_grad_list:
            time.sleep(self.sleep)

        # print(f"{cid}, {self.bottom_grad_list[cid]}")
        batch_gradient = []
        for item in self.bottom_grad_list[cid]:
            single_grad = vfl_server_service_pb2.internal_split_grad(
                grad=item
            )
            batch_gradient.append(single_grad)

        response = vfl_server_service_pb2.mlp_train_batch_gradient_response(
            cid=cid,
            batch_gradient=batch_gradient,
            early_stop=self.early_stop
        )
        self.num_mlp_response += 1
        while self.num_mlp_response % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            self.__reset_mlp_train_status()
            # if (self.current_epoch + 1) == self.epochs:
            #     print(">>>Train finish.")
        time.sleep(self.sleep)

        return response

    def gather_mlp_test_bottom_forward(self, request, context):
        cid = request.cid
        batch_index = request.batch_index
        bottom_forward = self.__get_mlp_bottom_forward_rpc_msg(request)

        first_request_cid = None
        if not len(self.mlp_client_id):
            first_request_cid = cid
        self.mlp_client_id.append(cid)
        self.concat_list[cid] = torch.tensor(bottom_forward)
        self.num_mlp_request += 1

        while self.num_mlp_request % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            concat_bottom_forward = torch.cat(self.concat_list, dim=1)
            response = self.__send_mlp_test_forward_result_to_label_owner(concat_bottom_forward,
                                                                          batch_index)
            self.continue_test_iter_flag = response.receive_flag
            self.is_calculate_accuracy = True

        while not self.is_calculate_accuracy:
            time.sleep(self.sleep)

        response = vfl_server_service_pb2.mlp_test_bottom_forward_response(
            cid=cid,
            continue_iter=self.continue_test_iter_flag
        )
        self.num_mlp_response += 1

        while self.num_mlp_response % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            self.__reset_mlp_test_status()
        time.sleep(self.sleep)

        return response



    def get_client_selected_lsh_index(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        cid = request.cid
        hash_infos = request.hash_info

        print("accept")
        first_request_cid = None
        if not len(self.client_id):
            first_request_cid = cid
        self.lsh_add_info_queue.append(cid)
        self.client_id.append(cid)
        while self.lsh_add_info_queue[0] != cid:
            time.sleep(self.sleep)
        if cid == first_request_cid:
            self.num_samples = len(hash_infos)
            print(self.num_samples)
        origin_bit_length = self.allocated_hash_funs[cid]
        bit_length = origin_bit_length + 8 - origin_bit_length % 8
        hash_bits = [bitarray(bit_length) for _ in range(len(hash_infos))]
        for i in range(len(hash_bits)):
            hash_bits[i].frombytes(hash_infos[i])
            hash_bits[i] = hash_bits[i][bit_length:bit_length+origin_bit_length]
        if cid == first_request_cid:
            self.hash_bits = hash_bits
        else:
            for i in range(len(hash_bits)):
                self.hash_bits[i].extend(hash_bits[i])
        self.lsh_add_info_queue.popleft()
        self.num_lsh_request += 1
        while self.num_lsh_request % self.num_clients != 0:
            time.sleep(self.sleep)

        print(len(self.hash_bits[0]))

        if cid == first_request_cid:
            self.hash_table = [bits.tobytes() for bits in self.hash_bits]
            response = self.__send_hash_info_to_label_owner()
            self.selected_index = response.selected_index

        while not self.selected_index:
            time.sleep(self.sleep)

        response = vfl_server_service_pb2.vfl_server_selected_lsh_index_response(
            cid=cid,
            selected_index=self.selected_index
        )
        self.num_lsh_response += 1
        while self.num_lsh_response % self.num_clients != 0:
            time.sleep(self.sleep)

        self.__reset_lsh_index_status()
        time.sleep(self.sleep)

        return response

    def get_client_weight_allocation(self, request, context):

        cid = request.cid
        weight = request.weight
        # num_features = request.num_features
        self.weight_list.append(weight)
        # self.num_features += num_features

        first_request_cid = None
        if not len(self.client_id):
            first_request_cid = cid
        self.weight_add_info_queue.append(cid)
        self.client_id.append(cid)
        while self.weight_add_info_queue[0] != cid:
            time.sleep(self.sleep)
        time.sleep(self.sleep)
        while len(self.weight_list) % self.num_clients != 0:
            time.sleep(self.sleep)
        allocated_hash_funs = round(self.num_hash_funs * (weight / np.sum(self.weight_list)))

        self.allocated_hash_funs.append(allocated_hash_funs)
        self.weight_add_info_queue.popleft()

        while len(self.allocated_hash_funs) % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
           while np.sum(self.allocated_hash_funs) != self.num_hash_funs:
                max_index = np.argmax(self.allocated_hash_funs)
                if np.sum(self.allocated_hash_funs) > self.num_hash_funs:
                    self.allocated_hash_funs[max_index] -= 1
                else:
                    self.allocated_hash_funs[max_index] += 1

        while np.sum(self.allocated_hash_funs) != self.num_hash_funs:
            time.sleep(self.sleep)

        response = vfl_server_service_pb2.client_weight_allocation_response(
            cid=cid,
            num_hash_functions=self.allocated_hash_funs[cid]
        )
        # response = vfl_server_service_pb2.client_weight_allocation_response(
        #     cid=cid,
        #     num_hash_functions=32
        # )
        self.num_weight_response += 1
        while self.num_weight_response % self.num_clients != 0:
            time.sleep(self.sleep)

        self.__reset_weight_index_status()
        time.sleep(self.sleep)

        return response

    def __reset_lr_train_status(self):
        self.lr_client_id = []
        self.lr_forward_res = []
        self.receive_flag = False
        self.num_lr_request = 0
        self.num_lr_response = 0
        self.current_batch_index = None
        self.sum_lr_forward = None
        self.is_summed = False
        self.batch_grad = None
        # print(">>>Reset Status.")

    def __reset_lr_test_status(self):
        self.lr_client_id = []
        self.lr_forward_res = []
        self.num_lr_request = 0
        self.num_lr_response = 0
        self.current_batch_index = None
        self.sum_lr_forward = None
        self.is_summed = False
        self.is_calculate_accuracy = False
        self.continue_test_iter_flag = False

    def __reset_mlp_train_status(self):
        self.mlp_client_id = []
        self.num_mlp_request = 0
        self.num_mlp_response = 0
        # self.current_epoch = 0
        self.concat_list = [0 for _ in range(self.num_clients)]
        self.bottom_grad_list = None

    def __reset_mlp_test_status(self):
        self.mlp_client_id = []
        self.num_mlp_request = 0
        self.num_mlp_response = 0
        self.concat_list = [0 for _ in range(self.num_clients)]
        self.is_calculate_accuracy = False
        self.continue_test_iter_flag = False



    def __reset_lsh_index_status(self):
        self.client_id = []
        self.num_lsh_request = 0
        self.num_lsh_response = 0
        self.lsh_add_info_queue = deque()
        # self.align_cluster_label = []
        # self.align_distance = []
        # self.align_info = []
        self.selected_index = None

    def __reset_weight_index_status(self):
        self.client_id = []
        self.num_weight_response = 0
        self.weight_add_info_queue = deque()
        self.weight_list = []

    def __get_vfl_label_owner_rpc_stub(self):
        vfl_label_owner_address = self.cfg.server_conf.vfl_label_owner.host + ":" + \
                                  self.cfg.server_conf.vfl_label_owner.port
        vfl_label_owner_channel = grpc.insecure_channel(vfl_label_owner_address,
                                                        options=self.options)
        vfl_label_owner_stub = vfl_label_owner_service_pb2_grpc.VFLLabelOwnerServiceStub(vfl_label_owner_channel)

        return vfl_label_owner_stub

    def __sum_lr_forward(self):
        """
        add the forward results from clients together
        :return:
        """
        self.sum_lr_forward = sum(self.lr_forward_res).numpy()
        self.is_summed = True
        # print(self.sum_lr_forward)
        # print(self.sum_lr_forward.decrypt())

    def __send_lr_train_forward_result_to_label_owner(self):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub()

        request = vfl_label_owner_service_pb2.lr_train_forward_request_label_owner(
            server_id=1,
            batch_index=self.current_batch_index,
            # summed_forward_result=self.sum_lr_forward.serialize()
            summed_forward_result=self.sum_lr_forward
        )

        response = vfl_label_owner_stub.get_lr_train_batch_gradient(request)

        return response

    def __send_lr_train_batch_gradient_to_client(self, key, batch_grad):
        lr_client_address = self.cfg.server_conf[key].host + ":" + self.cfg.server_conf[key].port

        lr_client_channel = grpc.insecure_channel(lr_client_address,
                                                  options=self.options)
        lr_client_stub = vfl_client_service_pb2_grpc.VFLClientServiceStub(lr_client_channel)

        request = vfl_client_service_pb2.lr_train_batch_gradient_request(
            server_id=1,
            batch_index=self.current_batch_index,
            # batch_gradient=batch_grad.serialize()
            batch_gradient=batch_grad
        )

        response = lr_client_stub.send_lr_train_batch_gradient(request)

        return response

    def __get_lr_train_batch_gradient_from_label_owner(self):
        assert self.is_summed is True

        response = self.__send_lr_train_forward_result_to_label_owner()
        self.batch_grad = response.batch_gradient
        self.early_stop = response.early_stop


        # return batch_grad

    def __send_lr_test_forward_result_to_label_owner(self):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub()
        request = vfl_label_owner_service_pb2.lr_test_forward_request_label_owner(
            server_id=1,
            batch_index=self.current_batch_index,
            # summed_test_result=self.sum_lr_forward.serialize()
            summed_test_result=self.sum_lr_forward
        )
        response = vfl_label_owner_stub.calculate_lr_test_accuracy(request)

        return response

    def __get_mlp_train_batch_gradient_from_label_owner(self, concat_bottom_f, batch_index):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub()
        top_forward = self.__mlp_top_forward_iteration(concat_bottom_f)
        top_f_numpy = top_forward.detach().cpu().numpy()
        msg = self.__generate_mlp_top_forward_rpc_msg(top_f_numpy)

        request = vfl_label_owner_service_pb2.mlp_train_top_forward_request(
            server_id=1,
            batch_index=batch_index,
            top_forward=msg
        )

        response = vfl_label_owner_stub.get_mlp_train_batch_gradient(request)
        batch_grad = response.batch_gradient
        self.early_stop = response.early_stop

        grad = []
        for item in batch_grad:
            grad.append(item.grad)
        grad = torch.tensor(grad)
        grad = grad.to(self.device)

        return top_forward, grad

    def __send_mlp_test_forward_result_to_label_owner(self, concat_bottom_f, batch_index):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub()
        top_forward = self.__mlp_top_forward_iteration(concat_bottom_f)
        top_f_numpy = top_forward.detach().cpu().numpy()
        msg = self.__generate_mlp_top_forward_rpc_msg(top_f_numpy)

        request = vfl_label_owner_service_pb2.mlp_test_top_forward_request(
            server_id=1,
            batch_index=batch_index,
            top_forward=msg
        )

        response = vfl_label_owner_stub.calculate_mlp_test_accuracy(request)

        return response



    def __send_cluster_info_to_label_owner(self):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub()
        # print(len(self.align_cluster_label))
        # print(len(self.align_cluster_label[0]))
        request = vfl_label_owner_service_pb2.server_cluster_info_request(
            server_id=1,
            info=self.align_info
        )
        response = vfl_label_owner_stub.calculate_client_align_index(request)

        return response

    def __send_hash_info_to_label_owner(self):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub()
        request = vfl_label_owner_service_pb2.server_hash_info_request(
            server_id=1,
            hash_info=self.hash_table
        )
        response = vfl_label_owner_stub.calculate_client_selected_lsh_index(request)

        return response

    def __mlp_top_forward_iteration(self, concat_bottom_f):
        concat_bottom_f = concat_bottom_f.to(self.device)
        top_forward = self.top_model(concat_bottom_f)
        # top_f_numpy = top_forward.detach().cpu().numpy()
        # print(top_f_numpy)

        return top_forward

    def __adjust_mlp_learning_rate(self, epoch):
        if self.current_epoch != epoch:
            self.current_epoch = epoch
            if self.current_epoch in self.lr_step:
                self.lr *= self.lr_gamma
                for param_group in self.top_optimizer.param_groups:
                    param_group['lr'] = self.lr

                print(f">>>Epoch{epoch}: Learning rate decay.")
                print(">>>Learning rate: ", self.lr)
        else:
            return

    def __split_bottom_grad(self, bottom_grad):
        shape_list = [self.n_bottom_out] * self.num_clients
        split_list = list(torch.split(bottom_grad, shape_list, dim=-1))
        bottom_grad_list = []
        for item in split_list:
            bottom_grad_list.append(item.numpy())

        self.bottom_grad_list = bottom_grad_list

    def __get_mlp_bottom_forward_rpc_msg(self, request):
        bottom_forward = []
        for item in request.bottom_forward:
            bottom_forward.append(item.forward)

        return bottom_forward

    def __generate_mlp_top_forward_rpc_msg(self, top_f_numpy):
        msg = []
        for item in top_f_numpy:
            single_msg = vfl_label_owner_service_pb2.internal_top_forward(
                forward=item
            )
            msg.append(single_msg)

        return msg



    def _adjust_learning_rate(self, epoch, optimizer):
        if epoch in self.lr_step:
            self.lr = self.lr * self.lr_gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
            tqdm.write(f"Learning rate decay: {self.lr}")

    @staticmethod
    def calculate_hamming_distance(s1, s2):
        hamming_distance = sum(c1 != c2 for c1, c2 in zip(s1, s2))
        return hamming_distance
