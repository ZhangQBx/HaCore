import time
import json
import rpc.grpc_file.vfl_label_owner_service_pb2_grpc as vfl_label_owner_service_pb2_grpc
import rpc.grpc_file.vfl_label_owner_service_pb2 as vfl_label_owner_service_pb2
from trainer import LRTrainer, MLPTrainer
from utils import prepare_logger
import torch
import numpy as np
import copy
import math
from bitarray import bitarray

class VFLLabelOwner(vfl_label_owner_service_pb2_grpc.VFLLabelOwnerServiceServicer):
    def __init__(self, rank, log_path, cfg, device, ip_address,
                 is_label_owner=True, is_regression=False, trainer='lr'):
        logger = prepare_logger(rank, log_path, cfg.defs.mode)
        if trainer == 'lr':
            self.trainer = LRTrainer(rank, logger, cfg, device, ip_address,
                                     trainer, is_label_owner, is_regression)

            self.criterion = torch.nn.BCELoss()
            self.num_hash_func = cfg.lr_conf.num_hash_funs
        elif trainer == 'mlp':
            self.trainer = MLPTrainer(rank, logger, cfg, device, ip_address,
                                      trainer, is_label_owner, is_regression)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.num_hash_func = cfg.mlp_conf.num_hash_funs


        if self.trainer.is_regression:
            self.criterion = torch.nn.MSELoss()

        # For train and test model
        self.epoch = 0
        self.epoch_loss = 0
        self.epoch_loss_list = []
        self.right_pred = 0
        self.current_batch_index = -1
        self.test_success_flag = False
        self.use_weight = cfg.defs.use_weight
        self.hash_info = []
        self.weight_list = None

    def get_lr_train_batch_gradient(self, request, context):
        """
        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        server_id = request.server_id
        batch_index = request.batch_index

        assert ((self.current_batch_index + 1) == batch_index) or \
               ((self.current_batch_index - batch_index) == self.trainer.num_train_batches - 1)
        self.current_batch_index = batch_index

        summed_forward_result = request.summed_forward_result
        summed_forward_result_tensor = torch.tensor(summed_forward_result, requires_grad=True)

        grad, early_stop = self.__calculate_lr_train_batch_gradient(summed_forward_result_tensor)
        grad = grad.numpy()
        # print(grad)

        response = vfl_label_owner_service_pb2.lr_train_forward_response_label_owner(
            server_id=server_id,
            batch_gradient=grad,
            early_stop=early_stop
        )

        return response

    def calculate_lr_test_accuracy(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        server_id = request.server_id
        batch_index = request.batch_index
        assert ((self.current_batch_index + 1) == batch_index) or \
               ((self.current_batch_index - batch_index) == self.trainer.num_train_batches - 1)
        self.current_batch_index = batch_index

        test_result = request.summed_test_result
        test_result_tensor = torch.tensor(test_result)

        self.__calculate_lr_test_batch_accuracy(test_result_tensor)
        response = vfl_label_owner_service_pb2.lr_test_forward_response_label_owner(
            server_id=server_id,
            receive_flag=self.test_success_flag
        )

        self.test_success_flag = False

        return response

    def get_mlp_train_batch_gradient(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        server_id = request.server_id
        batch_index = request.batch_index
        # assert ((self.current_batch_index + 1) == batch_index) or \
        #        ((self.current_batch_index - batch_index) == self.trainer.num_train_batches - 1)
        self.current_batch_index = batch_index

        top_forward = self.__get_mlp_top_forward_rpc_msg(request)
        # print(f">>>{top_forward}")
        top_forward_tensor = torch.tensor(top_forward, dtype=torch.float, requires_grad=True)
        # print(top_forward_tensor)
        grad, early_stop = self.__calculate_mlp_train_batch_gradient(top_forward_tensor)
        grad = grad.numpy()

        batch_grad = []
        for item in grad:
            single_grad = vfl_label_owner_service_pb2.internal_batch_gradient(
                grad=item
            )
            batch_grad.append(single_grad)

        response = vfl_label_owner_service_pb2.mlp_train_top_forward_response(
            server_id=server_id,
            batch_gradient=batch_grad,
            early_stop=early_stop
        )

        return response

    def calculate_mlp_test_accuracy(self, request, context):
        server_id = request.server_id
        batch_index = request.batch_index

        # assert ((self.current_batch_index + 1) == batch_index) or \
        #        ((self.current_batch_index - batch_index) == self.trainer.num_train_batches - 1)
        self.current_batch_index = batch_index

        top_forward = self.__get_mlp_top_forward_rpc_msg(request)
        top_forward_tensor = torch.tensor(top_forward)

        self.__calculate_mlp_test_batch_accuracy(top_forward_tensor)
        response = vfl_label_owner_service_pb2.mlp_test_top_forward_response(
            server_id=server_id,
            receive_flag=self.test_success_flag
        )

        self.test_success_flag = False

        return response



    def calculate_client_selected_lsh_index(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        server_id = request.server_id
        hash_info = request.hash_info
        self.hash_info = ['' for _ in range(len(hash_info))]
        for i, hash_bytes in enumerate(hash_info):
            hash_bit = bitarray(self.num_hash_func)
            hash_bit.frombytes(hash_bytes)
            hash_bit = hash_bit[self.num_hash_func:]
            self.hash_info[i] = hash_bit

        selected_index, weight_list = self.trainer.select_lsh_index(self.hash_info)

        with open('id.txt', 'w', encoding='utf-8') as file:
            for id in selected_index:
                file.write(str(id) + '\n')
        file.close()
        with open('weight.txt', 'w', encoding='utf-8') as file:
            for w in weight_list:
                file.write(str(w) + '\n')
        file.close()
        response = vfl_label_owner_service_pb2.label_owner_selected_lsh_index_response(
            server_id=server_id,
            selected_index=selected_index,
        )
        weight_list = np.array(weight_list)
        self.weight_list = torch.from_numpy(weight_list.reshape(-1, 1)).float()
        return response


    def __reset_train_status(self):
        self.current_batch_index = -1
        self.epoch = 0
        self.epoch_loss = 0

    def __reset_test_status(self):
        self.current_batch_index = -1
        self.right_pred = 0
        self.test_success_flag = False

    def __calculate_lr_train_batch_gradient(self, lr_f):
        """
        use lr forward result to calculate gradient
        :return:
        """
        # loss = None
        grad = None
        early_stop = False
        for batch_index, label in enumerate(self.trainer.train_loader):
            if batch_index != self.current_batch_index:
                continue
            num_train_batches = self.trainer.num_train_batches
            batch_size = self.trainer.batch_size
            batch_weight = None
            if self.weight_list is None and self.trainer.weight_list is not None:
                self.weight_list = torch.from_numpy(np.array(self.trainer.weight_list).reshape(-1, 1)).float()
            if self.weight_list is not None:
                start = batch_index * batch_size
                if batch_index < num_train_batches:
                    end = start + batch_size
                    batch_weight = self.weight_list[start:end, :].reshape(-1)
                else:
                    batch_weight = self.weight_list[start:, :].reshape(-1)

            lr_f.retain_grad()
            # print(f">>>{forward_result_cuda}")
            # label = label.unsqueeze(dim=1)
            # print(label.shape)
            # print(forward_result.shape)
            if self.trainer.is_regression:
                if self.use_weight:
                    criterion = torch.nn.MSELoss(reduction='none')
                    loss = criterion(lr_f, label)
                    loss *= batch_weight
                    loss = torch.mean(loss)

                else:
                    loss = self.criterion(lr_f, label)
            else:
                h = torch.sigmoid(lr_f)
                if batch_weight is None:
                    loss = self.criterion(h, label)
                else:
                    if self.use_weight:
                        criterion = torch.nn.BCELoss(batch_weight)
                    else:
                        criterion = self.criterion
                    loss = criterion(h, label)
            loss.backward()
            # print(f">>>{forward_result_cuda.grad}")
            grad = lr_f.grad
            self.epoch_loss += loss

            if self.current_batch_index == num_train_batches - 1:
                print(f">>>Epoch:{self.epoch + 1}, complete. (Label Owner).")
                print(f">>>Loss {float(self.epoch_loss)}. (Label Owner).")
                self.trainer.logger.info(f">>>Epoch: {self.epoch + 1}, train loss: {self.epoch_loss}")
                self.epoch += 1
                self.epoch_loss_list.append(self.epoch_loss.item())
                # print(self.epoch_loss_list)
                if len(self.epoch_loss_list) >= 5:
                    print(self.epoch_loss_list[-5:])
                    max_loss = max(self.epoch_loss_list[-5:])
                    min_loss = min(self.epoch_loss_list[-5:])
                    if max_loss - min_loss <= 1e-4:
                        early_stop = True
                        print(">>>Early Stop")
                        print(">>>LR Train finish.")
                self.epoch_loss = 0

                if self.trainer.epochs == self.epoch:
                    self.__reset_train_status()
                    print(">>>LR Train finish.")
            break

        # print(loss)
        return grad, early_stop

    def __calculate_lr_test_batch_accuracy(self, lr_f):
        """
        use lr forward result to calculate accuracy
        :return:
        """
        for batch_index, (data,label) in enumerate(self.trainer.test_loader):
            if batch_index != self.current_batch_index:
                continue
            if self.trainer.is_regression:
                self.right_pred += torch.sum((label - lr_f) ** 2)
            else:
                y_pred_sig = torch.sigmoid(lr_f).detach().numpy()
                prediction = (y_pred_sig > 0.5).astype(int)
                self.right_pred += np.sum(prediction == label.numpy())

            if self.current_batch_index == (self.trainer.num_test_batches - 1):
                if self.trainer.is_regression:
                    print(f">>>Test MSE:{self.right_pred / len(self.trainer.test_dataset)}")
                    print(f">>>Test RMSE:{math.sqrt(self.right_pred / len(self.trainer.test_dataset)) }")
                    self.trainer.logger.warning(f">>>Test MSE:{self.right_pred / len(self.trainer.test_dataset)}")
                else:
                    print(f">>>Test acc:{self.right_pred / len(self.trainer.test_dataset)}")
                    self.trainer.logger.warning(f">>>Test acc:{self.right_pred / len(self.trainer.test_dataset)}")
                self.__reset_test_status()
            self.test_success_flag = True

    def __calculate_mlp_train_batch_gradient(self, mlp_f):
        """
        use lr forward result to calculate gradient
        :param mlp_f:
        :return:
        """
        grad = None
        early_stop = False
        for batch_index, label in enumerate(self.trainer.train_loader):
            if batch_index != self.current_batch_index:
                continue
            num_train_batches = self.trainer.num_train_batches
            batch_size = self.trainer.batch_size
            batch_weight = None
            if self.weight_list is None and self.trainer.weight_list is not None:
                self.weight_list = torch.from_numpy(np.array(self.trainer.weight_list).reshape(-1, 1)).float()
            if self.weight_list is not None:
                start = batch_index * batch_size

                if batch_index < num_train_batches:
                    end = start + batch_size
                    batch_weight = self.weight_list[start:end, :].reshape(-1)
                else:
                    batch_weight = self.weight_list[start:, :].reshape(-1)
            # print(batch_index)
            mlp_f.retain_grad()
            # loss = self.criterion(mlp_f, label.long())
            if batch_weight is None:
                loss = self.criterion(mlp_f, label)
            else:
                if self.use_weight:
                    criterion = torch.nn.CrossEntropyLoss(reduction='none')
                    if self.trainer.is_regression:
                        label = label.unsqueeze(1)
                        batch_weight = batch_weight.unsqueeze(1)
                        criterion = torch.nn.MSELoss(reduction='none')
                    loss = criterion(mlp_f, label)
                    loss *= batch_weight
                    loss = torch.mean(loss)
                else:
                    loss = self.criterion(mlp_f, label)

            loss.backward()
            grad = mlp_f.grad

            self.epoch_loss += loss

            if self.current_batch_index == (num_train_batches - 1):
                print(f">>>Epoch:{self.epoch + 1}, complete. (Label Owner).")
                self.trainer.logger.info(f">>>Epoch: {self.epoch + 1}, train loss: {self.epoch_loss}")
                self.epoch += 1
                self.epoch_loss_list.append(self.epoch_loss.item())
                # print(self.epoch_loss_list)
                if len(self.epoch_loss_list) >= 5:
                    print(self.epoch_loss_list[-5:])
                    max_loss = max(self.epoch_loss_list[-5:])
                    min_loss = min(self.epoch_loss_list[-5:])
                    if max_loss - min_loss <= 1e-4:
                        early_stop = True
                        print(">>>Early Stop")
                        print(">>>MLP Train finish.")
                self.epoch_loss = 0

                if self.trainer.epochs == self.epoch:
                    self.__reset_train_status()
                    print(">>>MLP Train finish.")
            break

        return grad, early_stop

    def __calculate_mlp_test_batch_accuracy(self, mlp_f):
        for batch_index, (data, label) in enumerate(self.trainer.test_loader):
            if batch_index != self.current_batch_index:
                continue
            if self.trainer.is_regression:
                # predicted = torch.from_numpy(mlp_f.nonzero().numpy())
                self.right_pred += torch.sum((label - mlp_f) ** 2)
            else:
                _, predicted = mlp_f.max(1)
                label = torch.from_numpy(label.nonzero().numpy()[:, 1])
                self.right_pred += predicted.eq(label).sum().item()

            if self.current_batch_index == self.trainer.num_test_batches - 1:
                if self.trainer.is_regression:
                    print(f">>>Test MSE:{self.right_pred / len(self.trainer.test_dataset)}")
                    print(f">>>Test RMSE:{math.sqrt(self.right_pred / len(self.trainer.test_dataset)) }")
                    self.trainer.logger.warning(f">>>Test MSE:{self.right_pred / len(self.trainer.test_dataset)}")
                else:
                    print(f">>>Test acc:{self.right_pred / len(self.trainer.test_dataset)}")
                    self.trainer.logger.warning(f">>>Test acc:{self.right_pred / len(self.trainer.test_dataset)}")
                self.__reset_test_status()
            self.test_success_flag = True

    def __get_mlp_top_forward_rpc_msg(self, request):
        top_forward = []
        for item in request.top_forward:
            top_forward.append(item.forward)

        return top_forward

