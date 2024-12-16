import random

import pandas as pd
import torch
import torch.utils.data as tud
from datasets import dataset
import numpy as np
from utils import timer
import rpc.grpc_file.vfl_server_service_pb2_grpc as vfl_server_service_pb2_grpc
import rpc.grpc_file.vfl_server_service_pb2 as vfl_server_service_pb2
import rpc.grpc_file.vfl_client_service_pb2_grpc as vfl_client_service_pb2_grpc
import rpc.grpc_file.vfl_label_owner_service_pb2_grpc as vfl_label_owner_service_pb2_grpc
from utils.lsh import BinaryLSH
import time
import grpc
from tqdm import tqdm
from copy import deepcopy
from bitarray import bitarray
from collections import Counter
from itertools import product
import multiprocessing as mp
import tenseal as ts


def c_hamming_distance(x, y):
    """Compute the Hamming distance between two binary vectors."""
    return (x ^ y).count()


def sample_bits(bitarr, indices):
    sampled_bits = bitarr[indices]
    return sampled_bits


def find_keys_by_value(d, value):
    return [k for k, v in d.items() if v == value]


class BasicTrainer:
    def __init__(self, rank, logger, cfg, device, ip_address,
                 type_, is_label_owner=False, is_regression=False):
        self.type = type_
        self.rank = rank
        self.logger = logger
        self.cfg = cfg
        self.device = device
        self.is_label_owner = is_label_owner
        self.epochs = None
        self.batch_size = None
        self.batch_size_normal = None
        self.batch_size_lsh = None
        self.lr = None
        self.lr_gamma = None
        self.lr_step = None
        self.optimizer = None
        self.early_stop = False
        self.is_regression = is_regression
        self.sleep = 0.001
        self.ip_address = ip_address

        max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', max_msg_size),
                        ('grpc.max_receive_message_length', max_msg_size)]

        # Initialize seed
        self.seed = self.cfg.defs.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.origin_train_dataset = dataset(cfg, rank, True, is_label_owner, self.type)
        self.lsh_train_dataset = None
        self.train_dataset = None
        self.test_dataset = dataset(cfg, rank, False, is_label_owner, self.type)

        self.lsh = None
        self.hash_tables = None
        self.hash_info = []
        self.bucket_width = None
        self.weight = None
        self.num_hash_functions = None
        self.selected_index = None
        self.bucket_width = None
        self.total_num_hash_func = None

        self.origin_train_loader = None
        self.lsh_train_loader = None

        self.train_loader = None
        self.test_loader = None

        self.num_origin_train_batches = None
        self.num_lsh_train_batches = None

        self.num_train_batches = None
        self.num_test_batches = None

        # For KMEANS
        self.k = None
        self.weight_list = None

        self.use_lsh = self.cfg.defs.use_lsh

        self.sens = None


    @timer
    def train_test_vertical_model(self):
        if self.cfg.defs.vertical_fl.train_type == "lsh":
            if self.use_lsh:
                if not self.is_label_owner:
                    self._calculate_weight()
                    self._hash_data()
                while not self.selected_index:
                    time.sleep(self.sleep)
            else:
                self._get_selected_index()
            self.lsh_train_dataset = self.origin_train_dataset.update_dataset_via_indexes(self.selected_index)
            self.__generate_lsh_train_dataloader()
            time.sleep(1)
        elif self.cfg.defs.vertical_fl.train_type == 'random':
            np.random.seed(self.seed)
            random_sample = np.random.randint(0, self.origin_train_dataset.num_train_data, self.k)
            self.lsh_train_dataset = self.origin_train_dataset.update_dataset_via_indexes(random_sample)
            self.__generate_lsh_train_dataloader()
        if not self.is_label_owner:
            self._train()
            time.sleep(1)
            self._test()

    @timer
    def _hash_data(self):
        start_time = time.time()
        self.lsh = BinaryLSH(self.num_hash_functions, self.origin_train_dataset.num_features,
                             self.rank, self.cfg.defs.seed)
        signatures = self.lsh.insert(self.origin_train_dataset.train_data_numpy)
        bit_length = self.num_hash_functions + 8 - self.num_hash_functions % 8
        hash_bit = [bitarray(bit_length)
                    for _ in range(len(self.origin_train_dataset.train_label_numpy))]

        for i, sig in enumerate(signatures):
            hash_bit[i][:self.num_hash_functions] = bitarray(sig)

        hash_bytes = [bits.tobytes() for bits in hash_bit]
        end_time = time.time()
        if self.rank == 1:
            print("hashtime:", end_time - start_time)
        # self.test_HE(hash_bit)
        # time.sleep(100000)

        vfl_server_stub = self._get_vfl_server_rpc_stub()
        request = vfl_server_service_pb2.client_hash_info_request(
            cid=self.rank,
            hash_info=hash_bytes
        )
        response = vfl_server_stub.get_client_selected_lsh_index(request)
        self.selected_index = response.selected_index

    @timer
    def _calculate_weight(self):
        magnitudes = np.linalg.norm(self.origin_train_dataset.train_data_numpy - np.average(self.origin_train_dataset.train_data_numpy), axis=1)
        max_mag = np.max(magnitudes)
        min_mag = np.min(magnitudes)
        avg_mag = np.average(magnitudes)
        self.weight = avg_mag
        # self.weight = np.linalg.norm(self.origin_train_dataset.origin_data)
        # self.weight = np.linalg.norm(self.origin_train_dataset.train_data_numpy)

        print(self.weight)
        vfl_server_stub = self._get_vfl_server_rpc_stub()
        request = vfl_server_service_pb2.client_weight_request(
            cid=self.rank,
            weight=self.weight,
            # num_features=self.origin_train_dataset.num_features
        )
        response = vfl_server_stub.get_client_weight_allocation(request)
        self.num_hash_functions = response.num_hash_functions
        print(self.num_hash_functions)

    @timer
    def _train(self):
        assert self.epochs is not None
        if self.rank == 0:
            d = {'batch_size': self.batch_size, 'Clients': self.origin_train_dataset.num_clients}
            for epoch in tqdm(range(self.epochs), desc="Training Process", postfix=d):
                self._train_client_iteration(epoch)
                if self.early_stop:
                    tqdm.write(">>>Train finish.")
                    return
                # print(f">>>Epoch {epoch + 1}, complete. (Client {self.rank})")
            tqdm.write(">>>Train finish.")
        else:
            for epoch in range(self.epochs):
                self._train_client_iteration(epoch)
                if self.early_stop:
                    return
        # print(">>>Train finish.")

    def _test(self):
        self._test_client_iteration()
        if self.rank == 0:
            tqdm.write(">>>Test finish.")

    def _train_client_iteration(self, epoch):
        raise NotImplementedError

    def _test_client_iteration(self):
        raise NotImplementedError

    def _generate_origin_dataloader(self):
        if self.cfg.defs.vertical_fl.train_type == 'normal':
            self.__generate_origin_train_dataloader()
        self.__generate_test_dataloader()

    def _get_align_item_label(self, index):
        raise NotImplementedError

    def _adjust_learning_rate(self, epoch):

        """
        adjust lr while training
        :return:
        """
        raise NotImplementedError

    @timer
    def select_lsh_index(self, hash_info):
        central_index = []
        weight_list = []
        if self.is_regression:
            sample_index = np.array([i for i in range(len(hash_info))])
            # centroid, weight = self._k_medoids(sample_index, hash_info, self.k)
            centroid, weight = self._k_medoids(sample_index, hash_info, self.k)
            print(len(centroid))
            for i, w in zip(centroid, weight):
                central_index.append(i)
                weight_list.append(w*1000)
        else:
            if self.cfg.trainer == "mlp":
                label_counts = np.sum(self.origin_train_dataset.train_label_numpy, axis=0)
                print(label_counts)

                # multiprocessing
                process = []
                q = mp.Queue()
                # mp.set_start_method('spawn')
                for label in range(self.origin_train_dataset.train_label_numpy.shape[1]):
                    sample_index = np.where(self.origin_train_dataset.train_label_numpy[:, label] == 1)[0]
                    p = mp.Process(target=self._k_medoids, args=(sample_index, hash_info,
                                                                 round(self.k * label_counts[label] /
                                                                       self.origin_train_dataset.num_train_data), q))
                    p.start()
                    process.append(p)
                results = [q.get() for _ in process]
                for p in process:
                    p.join()

                for item in results:
                    for c, w in zip(item[0], item[1]):
                        central_index.append(c)
                        weight_list.append(w*10000)

                #
                # for label in range(self.origin_train_dataset.train_label_numpy.shape[1]):
                #     sample_index = np.where(self.origin_train_dataset.train_label_numpy[:, label] == 1)[0]
                #     print(f"sample_index: {sample_index}")
                #     time.sleep(10000)
                #     centroid, weight = self._k_medoids(sample_index, hash_info,
                #                                        round(self.k * label_counts[
                #                                            label] / self.origin_train_dataset.num_train_data))
                #     print(len(centroid))
                #     for i, w in zip(centroid, weight):
                #         central_index.append(i)
                #         weight_list.append(w / self.origin_train_dataset.train_label_numpy.shape[1])
            else:
                label_counts = Counter(self.origin_train_dataset.train_label_numpy)
                label_counts = dict(label_counts)
                process = []
                q = mp.Queue()
                for label in set(self.origin_train_dataset.train_label_numpy):
                    sample_index = np.where(self.origin_train_dataset.train_label_numpy == label)[0]
                    print(sample_index)
                    p = mp.Process(target=self._k_medoids, args=(sample_index, hash_info,
                                                                 round(self.k * label_counts[label] /
                                                                       self.origin_train_dataset.num_train_data), q))
                    p.start()
                    process.append(p)
                results = [q.get() for _ in process]
                for p in process:
                    p.join()
                print('test')
                for item in results:
                    for c, w in zip(item[0], item[1]):
                        central_index.append(c)
                        weight_list.append(w*10000)

        self.selected_index = central_index
        return central_index, weight_list

    def _get_selected_index(self):
        with open('id.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
        selected_id = [int(line.strip()) for line in lines]
        self.selected_index = selected_id
        file.close()
        with open('weight.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
        weight_list = [float(line.strip()) for line in lines]
        self.weight_list = weight_list

    def __generate_origin_train_dataloader(self):
        self.logger.warning(f"Origin train Dataset length:{len(self.origin_train_dataset)}")
        self.train_dataset = deepcopy(self.origin_train_dataset)

        self.origin_train_loader = tud.DataLoader(self.origin_train_dataset, self.batch_size_normal)
        self.train_loader = self.origin_train_loader

        self.num_origin_train_batches = len(self.origin_train_loader)
        self.num_train_batches = self.num_origin_train_batches
        self.batch_size = self.batch_size_normal

    def __generate_lsh_train_dataloader(self):
        self.logger.warning(f"LSH train Dataset length:{len(self.lsh_train_dataset)}")
        self.train_dataset = deepcopy(self.lsh_train_dataset)

        self.lsh_train_loader = tud.DataLoader(self.lsh_train_dataset, self.batch_size_lsh)
        self.train_loader = self.lsh_train_loader

        self.num_lsh_train_batches = len(self.lsh_train_loader)
        self.num_train_batches = self.num_lsh_train_batches
        self.batch_size = self.batch_size_lsh

    def __generate_test_dataloader(self):
        if self.test_loader is not None:
            return
        # batch_size = self.cfg.defs.vertical_fl.batch_size
        self.logger.warning(f"Test Dataset length:{len(self.test_dataset)}")
        self.test_loader = tud.DataLoader(self.test_dataset, self.batch_size_normal)
        self.num_test_batches = len(self.test_loader)

    def _get_vfl_server_rpc_stub(self):
        vfl_server_address = self.cfg.server_conf.vfl_server.host + ":" + \
                             self.cfg.server_conf.vfl_server.port
        vfl_server_channel = grpc.insecure_channel(vfl_server_address, options=self.options)
        vfl_server_stub = vfl_server_service_pb2_grpc.VFLServerServiceStub(vfl_server_channel)

        return vfl_server_stub

    def __get_vfl_label_owner_rpc_stub(self, ip_address):
        vfl_label_owner_channel = grpc.insecure_channel(ip_address, options=self.options)
        vfl_label_owner_stub = vfl_label_owner_service_pb2_grpc.VFLLabelOwnerServiceStub(vfl_label_owner_channel)

        return vfl_label_owner_stub

    def __get_vfl_client_rpc_stub(self, ip_address):
        vfl_client_channel = grpc.insecure_channel(ip_address, options=self.options)
        vfl_client_stub = vfl_client_service_pb2_grpc.VFLClientServiceStub(vfl_client_channel)

        return vfl_client_stub

    @staticmethod
    def _k_center(sample_index, hash_info, k):
        centroids = []
        x = random.sample(range(len(sample_index)), 1)
        centroids.append(x[0])
        min_dist_list = [np.inf for i in range(len(sample_index))]
        center_ids = np.array([0 for _ in range(len(sample_index))])
        while len(centroids) < k + 1:
            for i in range(len(sample_index)):
                centroid = centroids[-1]
                distance = c_hamming_distance(hash_info[sample_index[i]], hash_info[sample_index[centroid]])
                if distance < min_dist_list[i]:
                    min_dist_list[i] = distance
                    center_ids[i] = centroid
            new_centroid = np.argmax(min_dist_list)
            if new_centroid not in centroids:
                if len(centroids) == k:
                    break
                centroids.append(new_centroid)
            else:
                break
        new_centroids = []
        time1 = time.time()
        for centroid in centroids:
            sample_id = np.where(center_ids == centroid)[0]
            min_dis = np.inf
            min_centroid = None
            for id in sample_id:
                dist = 0
                for id2 in sample_id:
                    dist += c_hamming_distance(hash_info[sample_index[id]], hash_info[sample_index[id2]])
                if dist < min_dis:
                    min_dis = dist
                    min_centroid = id
            new_centroids.append(min_centroid)
            for id in sample_id:
                center_ids[id] = min_centroid
        centroids = new_centroids
        time2 = time.time()
        print("time:", time2 - time1)
        weight_list = []
        for centroid in centroids:
            sample_id = np.where(center_ids == centroid)[0]
            weight_list.append(len(sample_id) / len(sample_index))
        return centroids, weight_list

    @timer
    def _k_medoids(self, sample_index, hash_info, k, q=None, max_iters=3):
        # initialize centroids
        print('kmedoids')
        centroids = random.sample(list(sample_index), k)
        # labels = np.array([None for _ in range(len(sample_index))])
        labels = np.full(len(sample_index), None, dtype=object)
        sample_bits_id = np.random.randint(0, self.total_num_hash_func, size=self.bucket_width).tolist()
        binary_combo = product('01', repeat=self.bucket_width)
        candidate_bitarrays = [bitarray(''.join(combo)) for combo in binary_combo]
        candidate_buckets = [None for _ in range(len(sample_index))]
        num_signature_bits = self.total_num_hash_func
        weight_list = []
        for _ in range(max_iters):
            time1 = time.time()
            bucket = {i: [] for i in range(2 ** self.bucket_width)}
            for centroid in centroids:
                # bit_sample = sample_bits(hash_info[centroid], sample_bits_id)
                bit_sample = hash_info[centroid][sample_bits_id]
                bit2int = int(bit_sample.to01(), 2)
                bucket[bit2int].append(centroid)
            # allocate point to its nearest centroid
            for id in range(len(sample_index)):
                if sample_index[id] in centroids:
                    labels[id] = sample_index[id]
                    continue
                sample_bitarray = hash_info[sample_index[id]]
                if candidate_buckets[id] is None:
                    candidate_bucket = set()
                    # id_sample_bits = sample_bits(sample_bitarray, sample_bits_id)
                    id_sample_bits = sample_bitarray[sample_bits_id]
                    for c_bitarray in candidate_bitarrays:
                        if c_hamming_distance(id_sample_bits, c_bitarray) <= 2:
                            candidate_bucket.add(int(c_bitarray.to01(), 2))
                    candidate_buckets[id] = candidate_bucket
                assert candidate_buckets[id] is not None
                candidate_centroids = []
                for candidate_bit_id in candidate_buckets[id]:
                    candidate_centroids += bucket[candidate_bit_id]
                if candidate_centroids:
                    # sample_bitarray = hash_info[sample_index[id]]
                    dist_list = [c_hamming_distance(sample_bitarray, hash_info[candidate_centroid])
                                 for candidate_centroid in candidate_centroids]
                    labels[id] = candidate_centroids[np.argmin(dist_list)]
                else:
                    min_dis = np.inf
                    min_centroid = None
                    for centroid in centroids:
                        dist = c_hamming_distance(sample_bitarray, hash_info[centroid])
                        if dist < min_dis:
                            min_dis = dist
                            min_centroid = centroid
                    labels[id] = min_centroid

            distinct_labels = set(labels)
            print("distinct_labels:", len(distinct_labels))
            labels = np.array(labels)
            # renew centroids
            time2 = time.time()
            new_centroids = []
            new_weight_list = []
            for label in distinct_labels:
                sample_id = np.where(labels == label)[0]
                num_sample_id = len(sample_id)
                # bit_counts = [[0, 0] for _ in range(num_signature_bits)]
                bit_counts = [0 for _ in range(num_signature_bits)]
                for id in sample_id:
                    hash_info_id = hash_info[sample_index[id]]
                    for i in range(num_signature_bits):
                        if hash_info_id[i] == 1:
                            bit_counts[i] += 1
                min_dis = np.inf
                min_centroid = None
                for id in sample_id:
                    hash_info_id = hash_info[sample_index[id]]
                    dist_per_bit = [bit_counts[i] if hash_info_id[i] == 0 else num_sample_id - bit_counts[i]
                                    for i in range(num_signature_bits)]
                    dist = np.sum(dist_per_bit)
                    # print(dist)
                    if dist < min_dis:
                        min_dis = dist
                        min_centroid = id
                new_centroids.append(sample_index[min_centroid])
                new_weight_list.append(num_sample_id/len(sample_index))

            if len(new_centroids) == len(centroids):
                if np.allclose(np.array(new_centroids), np.array(centroids)):
                    print("early stop")
                    break
            centroids = deepcopy(new_centroids)
            weight_list = deepcopy(new_weight_list)
            print(len(centroids))
            time3 = time.time()
            print("time1:", time2 - time1)
            print("time2:", time3 - time2)

        # for centroid in centroids:
        #     sample_id = np.where(labels == centroid)[0]
        #     weight_list.append(len(sample_id) / len(sample_index))

        if q:
            q.put([centroids, weight_list])
            return
        return centroids, weight_list
        # return

    @timer
    def _old_k_medoids(self, sample_index, hash_info, k, q=None, max_iters=3):
        # initialize centroids
        print('kmedoids')
        centroids = random.sample(list(sample_index), k)
        # labels = np.array([None for _ in range(len(sample_index))])
        labels = np.full(len(sample_index), None, dtype=object)
        binary_combo = product('01', repeat=self.bucket_width)
        num_signature_bits = self.total_num_hash_func
        weight_list = []
        for _ in range(max_iters):
            time1 = time.time()

            # allocate point to its nearest centroid
            for id in range(len(sample_index)):
                if sample_index[id] in centroids:
                    labels[id] = sample_index[id]
                    continue
                min_dis = np.inf
                min_centroid = None
                for centroid in centroids:
                    dist = c_hamming_distance(hash_info[id], hash_info[centroid])
                    if dist < min_dis:
                        min_dis = dist
                        min_centroid = centroid
                labels[id] = min_centroid

            distinct_labels = set(labels)
            print("distinct_labels:", len(distinct_labels))
            labels = np.array(labels)
            # renew centroids
            time2 = time.time()
            new_centroids = []
            new_weight_list = []
            for label in distinct_labels:
                sample_id = np.where(labels == label)[0]
                num_sample_id = len(sample_id)

                min_dis = np.inf
                min_centroid = None
                for id in sample_id:
                    dist = 0
                    for id2 in sample_id:
                        dist += c_hamming_distance(hash_info[sample_index[id]], hash_info[sample_index[id2]])
                    if dist < min_dis:
                        min_dis = dist
                        min_centroid = id
                new_centroids.append(sample_index[min_centroid])
                new_weight_list.append(num_sample_id/len(sample_index))
            if len(new_centroids) == len(centroids):
                if np.allclose(np.array(new_centroids), np.array(centroids)):
                    print("early stop")
                    break
            centroids = deepcopy(new_centroids)
            weight_list = deepcopy(new_weight_list)
            print(len(centroids))
            time3 = time.time()
            print("time1:", time2 - time1)
            print("time2:", time3 - time2)

        # for centroid in centroids:
        #     sample_id = np.where(labels == centroid)[0]
        #     weight_list.append(len(sample_id) / len(sample_index))

        if q:
            q.put([centroids, weight_list])
            return
        return centroids, weight_list
        # return

