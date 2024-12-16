import time

import numpy as np


class BinaryLSH:
    def __init__(self, tables_num, num_feature, cid, origin_seed):
        self.seed = np.random.randint(0, 1000, 8)
        np.random.seed(self.seed[cid])
        self.R = np.random.randn(num_feature, tables_num)
        np.random.seed(origin_seed)
        self.hash_tables = [dict() for _ in range(tables_num)]

    def _hash(self, inputs):
        lsh_signature = np.sign(np.matmul(inputs, self.R))/2 + 0.5
        signatures = lsh_signature.astype(np.int32).tolist()

        return signatures

    def insert(self, inputs):
        inputs = np.array(inputs)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape([1, -1])

        signatures = self._hash(inputs)

        return signatures
