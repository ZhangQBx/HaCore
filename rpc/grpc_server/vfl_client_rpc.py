import rpc.grpc_file.vfl_client_service_pb2_grpc as vfl_client_service_pb2_grpc
from trainer import LRTrainer, MLPTrainer
from utils import prepare_logger


class VFLClient(vfl_client_service_pb2_grpc.VFLClientServiceServicer):
    def __init__(self, rank, log_path, cfg, device, ip_address,
                 is_label_owner=False, is_regression=False, trainer='lr'):
        logger = prepare_logger(rank, log_path, cfg.defs.mode)
        if trainer == 'lr':
            self.trainer = LRTrainer(rank, logger, cfg, device, ip_address,
                                     trainer, is_label_owner, is_regression)
        elif trainer == 'mlp':
            self.trainer = MLPTrainer(rank, logger, cfg, device, ip_address,
                                      trainer, is_label_owner, is_regression)

