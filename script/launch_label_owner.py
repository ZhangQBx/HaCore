import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grpc
from concurrent import futures
import hydra
from omegaconf import DictConfig
from rpc.grpc_server.vfl_label_owner_rpc import VFLLabelOwner
from rpc.grpc_file import vfl_label_owner_service_pb2_grpc
from utils import get_cuda_device


def get_vfl_label_owner_address_and_grpc_server(host, port):
    vfl_label_owner_address = host + ":" + str(port)
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size),
               ('grpc.max_receive_message_length', max_msg_size)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=options)

    return vfl_label_owner_address, server


def init_lr_label_owner(host, port, device, cfg):
    vfl_label_owner_address, server = get_vfl_label_owner_address_and_grpc_server(host, port)
    rank = cfg.defs.num_clients
    log_path = cfg.defs.log_path
    is_regression = cfg.lr_conf.is_regression

    vfl_label_owner_server = VFLLabelOwner(rank, log_path, cfg, device, vfl_label_owner_address,
                                           is_label_owner=True, is_regression=is_regression, trainer='lr')

    vfl_label_owner_service_pb2_grpc.add_VFLLabelOwnerServiceServicer_to_server(vfl_label_owner_server,
                                                                                server)
    server.add_insecure_port(vfl_label_owner_address)
    server.start()
    print(">>>grpc LRLabelOwner start.")

    vfl_label_owner_server.trainer.train_test_vertical_model()

    server.wait_for_termination()


def init_mlp_label_owner(host, port, device, cfg):
    vfl_label_owner_address, server = get_vfl_label_owner_address_and_grpc_server(host, port)
    rank = cfg.defs.num_clients
    log_path = cfg.defs.log_path
    is_regression = cfg.mlp_conf.is_regression
    vfl_label_owner_server = VFLLabelOwner(rank, log_path, cfg, device, vfl_label_owner_address,
                                           is_label_owner=True, is_regression=is_regression, trainer='mlp')

    vfl_label_owner_service_pb2_grpc.add_VFLLabelOwnerServiceServicer_to_server(vfl_label_owner_server,
                                                                                server)
    server.add_insecure_port(vfl_label_owner_address)
    server.start()
    print(">>>grpc MLPLabelOwner start.")

    vfl_label_owner_server.trainer.train_test_vertical_model()

    server.wait_for_termination()





def launch_lr_label_owner(cfg):
    host = cfg.server_conf.vfl_label_owner.host
    port = int(cfg.server_conf.vfl_label_owner.port)
    device = get_cuda_device(1, trainer='lr')

    init_lr_label_owner(host, port, device, cfg)


def launch_mlp_label_owner(cfg):
    host = cfg.server_conf.vfl_label_owner.host
    port = int(cfg.server_conf.vfl_label_owner.port)
    device = get_cuda_device(1, trainer='mlp')

    init_mlp_label_owner(host, port, device, cfg)





@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def launch_vfl_label_owner(cfg: DictConfig):
    if cfg.trainer == 'lr':
        launch_lr_label_owner(cfg)
    elif cfg.trainer == 'mlp':
        launch_mlp_label_owner(cfg)



if __name__ == "__main__":
    # launch_lr_clients()
    # launch_mlp_clients()
    # args = get_args()
    launch_vfl_label_owner()
