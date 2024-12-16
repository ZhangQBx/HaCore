import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grpc
import hydra
from omegaconf import DictConfig
from concurrent import futures
from rpc.grpc_server.vfl_server_rpc import VFLServer
from rpc.grpc_file import vfl_server_service_pb2_grpc
from utils import get_cuda_device


def get_vfl_server_address_and_grpc_server(host, port):
    vfl_server_address = host + ":" + str(port)
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size),
               ('grpc.max_receive_message_length', max_msg_size)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)

    return vfl_server_address, options, server


def init_lr_server(host, port, device, cfg):
    """
    launch LR server
    :param device: cuda device
    :param host: address host
    :param port: address port
    :param cfg: config file
    :return:
    """
    vfl_server_address, options, server = get_vfl_server_address_and_grpc_server(host, port)

    pk_file = "../ts_ckks.config"
    vfl_server = VFLServer(cfg.defs.num_clients, options, device, cfg, trainer='lr')
    vfl_server_service_pb2_grpc.add_VFLServerServiceServicer_to_server(vfl_server, server)
    server.add_insecure_port(vfl_server_address)
    server.start()
    print(">>>grpc LRServer start.")

    server.wait_for_termination()


def init_mlp_server(host, port, device, cfg):
    """
    launch MLP server
    :param device: cuda device
    :param host: address host
    :param port: address port
    :param cfg: config file
    :return:
    """
    vfl_server_address, options, server = get_vfl_server_address_and_grpc_server(host, port)

    pk_file = "../ts_ckks.config"
    vfl_server = VFLServer(cfg.defs.num_clients, options, device, cfg, trainer='mlp')
    vfl_server_service_pb2_grpc.add_VFLServerServiceServicer_to_server(vfl_server, server)
    server.add_insecure_port(vfl_server_address)
    server.start()
    print(">>>grpc MLPServer start.")

    server.wait_for_termination()





def launch_lr_server(cfg):
    host = cfg.server_conf.vfl_server.host
    port = int(cfg.server_conf.vfl_server.port)

    device = get_cuda_device(1, trainer='lr')
    init_lr_server(host, port, device, cfg)


def launch_mlp_server(cfg):
    host = cfg.server_conf.vfl_server.host
    port = int(cfg.server_conf.vfl_server.port)

    device = get_cuda_device(1, trainer='mlp')
    init_mlp_server(host, port, device, cfg)




@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def launch_vfl_server(cfg: DictConfig):
    if cfg.trainer == 'lr':
        launch_lr_server(cfg)
    elif cfg.trainer == 'mlp':
        launch_mlp_server(cfg)



if __name__ == "__main__":
    # launch_lr_clients()
    # launch_mlp_clients()
    # args = get_args()
    launch_vfl_server()
