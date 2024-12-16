import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grpc
from concurrent import futures
import hydra
import multiprocessing as mp
from omegaconf import DictConfig
from rpc.grpc_server.vfl_client_rpc import VFLClient
from rpc.grpc_file import vfl_client_service_pb2_grpc
from utils import get_cuda_device


def get_vfl_client_address_and_grpc_server(host, port):
    vfl_client_address = host + ":" + str(port)
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size),
               ('grpc.max_receive_message_length', max_msg_size)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=options)

    return vfl_client_address, server


def init_lr_client(host, port, rank, device, cfg):
    vfl_client_address, server = get_vfl_client_address_and_grpc_server(host, port)
    log_path = cfg.defs.log_path
    is_regression = cfg.lr_conf.is_regression

    vfl_client_server = VFLClient(rank, log_path, cfg, device, vfl_client_address,
                                  is_regression=is_regression, trainer='lr')
    vfl_client_service_pb2_grpc.add_VFLClientServiceServicer_to_server(vfl_client_server,
                                                                       server)

    server.add_insecure_port(vfl_client_address)
    server.start()
    print(f">>>grpc LRClient{rank} start.")

    vfl_client_server.trainer.train_test_vertical_model()

    server.wait_for_termination()


def init_mlp_client(host, port, rank, device, cfg):
    vfl_client_address, server = get_vfl_client_address_and_grpc_server(host, port)
    log_path = cfg.defs.log_path
    is_regression = cfg.mlp_conf.is_regression

    vfl_client_server = VFLClient(rank, log_path, cfg, device, vfl_client_address,
                                  is_regression=is_regression, trainer='mlp')
    vfl_client_service_pb2_grpc.add_VFLClientServiceServicer_to_server(vfl_client_server,
                                                                       server)

    server.add_insecure_port(vfl_client_address)
    server.start()
    print(f">>>grpc MLPClient{rank} start.")

    vfl_client_server.trainer.train_test_vertical_model()

    server.wait_for_termination()





def launch_lr_clients(cfg):
    process = []
    num_clients = cfg.defs.num_clients

    mp.set_start_method("spawn")
    for rank in range(num_clients):
        key = 'vfl_client_' + str(rank + 1)
        host = cfg.server_conf[key].host
        port = int(cfg.server_conf[key].port)

        # num_gpu = torch.cuda.device_count()
        # gpu_id = rank // num_gpu
        device = get_cuda_device(0, trainer='lr')

        p = mp.Process(target=init_lr_client, args=(host, port, rank, device, cfg))
        p.start()
        process.append(p)

    for p in process:
        p.join()


def launch_mlp_clients(cfg):
    process = []
    num_clients = cfg.defs.num_clients

    mp.set_start_method("spawn")
    for rank in range(num_clients):
        key = 'vfl_client_' + str(rank + 1)
        host = cfg.server_conf[key].host
        port = int(cfg.server_conf[key].port)

        # num_gpu = torch.cuda.device_count()
        # gpu_id = rank // num_gpu
        device = get_cuda_device(0, trainer='mlp')

        p = mp.Process(target=init_mlp_client, args=(host, port, rank, device, cfg))
        p.start()
        process.append(p)

    for p in process:
        p.join()





@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def launch_vfl_clients(cfg):
    if cfg.trainer == 'lr':
        launch_lr_clients(cfg)
    elif cfg.trainer == 'mlp':
        launch_mlp_clients(cfg)



if __name__ == "__main__":
    # launch_lr_clients()
    # launch_mlp_clients()
    # args = get_args()
    launch_vfl_clients()