import os
import logging
import time


class Log:
    def __init__(self, rank, log_path, mode, time=None, customize=False):
        # cooperate with hydra
        self.customize = customize
        self.rank = rank
        self.log_path = log_path
        self.mode = mode
        self.logger = None
        self.__get_logger()
        self.__add_handler(time)

    def __get_logger(self):
        self.logger = logging.getLogger(f"rank{self.rank}")
        self.logger.setLevel(logging.DEBUG)

    def __add_handler(self, time):
        if not self.customize:
            path = os.path.join(self.log_path, str(self.mode), f"rank{self.rank}")
        else:
            path = os.path.join(self.log_path, time[:10], time[11:], f"rank{self.rank}")
        if not os.path.exists(path):
            os.makedirs(path)

        log_path = os.path.join(path, f"rank{self.rank}.log")
        fh = logging.FileHandler(log_path, mode='w')
        formatter = logging.Formatter("%(levelname)s - %(process)d - %(asctime)s - %(filename)s: %(message)s")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)


def prepare_logger(rank, log_path, activate_code, time=None, customize=False):
    log = Log(rank, log_path, activate_code, time, customize)
    log.logger.debug("Initialized.")

    return log.logger


def timer(func):
    def wrapper(*args, **kwargs):
        st = time.time()
        ret = func(*args, **kwargs)
        print(f">>>{func.__name__.upper()}'s time-cost: {time.time() - st}")
        return ret

    return wrapper
