import pandas as pd
from utils import timer
from .basic_dataset import BasicDataset


class YearPrediction(BasicDataset):
    """
    Generate year prediction dataset for vfl
    """

    def __init__(self, cfg, client_rank, train=True, is_label_owner=False):
        super(YearPrediction, self).__init__(cfg, client_rank, train,
                                             is_label_owner)
        key = ''
        if cfg.trainer == 'mlp':
            key = 'mlp_conf'
        elif cfg.trainer == 'lr':
            key = 'lr_conf'
        self.csv_path = self.cfg[key].data_path
        self.shuffle_data = False
        self.num_train_data = 463715
        self._get_train_and_test_tensor_data()
        # print(self.train_data_tensor.shape)
        # print(self.train_label_tensor.shape)
        # print(self.test_data_tensor.shape)
        # print(self.test_label_tensor.shape)

    # @timer
    def _load_data_from_csv(self):
        """

        :param
        :return:
        """

        yp_data = pd.read_csv(self.csv_path, header=None)
        yp_data = yp_data.rename(columns={0: "year"})
        # yp_data =yp_data.sample(10000)
        x = yp_data.iloc[:, 1:]
        y = yp_data.iloc[:, 0]

        return x, y


