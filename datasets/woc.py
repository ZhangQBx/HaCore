import pandas as pd
from .basic_dataset import BasicDataset


class Woc(BasicDataset):
    def __init__(self, cfg, rank, train=True, is_label_owner=False):
        super(Woc, self).__init__(cfg, rank, train,
                                  is_label_owner)

        key = ''
        if cfg.trainer == 'mlp':
            key = 'mlp_conf'
        elif cfg.trainer == 'lr':
            key = 'lr_conf'
        self.csv_path = self.cfg[key].data_path

        self._get_train_and_test_tensor_data()

    def _load_data_from_csv(self):
        data = pd.read_csv(self.csv_path)
        x = data.drop(['ID', 'label'], axis=1)
        y = data['label']

        return x, y