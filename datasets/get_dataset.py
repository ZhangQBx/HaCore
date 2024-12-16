import hydra
from omegaconf import DictConfig
from datasets.year_prediction import YearPrediction
from datasets.bank import Bank
from datasets.body_performance import BodyPerformance
from datasets.mushrooms import Mushrooms
from datasets.higgs import Higgs
from datasets.rice import Rice
from datasets.zboson import Zboson
from datasets.susy import SUSY
from datasets.house_pricing import HousePricing
from datasets.covtype import Covtype
from datasets.woc import Woc


def dataset(cfg, rank, train=True, is_label_owner=False, trainer_type='lr'):
    """

    :param trainer_type: trainer type
    :param rank: client rank
    :param cfg: config file
    :param train: train or test
    :param is_label_owner: if this client hold the label
    :return: dataset object
    """
    name = None
    vfl_dataset = None
    if trainer_type == 'lr':
        name = cfg.lr_conf.dataset_name
    elif trainer_type == 'mlp':
        name = cfg.mlp_conf.dataset_name
    elif trainer_type == 'knn':
        name = cfg.knn_conf.dataset_name

    if name == 'yp':
        vfl_dataset = YearPrediction(cfg, rank, train, is_label_owner)
    elif name == 'bank':
        vfl_dataset = Bank(cfg, rank, train, is_label_owner)
    elif name == 'bp':
        vfl_dataset = BodyPerformance(cfg, rank, train, is_label_owner)
    elif name == 'mushrooms':
        vfl_dataset = Mushrooms(cfg, rank, train, is_label_owner)
    elif name == 'higgs':
        vfl_dataset = Higgs(cfg, rank, train, is_label_owner)
    elif name == 'rice':
        vfl_dataset = Rice(cfg, rank, train, is_label_owner)
    elif name == 'zboson':
        vfl_dataset = Zboson(cfg, rank, train, is_label_owner)
    elif name == 'susy':
        vfl_dataset = SUSY(cfg, rank, train, is_label_owner)
    elif name == 'hp':
        vfl_dataset = HousePricing(cfg, rank, train, is_label_owner)
    elif name == 'covtype':
        vfl_dataset = Covtype(cfg, rank, train, is_label_owner)
    elif name == 'woc':
        vfl_dataset = Woc(cfg, rank, train, is_label_owner)

    return vfl_dataset


@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def test(cfg: DictConfig):
    # get_dataset(1, cfg)
    dataset(cfg, 0)
    dataset(cfg, 1)
    dataset(cfg, 2)


if __name__ == "__main__":
    test()
