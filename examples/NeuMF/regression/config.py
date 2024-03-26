import argparse
import inspect

import torch


class Config:
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    train_epochs = 10
    batch_size = 128
    learning_rate = 0.01
    l2_regularization = 1e-3  # 正则化系数
    learning_rate_decay = 0.99  # 学习率衰减程度

    dataset_file = 'rllm/rllm/datasets/rel-movielens1m/regression/ratings/'

    mf_dim = 10
    mlp_layers = [32, 16, 8]

    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str
