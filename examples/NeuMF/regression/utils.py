import time
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from config import Config


class NCFDataset(Dataset):
    def __init__(self, df):
        self.user_id = torch.LongTensor(df['userID'].to_list())
        print(len(df['itemID']))
        self.item_id = torch.LongTensor(df['itemID'].to_list())
        self.rating = torch.Tensor(df['rating'].to_list())

    def __getitem__(self, idx):
        return self.user_id[idx], self.item_id[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def predict_mse(trained_model, dataloader, device):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_id, item_id, ratings = [i.to(device) for i in batch]
            #print(user_id, item_id)
            #print(user_id.size())
            predict = trained_model(user_id, item_id)
            mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count  # dataloader上的均方误差
