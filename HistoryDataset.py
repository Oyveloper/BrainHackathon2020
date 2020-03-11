import pandas as pd
import utils
import torch
from torch.utils.data import Dataset
import os

from FetchData import FetchData



class HistoryDataset(Dataset):
    def __init__(self, training=True):
        datafetcher = FetchData()
        self.data = datafetcher.get_data()
        train_ammount = 0.8
        train_split = int(0.8 * len(self.data))
        self.element_size = self.data.shape[1] - 1

        # Splitting training and testing data 
        if training:
            self.data = self.data.iloc[:train_split, :]
        else:
            self.data = self.data.iloc[train_split:, :]

        self.data_X = torch.tensor(self.data.drop('ActivePower (Average)', axis=1).values)
        self.data_Y = torch.tensor(self.data['ActivePower (Average)'].values).view(-1, 1)


    def get_element_size(self):
        return self.element_size
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data_X[idx], self.data_Y[idx]
