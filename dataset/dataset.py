import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.utils.rnn as rnn_utils


class ForceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config, is_train=True):
        self.df = df
        self.is_train = is_train
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pos_x = self.df.iloc[idx, 0]  # position_x
        pos_y = self.df.iloc[idx, 1]  # position_y
        pos_z = self.df.iloc[idx, 2]  # position_z

        inputs = torch.tensor([pos_x, pos_y, pos_z], dtype=torch.float32)

        if self.is_train:
            label = torch.tensor(self.df.iloc[idx, 3], dtype=torch.float32)
            return inputs, label
        else:
            return inputs


class EnergyDataset(Dataset):
    def __init__(self, data_set, is_train=True):
        self.is_train = is_train
        if self.is_train:
            self.sequences = torch.tensor(data_set[0], dtype=torch.float32)
            self.mask = torch.tensor(data_set[1], dtype=torch.bool)
            self.label = torch.tensor(data_set[2], dtype=torch.float32)
        else:
            self.sequences = torch.tensor(data_set[0], dtype=torch.float32)
            self.mask = torch.tensor(data_set[1], dtype=torch.bool)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        inputs = self.sequences[idx]
        mask = self.mask[idx]  # return mask for validation
        if self.is_train:
            label = self.label[idx]
            return inputs, mask, label
        else:
            return inputs, mask


def get_loader(train_data: Dataset, valid_data: Dataset, config: dict) -> DataLoader:
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=config[config["task"]]["trainer"]["batch"]
    )
    valid_loader = DataLoader(
        valid_data,
        shuffle=False,
        batch_size=config[config["task"]]["trainer"]["batch"],
    )

    return train_loader, valid_loader
