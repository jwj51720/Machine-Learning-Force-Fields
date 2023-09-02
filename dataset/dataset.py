import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class ForceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train=True):
        self.df = df
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        pos_x = self.df.loc[idx, "position_x"]
        pos_y = self.df.loc[idx, "position_y"]
        pos_z = self.df.loc[idx, "position_z"]

        inputs = torch.tensor([pos_x, pos_y, pos_z], dtype=torch.float32)

        if self.is_train:
            label = torch.tensor(self.df.loc[idx, "force"], dtype=torch.float32)
            return inputs, label
        else:
            return inputs


def get_loader(train_data: Dataset, valid_data: Dataset, config: dict) -> DataLoader:
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=config["batch_size"],
    )
    valid_loader = DataLoader(
        valid_data,
        shuffle=False,
        batch_size=config["batch_size"],
    )

    return train_loader, valid_loader
