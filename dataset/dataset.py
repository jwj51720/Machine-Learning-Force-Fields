import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


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
        # inputs = [pos_x, pos_y, pos_z]

        if self.is_train:
            label = torch.tensor(self.df.iloc[idx, 3], dtype=torch.float32)
            # label = self.df.iloc[idx, 3]
            return inputs, label
        else:
            return inputs


def get_loader(train_data: Dataset, valid_data: Dataset, config: dict) -> DataLoader:
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=config["trainer"]["batch"]
    )
    valid_loader = DataLoader(
        valid_data,
        shuffle=False,
        batch_size=config["trainer"]["batch"],
    )

    return train_loader, valid_loader
