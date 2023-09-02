from torch.utils.data import DataLoader
import torch.nn as nn


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: dict,
    ):
        pass
