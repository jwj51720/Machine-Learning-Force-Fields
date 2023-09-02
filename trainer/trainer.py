from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm
import torch
import copy


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.cfg_trainer = self.config["trainer"]

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.cfg_trainer["learning_rate"]
        )
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.cfg_trainer["lr_scheduler"]["patience"],
            factor=self.cfg_trainer["lr_scheduler"]["factor"],
            mode=self.cfg_trainer["lr_scheduler"]["mode"],
            verbose=True,
        )

        self.epochs = self.cfg_trainer["epochs"]
        self.start_epoch = 1
        self.save_dir = self.config["data"]["save_dir"]
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.min_val_loss = float("inf")
        self.min_val_EF = float("inf")

    def training(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"..{epoch} epoch..")
            self.model.train()
            for inputs, labels in tqdm(self.train_loader):
                inputs = inputs.to(self.config["device"])
                labels = labels.to(self.config["device"])
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

            val_loss, val_score = self.validation()
            print(
                f"Train Loss: {loss:.4f}, Valid Loss: {val_loss:.4f}, Valid Score: {val_score:.4f}"
            )
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                best_model = copy.deepcopy(self.model)
                print("..min valid loss..")
                print("..new best_model..")

            if self.config["trainer"]["scheduler"]:
                self.lr_scheduler.step(val_loss)

        return best_model

    def validation(self):
        val_loss = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(self.valid_loader):
                inputs = inputs.to(self.config["device"])
                labels = labels.to(self.config["device"])
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                val_loss += loss.item()
                val_score = 1
                breakpoint()
        return (val_loss / len(self.valid_loader), val_score)
