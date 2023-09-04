from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm
import torch
import copy
import numpy as np
import math


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
        if self.config["task"] == "force":
            self.cfg_trainer = self.config["force"]["trainer"]
        elif self.config["task"] == "energy":
            self.cfg_trainer = self.config["energy"]["trainer"]

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
        if config["task"] == "force":
            self.save_dir = self.config["data"]["force_dir"]
        elif config["task"] == "energy":
            self.save_dir = self.config["data"]["energy_dir"]

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.min_val_loss = float("inf")

    def training_force(self):
        es_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            total_loss = 0
            print(f"..{epoch} epoch..")
            self.model.train()
            for inputs, labels in tqdm(self.train_loader):
                inputs = inputs.to(self.config["device"])
                labels = labels.to(self.config["device"])
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                total_loss += loss
                loss.backward()
                self.optimizer.step()

            val_loss, val_score = self.validation_force()
            print(
                f"Train Loss: {total_loss/len(self.train_loader):.6f}, Valid Loss: {val_loss:.6f}, Valid Score: {val_score:.6f}"
            )
            es_count += 1
            if val_loss < self.min_val_loss:
                es_count = 0
                self.min_val_loss = val_loss
                best_model = copy.deepcopy(self.model)
                print("..min valid loss..")
                print("..new best_model..")

            if es_count == self.config["force"]["trainer"]["early_stopping"]:
                return best_model

            if self.config["force"]["trainer"]["scheduler"]:
                self.lr_scheduler.step(val_loss)

            if epoch % 10 == 0:
                torch.save(best_model.state_dict(), "../current_best.pt")

        return best_model

    def training_energy(self):
        es_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            total_loss = 0
            print(f"..{epoch} epoch..")
            self.model.train()
            for inputs, masks, labels in tqdm(self.train_loader):
                inputs = inputs.to(self.config["device"])
                labels = labels.to(self.config["device"])
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.criterion(output.squeeze(), labels)
                total_loss += loss
                loss.backward()
                self.optimizer.step()

            val_loss, val_score = self.validation_energy()
            print(
                f"Train Loss: {total_loss/len(self.train_loader):.6f}, Valid Loss: {val_loss:.6f}, Valid Score: {val_score:.6f}"
            )
            es_count += 1
            if val_loss < self.min_val_loss:
                es_count = 0
                self.min_val_loss = val_loss
                best_model = copy.deepcopy(self.model)
                print("..min valid loss..")
                print("..new best_model..")

            if es_count == self.config["force"]["trainer"]["early_stopping"]:
                return best_model

            if self.config["force"]["trainer"]["scheduler"]:
                self.lr_scheduler.step(val_loss)

            if epoch % 10 == 0:
                torch.save(best_model.state_dict(), "../current_best.pt")

        return best_model

    def validation_force(self):
        val_loss = 0
        squared_error = 0
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(self.valid_loader):
                inputs = inputs.to(self.config["device"])
                labels = labels.to(self.config["device"])
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                val_loss += loss.item()
                squared_error += torch.sum(torch.square(output - labels)).item()
                total_samples += inputs.size(0)
        return (
            val_loss / len(self.valid_loader),
            math.sqrt(squared_error / total_samples / 3) * 40,  # EF metric
        )

    def validation_energy(self):
        val_loss = 0
        squared_error = 0
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, masks, labels in tqdm(self.valid_loader):
                inputs = inputs.to(self.config["device"])
                masks = masks.to(self.config["device"])
                labels = labels.to(self.config["device"])
                output = self.model(inputs)
                output = output.squeeze()
                loss = self.criterion(output, labels)
                val_loss += loss.item()
                squared_error += torch.sum(
                    torch.square((output - labels) / torch.sum(masks, dim=1))
                ).item()
                total_samples += inputs.size(0)
        return (
            val_loss / len(self.valid_loader),
            math.sqrt(squared_error / total_samples) * 1000,  # EF metric
        )
