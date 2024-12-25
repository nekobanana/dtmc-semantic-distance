import json
import random
import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F

class SiameseNetwork(pl.LightningModule):
    def __init__(self, max_dtmc_size, lr=0.001, margin=1.0):
        super().__init__()
        input_size = max_dtmc_size * max_dtmc_size
        hidden_size = int(input_size * 0.8)
        self.save_hyperparameters(input_size, hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.lr = lr
        self.margin = margin

    def forward(self, x1, x2):
        encoded_x1 = self.encoder(x1.reshape(x1.shape[0], 1, -1).squeeze(1))
        encoded_x2 = self.encoder(x2.reshape(x2.shape[0], 1, -1).squeeze(1))
        return torch.norm(encoded_x1 - encoded_x2, dim=-1, p=1)

    def contrastive_loss(self, distance, label):
        loss_same = (torch.tensor(1, device="cuda").repeat(distance.shape) - label) * torch.pow(distance, 2)
        loss_diff = label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return torch.mean(loss_same + loss_diff)

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        encoded_distance = self(x1, x2)
        loss = self.contrastive_loss(encoded_distance, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        distances = self(x1, x2)
        loss = self.contrastive_loss(distances, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x1, x2, y = batch
        distances = self(x1, x2)
        loss = self.contrastive_loss(distances, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
