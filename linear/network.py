import torch
import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F

class SiameseNetworkEncoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        # nn.init.zeros_(self.linear1.bias)
        # nn.init.zeros_(self.linear2.bias)

    def forward(self, x):

        x = self.linear1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.linear2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        return x

class SiameseNetwork(pl.LightningModule):
    def __init__(self, max_dtmc_size, lr=0.001, margin=1.0, dl_hparams=None):
        super().__init__()
        input_size = max_dtmc_size * max_dtmc_size
        hidden_size = int(input_size * 1.8)
        self.loss_fn = self.mse_loss
        hparams = {
            "max_dtmc_size": max_dtmc_size,
            "lr": lr,
            "margin": margin,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "loss_fn": self.loss_fn.__name__
        }
        if dl_hparams is not None:
            hparams.update(dl_hparams)
        self.save_hyperparameters(hparams)
        self.encoder = SiameseNetworkEncoder(input_size, hidden_size)
        self.lr = lr
        self.margin = margin

    def forward(self, x1, x2):
        encoded_x1 = self.encoder(x1.reshape(x1.shape[0], 1, -1).squeeze(1))
        encoded_x2 = self.encoder(x2.reshape(x2.shape[0], 1, -1).squeeze(1))
        return torch.norm(encoded_x1 - encoded_x2, dim=-1, p=1)

    def contrastive_loss(self, distance, label):
        loss_same = (1 - label) * (distance ** 2)
        loss_diff = label * (torch.clamp(self.margin - distance, min=0.0) ** 2)
        return torch.mean(loss_same + loss_diff)

    def mse_loss(self, distance, label):
        return F.mse_loss(distance, label)

    def training_step(self, batch, batch_idx):
        dtmc1, dtmc2, label = batch
        distance = self(dtmc1, dtmc2)
        loss = self.loss_fn(distance, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        dtmc1, dtmc2, label = batch
        distance = self(dtmc1, dtmc2)
        loss = self.loss_fn(distance, label)
        self.log("val_loss", loss)
        return loss

    # def on_validation_epoch_end(self) -> None:
    #     print(f'val_loss = {self.val_loss_acc / }')

    def test_step(self, batch, batch_idx):
        dtmc1, dtmc2, label = batch
        distance = self(dtmc1, dtmc2)
        loss = self.loss_fn(distance, label)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
