
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
    def __init__(self, max_dtmc_size, lr=0.001, margin=1.0, dl_hparams=None, checkpoint_name = None):
        super().__init__()
        input_size = max_dtmc_size * max_dtmc_size
        hidden_size = int(input_size * 1.8)
        self.loss_fn = self.contrastive_loss
        hparams = {
            "max_dtmc_size": max_dtmc_size,
            "lr": lr,
            "margin": margin,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "loss_fn": self.loss_fn.__name__,
            "checkpoint_name": checkpoint_name
        }
        if dl_hparams is not None:
            hparams.update(dl_hparams)
        self.save_hyperparameters(hparams)
        self.encoder = SiameseNetworkEncoder(input_size, hidden_size)
        self.lr = lr
        self.margin = margin

        self.test_output = []

    def forward(self, x1, x2):
        encoded_x1 = self.encoder(x1.reshape(x1.shape[0], 1, -1).squeeze(1))
        encoded_x2 = self.encoder(x2.reshape(x2.shape[0], 1, -1).squeeze(1))
        return torch.linalg.vector_norm(encoded_x1 - encoded_x2, dim=1, ord=2)

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
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/distance", torch.mean(torch.abs(distance - label)), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        dtmc1, dtmc2, label = batch
        distance = self(dtmc1, dtmc2)
        loss = self.loss_fn(distance, label)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/distance", torch.mean(torch.abs(distance - label)), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        dtmc1, dtmc2, label = batch
        distance = self(dtmc1, dtmc2)
        loss = self.loss_fn(distance, label)
        self.log("test/loss", loss, on_step=False, on_epoch=True, reduce_fx="mean")
        self.log("test/distance", torch.mean(torch.abs(distance - label)), on_step=False, on_epoch=True, reduce_fx="mean")
        for b in batch:
            self.test_output.append((label, distance, torch.abs(distance - label)))
        return loss

    def on_test_end(self) -> None:
        global_diff_list = []
        for r in self.test_output:
            for label, model, difference in zip(*r):
                print(f'Real distance: {label:.4f}, model distance: {model:.4f}, diff: {difference:.4f}')
                global_diff_list.append(difference)
        print(f'Difference avg: {sum(global_diff_list) / len(global_diff_list):.4f}')


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

