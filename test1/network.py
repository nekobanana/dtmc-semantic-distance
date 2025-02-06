import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.stats as stats


class MLPModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.output_activation = nn.Softplus()  # Per garantire parametri positivi

    def forward(self, x):
        return self.output_activation(self.model(x))
        # return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        js_div = js_divergence(y, y_hat)

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/js_divergence', js_div, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)

        js_div = js_divergence(y, y_hat)
        self.log('val/loss', val_loss, prog_bar=True)
        self.log('val/js_divergence', js_div, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        test_loss = F.mse_loss(y_hat, y)

        js_div = js_divergence(y, y_hat)
        self.log('test/loss', test_loss, prog_bar=True)
        self.log('test/js_divergence', js_div, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

def kl_div_lognormal_torch(mu1, sigma1, mu2, sigma2):
    return torch.log(sigma2 / sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) - 0.5

def js_divergence(y1, y2):
    t1 = y1.transpose(0, 1)
    mu1 = torch.log(t1[2])
    sigma1 = t1[0]
    t2 = y2.transpose(0, 1)
    mu2 = torch.log(t2[2])
    sigma2 = t2[0]

    mu_m = (mu1 + mu2) / 2
    sigma_m = torch.sqrt((sigma1 ** 2 + sigma2 ** 2) / 2)

    # Calcoliamo la KL-divergence tra ogni distribuzione e la distribuzione media
    kl1 = kl_div_lognormal_torch(mu1, sigma1, mu_m, sigma_m)
    kl2 = kl_div_lognormal_torch(mu2, sigma2, mu_m, sigma_m)

    # JSD = media delle due KL-divergences
    jsd_batch = (kl1 + kl2) / 2

    return jsd_batch.mean()



