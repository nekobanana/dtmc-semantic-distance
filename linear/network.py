import os
import random

import scipy
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
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        return x

class SiameseNetwork(pl.LightningModule):
    def __init__(self, max_dtmc_size, lr=0.001, margin=1.0, dl_hparams=None, checkpoint_name = None, log_dir = 'logs'):
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
            "loss_fn": self.loss_fn.__name__,
            "checkpoint_name": checkpoint_name,
            "log_dir": log_dir,
        }
        if dl_hparams is not None:
            hparams.update(dl_hparams)
        self.save_hyperparameters(hparams)
        self.encoder = SiameseNetworkEncoder(input_size, hidden_size)
        self.lr = lr
        self.margin = margin

        self.test_output = []
        self.output_file = os.path.join(log_dir, 'out.txt')
        os.makedirs(log_dir, exist_ok=True)
        if not os.path.exists(self.output_file):
            with open( self.output_file, 'w'): pass

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
        dtmc1, dtmc2, label, _ = batch
        distance = self(dtmc1, dtmc2)
        loss = self.loss_fn(distance, label)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/err_abs", torch.mean(torch.abs(distance - label)), on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        dtmc1, dtmc2, label, _ = batch
        distance = self(dtmc1, dtmc2)
        loss = self.loss_fn(distance, label)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/err_abs", torch.mean(torch.abs(distance - label)), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        dtmc1, dtmc2, label, couple_idx = batch
        distance = self(dtmc1, dtmc2)
        loss = self.loss_fn(distance, label)
        self.log("test/loss", loss, on_step=False, on_epoch=True, reduce_fx="mean", sync_dist=True)
        self.log("test/err_abs", torch.abs(distance - label), on_step=False, on_epoch=True, reduce_fx="mean", sync_dist=True)
        # batch_size = 1
        self.test_output.append((label, distance, torch.abs(distance - label), couple_idx))
        return loss

    def on_test_end(self) -> None:
        global_diff_list = []
        global_rel_diff_list = []
        # batch_size = 1
        with open(self.output_file, "a") as f:
            for label, model, difference, couple_idx in self.test_output:
                f.write(f'Couple: ({int(couple_idx[0])}, {int(couple_idx[1])}), real distance: {float(label):.4f}, model distance: {float(model):.4f}, abs. err.: {float(difference):.4f}, rel. err.: {float(difference/label):.4f}\n')
                global_diff_list.append(difference)
                global_rel_diff_list.append(difference/label)
            f.write(f'Abs. err. avg: {float(sum(global_diff_list) / len(global_diff_list)):.4f}, rel. err. avg: {float(sum(global_rel_diff_list) / len(global_rel_diff_list)):.4f}\n\n')


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def test_ranking_accuracy(self, test_dataloader, k):
        """Valuta quanto l'ordinamento delle distanze predette si avvicina a quello reale."""
        all_spearman_corrs = []

        for dtmc_ref, a, b, couple_idx in test_dataloader.dataset:
            dtmc_ref = dtmc_ref.unsqueeze(0)  # Aggiunge la dimensione batch
            sampled_pairs_idx = random.sample(range(len(test_dataloader.dataset.dtmc_data)), k)
            sampled_pairs = [test_dataloader.dataset.dtmc_data[idx] for idx in sampled_pairs_idx]
            sampled_pairs = torch.stack([torch.tensor(p, dtype=torch.float) for p in sampled_pairs])

            # Calcola le distanze predette dal modello
            predicted_distances = self(dtmc_ref.repeat(10, 1, 1), sampled_pairs).detach().cpu().numpy()

            # Recupera le distanze reali
            true_distances = []
            for idx in sampled_pairs_idx:
                label1 = torch.tensor(test_dataloader.dataset.labels[couple_idx[0]])
                label2 = torch.tensor(test_dataloader.dataset.labels[idx])
                true_distances.append(test_dataloader.dataset.get_couples_and_label_diff(label1, label2, None, None)[2])
            true_distances = torch.tensor(true_distances, dtype=torch.float).numpy()

            # Calcola la correlazione di Spearman tra l'ordinamento predetto e quello reale
            spearman_corr, _ = scipy.stats.spearmanr(predicted_distances, true_distances)
            all_spearman_corrs.append(spearman_corr)

        # Stampa la media della correlazione
        avg_spearman_corr = sum(all_spearman_corrs) / len(all_spearman_corrs)
        with open(self.output_file, "a") as f:
            f.write(f'Avg. Spearman correlation ({k=}): {avg_spearman_corr:.4f}\n\n')
        return avg_spearman_corr