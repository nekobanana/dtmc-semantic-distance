# Esempio di file DTMC
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from dataset import DTMCEmbeddingDataset
from nnetwork import SiameseDTMC
torch.set_float32_matmul_precision('medium')
# torch.set_float32_matmul_precision('high')

def main():
    dtmc_folder = 'data/dtmcs'
    label_folder = 'data/labels'

    dataset = DTMCEmbeddingDataset(dtmc_folder, label_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Inizializza il modello
    model = SiameseDTMC()

    # Allenamento con PyTorch Lightning
    trainer = Trainer(max_epochs=None, accelerator="gpu", log_every_n_steps=1)
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    main()