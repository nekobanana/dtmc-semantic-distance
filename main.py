# Esempio di file DTMC
import torch
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader

from dataset import DTMCDataset
from nnetwork import SiameseDTMC
torch.set_float32_matmul_precision('medium')
# torch.set_float32_matmul_precision('high')

def main():
    dtmc_folder = 'data/dtmcs'
    label_folder = 'data/labels'

    dataset = DTMCDataset(dtmc_folder, label_folder, label_length=8) # label_length deve essere uguale a out_channels
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Inizializza il modello
    model = SiameseDTMC(in_channels=2, hidden_channels=16, out_channels=8, num_heads=4, learning_rate=1e-4)

    # Allenamento con PyTorch Lightning
    trainer = Trainer(max_epochs=1000, accelerator="gpu", log_every_n_steps=1)
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    main()