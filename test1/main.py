# Parametri
import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
import scipy.stats as stats
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from test1.dataset import MatrixDataset
from test1.network import MLPModel
from torch.utils.data import random_split

def main():
    logger = TensorBoardLogger("logs", name="markov_model")

    # Creazione dataset e split
    dataset = MatrixDataset("../data/max50_random_3/raw")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)
    test_loader = DataLoader(test_dataset, batch_size=1)  # Batch=1 per plotting

    # Definizione modello
    input_dim = dataset.max_size ** 2
    output_dim = len(dataset[0][1])  # Numero di parametri della distribuzione
    model = MLPModel(input_dim, output_dim)

    # Trainer
    trainer = pl.Trainer(max_epochs=150, logger=logger, accelerator="gpu", log_every_n_steps=1)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # Visualizzazione per alcuni esempi di test
    test_examples = 20
    model.eval()

    with torch.no_grad():
        for i, (x, y, file_name) in enumerate(test_loader):
            if i >= test_examples:
                break
            y_hat = model(x).squeeze().cpu().numpy()  # Parametri predetti
            y_true = y.squeeze().cpu().numpy()  # Parametri reali

            plt.figure(figsize=(6, 4))
            file_name = file_name[0]
            with open(os.path.join(dataset.raw_label_data_dir, file_name), 'r') as f:
                distr = json.load(f)['distribution']
            distr_int = dict(map(lambda x: (int(x[0]), int(x[1])), distr.items()))
            steps = []
            for x_val in distr_int.keys():
                steps += [x_val] * distr_int[x_val]
            steps = np.array(steps)
            plt.hist(steps, bins=max(steps) - min(steps), density=True, alpha=0.6, label="True Distribution")
            plot_lognormal(y_true, "True fitted Distribution", "blue")
            plot_lognormal(y_hat, "Predicted Distribution", "red")

            plt.legend()
            plt.title(f"Test Sample {file_name}")
            plt.show()

def plot_lognormal(params, label, color):
    x = np.linspace(0, 500, 500)
    pdf = stats.lognorm.pdf(x, *params)
    plt.plot(x, pdf, label=label, color=color)


if __name__ == "__main__":
    main()