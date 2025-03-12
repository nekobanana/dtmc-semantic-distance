import os
import random
import sys
import json
from multiprocessing import set_start_method

import scipy
from lightning_fabric import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader import DTMCDataLoader, LabelType
from network import SiameseNetwork

torch.set_float32_matmul_precision('high')
seed_everything(8, workers=True)

def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4, default=str)

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_checkpoint_dir(name, log_dir):
    base_dir = "checkpoints_new"
    checkpoint_dir = os.path.join(base_dir, f"{name}/{log_dir.split('/')[-1]}")
    print(f'checkpoint: {checkpoint_dir}')
    return checkpoint_dir

def test_ranking_accuracy(model, test_dataloader, k):
    """Valuta quanto l'ordinamento delle distanze predette si avvicina a quello reale."""
    all_spearman_corrs = []

    for dtmc_ref, a, b, couple_idx in test_dataloader.dataset:
        dtmc_ref = dtmc_ref.unsqueeze(0)  # Aggiunge la dimensione batch
        sampled_pairs_idx = random.sample(range(len(test_dataloader.dataset.dtmc_data)), k)
        sampled_pairs = [test_dataloader.dataset.dtmc_data[idx] for idx in sampled_pairs_idx]
        sampled_pairs = torch.stack([torch.tensor(p, dtype=torch.float) for p in sampled_pairs])

        # Calcola le distanze predette dal modello
        predicted_distances = model(dtmc_ref.repeat(10, 1, 1), sampled_pairs).detach().cpu().numpy()

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
    print(f'Avg. Spearman correlation ({k=}): {avg_spearman_corr:.4f}')
    return avg_spearman_corr

def train_model(base_folder):
    # base_folder = '../data/max50_random'
    dtmc_folder = f'{base_folder}/ready/dtmcs'
    label_folder = f'{base_folder}/ready/labels'
    max_dtmc_size = 50
    label_type = LabelType.HISTOGRAM_JS

    lr = 0.001
    max_epochs = 500

    log_dir = 'lightning_logs_new'

    name = f'{str(label_type).lower().split(".")[-1]}_{base_folder.split("/")[-1]}'
    logger = TensorBoardLogger(log_dir, name=name)
    log_dir = logger.log_dir

    checkpoint_dir = get_checkpoint_dir(name, log_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataloader = DTMCDataLoader(dtmc_folder, label_folder, label_type=label_type,
                                dtmc_max_size=max_dtmc_size, ds_same_dtmc_fraction=0.01,
                                train_size=0.9, val_size=0.1,
                                ds_size=5000, batch_size=1024, seed=0, num_workers=8)
    model = SiameseNetwork(max_dtmc_size=max_dtmc_size, lr=lr, dl_hparams=dataloader.h_params)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=2, monitor="val/loss")
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", deterministic=True,
                         log_every_n_steps=1, logger=logger, callbacks=[checkpoint_callback])

    config = {
        "name": name,
        "base_folder": base_folder,
        "dataloader_params": dataloader.h_params,
        "model_params": {"max_dtmc_size": max_dtmc_size, "lr": lr},
        "trainer_params": {"max_epochs": max_epochs, "accelerator": "gpu", "log_every_n_steps": 1},
        "logdir": log_dir
    }
    save_config(config, os.path.join(checkpoint_dir, "config.json"))

    trainer.fit(model=model, train_dataloaders=dataloader.train_dataloader(), val_dataloaders=dataloader.val_dataloader())

    # test_results = trainer.test(model=model, dataloaders=dataloader.test_dataloader())
    # print(test_results)

def get_logger_from_config(config):
    name = config["name"]
    log_dir = config["logdir"]
    logger = TensorBoardLogger(log_dir, name=name)
    return logger


def test_model(checkpoint_path, test_folder):
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    config = load_config(config_path)
    logger = get_logger_from_config(config)
    config["dataloader_params"]["dtmc_folder"] = os.path.join(test_folder, "dtmcs")
    config["dataloader_params"]["label_folder"] = os.path.join(test_folder, "labels")
    config["dataloader_params"]["label_type"] = LabelType[config["dataloader_params"]["label_type"].split(".")[-1]]
    config["dataloader_params"]["batch_size"] = 1

    dataloader = DTMCDataLoader(**config["dataloader_params"])
    model = SiameseNetwork(**config["model_params"], checkpoint_name=checkpoint_path, dl_hparams=dataloader.h_params)
    trainer = pl.Trainer(logger=logger, **config["trainer_params"], devices=1, num_nodes=1)
    test_results_same = trainer.test(model=model, dataloaders=dataloader.test_dataloader_same(), ckpt_path=checkpoint_path)
    print(test_results_same)
    test_results_different = trainer.test(model=model, dataloaders=dataloader.test_dataloader_different(), ckpt_path=checkpoint_path)
    print(test_results_different)
    test_ranking_accuracy(model, dataloader.test_dataloader_same(), 10)


if __name__ == '__main__':
    set_start_method('spawn')

    mode = sys.argv[1]
    if mode == "train":
        train_model(sys.argv[2])
    elif mode == "test":
        test_model(sys.argv[2], sys.argv[3])
    else:
        print("Usage:")
        print("  train <base_folder_path>")
        print("  test <checkpoint_path> <test_folder>")
