import os
import sys
import json
from multiprocessing import set_start_method
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader import DTMCDataLoader, LabelType
from network import SiameseNetwork

torch.set_float32_matmul_precision('high')

def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4, default=str)

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_versioned_dir(name):
    base_dir = "checkpoints"
    version = 0
    while os.path.exists(os.path.join(base_dir, f"{name}/version_{version}")):
        version += 1
    return os.path.join(base_dir, f"{name}/version_{version}")

def main(base_folder):
    # base_folder = '../data/max50_random'
    dtmc_folder = f'{base_folder}/ready/dtmcs'
    label_folder = f'{base_folder}/ready/labels'
    max_dtmc_size = 50
    label_type = LabelType.HISTOGRAM_JS

    lr = 0.001
    max_epochs = 500

    name = f'{str(label_type).lower().split(".")[-1]}_{base_folder.split("/")[-1]}'
    logger = TensorBoardLogger("lightning_logs", name=name)

    checkpoint_dir = get_versioned_dir(name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataloader = DTMCDataLoader(dtmc_folder, label_folder, label_type=label_type,
                                dtmc_max_size=max_dtmc_size, ds_same_dtmc_fraction=0.01,
                                train_size=0.9, val_size=0.1, test_size=0,
                                ds_size=100000, batch_size=4096, seed=0, num_workers=8)
    model = SiameseNetwork(max_dtmc_size=max_dtmc_size, lr=lr, dl_hparams=dataloader.h_params)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=2, monitor="val/loss")
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", log_every_n_steps=1, logger=logger, callbacks=[checkpoint_callback])

    config = {
        "name": name,
        "base_folder": base_folder,
        "dataloader_params": dataloader.h_params,
        "model_params": {"max_dtmc_size": max_dtmc_size, "lr": lr},
        "trainer_params": {"max_epochs": max_epochs, "accelerator": "gpu", "log_every_n_steps": 1}
    }
    save_config(config, os.path.join(checkpoint_dir, "config.json"))

    trainer.fit(model=model, train_dataloaders=dataloader.train_dataloader(), val_dataloaders=dataloader.val_dataloader())

    test_results = trainer.test(model=model, dataloaders=dataloader.test_dataloader())
    print(test_results)

def get_logger_from_config(config):
    name = config["name"]
    logger = TensorBoardLogger("lightning_logs", name=name)
    return logger


if __name__ == '__main__':
    set_start_method('spawn')

    mode = sys.argv[1]
    if mode == "train":
        main(sys.argv[2])
    elif mode == "resume":
        checkpoint_path = sys.argv[2]
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        config = load_config(config_path)
        logger = get_logger_from_config(config)
        dataloader = DTMCDataLoader(**config["dataloader_params"])
        model = SiameseNetwork(**config["model_params"], dl_hparams=dataloader.h_params)
        trainer = pl.Trainer(logger=logger, **config["trainer_params"])
        trainer.fit(model=model, ckpt_path=checkpoint_path)
    elif mode == "test":
        checkpoint_path = sys.argv[2]
        test_folder = sys.argv[3]
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        config = load_config(config_path)
        logger = get_logger_from_config(config)
        config["dataloader_params"]["dtmc_folder"] = os.path.join(test_folder, "dtmcs")
        config["dataloader_params"]["label_folder"] = os.path.join(test_folder, "labels")
        config["dataloader_params"]["train_size"] = 0
        config["dataloader_params"]["val_size"] = 0
        config["dataloader_params"]["test_size"] = 1
        config["dataloader_params"]["label_type"] = LabelType[config["dataloader_params"]["label_type"].split(".")[-1]]
        # config["dataloader_params"]["ds_size"] = 5000
        # config["dataloader_params"]["ds_same_dtmc_fraction"] = 0.2

        dataloader = DTMCDataLoader(**config["dataloader_params"])
        model = SiameseNetwork(**config["model_params"], checkpoint_name=checkpoint_path, dl_hparams=dataloader.h_params)

        trainer = pl.Trainer(logger=logger, **config["trainer_params"])
        test_results = trainer.test(model=model, dataloaders=dataloader.test_dataloader(), ckpt_path=checkpoint_path)
        print(test_results)
    else:
        print("Usage:")
        print("  train <base_folder_path>")
        print("  resume <checkpoint_path>")
        print("  test <checkpoint_path> <test_folder>")
