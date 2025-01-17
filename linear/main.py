import os
import sys
from multiprocessing import set_start_method
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader import DTMCDataLoader, LabelType
from network import SiameseNetwork

# torch.set_float32_matmul_precision('medium')
torch.set_float32_matmul_precision('high')

def main(base_folder):
    # base_folder = '../data/only50_random_30-50_2full'
    dtmc_folder = f'{base_folder}/ready/dtmcs'
    label_folder = f'{base_folder}/ready/labels'
    max_dtmc_size = 50

    label_type = LabelType.SPECTRAL_DISTANCE

    # name = f'test_new_ds_modified'
    name = f'{str(label_type).lower().split('.')[-1]}_{base_folder.split("/")[-1]}'
    logger = TensorBoardLogger("lightning_logs", name=name)

    dataloader = DTMCDataLoader(dtmc_folder, label_folder, label_type=label_type,
                                dtmc_max_size=max_dtmc_size, ds_same_dtmc_fraction=0.2,
                                ds_size=1000, batch_size=1000, seed=2, num_workers=8)

    model = SiameseNetwork(max_dtmc_size=max_dtmc_size, lr=0.001, dl_hparams=dataloader.h_params)

    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{name}", save_top_k=2, monitor="val/loss")
    trainer = pl.Trainer(max_epochs=50, accelerator="gpu", log_every_n_steps=1, logger=logger, callbacks=[checkpoint_callback])
    # trainer = pl.Trainer(max_epochs=1000, accelerator="gpu", gradient_clip_val=1.0, log_every_n_steps=1)
    trainer.fit(model=model, train_dataloaders=dataloader.train_dataloader(), val_dataloaders=dataloader.val_dataloader())
    test_results = trainer.test(model=model, dataloaders=dataloader.test_dataloader())
    print(test_results)
    # torch.save(model.state_dict(), 'save/model.pt')

if __name__ == '__main__':
    set_start_method('spawn')
    main(sys.argv[1])
