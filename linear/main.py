from multiprocessing import set_start_method

import torch
import pytorch_lightning as pl

from linear.DataLoader import DTMCDataLoader, LabelType
from linear.network import SiameseNetwork

# torch.set_float32_matmul_precision('medium')
torch.set_float32_matmul_precision('high')

def main():
    base_folder = '../data/only32'
    dtmc_folder = f'{base_folder}/ready/dtmcs'
    label_folder = f'{base_folder}/ready/labels'

    max_dtmc_size = 32

    dataloader = DTMCDataLoader(dtmc_folder, label_folder, label_type=LabelType.HISTOGRAM_TOTAL_VAR, dtmc_max_size=max_dtmc_size,
                                ds_size=10000, batch_size=8, seed=2, num_workers=8)

    model = SiameseNetwork(max_dtmc_size=max_dtmc_size, lr=0.001)

    trainer = pl.Trainer(max_epochs=1000, accelerator="gpu", log_every_n_steps=1)
    # trainer = pl.Trainer(max_epochs=1000, accelerator="gpu", gradient_clip_val=1.0, log_every_n_steps=1)
    trainer.fit(model=model, train_dataloaders=dataloader.train_dataloader(), val_dataloaders=dataloader.val_dataloader())
    test_results = trainer.test(model=model, dataloaders=dataloader.test_dataloader())
    print(test_results)
    torch.save(model.state_dict(), 'save/model.pt')

if __name__ == '__main__':
    set_start_method('spawn')
    main()
