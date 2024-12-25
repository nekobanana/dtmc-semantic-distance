from enum import Enum, auto

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import random_split
from torchvision import transforms

from linear.dataset import HistogramTotalVarDTMCDataset, HistogramJSDTMCDataset, MixingTimeDTMCDataset


class LabelType(Enum):
    HISTOGRAM_JS = auto()
    HISTOGRAM_TOTAL_VAR = auto()
    MIXING_TIME = auto()


class DTMCDataLoader(pl.LightningModule):
    def __init__(self, dtmc_folder, label_folder, label_type: LabelType, dtmc_max_size=300, ds_size = None,
                 train_size = 0.8, val_size = 0.1, test_size = 0.1, batch_size = 32, seed = 42,
                 num_workers = None):
        super(DTMCDataLoader, self).__init__()
        self.dtmc_folder = dtmc_folder
        self.label_folder = label_folder
        self.label_type = label_type
        self.dtmc_max_size = dtmc_max_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_size = ds_size
        self.num_workers = num_workers
        torch.manual_seed(self.seed)
        self.save_hyperparameters(label_type, train_size, val_size, test_size, batch_size, seed)
        assert(train_size + val_size + test_size == 1)
        match self.label_type:
            case LabelType.HISTOGRAM_TOTAL_VAR:
                dataset = HistogramTotalVarDTMCDataset(dtmc_folder, label_folder, ds_max_size=ds_size, dtmc_max_size=dtmc_max_size)
            case LabelType.HISTOGRAM_JS:
                dataset = HistogramJSDTMCDataset(dtmc_folder, label_folder, ds_max_size=ds_size, dtmc_max_size=dtmc_max_size)
            case LabelType.MIXING_TIME:
                dataset = MixingTimeDTMCDataset(dtmc_folder, label_folder, ds_max_size=ds_size, dtmc_max_size=dtmc_max_size)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [self.train_size, self.val_size, self.test_size])

    def train_dataloader(self):
        train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True)
        return train_data_loader

    def val_dataloader(self):
        val_data_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False)
        return val_data_loader

    def test_dataloader(self):
        test_data_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False)
        return test_data_loader
