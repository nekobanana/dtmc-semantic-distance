from enum import Enum, auto

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from torchvision import transforms

from linear.dataset import HistogramTotalVarDTMCDataset, HistogramJSDTMCDataset, SpectralDistanceDTMCDataset
from linear.example_dataset import ExampleDataset


class LabelType(Enum):
    HISTOGRAM_JS = auto()
    HISTOGRAM_TOTAL_VAR = auto()
    SPECTRAL_DISTANCE = auto()
    EXAMPLE_DATASET = auto()


class DTMCDataLoader(pl.LightningDataModule):
    def __init__(self, dtmc_folder, label_folder, label_type: LabelType, dtmc_max_size=300, ds_size = None,
                 ds_same_dtmc_fraction = 0.2, train_size = 0.8, val_size = 0.1, test_size = 0.1, batch_size = 32, seed = 42,
                 num_workers = None):
        super(DTMCDataLoader, self).__init__()
        self.dtmc_folder = dtmc_folder
        self.label_folder = label_folder
        self.label_type = label_type
        self.dtmc_max_size = dtmc_max_size
        self.ds_same_dtmc_fraction = ds_same_dtmc_fraction
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_size = ds_size
        self.num_workers = num_workers
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        torch.manual_seed(self.seed)
        self.h_params = {
            'dtmc_folder': self.dtmc_folder,
            'label_folder': self.label_folder,
            'label_type': self.label_type,
            'dtmc_max_size': self.dtmc_max_size,
            'ds_same_dtmc_fraction': self.ds_same_dtmc_fraction,
            'ds_size': self.dataset_size,
            'train_size': self.train_size,
            'val_size': self.val_size,
            'test_size': self.test_size,
            'batch_size': self.batch_size,
            'seed': self.seed,
            'num_workers': self.num_workers
        }
        assert(self.train_size + self.val_size + self.test_size == 1)
        match self.label_type:
            case LabelType.HISTOGRAM_TOTAL_VAR:
                dataset = HistogramTotalVarDTMCDataset(self.dtmc_folder, self.label_folder, ds_max_size=self.dataset_size,
                                                       dtmc_max_size=self.dtmc_max_size, ds_same_DTMC_fraction=self.ds_same_dtmc_fraction)
            case LabelType.HISTOGRAM_JS:
                dataset = HistogramJSDTMCDataset(self.dtmc_folder, self.label_folder, ds_max_size=self.dataset_size,
                                                 dtmc_max_size=self.dtmc_max_size, ds_same_DTMC_fraction=self.ds_same_dtmc_fraction)
            case LabelType.SPECTRAL_DISTANCE:
                dataset = SpectralDistanceDTMCDataset(self.dtmc_folder, None, ds_max_size=self.dataset_size,
                                                      dtmc_max_size=self.dtmc_max_size, ds_same_DTMC_fraction=self.ds_same_dtmc_fraction)
            case LabelType.EXAMPLE_DATASET:
                dataset = ExampleDataset('../test/markov_chain_results.json', dtmc_max_size=self.dtmc_max_size,)
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
