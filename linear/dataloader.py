from enum import Enum, auto

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from torchvision import transforms

from dataset import HistogramTotalVarDTMCDataset, HistogramJSDTMCDataset, SpectralDistanceDTMCDataset
from example_dataset import ExampleDataset


class LabelType(Enum):
    HISTOGRAM_JS = auto()
    HISTOGRAM_TOTAL_VAR = auto()
    SPECTRAL_DISTANCE = auto()
    EXAMPLE_DATASET = auto()

# TODO: test non verrrà mai usato insieme a train: fai in modo di separare DTMC uguali e diverse

class DTMCDataLoader(pl.LightningDataModule):
    def __init__(self, dtmc_folder, label_folder, label_type: LabelType, dtmc_max_size=50, ds_size = None,
                 ds_same_dtmc_fraction = 0.2, train_size = 0.8, val_size = 0.1, batch_size = 32, seed = 42,
                 num_workers = None):
        super(DTMCDataLoader, self).__init__()
        self.dtmc_folder = dtmc_folder
        self.label_folder = label_folder
        self.label_type = label_type
        self.dtmc_max_size = dtmc_max_size
        self.ds_same_dtmc_fraction = ds_same_dtmc_fraction
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_size = ds_size
        self.num_workers = num_workers
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
            'batch_size': self.batch_size,
            'seed': self.seed,
            'num_workers': self.num_workers
        }
        assert(self.train_size + self.val_size == 1)
        self.train_dl = None
        self.val_dl = None

    def get_dataset(self, ds_same_dtmc_fraction):
        match self.label_type:
            case LabelType.HISTOGRAM_TOTAL_VAR:
                dataset = HistogramTotalVarDTMCDataset(self.dtmc_folder, self.label_folder,
                                                            ds_max_size=self.dataset_size,
                                                            dtmc_max_size=self.dtmc_max_size,
                                                            ds_same_DTMC_fraction=ds_same_dtmc_fraction)
            case LabelType.HISTOGRAM_JS:
                dataset = HistogramJSDTMCDataset(self.dtmc_folder, self.label_folder,
                                                      ds_max_size=self.dataset_size,
                                                      dtmc_max_size=self.dtmc_max_size,
                                                      ds_same_DTMC_fraction=ds_same_dtmc_fraction)
            case LabelType.SPECTRAL_DISTANCE:
                dataset = SpectralDistanceDTMCDataset(self.dtmc_folder, None, ds_max_size=self.dataset_size,
                                                           dtmc_max_size=self.dtmc_max_size,
                                                           ds_same_DTMC_fraction=ds_same_dtmc_fraction)
            case LabelType.EXAMPLE_DATASET:
                dataset = ExampleDataset('../test/markov_chain_results.json', dtmc_max_size=self.dtmc_max_size, )
        return dataset


    def init_train_val_dataloader(self):
        dataset = self.get_dataset(self.ds_same_dtmc_fraction)
        train_dataset, val_dataset = random_split(dataset, [self.train_size, self.val_size])
        self.train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True)
        self.val_dl = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False)

    def train_dataloader(self):
        if self.train_dl is None:
            self.init_train_val_dataloader()
        return self.train_dl

    def val_dataloader(self):
        if self.val_dl is None:
            self.init_train_val_dataloader()
        return self.val_dl

    def test_dataloader_same(self):
        test_dataset_same = self.get_dataset(1)
        return torch.utils.data.DataLoader(
            test_dataset_same,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False)

    def test_dataloader_different(self):
        test_dataset_different = self.get_dataset(0)
        return torch.utils.data.DataLoader(
            test_dataset_different,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False)
