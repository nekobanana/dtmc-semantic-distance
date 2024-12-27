import abc
import itertools
import json
import math
import os
from abc import ABC
from collections.abc import Iterator

from torchvision import transforms

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class DTMCDataset(Dataset):
    __metaclass__ = abc.ABCMeta
    def __init__(self, dtmc_folder, label_folder, ds_max_size = None, dtmc_max_size=300):
        self.ds_max_size = ds_max_size
        self.dtmc_data = []
        self.labels = []
        self.dtmc_max_size = dtmc_max_size
        file_list = sorted(os.listdir(dtmc_folder), key=lambda x: int(x.split('.')[0]))
        self.couples = list(itertools.combinations(range(len(file_list)), 2))
        if ds_max_size is not None:
            self.couples = self.couples[:ds_max_size]
        for file_name in file_list:
            dtmc_path = os.path.join(dtmc_folder, file_name)
            label_path = os.path.join(label_folder, file_name)

            with open(dtmc_path, 'r') as f:
                dtmc = json.load(f)

            with open(label_path, 'r') as f:
                distr = json.load(f)

            self.dtmc_data.append(dtmc)
            self.labels.append(distr)

    def __len__(self):
        return len(self.couples)

    def __getitem__(self, idx):
        couple_idx = self.couples[idx]
        dtmc1 = torch.tensor(self.dtmc_data[couple_idx[0]], dtype=torch.float)
        dtmc2 = torch.tensor(self.dtmc_data[couple_idx[1]], dtype=torch.float)
        distr1 = torch.tensor(self.labels[couple_idx[0]], dtype=torch.float)
        distr2 = torch.tensor(self.labels[couple_idx[1]], dtype=torch.float)
        label_diff = self.get_label_diff(distr1, distr2, dtmc1, dtmc2)
        return label_diff

    @abc.abstractmethod
    def get_label_diff(self, label1, label2, dtmc1, dtmc2):
        pass


def get_label_diff_preparation(label1, label2):
    max_value = max(label1.shape[0], label2.shape[0])
    label1 = F.pad(F.normalize(label1, p=1, dim=-1), (0, max_value - label1.shape[0]))
    label2 = F.pad(F.normalize(label2, p=1, dim=-1), (0, max_value - label2.shape[0]))
    return label1, label2


class HistogramTotalVarDTMCDataset(DTMCDataset):
    def get_label_diff(self, label1, label2, dtmc1, dtmc2):
        label1, label2 = get_label_diff_preparation(label1, label2)
        label_diff = 0.5 * torch.linalg.norm(label1 - label2, ord=1)
        return dtmc1, dtmc2, label_diff

class HistogramJSDTMCDataset(DTMCDataset):
    def get_label_diff(self, label1, label2, dtmc1, dtmc2):
        label1, label2 = get_label_diff_preparation(label1, label2)
        m = 0.5 * (label1 + label2)
        label_diff = 0.5 * (torch.sum(label1 * torch.log(label1 / m + 1e-8)) +
                            torch.sum(label2 * torch.log(label2 / m + 1e-8)))
        return dtmc1, dtmc2, label_diff

class MixingTimeDTMCDataset(DTMCDataset):
    def get_label_diff(self, label1, label2, dtmc1, dtmc2):
        label_diff = label1 - label2
        return dtmc1, dtmc2, label_diff
