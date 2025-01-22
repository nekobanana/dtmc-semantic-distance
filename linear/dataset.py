import itertools
import json
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class DTMCDataset(Dataset):
    def __init__(self, dtmc_folder, label_folder, ds_max_size = None, ds_same_DTMC_fraction = 0.2, dtmc_max_size=300):
        self.ds_max_size = ds_max_size
        self.dtmc_data = []
        self.labels = []
        self.dtmc_max_size = dtmc_max_size
        file_list = sorted(os.listdir(dtmc_folder), key=lambda x: int(x.split('.')[0]))
        different_couples = list(itertools.combinations(range(len(file_list)), 2))
        if ds_max_size is not None:
            different_DTMC_size = int(ds_max_size * (1 - ds_same_DTMC_fraction))
            same_DTMC_size = ds_max_size - different_DTMC_size
        else:
            different_DTMC_size = len(different_couples)
            same_DTMC_size = int(len(different_couples) / (1 - ds_same_DTMC_fraction) * ds_same_DTMC_fraction)

        different_couples = random.sample(different_couples, different_DTMC_size)
        same_DTMC_indeces = random.sample(range(len(file_list)), same_DTMC_size)
        same_DTMC_couples = [(i, i) for i in same_DTMC_indeces]
        self.couples = different_couples + same_DTMC_couples
        # random.shuffle(self.couples)

        for file_name in file_list:
            dtmc_path = os.path.join(dtmc_folder, file_name)
            with open(dtmc_path, 'r') as f:
                dtmc = json.load(f)
            self.dtmc_data.append(dtmc)

            if label_folder is not None:
                label_path = os.path.join(label_folder, file_name)
                with open(label_path, 'r') as f:
                    distr = json.load(f)
                self.labels.append(distr)

    def __len__(self):
        return len(self.couples)

    def __getitem__(self, idx):
        couple_idx = self.couples[idx]
        dtmc1 = torch.tensor(self.dtmc_data[couple_idx[0]], dtype=torch.float)
        if couple_idx[0] == couple_idx[1]:
            return dtmc1, dtmc1[:], torch.tensor(0.)
        dtmc2 = torch.tensor(self.dtmc_data[couple_idx[1]], dtype=torch.float)
        distr1 = torch.tensor(self.labels[couple_idx[0]], dtype=torch.float)
        distr2 = torch.tensor(self.labels[couple_idx[1]], dtype=torch.float)
        label_diff = self.get_label_diff(distr1, distr2, dtmc1, dtmc2)
        return label_diff

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
        eps = 1e-8  # Piccola costante per evitare problemi con zeri
        label1 = label1 + eps
        label2 = label2 + eps
        label1 = label1 / label1.sum()
        label2 = label2 / label2.sum()
        m = 0.5 * (label1 + label2)
        label_diff = 0.5 * (torch.sum(label1 * torch.log(label1 / m)) +
                            torch.sum(label2 * torch.log(label2 / m)))
        return dtmc1, dtmc2, label_diff

class SpectralDistanceDTMCDataset(DTMCDataset):
    def __getitem__(self, idx):
        couple_idx = self.couples[idx]
        dtmc1 = torch.tensor(self.dtmc_data[couple_idx[0]], dtype=torch.float)
        dtmc2 = torch.tensor(self.dtmc_data[couple_idx[1]], dtype=torch.float)
        eigvals_m1 = torch.linalg.eig(dtmc1)
        eigvals_m2 = torch.linalg.eig(dtmc2)
        label = torch.linalg.vector_norm(eigvals_m1.eigenvalues - eigvals_m2.eigenvalues, ord=2)
        return dtmc1, dtmc2, label
