import numpy as np
import torch
import torch.nn.functional as F
import json
import os
from torch.utils.data import Dataset
import scipy.optimize as opt
import scipy.stats as stats
import matplotlib.pyplot as plt


class MatrixDataset(Dataset):
    def __init__(self, root_dir):
        self.matrix_dir = os.path.join(root_dir, "dtmcs")
        self.raw_label_data_dir = os.path.join(root_dir, "labels")
        self.curve_dir = os.path.join(root_dir, "fitted")
        self.files = sorted([f for f in os.listdir(self.matrix_dir) if f.endswith('.json')],
                            key=lambda x: int(x.split('.')[0]))

        # Troviamo la dimensione massima per il padding
        self.max_size = self._find_max_size()
        self._preprocess()

    def _preprocess(self):
        if not os.path.isdir(self.curve_dir) or len(os.listdir(self.curve_dir)) == 0:
            os.makedirs(self.curve_dir, exist_ok=True)
            file_list = os.listdir(self.raw_label_data_dir)
            for file_name in file_list:
                with open(os.path.join(self.raw_label_data_dir, file_name), 'r') as f:
                    distr = json.load(f)['distribution']
                    fitted = fit_distribution(distr)
                    with open(os.path.join(self.curve_dir, file_name), 'w') as f2:
                        json.dump(fitted, f2)

    def _find_max_size(self):
        max_dim = 0
        for file in self.files:
            with open(os.path.join(self.matrix_dir, file), 'r') as f:
                matrix = json.load(f)['dtmc']
                max_dim = max(max_dim, len(matrix))
        return max_dim

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]

        with open(os.path.join(self.matrix_dir, file_name), 'r') as f:
            matrix = torch.tensor(json.load(f)['dtmc'], dtype=torch.float32)
        padded_matrix = F.pad(matrix, (0, self.max_size - matrix.shape[1], 0, self.max_size - matrix.shape[0]), value=0)

        matrix_flat = padded_matrix.view(-1)

        with open(os.path.join(self.curve_dir, file_name), 'r') as f:
            curve = torch.tensor(json.load(f), dtype=torch.float32)
        return matrix_flat, curve, file_name


def fit_distribution(distribution):
    distr_int = dict(map(lambda x: (int(x[0]), int(x[1])), distribution.items()))
    y_vals = []
    for x_val in distr_int.keys():
        y_vals += [x_val] * distr_int[x_val]
    y_vals = np.array(y_vals)
    dist = stats.lognorm
    params = dist.fit(y_vals)
    return params
