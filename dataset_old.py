import json
import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class DTMCEmbeddingDatasetOld(Dataset):
    def __init__(self, dtmc_folder, label_folder):
        self.dtmc_data = []
        self.labels = []
        file_list = sorted(os.listdir(dtmc_folder), key=lambda x: int(x.split('.')[0]))
        for file_name in file_list:
            dtmc_path = os.path.join(dtmc_folder, file_name)
            label_path = os.path.join(label_folder, file_name)

            with open(dtmc_path, 'r') as f:
                dtmc = torch.tensor(json.load(f)['dtmc'], dtype=torch.float32, device='cuda')

            with open(label_path, 'r') as f:
                distr_raw = json.load(f)['distribution']

            distr_raw = {int(k): v for k, v in distr_raw.items()}
            max_key = max(distr_raw.keys())
            distr = torch.tensor([distr_raw.get(idx, 0) for idx in range(max_key)], dtype=torch.float64)
            distr = torch.nn.functional.normalize(distr, p=2, dim=-1)
            self.dtmc_data.append(dtmc)
            self.labels.append(distr)

    def __len__(self):
        return len(self.dtmc_data)

    def __getitem__(self, idx):
        dtmc1 = self.dtmc_data[idx]
        dtmc2 = self.dtmc_data[(idx + 1) % len(self.dtmc_data)]  # Pair with next DTMC
        distr1 = self.labels[idx]
        distr2 = self.labels[(idx + 1) % len(self.labels)]
        # label_diff = torch.tensor(abs(self.labels[idx] - self.labels[(idx + 1) % len(self.labels)]), dtype=torch.float)
        max_value = max(distr1.shape[0], distr2.shape[0])
        distr1 = F.pad(distr1, (0, max_value - distr1.shape[0]))
        distr2 = F.pad(distr2, (0, max_value - distr2.shape[0]))
        m = 0.5 * (distr1 + distr2)
        js_distance = 0.5 * (torch.sum(distr1 * torch.log(distr1 / m + 1e-8)) +
                             torch.sum(distr2 * torch.log(distr2 / m + 1e-8)))
        return (
            torch.tensor(dtmc1.flatten(), dtype=torch.float),
            torch.tensor(dtmc2.flatten(), dtype=torch.float),
            js_distance
        )
