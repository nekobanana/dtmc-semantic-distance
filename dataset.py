import json
import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer


def _2d_tensor_to_string(tensor):
    return '\n '.join([' '.join([str(inner) for inner in outer]) for outer in tensor.tolist()])


class DTMCEmbeddingDataset(Dataset):
    def __init__(self, dtmc_folder, label_folder, tokenizer_name="bert-base-uncased", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.dtmc_data = []
        self.labels = []
        file_list = sorted(os.listdir(dtmc_folder), key=lambda x: int(x.split('.')[0]))
        for file_name in file_list:
            dtmc_path = os.path.join(dtmc_folder, file_name)
            label_path = os.path.join(label_folder, file_name)

            with open(dtmc_path, 'r') as f:
                dtmc = torch.tensor(json.load(f)['dtmc'], dtype=torch.float, device='cuda')

            with open(label_path, 'r') as f:
                distr_raw = json.load(f)['distribution']

            distr_raw = {int(k): v for k, v in distr_raw.items()}
            max_key = max(distr_raw.keys())
            distr = torch.tensor([distr_raw.get(idx, 0) for idx in range(max_key)], dtype=torch.float)
            distr = F.normalize(distr, p=1, dim=-1)
            self.dtmc_data.append(dtmc)
            self.labels.append(distr)

    def __len__(self):
        return len(self.dtmc_data)

    def __getitem__(self, idx):

        dtmc1 = self.tokenizer(
            _2d_tensor_to_string(self.dtmc_data[idx]), padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_length
        )
        dtmc2 = self.tokenizer(
            _2d_tensor_to_string(self.dtmc_data[(idx + 1) % len(self.dtmc_data)]), padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_length
        )
        distr1 = self.labels[idx]
        distr2 = self.labels[(idx + 1) % len(self.labels)]
        max_value = max(distr1.shape[0], distr2.shape[0])
        distr1 = F.pad(distr1, (0, max_value - distr1.shape[0]))
        distr2 = F.pad(distr2, (0, max_value - distr2.shape[0]))
        # m = 0.5 * (distr1 + distr2)
        # js_divergence = 0.5 * (kl_div(distr1, m) + kl_div(distr2, m))
        # return dtmc1, dtmc2, js_divergence
        total_value_double = torch.linalg.norm(distr1 - distr2, dim=-1, ord=1)
        return dtmc1, dtmc2, total_value_double


# def kl_div(p, q):
#     return torch.sum(p * torch.log(p / q))