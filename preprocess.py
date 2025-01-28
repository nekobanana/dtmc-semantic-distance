import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def preprocess_distributions(dtmc_folder, label_folder, processed_folder, dtmc_max_size, process_labels=True):
    processed_dtmc_folder = os.path.join(processed_folder, dtmc_folder.split('/')[-1])
    os.makedirs(processed_dtmc_folder, exist_ok=True)
    processed_label_folder = os.path.join(processed_folder, label_folder.split('/')[-1])
    os.makedirs(processed_label_folder, exist_ok=True)
    file_list = sorted(os.listdir(label_folder if process_labels else dtmc_folder), key=lambda x: int(x.split('.')[0]))
    for file_name in file_list:
        dtmc_path = os.path.join(dtmc_folder, file_name)
        label_path = os.path.join(label_folder, file_name)

        with open(dtmc_path, 'r') as f:
            dtmc_raw = json.load(f)['dtmc']

        if process_labels:
            with open(label_path, 'r') as f:
                distr_raw = json.load(f)['distribution']

            distr_raw = {int(k): v for k, v in distr_raw.items()}
            max_key = max(distr_raw.keys())
            distr = np.array([distr_raw.get(idx, 0) for idx in range(max_key + 1)], dtype=float)
            if distr.sum() == 0:
                print(f'Error for file {file_name}')
            distr = distr / distr.sum()

        dtmc = np.array(dtmc_raw, dtype=float)
        # dtmc = torch.tensor(dtmc_raw)
        dtmc_n = dtmc.shape[0]
        assert dtmc_n == dtmc.shape[1]
        pad_before = int((dtmc_max_size - dtmc_n) / 2)
        pad_after = dtmc_max_size - dtmc_n - pad_before
        try:
            dtmc = np.pad(dtmc, (pad_before, pad_after), mode='constant', constant_values=0)
            # dtmc = F.pad(dtmc, (0, dtmc_max_size - dtmc_n, 0, dtmc - dtmc_n), mode='constant', value=0)

        except ValueError:
            print(f'Error for file {file_name}')
        with open(os.path.join(processed_dtmc_folder, file_name), 'w') as f:
            json.dump(dtmc.tolist(), f)
        if process_labels:
            with open(os.path.join(processed_label_folder, file_name), 'w') as f:
                json.dump(distr.tolist(), f)


# def preprocess_eigenvalues(dtmc_folder, label_folder, processed_folder):
#     processed_label_folder = os.path.join(processed_folder, 'eigvals')
#     os.makedirs(processed_label_folder, exist_ok=True)
#     file_list = sorted(os.listdir(label_folder), key=lambda x: int(x.split('.')[0]))
#     for file_name in file_list:
#         dtmc_path = os.path.join(dtmc_folder, file_name)
#
#         with open(dtmc_path, 'r') as f:
#             dtmc_raw = json.load(f)['dtmc']
#         dtmc = np.array(dtmc_raw, dtype=float)
#         eigen_values = np.linalg.eigvals(dtmc)
#         with open(os.path.join(processed_label_folder, file_name), 'w') as f:
#             json.dump(eigen_values.tolist(), f)


if __name__ == '__main__':
    base_folder = sys.argv[1]
    # base_folder = 'data/only50_random_30-50_2full'
    preprocess_distributions(f'{base_folder}/raw/dtmcs', f'{base_folder}/raw/labels',
                             f'{base_folder}/ready',50,
                             process_labels=True)
    # preprocess_eigenvalues(f'{base_folder}/raw/dtmcs', f'{base_folder}/raw/labels',f'{base_folder}/ready')