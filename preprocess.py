import json
import os
from pathlib import Path

import numpy as np


def preprocess_distributions(dtmc_folder, label_folder, processed_folder, dtmc_max_size):
    processed_dtmc_folder = os.path.join(processed_folder, dtmc_folder.split('/')[-1])
    os.makedirs(processed_dtmc_folder, exist_ok=True)
    processed_label_folder = os.path.join(processed_folder, label_folder.split('/')[-1])
    os.makedirs(processed_label_folder, exist_ok=True)
    file_list = sorted(os.listdir(label_folder), key=lambda x: int(x.split('.')[0]))
    for file_name in file_list:
        dtmc_path = os.path.join(dtmc_folder, file_name)
        label_path = os.path.join(label_folder, file_name)

        with open(label_path, 'r') as f:
            distr_raw = json.load(f)['distribution']
        with open(dtmc_path, 'r') as f:
            dtmc_raw = json.load(f)['dtmc']

        distr_raw = {int(k): v for k, v in distr_raw.items()}
        max_key = max(distr_raw.keys())
        distr = np.array([distr_raw.get(idx, 0) for idx in range(max_key + 1)], dtype=float)
        if distr.sum() == 0:
            print(f'Error for file {file_name}')
        distr = distr / distr.sum()
        dtmc = np.array(dtmc_raw, dtype=float)
        dtmc_n = dtmc.shape[0]
        assert dtmc_n == dtmc.shape[1]
        pad_before = int((dtmc_max_size - dtmc_n) / 2)
        pad_after = dtmc_max_size - dtmc_n - pad_before
        dtmc = np.pad(dtmc, (pad_before, pad_after), mode='constant', constant_values=0)
        with open(os.path.join(processed_dtmc_folder, file_name), 'w') as f:
            json.dump(dtmc.tolist(), f)
        with open(os.path.join(processed_label_folder, file_name), 'w') as f:
            json.dump(distr.tolist(), f)


if __name__ == '__main__':
    base_folder = 'data/only32'
    preprocess_distributions(f'{base_folder}/raw/dtmcs', f'{base_folder}/raw/labels',
                             f'{base_folder}/ready',32)