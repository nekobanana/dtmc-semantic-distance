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
        distr = np.array([distr_raw.get(idx, 0) for idx in range(max_key)], dtype=float)
        distr = distr / distr.sum()
        dtmc = np.array(dtmc_raw, dtype=float)
        dtmc = np.pad(dtmc, dtmc_max_size)
        with open(os.path.join(processed_dtmc_folder, file_name), 'w') as f:
            json.dump(dtmc.tolist(), f)
        with open(os.path.join(processed_label_folder, file_name), 'w') as f:
            json.dump(distr.tolist(), f)


if __name__ == '__main__':
    preprocess_distributions('data/max100/raw/dtmcs', 'data/max100/raw/labels',
                             'data/max100/ready',100)