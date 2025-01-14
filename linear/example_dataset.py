import json
import torch
from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self, file_path, dtmc_max_size):
        """
        Args:
            file_path (str): Path to the JSON file containing the dataset.
            dtmc_max_size (int): The maximum number of rows and columns for the matrices.
        """
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.dtmc_max_size = dtmc_max_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            tuple: (m1, m2, spectral_distance)
                - m1 (torch.Tensor): Padded flattened matrix 1.
                - m2 (torch.Tensor): Padded flattened matrix 2.
                - spectral_distance (torch.Tensor): Spectral distance between the matrices.
        """
        sample = self.data[idx]

        # Extract flattened matrices m1 and m2
        m1 = torch.tensor(sample['m1'], dtype=torch.float32)
        m2 = torch.tensor(sample['m2'], dtype=torch.float32)

        # Calculate the original size (assuming square matrices)
        original_size = int(len(m1) ** 0.5)

        # Reshape to square matrices
        m1 = m1.view(original_size, original_size)
        m2 = m2.view(original_size, original_size)

        # Pad to dtmc_max_size x dtmc_max_size
        m1 = torch.nn.functional.pad(m1, (0, self.dtmc_max_size - original_size, 0, self.dtmc_max_size - original_size),
                                     mode='constant', value=0)
        m2 = torch.nn.functional.pad(m2, (0, self.dtmc_max_size - original_size, 0, self.dtmc_max_size - original_size),
                                     mode='constant', value=0)

        # Flatten the padded matrices
        m1 = m1.flatten()
        m2 = m2.flatten()

        # Extract spectral distance
        spectral_distance = torch.tensor(sample['distances']['spectral'], dtype=torch.float32)

        return m1, m2, spectral_distance

# Example usage:
# dataset = DistanceDataset("path_to_your_file.json", dtmc_max_size=50)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
#
# for m1, m2, spectral_distance in dataloader:
#     print(m1, m2, spectral_distance)
