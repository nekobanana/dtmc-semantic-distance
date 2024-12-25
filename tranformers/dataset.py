import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data


class DTMCDatasetTransformer(Dataset):
    def __init__(self, dtmc_folder, label_folder, label_length):
        self.graph_data = []
        self.labels = []
        self.label_length = label_length
        file_list = sorted(os.listdir(dtmc_folder), key=lambda x: int(x.split('.')[0]))

        for file_name in file_list:
            dtmc_path = os.path.join(dtmc_folder, file_name)
            label_path = os.path.join(label_folder, file_name)

            # Carica la matrice di transizione DTMC
            with open(dtmc_path, 'r') as f:
                dtmc = torch.tensor(json.load(f)['dtmc'], dtype=torch.float)

            # Calcola le feature dei nodi: indice normalizzato o media dei vicini
            num_nodes = dtmc.size(0)
            node_indices = torch.arange(0, num_nodes, dtype=torch.float).view(-1, 1) / (num_nodes - 1)
            neighbor_features = dtmc.mean(dim=1, keepdim=True)

            # Scegli la feature da usare (una delle due)
            # node_features = node_indices  # Usa l'indice normalizzato
            # node_features = neighbor_features  # Oppure usa la media dei vicini
            node_features = torch.stack([neighbor_features, node_indices], dim=1).squeeze(2)

            # Edge index (connettività completa per la matrice di transizione)
            edge_index = torch.nonzero(dtmc > 0, as_tuple=False).t().contiguous()

            # Crea il grafo
            graph = Data(x=node_features, edge_index=edge_index, edge_attr=dtmc[edge_index[0], edge_index[1]].unsqueeze(1))
            self.graph_data.append(graph)

            # Carica la label
            with open(label_path, 'r') as f:
                distr_raw = json.load(f)['distribution']
            distr_raw = {int(k): v for k, v in distr_raw.items()}
            max_key = max(distr_raw.keys())
            distr = torch.tensor([distr_raw.get(idx, 0) for idx in range(max_key)], dtype=torch.float)
            distr = F.normalize(distr, p=1, dim=-1)
            self.labels.append(distr)

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        graph1 = self.graph_data[idx]
        graph2 = self.graph_data[(idx + 1) % len(self.graph_data)]
        distr1 = self.labels[idx]
        distr2 = self.labels[(idx + 1) % len(self.labels)]

        max_value = max(distr1.shape[0], distr2.shape[0])
        distr1 = F.pad(distr1, (0, max_value - distr1.shape[0]))
        distr2 = F.pad(distr2, (0, max_value - distr2.shape[0]))
        # total_value_double = torch.linalg.norm(distr1 - distr2, dim=-1, ord=1)
        total_value_double = adaptive_maxpool1d(distr1 - distr2, self.label_length)

        return graph1, graph2, total_value_double

# def kl_div(p, q):
#     return torch.sum(p * torch.log(p / q))

def adaptive_maxpool1d(input_tensor, target_length):
    """
    Riduce la lunghezza di un tensore 1D alla lunghezza target usando MaxPool1d.

    Args:
        input_tensor (torch.Tensor): Tensore di forma (batch_size, 1, input_length)
        target_length (int): Lunghezza desiderata dell'output.

    Returns:
        torch.Tensor: Tensore ridotto alla lunghezza desiderata.
    """
    input_length = input_tensor.shape[-1]
    if input_length < target_length:
        input_tensor = F.pad(input_tensor, (0, target_length - input_length))
        input_length = input_tensor.shape[-1]

    # Calcola kernel_size e stride per ottenere esattamente target_length
    kernel_size = input_length // target_length
    stride = kernel_size

    # Aggiusta kernel_size se il rapporto non è esatto
    if input_length % target_length != 0:
        kernel_size += 1

    output_tensor = F.max_pool1d(input_tensor.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=stride)

    # Se l'output è più lungo della lunghezza target, tronca l'output
    if output_tensor.shape[-1] > target_length:
        output_tensor = output_tensor[:, :, :target_length]

    return output_tensor.squeeze(0).squeeze(0)