import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import json
from torch.nn.utils.rnn import pad_sequence

torch.set_default_device('cuda')

# Dataset per coppie di DTMC e distanze
class SiameseDTMCDataset(Dataset):
    def __init__(self, dtmc_dir, labels_dir):
        self.data = []

        # Carica le DTMC e i relativi istogrammi
        file_list = sorted(os.listdir(dtmc_dir), key=lambda x: int(x.split('.')[0]))
        for i in range(len(file_list)):
            for j in range(i + 1, len(file_list)):
                dtmc_path1 = os.path.join(dtmc_dir, file_list[i])
                dtmc_path2 = os.path.join(dtmc_dir, file_list[j])
                label_path1 = os.path.join(labels_dir, file_list[i])
                label_path2 = os.path.join(labels_dir, file_list[j])

                with open(dtmc_path1, 'r') as f:
                    dtmc1 = torch.tensor(json.load(f)['dtmc'], dtype=torch.float32, device='cuda')

                with open(dtmc_path2, 'r') as f:
                    dtmc2 = torch.tensor(json.load(f)['dtmc'], dtype=torch.float32, device='cuda')

                with open(label_path1, 'r') as f:
                    distr1_raw = json.load(f)['distribution']
                    max_value_1 = int(max(distr1_raw.keys(), key=lambda x: int(x)))

                with open(label_path2, 'r') as f:
                    distr2_raw = json.load(f)['distribution']
                    max_value_2 = int(max(distr2_raw.keys(), key=lambda x: int(x)))

                max_value = max(max_value_1, max_value_2)
                distr1 = np.array([distr1_raw.get(str(idx), 0) for idx in range(max_value)], dtype=np.float32)
                distr1 /= distr1.sum()

                distr2 = np.array([distr2_raw.get(str(idx), 0) for idx in range(max_value)], dtype=np.float32)
                distr2 /= distr2.sum()

                # Calcola la distanza di Jensen-Shannon tra le due distribuzioni
                m = 0.5 * (distr1 + distr2)
                js_distance = 0.5 * (np.sum(distr1 * np.log(distr1 / m + 1e-8)) + np.sum(distr2 * np.log(distr2 / m + 1e-8)))

                self.data.append((dtmc1, dtmc2, js_distance))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dtmc1, dtmc2, js_distance = self.data[idx]
        return dtmc1, dtmc2, js_distance

# Architettura Siamese basata su Transformer
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(embed_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return x.mean(dim=1)  # Riduzione sulla dimensione delle righe

class SiameseModel(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2):
        super(SiameseModel, self).__init__()
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_layers)

    def forward_one(self, x):
        return self.encoder(x)

    def forward(self, x1_list, x2_list):
        # Elabora ogni sequenza separatamente
        out1 = [self.forward_one(x) for x in x1_list]
        out2 = [self.forward_one(x) for x in x2_list]
        # Converte in tensor per calcolare la loss
        out1 = torch.stack(out1)
        out2 = torch.stack(out2)
        return out1, out2

# Funzione per calcolare la distanza di Jensen-Shannon
class JSLoss(nn.Module):
    def __init__(self):
        super(JSLoss, self).__init__()

    def forward(self, output1, output2, target):
        m = 0.5 * (output1 + output2)
        js_distance = 0.5 * (torch.sum(output1 * torch.log(output1 / m + 1e-8), dim=1) +
                             torch.sum(output2 * torch.log(output2 / m + 1e-8), dim=1))
        loss = torch.mean((js_distance - target) ** 2)
        return loss

# Parametri
batch_size = 32
epochs = 20
learning_rate = 0.001
embed_dim = 64
num_heads = 4
num_layers = 2
dtmc_dir = "data/dtmcs"
labels_dir = "data/labels"

# Dataset e DataLoader
generator = torch.Generator(device='cuda')  # Crea un generatore su CUDA

def collate_fn(batch):
    dtmc1, dtmc2, target = zip(*batch)
    return list(dtmc1), list(dtmc2), torch.tensor(target, dtype=torch.float32, device='cuda')

dataset = SiameseDTMCDataset(dtmc_dir=dtmc_dir, labels_dir=labels_dir)
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    generator=generator  # Passa il generatore al DataLoader
)

# Modello, loss e ottimizzatore
model = SiameseModel(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
criterion = JSLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for dtmc1, dtmc2, target in data_loader:
        optimizer.zero_grad()
        output1, output2 = model(dtmc1, dtmc2)
        loss = criterion(output1, output2, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(data_loader):.4f}")

print("Training completato!")
