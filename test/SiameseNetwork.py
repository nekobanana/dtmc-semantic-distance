import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SiameseNetwork, self).__init__()

        # Shared network for encoding Markov matrices
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward_one(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        encoded_x1 = self.forward_one(x1)
        encoded_x2 = self.forward_one(x2)

        # Compute the distance between encoded representations
        return torch.norm(encoded_x1 - encoded_x2, p=2, dim=1)


# Loss function for the Siamese network
def contrastive_loss(distance, label, margin):
    # label: 1 if different, 0 if same
    loss_same = (1 - label) * torch.pow(distance, 2)
    loss_diff = label * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return torch.mean(loss_same + loss_diff)


# Example training process
def train_siamese_network(transition_matrices_1, transition_matrices_2, labels, input_size, hidden_size, epochs=50,
                          lr=0.001):
    # Create the network
    model = SiameseNetwork(input_size, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        # Convert data to tensors
        x1 = torch.tensor(transition_matrices_1, dtype=torch.float32)
        x2 = torch.tensor(transition_matrices_2, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)

        # Forward pass
        distances = model(x1, x2)

        # Compute loss
        loss = contrastive_loss(distances, y, margin=1.0)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return model


def pad_flattened_square_matrix(flat_matrix, target_size):
    # Calculate the size of the original square matrix
    original_size = int(len(flat_matrix) ** 0.5)
    if original_size ** 2 != len(flat_matrix):
        raise ValueError("Input list is not a valid flattened square matrix.")

    # Reshape the flattened matrix into 2D
    matrix_2d = np.array(flat_matrix).reshape((original_size, original_size))

    # Create the padded matrix
    padded_matrix = np.zeros((target_size, target_size))
    padded_matrix[:original_size, :original_size] = matrix_2d

    # Flatten the padded matrix back if needed
    return padded_matrix.flatten()



# Nome del file JSON che contiene i risultati
filename = "markov_chain_results.json"

# Leggi il file e carica i dati
with open(filename, 'r') as f:
    data = json.load(f)  # data sarà tipicamente una lista di record

complete_dataset = []

# Itera sui risultati per ottenere le coppie di matrici (flat) e le distanze
for idx, record in enumerate(data):
    m1 = record["m1"]  # matrice m1
    m2 = record["m2"]  # matrice m2
    distances = record["distances"]  # dizionario con tutte le distanze

    print(f"Coppia n.{idx + 1}:")
    print("Matrice m1 (flat):", m1)
    print("Matrice m2 (flat):", m2)
    print("Distanze:", distances)
    print("-" * 40)

    complete_dataset.append((m1,m2,distances))

random.shuffle(complete_dataset)

split_index = int(0.8 * len(complete_dataset))
train_set = complete_dataset[:split_index]
test_set = complete_dataset[split_index:]

m1_list = []
m2_list = []
distances_list = []
total_variation_distances = []
kl_divergence_distances = []
l2_distances = []
# wasserstein_distances = []
spectral_distances = []
max_dim = 50
for m1,m2,distances in train_set:
    m1_list.append(pad_flattened_square_matrix(m1,max_dim))
    m2_list.append(pad_flattened_square_matrix(m2,max_dim))
    distances_list.append(distances)
    total_variation_distances.append(distances["total_variation"])
    kl_divergence_distances.append(distances["kl_divergence"])
    l2_distances.append(distances["l2_distance"])
    # wasserstein_distances.append(distances["wasserstein"])
    spectral_distances.append(distances["spectral"])

m1_list_test = []
m2_list_test = []
distances_list_test = []
total_variation_distances_test = []
kl_divergence_distances_test = []
l2_distances_test = []
# wasserstein_distances_test = []
spectral_distances_test = []
max_dim = 50
for m1,m2,distances in test_set:
    m1_list_test.append(pad_flattened_square_matrix(m1,max_dim))
    m2_list_test.append(pad_flattened_square_matrix(m2,max_dim))
    distances_list_test.append(distances)
    total_variation_distances_test.append(distances["total_variation"])
    kl_divergence_distances_test.append(distances["kl_divergence"])
    l2_distances_test.append(distances["l2_distance"])
    # wasserstein_distances_test.append(distances["wasserstein"])
    spectral_distances_test.append(distances["spectral"])

labels = spectral_distances
input_size = 2500
hidden_size = 4500
train_model = True
if train_model:
    trained_model = train_siamese_network(m1_list, m2_list, labels, input_size, hidden_size)
    torch.save(trained_model.state_dict(), "model.pth")
else:
    trained_model = SiameseNetwork(input_size, hidden_size)
    trained_model.load_state_dict(torch.load("model.pth"))
    trained_model.eval()

error = 0.0
for i in range(len(m1_list_test)):
    with torch.no_grad():
        test_matrix_1 = torch.tensor(m1_list_test[i], dtype=torch.float32).unsqueeze(0)
        test_matrix_2 = torch.tensor(m2_list_test[i], dtype=torch.float32).unsqueeze(0)
        distance = trained_model(test_matrix_1, test_matrix_2)
        iteration_error = abs(spectral_distances_test[i] - distance.item())
        error = error + iteration_error
        print(f"Distance between couple : {i} " + f", d= {distance.item():.4f}" + f"  D= {spectral_distances_test[i]:.4f}" + f"  error = {iteration_error:.4f}")

error = error / len(m1_list_test)
print(f"Global error is : {error:.4f}")