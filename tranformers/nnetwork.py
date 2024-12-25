import torch
from torch import nn
import pytorch_lightning as pl
from torch_geometric.nn import TransformerConv, global_mean_pool
import torch.nn.functional as F

class GraphTransformerModule(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_heads: int,
                 out_channels: int,
                 num_layers: int = 2,
                 learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters(in_channels, hidden_channels, num_heads, out_channels, num_layers, learning_rate)
        # Transformer layers
        self.layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerConv(
                    in_channels=in_channels if i == 0 else hidden_channels,
                    out_channels=hidden_channels, # sono il numero di channels per ogni head
                    heads=num_heads,   # Multi-head attention
                    dropout=0.1,
                    edge_dim=1  # Include edge features (probabilità di transizione),
                )
            )
            self.linear_layers.append(
                nn.Linear(hidden_channels * num_heads, hidden_channels)
            )

        # Output MLP for graph representation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

        # Learning rate
        self.learning_rate = learning_rate

    def forward(self, data):
        # `data` è un oggetto di PyTorch Geometric con:
        # - data.x: Node features (dimensione [num_nodes, in_channels])
        # - data.edge_index: Indici degli archi (dimensione [2, num_edges])
        # - data.edge_attr: Attributi degli archi (dimensione [num_edges, edge_dim])
        # - data.batch: Batch index per i nodi

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Passaggio attraverso i Transformer layers
        for layer, linear in zip(self.layers, self.linear_layers):
            x = layer(x, edge_index, edge_attr)
            x = linear(x)
            x = torch.relu(x)

        # Pooling globale per ottenere una rappresentazione del grafo
        graph_embedding = global_mean_pool(x, batch)

        # Passaggio attraverso il MLP finale
        return self.mlp(graph_embedding)

    def training_step(self, batch, batch_idx):
        # Assume un task di regressione per il mixing time
        pred = self(batch)
        loss = nn.MSELoss()(pred, batch.y)  # batch.y: ground truth
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = nn.MSELoss()(pred, batch.y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)


class SiameseDTMC(pl.LightningModule):
    def __init__(self, in_channels=2, hidden_channels=16, out_channels=8, num_heads=4, learning_rate=1e-4):
        super(SiameseDTMC, self).__init__()
        self.save_hyperparameters()
        self.encoder = GraphTransformerModule(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_heads=num_heads, learning_rate=learning_rate)
        self.learning_rate = learning_rate

    def forward(self, input1, input2):
        embedding1 = self.encoder(input1)
        embedding2 = self.encoder(input2)
        return embedding1, embedding2

    def training_step(self, batch, batch_idx):
        dtmc1, dtmc2, label_diff = batch
        embedding1, embedding2 = self(dtmc1, dtmc2)
        # embedding_diff = torch.sum(torch.abs(embedding1 - embedding2))
        # loss = embedding_diff - label_diff
        embedding_diff = embedding1 - embedding2
        loss = F.mse_loss(embedding_diff, label_diff)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)



# dtmc1, dtmc2, label_diff = tuple(zip(*batch))  # label_diff: differenza tra i valori calcolati delle etichette
# embedding1, embedding2 = self(dtmc1[0], dtmc2[0]) # la batch size deve essere 1