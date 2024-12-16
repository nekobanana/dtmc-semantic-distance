import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from transformers import AutoModel, AutoTokenizer



class DTMCTransformer(nn.Module):
    def __init__(self, model_name="bert-base-uncased", embed_dim=1000):
        super(DTMCTransformer, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, embed_dim)

    def forward(self, x):
        input_ids = x['input_ids'].squeeze(0)
        # token_type_ids = x['token_type_ids'].squeeze(0)
        attention_mask = x['attention_mask'].squeeze(0)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Use [CLS] token representation
        return self.projection(pooled_output)


class SiameseDTMC(LightningModule):
    def __init__(self, model_name="bert-base-uncased", embed_dim=128, learning_rate=1e-4):
        super(SiameseDTMC, self).__init__()
        self.save_hyperparameters()
        self.encoder = DTMCTransformer(model_name, embed_dim)
        # self.loss = nn.MSELoss()  # Example loss function; modify as needed
        self.learning_rate = learning_rate

    def forward(self, input1, input2):
        embedding1 = self.encoder(input1)
        embedding2 = self.encoder(input2)
        return embedding1, embedding2

    def training_step(self, batch, batch_idx):
        dtmc1, dtmc2, label_diff = batch
        embedding1, embedding2 = self(dtmc1, dtmc2)
        embedding_diff = torch.sum(torch.abs(embedding1 - embedding2))
        loss = embedding_diff - label_diff
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)



# dtmc1, dtmc2, label_diff = tuple(zip(*batch))  # label_diff: differenza tra i valori calcolati delle etichette
# embedding1, embedding2 = self(dtmc1[0], dtmc2[0]) # la batch size deve essere 1