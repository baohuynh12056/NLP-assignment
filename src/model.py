import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class RDModel(nn.Module):
    def __init__(self, model_name):
        super(RDModel, self).__init__()

        # Load BERT backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # Projection Head
        self.projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 256),  # Project down to 256 dimensions
        )

    def forward(self, input_ids, attention_mask):
        # Get outputs from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_token = outputs.last_hidden_state[:, 0, :]

        # Projecting
        embedding = self.projection(cls_token)

        # Normalizing
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    # For indexing
    def encode_word(self, input_ids, attetion_mask):
        return self.forward(input_ids, attetion_mask)

    # For inference
    def encode_text(self, input_ids, attetion_mask):
        return self.forward(input_ids, attetion_mask)