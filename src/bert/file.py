import torch
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from enum import Enum

class BertVariant(str, Enum):
    BERTIMBAU    = "neuralmind/bert-base-portuguese-cased"
    MBERT        = "google-bert/bert-base-multilingual-cased"
    JABUTICABERT = "amadeusai/JabuticaBERT-Large"

class BertForTextEmbedding(nn.Module):

    def __init__(self, dropout=0.1, num_labels=2):
        super().__init__()
        self.bert: BertModel = BertModel.from_pretrained("google-bert/bert-base-multilingual-cased")

        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.linear(self.dropout(outputs.pooler_output))