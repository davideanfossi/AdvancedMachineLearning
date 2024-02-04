import torch
from torch import nn

# For general data processing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt

# to convert to dataset datatype - the transformers library does not work well with pandas
from utils.logger import logger
from torch.utils.data import Dataset


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
            ),
            num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Change the shape for transformer input
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Aggregate the transformer output
        x = self.fc(x)
        return x.squeeze(1)
