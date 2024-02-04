import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils.logger import logger


class TransformerClassifier(nn.Module):
    def __init__(
        self, num_classes, seq_length=1024, num_heads=8, hidden_size=512, num_layers=6
    ):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(
            seq_length, hidden_size
        )  # Use a Linear layer for embedding
        encoder_layers = TransformerEncoderLayer(hidden_size, num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, seq_length, hidden_size)
        x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)  # Permute to (seq_length, batch_size, hidden_size)

        # Forward pass through transformer encoder
        transformer_output = self.transformer_encoder(x)

        # Extract features and compute logits -> Taking output of the first token as features
        feat = transformer_output[0, :, :]
        logger.info(f"\nSTART TESTING...")
        logger.info(f"FEAT: {feat}")
        logger.info(f"FEAT SHAPE: {feat.shape}")

        # Compute logits
        logits = self.fc(feat)  # Fully connected layer for logits
        logger.info(f"LOGITS: {logits}")
        logger.info(f"LOGITS SHAPE: {logits.shape}")
        logger.info(f"END TESTING...\n")

        return logits, {"features": feat}
