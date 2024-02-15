import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils.logger import logger

class LSTM(nn.Module):
    def __init__(self, num_classes, batch_size): #* aggiusta i parametri, ad es. passa la batch come arg
        super(LSTM, self).__init__()
        self.input_size = 1024
        self.hidden_size = 512
        self.num_layers = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, 
                            bias=True, batch_first=True, dropout=0.5, bidirectional=False, 
                            proj_size=0, device=None, dtype=None)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states with the proper batch size
        # h0 and c0 shape = (num_layers, batch_size, hidden_size)=(1, 32, 32)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # aggiungo una dimensione 
        # we want x shape equal to (batch_size, sequence_length, input_size)=(32, 1, 1024)
        # prima: x.shape = (32, 1024)
        x = x.unsqueeze(1)
        # dopo: x.shape = (32, 1, 1024)

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0)) # _ = hn, cn (non ci servono al momento)
        # out: contains the output features (batch_size, sequence_length, hidden_size)=(32, 1, 32)
        # hn: final hidden state for each element in sequence, stessa size di h0
        # cn: final cell state for each element in sequence, stessa di c0

        # Reshape the output to be compatible with the fully connected layer
        feat = out.view(-1, self.hidden_size)
        # Pass through fully connected layer to get logits
        logits = self.fc(feat)
        #DEBUG
        #logger.info(f"######## => x.size(0): {x.size(0)} | bs: {self.batch_size} | x: {x} | x.shape: {x.shape} | l.shape: {logits.shape} | f.shape: {feat.shape} | logits: {logits} | feat: {feat}")
        
        return logits, {"features": feat} #(32, 8), (32, 1024)

class TransformerClassifier(nn.Module):
    def __init__(
        self, num_classes, input_size=1024, num_heads=8, hidden_size=1024, num_layers=6
    ):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layers = TransformerEncoderLayer(hidden_size, num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # * Embedding: Linear transformation of input features
        x = self.embedding(x)

        # * Reshape input tensor for Transformer encoder
        # addind a dimension -> from (batch_size, hidden_size) to (batch_size, 1, hidden_size)
        x = x.unsqueeze(1)
        # resulting shape (1, batch_size, hidden_size) -> where 1 is the sequence length
        x = x.permute(1, 0, 2)

        # * Forward pass through transformer encoder
        transformer_output = self.transformer_encoder(x)

        # * Extract features and compute logits -> Taking output of the first token as features
        feat = transformer_output[0, :, :]

        # * Compute logits
        logits = self.fc(feat)  # Fully connected layer for logits

        return logits, {"features": feat}
    
class ActionNetwork(nn.Module):
    def __init__(self, num_classes, input_size=16, batch_size=1): #* aggiusta i parametri, ad es. passa la batch come arg
        super(ActionNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = 5
        self.num_layers = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, 
                            bias=True, batch_first=True, dropout=0, bidirectional=False, 
                            proj_size=0, device=None, dtype=None)
        self.lstm2 = nn.LSTM(5, 50, self.num_layers, 
                            bias=True, batch_first=True, dropout=0.2, bidirectional=False, 
                            proj_size=0, device=None, dtype=None)
        # self.fc2 = nn.Linear(50, 50)
        # self.fc = nn.Linear(self.hidden_size, num_classes)
        # print(f"2)  {input_size}")

    def forward(self, x):
        #* x.shape = (1, 100, 16)
        # x = torch.tensor(x, dtype=torch.int32)

        # print(f"IIIII: {x}, {x.shape}")

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0)) #? Lui vuole 100x5 come output

        h02 = torch.zeros(self.num_layers, out.size(0), 50).to(x.device)
        c02 = torch.zeros(self.num_layers, out.size(0), 50).to(x.device)

        # out2, _ = self.lstm2(out, (h02, c02))

        # Ridimensiona l'output per adattarlo al layer fully connected
        # feat = out[:, -1, :]  # Prendi solo l'output dell'ultimo timestep
        logits = self.fc(out)  # Passa l'output attraverso il layer fully connected

        # print(logits.shape, out.shape, out2.shape)

        return logits, {"features": out}    #* (1, 100, 20) - (1, 100, 5)
        #* logits contiene gli score relativi alle 20 label