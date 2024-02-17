import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_classes, batch_size): # 1024, 8
        super(MLP, self).__init__()
        self.input_size = 1024
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, 512), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        logits = self.mlp(x)
        return logits, {"features": {}}


class MLP(nn.Module):
    def __init__(self, num_classes, batch_size): # 1024, 8
        super(MLP, self).__init__()
        self.input_size = 1024
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, 512), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        logits = self.mlp(x)
        return logits, {"features": {}}

class LSTM(nn.Module):
    def __init__(self, num_classes, batch_size): #* aggiusta i parametri, ad es. passa la batch come arg
        super(LSTM, self).__init__()
        self.input_size = 1024
        self.lstm_hidden_size = 512
        self.num_layers = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.lstm_hidden_size, self.num_layers, 
                            bias=True, batch_first=True, dropout=0.5, bidirectional=False, 
                            proj_size=0, device=None, dtype=None)
        self.fc = nn.Linear(self.lstm_hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states with the proper batch size
        # h0 and c0 shape = (num_layers, batch_size, lstm_hidden_size)=(1, 32, 32)
        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size).to(x.device)

        # aggiungo una dimensione 
        # we want x shape equal to (batch_size, sequence_length, input_size)=(32, 1, 1024)
        # prima: x.shape = (32, 1024)
        x = x.unsqueeze(1)
        # dopo: x.shape = (32, 1, 1024)

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0)) # _ = hn, cn (non ci servono al momento)
        # out: contains the output features (batch_size, sequence_length, lstm_hidden_size)=(32, 1, 512)
        # hn: final hidden state for each element in sequence, stessa size di h0
        # cn: final cell state for each element in sequence, stessa di c0

        # Reshape the output to be compatible with the fully connected layer
        feat = out.view(-1, self.lstm_hidden_size)
        # Pass through fully connected layer to get logits
        logits = self.fc(feat)
        #DEBUG
        #logger.info(f"######## => x.size(0): {x.size(0)} | bs: {self.batch_size} | x: {x} | x.shape: {x.shape} | l.shape: {logits.shape} | f.shape: {feat.shape} | logits: {logits} | feat: {feat}")
        
        return logits, {"features": feat} #(32, 8), (32, 1024)

class TransformerClassifier(nn.Module):
    def __init__(
        self, num_classes, input_size=1024, num_heads=8, lstm_hidden_size=1024, num_layers=6
    ):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Linear(input_size, lstm_hidden_size)
        encoder_layers = TransformerEncoderLayer(lstm_hidden_size, num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        # * Embedding: Linear transformation of input features
        x = self.embedding(x)

        # * Reshape input tensor for Transformer encoder
        # addind a dimension -> from (batch_size, lstm_hidden_size) to (batch_size, 1, lstm_hidden_size)
        x = x.unsqueeze(1)
        # resulting shape (1, batch_size, lstm_hidden_size) -> where 1 is the sequence length
        x = x.permute(1, 0, 2)

        # * Forward pass through transformer encoder
        transformer_output = self.transformer_encoder(x)

        # * Extract features and compute logits -> Taking output of the first token as features
        feat = transformer_output[0, :, :]

        # * Compute logits
        logits = self.fc(feat)  # Fully connected layer for logits

        return logits, {"features": feat}
    
class ActionNetwork(nn.Module):
    def __init__(self, num_classes, input_size, batch_size): #* aggiusta i parametri, ad es. passa la batch come arg
        super(ActionNetwork, self).__init__()
        self.input_size = 1600 # input_size
        self.lstm_hidden_size = 800
        self.lstm2_hidden_size = 50
        self.num_layers = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.lstm_hidden_size, self.num_layers, 
                            bias=True, batch_first=True, dropout=0, bidirectional=False, 
                            proj_size=0, device=None, dtype=None)
        self.lstm2 = nn.LSTM(self.lstm_hidden_size, self.lstm2_hidden_size, self.num_layers, 
                            bias=True, batch_first=True, dropout=0, bidirectional=False, 
                            proj_size=0, device=None, dtype=None)
        
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.fc1 = nn.Linear(50, num_classes)
        self.fc2 = nn.Linear(num_classes, num_classes)

    def forward_old(self, x):
        # x.shape = (32, 100, 16)

        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0)) # (32, 100, 5)

        h02 = torch.zeros(self.num_layers, out.size(0), self.lstm2_hidden_size).to(x.device)
        c02 = torch.zeros(self.num_layers, out.size(0), self.lstm2_hidden_size).to(x.device)

        out2, _ = self.lstm2(out, (h02, c02)) # (32, 100, 1)
        out2 = out2.squeeze(2)  # (32, 100)
        
        #out3 = self.dropout(out2) 

        #logits = self.fc2(torch.relu(self.fc1(out2)))  #(32, 20)
        logits = self.fc1(out2)  # (32, 20)

        #* SoftMax
        predicted_activity = torch.argmax(F.softmax(logits, dim=1), dim=1)
        #print(out.shape, out2.shape, out3.shape, logits.shape)

        return logits, {"features": out2}

    def forward(self, x):
        # x.shape = (32, 100, 16)
        x = x.reshape(32, 1600)  # (32, 100, 16) -> (32, 1600)
        x = x.unsqueeze(1) # (32, 1, 1600)

        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0)) # (32, 1, 800)

        h02 = torch.zeros(self.num_layers, out.size(0), self.lstm2_hidden_size).to(x.device)
        c02 = torch.zeros(self.num_layers, out.size(0), self.lstm2_hidden_size).to(x.device)

        out2, _ = self.lstm2(out, (h02, c02)) # (32, 1, 50)
        
        out3 = self.dropout(out2)  

        feat = out3.view(-1, self.lstm2_hidden_size) # (32, 50)
        logits = self.fc2(torch.relu(self.fc1(feat)))  # (32, 20)
        #logits = self.fc1(feat)  # (32, 20)

        #* SoftMax
        predicted_activity = torch.argmax(F.softmax(logits, dim=1), dim=1)
        #print(out.shape, out2.shape, out3.shape, logits.shape)

        return logits, {"features": feat}
    

class ActionNetwork_fusion(nn.Module):
    def __init__(self, num_classes, batch_size): #* aggiusta i parametri, ad es. passa la batch come arg
        super(ActionNetwork_fusion, self).__init__()
        self.input_size = 7200 # input_size
        self.lstm_hidden_size = 3000
        self.lstm2_hidden_size = 50
        self.num_layers = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.lstm_hidden_size, self.num_layers, 
                            bias=True, batch_first=True, dropout=0, bidirectional=False, 
                            proj_size=0, device=None, dtype=None)
        self.lstm2 = nn.LSTM(self.lstm_hidden_size, self.lstm2_hidden_size, self.num_layers, 
                            bias=True, batch_first=True, dropout=0, bidirectional=False, 
                            proj_size=0, device=None, dtype=None)
        
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.fc1 = nn.Linear(50, num_classes)
        self.fc2 = nn.Linear(num_classes, num_classes)

    def forward_old(self, x):
        # x.shape = (32, 100, 16)

        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0)) # (32, 100, 5)

        h02 = torch.zeros(self.num_layers, out.size(0), self.lstm2_hidden_size).to(x.device)
        c02 = torch.zeros(self.num_layers, out.size(0), self.lstm2_hidden_size).to(x.device)

        out2, _ = self.lstm2(out, (h02, c02)) # (32, 100, 1)
        out2 = out2.squeeze(2)  # (32, 100)
        
        #out3 = self.dropout(out2) 

        #logits = self.fc2(torch.relu(self.fc1(out2)))  #(32, 20)
        logits = self.fc1(out2)  # (32, 20)

        #* SoftMax
        predicted_activity = torch.argmax(F.softmax(logits, dim=1), dim=1)
        #print(out.shape, out2.shape, out3.shape, logits.shape)

        return logits, {"features": out2}

    def forward(self, x):
        # x.shape = (32, 450, 16)
        x = x.reshape(32, 7200)  # (32, 100, 16) -> (32, 1600)
        x = x.unsqueeze(1) # (32, 1, 1600)

        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0)) # (32, 1, 800)

        h02 = torch.zeros(self.num_layers, out.size(0), self.lstm2_hidden_size).to(x.device)
        c02 = torch.zeros(self.num_layers, out.size(0), self.lstm2_hidden_size).to(x.device)

        out2, _ = self.lstm2(out, (h02, c02)) # (32, 1, 50)
        
        out3 = self.dropout(out2)  

        feat = out3.view(-1, self.lstm2_hidden_size) # (32, 50)
        logits = self.fc2(torch.relu(self.fc1(feat)))  # (32, 20)
        #logits = self.fc1(feat)  # (32, 20)

        #* SoftMax
        predicted_activity = torch.argmax(F.softmax(logits, dim=1), dim=1)
        #print(out.shape, out2.shape, out3.shape, logits.shape)

        return logits, {"features": feat}