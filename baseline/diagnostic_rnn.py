import torch
import torch.nn as nn
import numpy as np

class DiagnosticRNN(nn.Module):

    def __init__(self, num_classes, batch_size=64, embedding_size=256, num_hidden=512):
        super(DiagnosticRNN, self).__init__()
        
        self.embedding = nn.Parameter(
            torch.empty((11, embedding_size), dtype=torch.float32)
        )

        # weights and biases
        self.lstm = nn.LSTM(embedding_size, num_hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(num_hidden, num_classes)
        self.softmax = nn.Softmax()
        

    def forward(self, x):
        emb = (
            torch.matmul(x, self.embedding)
            if self.training
            else self.embedding[x]
        )

        lstm_out, _ = self.lstm.forward(emb)
        fc_out = self.fc.forward(lstm_out).squeeze()
        return fc_out
