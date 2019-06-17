import torch
import torch.nn as nn
import numpy as np

class DiagnosticRNN(nn.Module):

    def __init__(self, num_classes, device, vocab_size=25, batch_size=1024, embedding_size=64, num_hidden=64):
        super(DiagnosticRNN, self).__init__()
        
        # self.embedding =  nn.Embedding(vocab_size, embedding_size)
        self.embedding = nn.Parameter(
            torch.empty((vocab_size, embedding_size), dtype=torch.float32)
        )

        # weights and biases
        self.lstm = nn.LSTM(embedding_size, num_hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(num_hidden, num_classes)

        self.num_hidden = num_hidden
        self.device = device

        nn.init.normal_(self.embedding, 0.0, 0.1)


    def forward(self, messages):
        emb = (
            torch.matmul(messages, self.embedding)
            if self.training
            else self.embedding[messages]
        )
        # emb = self.embedding(messages)

        _out, (h,_c) = self.lstm.forward(emb)
        h = h.squeeze()

        fc_out = self.fc.forward(h)
        return fc_out, h
