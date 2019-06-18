import torch
import torch.nn as nn
import numpy as np

import random

class MessagesReceiver(nn.Module):

    def __init__(
        self,
        num_classes_by_model,
        device,
        vocab_size=25,
        batch_size=1024,
        embedding_size=64,
        num_hidden=64):
        super(MessagesReceiver, self).__init__()
        
        self.embedding = nn.Parameter(
            torch.empty((vocab_size, embedding_size), dtype=torch.float32)
        )

        self.lstm = nn.LSTM(embedding_size, num_hidden, num_layers=1, batch_first=True)

        fc_layers = []
        
        for num_classes in num_classes_by_model:
            fc_layer = nn.Linear(num_hidden, num_classes)
            fc_layers.append(fc_layer)

        self.criterion = nn.CrossEntropyLoss()
        self.fc_layers = nn.ModuleList(fc_layers)
        self.num_hidden = num_hidden
        self.device = device

        nn.init.normal_(self.embedding, 0.0, 0.1)


    def forward(self, messages, targets, sample_count: int = None):
        if not sample_count:
            sample_count = len(self.fc_layers)

        emb = torch.matmul(messages, self.embedding) if self.training else self.embedding[messages]

        _, (h, _) = self.lstm.forward(emb)
        h = h.squeeze()

        indices = [i for i in range(len(self.fc_layers))]
        out = [None for _ in range(len(self.fc_layers))]
        for _ in range(sample_count):
            # pick randomly one of the models to forward
            i = random.choice(indices)

            # remove the picked index so we don't repeat the same model next iteration of the same forward pass
            indices.remove(i) 

            out[i] = self.fc_layers[i].forward(h)

        return out
