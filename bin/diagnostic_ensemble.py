import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random

from models.diagnostic_rnn import DiagnosticRNN


class DiagnosticEnsemble(nn.Module):
    def __init__(
            self,
            num_classes_by_model,
            device,
            vocab_size=25,
            batch_size=1024,
            embedding_size=64,
            num_hidden=64,
            learning_rate=1e-3):
        super(DiagnosticEnsemble, self).__init__()

        models = []
        self.optimizers = []
        self.criterions = []

        for num_classes in num_classes_by_model:
            diagnostic_rnn = DiagnosticRNN(
                num_classes=num_classes,
                device=device,
                vocab_size=vocab_size,
                embedding_size=embedding_size,
                num_hidden=num_hidden)

            diagnostic_rnn.to(device)

            models.append(diagnostic_rnn)
            self.criterions.append(nn.CrossEntropyLoss())

            self.optimizers.append(optim.Adam(diagnostic_rnn.parameters(), lr=learning_rate))

        self.models = nn.ModuleList(models)

    def forward(self, messages, targets, sample_count: int = None):
        if not sample_count:
            sample_count = len(self.models)

        accuracies = np.zeros((len(self.models),))
        losses = np.zeros((len(self.models),))

        indices = [i for i in range(len(self.models))]
        for _ in range(sample_count):
            # pick randomly one of the models to forward
            i = random.choice(indices)

            # remove the picked index so we don't repeat the same model next iteration of the same forward pass
            indices.remove(i) 

            if self.training:
                self.optimizers[i].zero_grad()

            current_targets = targets[:, i]
            out, _ = self.models[i].forward(messages)

            loss = self.criterions[i].forward(out, current_targets)

            if self.training:
                loss.backward()
                self.optimizers[i].step()

            losses[i] = loss.item()
            accuracies[i] = torch.mean((torch.argmax(out, dim=1) == current_targets).float()).item()

        return accuracies, losses
