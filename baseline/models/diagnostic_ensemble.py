import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

        self.models = []
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

            self.models.append(diagnostic_rnn)
            self.criterions.append(nn.CrossEntropyLoss())
            self.optimizers.append(optim.Adam(diagnostic_rnn.parameters(), lr=learning_rate))


    def forward(self, messages, targets):
        accuracies = np.zeros((len(self.models),))
        losses = np.zeros((len(self.models),))

        # for i, model in enumerate(self.models):
        i = 3
        model = self.models[i]
        current_targets = targets[:, i]
        out, _ = model.forward(messages)

        loss = self.criterions[i].forward(out, current_targets)
        
        if self.training:
            loss.backward()
            self.optimizers[i].step()
            self.optimizers[i].zero_grad()

        losses[i] = loss.item()
        accuracies[i] = torch.mean((torch.argmax(out, dim=1) == current_targets).float()).item()

        return accuracies, losses

    def eval(self):
        super().eval()

        for model in self.models:
            model.eval()

    def train(self, mode=True):
        super().train(mode)

        for model in self.models:
            model.train(mode)


    def __str__(self):
        result = 'DiagnosticEnsemble(\n'
        for model in self.models:
            result = f'{result}{model}\n'

        result = f'{result})'
        return result
