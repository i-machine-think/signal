import torch
import torch.nn as nn


class ShapesCNN(nn.Module):
    def __init__(self, n_out_features):
        super().__init__()

        n_filters = 20

        self.conv_net = nn.Sequential(
            nn.Conv2d(3, n_filters, 3, stride=2),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, 3, stride=2),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, 3, stride=2),
            nn.BatchNorm2d(n_filters),
            nn.ReLU()
        )
        self.lin = nn.Sequential(nn.Linear(80, n_out_features), nn.ReLU())

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        output = self.conv_net(x)
        output = output.view(batch_size, -1)
        output = self.lin(output)
        return output
