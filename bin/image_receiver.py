import torch
import torch.nn as nn
import numpy as np

import random

class ImageReceiver(nn.Module):

    def __init__(
        self,
        z_dim=11,
        hidden_dim=512,
        output_dim=30*30*3):
        super(ImageReceiver, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid())


    def forward(self, input):
        output = self.model.forward(input)
        return output
