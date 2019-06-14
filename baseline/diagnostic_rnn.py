################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

################################################################################

class DiagnosticRNN(nn.Module):

    def __init__(self, seq_length, num_classes, batch_size=64, input_dim=1, num_hidden=64, device='cpu'):
        super(DiagnosticRNN, self).__init__()
        # Initialization here ...
        # weights and biases
        self.Whx = torch.nn.Parameter(torch.randn(input_dim,num_hidden) / 1000)
        self.Whh = torch.nn.Parameter(torch.randn(num_hidden,num_hidden) / 1000)
        self.Wph = torch.nn.Parameter(torch.randn(num_hidden, num_classes) / 1000)
        self.bh = torch.nn.Parameter(torch.zeros(1, num_hidden))
        self.bp = torch.nn.Parameter(torch.zeros(1, num_classes))

        # number of timesteps
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_hidden = num_hidden

    def forward(self, x):
        # Implementation here ...
        h = torch.zeros(self.batch_size, self.num_hidden)
        for t in range(self.seq_length):

        	W1 = x[:,t].unsqueeze(dim=1) @ self.Whx
        	W2 = h @ self.Whh

        	h = torch.tanh(W1 + W2 + self.bh)

        p = h @ self.Wph + self.bp

        # print(x.shape, h.shape, p.shape)

        return p
