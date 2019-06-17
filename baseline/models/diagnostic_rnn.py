import torch
import torch.nn as nn
import numpy as np

class DiagnosticRNN(nn.Module):

	def __init__(self, num_classes, device, vocab_size=25, batch_size=1024, embedding_size=32, num_hidden=64):
		super(DiagnosticRNN, self).__init__()
		
		self.embedding =  nn.Embedding(vocab_size, embedding_size)

		# weights and biases
		self.lstm = nn.LSTM(embedding_size, num_hidden, num_layers=1, batch_first=True)
		self.fc = nn.Linear(num_hidden, num_classes)

		self.num_hidden = num_hidden
		self.device = device

	def forward(self, messages):
		emb = self.embedding(messages)

		_out, (h,_c) = self.lstm.forward(emb)
		h = h.squeeze()

		fc_out = self.fc.forward(h)
		return fc_out, h
