import torch
import torch.nn as nn
import numpy as np

class DiagnosticRNN(nn.Module):

	def __init__(self, num_classes, device, vocab_size=25, batch_size=1024, embedding_size=32, num_hidden=64):
		super(DiagnosticRNN, self).__init__()
		
		self.embedding =  nn.Embedding(vocab_size, embedding_size)

		# weights and biases
		# self.lstm_cell = nn.LSTMCell(embedding_size, num_hidden)
		self.lstm = nn.LSTM(embedding_size, num_hidden, num_layers=1, batch_first=True)
		self.fc = nn.Linear(num_hidden, num_classes)

		self.num_hidden = num_hidden
		self.device = device

		self.initialize_parameters()
		
	def initialize_parameters(self):
		# nn.init.normal_(self.embedding, 0.0, 0.1)
		# nn.init.xavier_uniform_(self.lstm_cell.weight_ih)
		# nn.init.orthogonal_(self.lstm_cell.weight_hh)
		# nn.init.constant_(self.lstm_cell.bias_ih, val=0)
		# nn.init.constant_(self.lstm_cell.bias_hh, val=0)
		# nn.init.constant_(
		# 	self.lstm_cell.bias_hh[self.num_hidden : 2 * self.num_hidden], val=1
		# )
		a = 0

	def forward(self, messages):
		batch_size = messages.shape[0]
		emb = self.embedding(messages.long())
		# emb = messages

		# emb_concat = emb.view(batch_size, 1, -1)

		# initialize hidden
		# h = torch.zeros([batch_size, self.num_hidden], device=self.device)
		# c = torch.zeros([batch_size, self.num_hidden], device=self.device)
		# h = (h, c)

		# # make sequence_length be first dim
		# seq_iterator = emb.transpose(0, 1)
		# for w in seq_iterator:
		# 	h = self.lstm_cell(w, h)

		# lstm_out, _ = h
		_out, (h,_c) = self.lstm.forward(emb)
		h = h.squeeze()
		# lstm_out, _ = self.lstm.forward(emb)
		# lstm_out = lstm_out[:, -1, :]

		fc_out = self.fc.forward(h)
		return fc_out, h
