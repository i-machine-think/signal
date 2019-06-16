import pickle
import argparse
import sys
import torch
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from helpers.game_helper import get_sender_receiver, get_trainer, get_training_data
from helpers.train_helper import TrainHelper
from helpers.file_helper import FileHelper
from helpers.metrics_helper import MetricsHelper
from helpers.metadata_helper import get_metadata_properties, get_shapes_metadata

from enums.dataset_type import DatasetType

from models.shapes_receiver import ShapesReceiver

from diagnostic_rnn import DiagnosticRNN


def parse_arguments(args):
	# Training settings
	parser = argparse.ArgumentParser(
		description="Training Sender/Receiver Agent on a task"
	)
	parser.add_argument(
		"--debugging",
		help="Enable debugging mode (default: False)",
		action="store_true",
	)
	parser.add_argument(
		"--single-model",
		help="Use a single model (default: False)",
		action="store_true",
	)
	parser.add_argument(
		"--dataset-type",
		type=str,
		default="meta",
		metavar="S",
		help="type of input used by dataset pick from raw/features/meta (default features)",
	)
	parser.add_argument(
		"--greedy",
		help="Use argmax at prediction time instead of sampling (default: False)",
		action="store_true",
	)
	parser.add_argument(
		"--iterations",
		type=int,
		default=10000,
		metavar="N",
		help="number of batch iterations to train (default: 10k)",
	)
	parser.add_argument(
		"--log-interval",
		type=int,
		default=200,
		metavar="N",
		help="number of iterations between logs (default: 200)",
	)
	parser.add_argument(
		"--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
	)
	parser.add_argument(
		"--embedding-size",
		type=int,
		default=64,
		metavar="N",
		help="embedding size for embedding layer (default: 64)",
	)
	parser.add_argument(
		"--hidden-size",
		type=int,
		default=64,
		metavar="N",
		help="hidden size for hidden layer (default: 64)",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=1024,
		metavar="N",
		help="input batch size for training (default: 1024)",
	)
	parser.add_argument(
		"--max-length",
		type=int,
		default=10,
		metavar="N",
		help="max sentence length allowed for communication (default: 10)",
	)
	parser.add_argument(
		"--k",
		type=int,
		default=3,
		metavar="N",
		help="Number of distractors (default: 3)",
	)
	parser.add_argument(
		"--vocab-size",
		type=int,
		default=25,
		metavar="N",
		help="Size of vocabulary (default: 25)",
	)
	parser.add_argument(
		"--darts",
		help="Use random architecture from DARTS space instead of random LSTMCell (default: False)",
		action="store_true",
		default=False,
	)
	parser.add_argument(
		"--num-nodes",
		type=int,
		default=4,
		metavar="N",
		help="Size of darts cell to use with random-darts (default: 4)",
	)
	parser.add_argument(
		"--lr",
		type=float,
		default=1e-3,
		metavar="N",
		help="Adam learning rate (default: 1e-3)",
	)
	parser.add_argument(
		"--sender-path",
		type=str,
		default=False,
		metavar="S",
		help="Sender to be loaded",
	)
	parser.add_argument(
		"--receiver-path",
		type=str,
		default=False,
		metavar="S",
		help="Receiver to be loaded",
	)
	parser.add_argument(
		"--freeze-sender",
		help="Freeze sender weights (do not train) ",
		action="store_true",
	)
	parser.add_argument(
		"--freeze-receiver",
		help="Freeze receiver weights (do not train) ",
		action="store_true",
	)
	parser.add_argument(
		"--obverter-setup",
		help="Enable obverter setup with shapes",
		action="store_true",
	)
	parser.add_argument(
		"--name",
		type=str,
		default=False,
		metavar="S",
		help="Name to append to run file name",
	)
	parser.add_argument(
		"--folder",
		type=str,
		default=False,
		metavar="S",
		help="Additional folder within runs/",
	)
	parser.add_argument("--disable-print",
						help="Disable printing", action="store_true")

	parser.add_argument(
		"--device",
		type=str,
		help="Device to be used. Pick from none/cpu/cuda. If default none is used automatic check will be done")
	parser.add_argument(
		"--patience",
		type=int,
		default=5,
		help="Amount of epochs to check for not improved validation score before early stopping",
	)

	args = parser.parse_args(args)

	return args


def baseline(args):
	args = parse_arguments(args)

	if args.device:
		device = torch.device(args.device)
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# get data from files
	train_msg = get_message_file('train')
	train_indices = get_indices_file('train')
	train_props = get_metadata_properties(DatasetType.Train)[train_indices]
	# valid_msg = get_message_file('validation')
	# valid_props = get_metadata_properties(DatasetType.Valid)[:valid_msg.shape[0],:]
	# test_msg = get_message_file('test')
	# test_props = get_metadata_properties(DatasetType.Test)[:test_msg.shape[0],:]

	batch_size = 1024

	models = []
	for i in range(5):
		model = DiagnosticRNN(
			2 if i == 2 else 3,
			batch_size=batch_size,
			device=device)

		model.to(device)
		models.append(model)

	# Setup the loss and optimizer
	criterions = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss(),
				  nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

	optimizers = [optim.SGD(models[0].parameters(), lr=0.001), optim.SGD(models[1].parameters(), lr=0.001), optim.SGD(models[2].parameters(), lr=0.001),
				  optim.SGD(models[3].parameters(), lr=0.001), optim.SGD(models[4].parameters(), lr=0.001)]

	for epoch in range(50):
		accuracies = [[], [], [], [], []]
		losses = [[], [], [], [], []]

		for i in range(0, train_msg.shape[0], batch_size):
		# for i in range(0, 32, batch_size):
			print(f'epoch: {epoch} | step: {i}		\r', end='')
			msg_batch = torch.tensor(
				train_msg[i:i+batch_size, :], device=device, dtype=torch.float32)
			
			for img_property in range(5):
				# get data in batches
				props_batch = torch.tensor(train_props[i:i+batch_size, img_property], device=device, dtype=torch.int64)

				out = models[img_property].forward(msg_batch)

				loss = criterions[img_property].forward(out, props_batch)
				loss.backward()

				losses[img_property].append(loss.item())

				optimizers[img_property].step()

				# optimizers[img_property].zero_grad()

				# accuracy
				acc = torch.mean(
					(torch.argmax(out, dim=1) == props_batch).float())
				
				accuracies[img_property].append(acc.item())
			if i % 1280 == 0:
				print(f'acc: {round(np.mean(accuracies[0]), 5)} | {round(np.mean(accuracies[1]), 4)} | {round(np.mean(accuracies[2]), 4)} | {round(np.mean(accuracies[3]), 4)} | {round(np.mean(accuracies[4]), 4)}')
				print(f'loss: {round(np.mean(losses[0]), 4)} | {round(np.mean(losses[1]), 4)} | {round(np.mean(losses[2]), 4)} | {round(np.mean(losses[3]), 4)} | {round(np.mean(losses[4]), 4)}')
				print("------------------")


def get_message_file(datatype, length=10, vocab=25, seed=7):
	path = 'data/messages/'
	train_msg = f'max_len_{length}_vocab_{vocab}_seed_{seed}.{datatype}.messages.npy'
	msg_file = np.load(os.path.join(path, train_msg))
	return msg_file


def get_indices_file(datatype, length=10, vocab=25, seed=7):
	path = 'data/messages/'
	train_msg = f'max_len_{length}_vocab_{vocab}_seed_{seed}.{datatype}.indices.npy'
	msg_file = np.load(os.path.join(path, train_msg))
	return msg_file

# def get_metadata_file(datatype):
# 	path = 'data/features/'
# 	meta_file = pickle.load(open(path+f'{datatype}.metadata.p','rb'))

# 	return meta_file


if __name__ == "__main__":
	baseline(sys.argv[1:])
