import argparse
import sys
import torch
import os
import numpy as np

from enums.dataset_type import DatasetType
from helpers.metadata_helper import get_metadata_properties
from models.diagnostic_ensemble import DiagnosticEnsemble

def parse_arguments(args):
    # Training settings
    parser = argparse.ArgumentParser(
        description="Training Sender/Receiver Agent on a task"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1000,
        metavar="N",
        help="number of batch epochs to train (default: 1k)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=32,
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
        "--vocab-size",
        type=int,
        default=25,
        metavar="N",
        help="Size of vocabulary (default: 25)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="N",
        help="Adam learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to be used. Pick from none/cpu/cuda. If default none is used automatic check will be done")

    args = parser.parse_args(args)

    return args

def print_results(accuracies, losses, epoch, dataset: str):
    mean_accuracies = np.mean(accuracies, axis=0)
    mean_losses = np.mean(losses, axis=0)

    print(f'epoch: {epoch} | {dataset} acc: {round(mean_accuracies[0], 5)} | {round(mean_accuracies[1], 4)} | {round(mean_accuracies[2], 4)} | {round(mean_accuracies[3], 4)} | {round(mean_accuracies[4], 4)}')
    print(f'epoch: {epoch} | {dataset} loss: {round(mean_losses[0], 4)} | {round(mean_losses[1], 4)} | {round(mean_losses[2], 4)} | {round(mean_losses[3], 4)} | {round(mean_losses[4], 4)}')

def perform_iteration(model, messages, properties, batch_size, device):
    accuracies = np.empty((5,))
    losses = np.empty((5,))
    
    for i in range(0, messages.shape[0], batch_size):
        messages_batch = torch.tensor(messages[i:i+batch_size, :], device=device, dtype=torch.int64)
        properties_batch = torch.tensor(properties[i:i+batch_size, :], device=device, dtype=torch.int64)
        
        current_accuracies, current_losses = model.forward(messages_batch, properties_batch)

        accuracies = np.vstack((accuracies, current_accuracies))
        losses = np.vstack((losses, current_losses))

    return accuracies, losses

def baseline(args):
    args = parse_arguments(args)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get data from files
    train_messages = get_message_file('train')
    train_indices = get_indices_file('train')
    train_properties = get_metadata_properties(DatasetType.Train)[train_indices]
    validation_messages = get_message_file('validation')
    validation_properties = get_metadata_properties(DatasetType.Valid)
    # test_messages = get_message_file('test')
    # test_properties = get_metadata_properties(DatasetType.Test)[:test_messages.shape[0],:]

    model = DiagnosticEnsemble(
        num_classes_by_model=[3, 3, 2, 3, 3],
        batch_size=args.batch_size,
        device=device,
        embedding_size=args.embedding_size,
        num_hidden=args.hidden_size,
        learning_rate=args.lr)

    model.to(device)

    # Setup the loss and optimizer

    for epoch in range(args.max_epochs):

        # TRAIN
        model.train()
        perform_iteration(model, train_messages, train_properties, args.batch_size, device)

        # VALIDATION

        model.eval()
        accuracies, losses = perform_iteration(model, validation_messages, validation_properties, args.batch_size, device)
        print_results(accuracies, losses, epoch, "validation")


def get_message_file(datatype, length=10, vocab=25, seed=7):
    path = 'data/messages/'
    messages_filename = f'max_len_{length}_vocab_{vocab}_seed_{seed}.{datatype}.messages.npy'
    messages_data = np.load(os.path.join(path, messages_filename))
    return messages_data


def get_indices_file(datatype, length=10, vocab=25, seed=7):
    path = 'data/messages/'
    indices_filename = f'max_len_{length}_vocab_{vocab}_seed_{seed}.{datatype}.indices.npy'
    indices_data = np.load(os.path.join(path, indices_filename))
    return indices_data

if __name__ == "__main__":
    baseline(sys.argv[1:])
