import argparse
import sys
import torch
import os
import numpy as np

from torch.utils import data

from datasets.diagnostic_dataset import DiagnosticDataset
from enums.dataset_type import DatasetType
from helpers.metadata_helper import get_metadata_properties
from models.diagnostic_ensemble import DiagnosticEnsemble
from metrics.average_ensemble_meter import AverageEnsembleMeter

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

def print_results(accuracies_meter: AverageEnsembleMeter, losses_meter: AverageEnsembleMeter, epoch, dataset: str):
    print(f'epoch: {epoch} | {dataset} | avg: {round(accuracies_meter.avg, 4)} | acc: {round(accuracies_meter.averages[0], 4)} | {round(accuracies_meter.averages[1], 4)} | {round(accuracies_meter.averages[2], 4)} | {round(accuracies_meter.averages[3], 4)} | {round(accuracies_meter.averages[4], 4)}')
    print(f'epoch: {epoch} | {dataset} | avg: {round(losses_meter.avg, 4)} | loss: {round(losses_meter.averages[0], 4)} | {round(losses_meter.averages[1], 4)} | {round(losses_meter.averages[2], 4)} | {round(losses_meter.averages[3], 4)} | {round(losses_meter.averages[4], 4)}')

def perform_iteration(model: DiagnosticEnsemble, dataloader, batch_size: int, device):
    accuracies_meter = AverageEnsembleMeter(number_of_values=5)
    losses_meter = AverageEnsembleMeter(number_of_values=5)
    
    for messages, properties in dataloader:
        messages = messages.long().to(device)
        properties = properties.long().to(device)

        current_accuracies, current_losses = model.forward(messages, properties)
        
        accuracies_meter.update(current_accuracies)
        losses_meter.update(current_losses, crash=False)

    return accuracies_meter, losses_meter

def generate_unique_name(length, vocabulary_size, seed):
    result = f'max_len_{length}_vocab_{vocabulary_size}_seed_{seed}'
    return result

def baseline(args):
    args = parse_arguments(args)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unique_name = generate_unique_name(
        length=args.max_length,
        vocabulary_size=args.vocab_size,
        seed=args.seed)

    train_dataset = DiagnosticDataset(unique_name, DatasetType.Train)
    train_dataloader = data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    validation_dataset = DiagnosticDataset(unique_name, DatasetType.Valid)
    validation_dataloader = data.DataLoader(validation_dataset, shuffle=False, batch_size=args.batch_size)

    # test_dataset = DiagnosticDataset(unique_name, DatasetType.Test)
    # test_dataloader = data.DataLoader(test_dataset, shuffle=False)

    diagnostic_model = DiagnosticEnsemble(
        num_classes_by_model=[3, 3, 2, 3, 3],
        batch_size=args.batch_size,
        device=device,
        embedding_size=args.embedding_size,
        num_hidden=args.hidden_size,
        learning_rate=args.lr)

    diagnostic_model.to(device)

    # Setup the loss and optimizer

    for epoch in range(args.max_epochs):

        # TRAIN
        diagnostic_model.train()
        perform_iteration(diagnostic_model, train_dataloader, args.batch_size, device)

        # VALIDATION

        diagnostic_model.eval()
        validation_accuracies_meter, validation_losses_meter = perform_iteration(diagnostic_model, validation_dataloader, args.batch_size, device)
        print_results(validation_accuracies_meter, validation_losses_meter, epoch, "validation")



if __name__ == "__main__":
    baseline(sys.argv[1:])
