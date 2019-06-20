import argparse
import sys
import torch
import os
import numpy as np
import time

from torch.utils import data

from datasets.diagnostic_dataset import DiagnosticDataset
from enums.dataset_type import DatasetType
from helpers.metadata_helper import get_metadata_properties
from models.diagnostic_ensemble import DiagnosticEnsemble
from metrics.average_ensemble_meter import AverageEnsembleMeter

import matplotlib.pyplot as plt

header = '  Time Epoch Iteration    Progress (%Epoch) | Loss-Avg  Acc-Avg | Loss-Color Loss-Shape Loss-Size Loss-PosH Loss-PosW | Acc-Color Acc-Shape Acc-Size Acc-PosH Acc-PosW |    Dataset'
log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,| {:>8.6f} {:>7.6f} | {:>10.6f} {:>10.6f} {:>9.6f} {:>9.6f} {:>9.6f} | {:>9.6f} {:>9.6f} {:>8.6f} {:>8.6f} {:>8.6f} | {:>10s}'.split(','))

start_time = time.time()

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
        "--seed", type=int, default=7, metavar="S", help="random seed (default: 42)"
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
    
    parser.add_argument(
        "--training-sample-count",
        type=int,
        default=None,
        help="How many models should be sampled while training. Default is None, meaning that all models will be used")
    parser.add_argument(
        "--inference",
        action="store_true",
        help="If sender model trained using inference step is used"
    )
    parser.add_argument(
        "--step3",
        help="If sender model trained using step3 is used",
        action="store_true"
    )

    args = parser.parse_args(args)

    return args

def print_results(accuracies_meter: AverageEnsembleMeter, losses_meter: AverageEnsembleMeter, epoch, max_epochs, dataset: str):
    print(log_template.format(
        time.time()-start_time,
        epoch,
        epoch,
        1 + epoch,
        max_epochs,
        100. * (1+epoch) / max_epochs,
        losses_meter.avg,
        accuracies_meter.avg,
        losses_meter.averages[0],
        losses_meter.averages[1],
        losses_meter.averages[2],
        losses_meter.averages[3],
        losses_meter.averages[4],
        accuracies_meter.averages[0],
        accuracies_meter.averages[1],
        accuracies_meter.averages[2],
        accuracies_meter.averages[3],
        accuracies_meter.averages[4],
        dataset
    ))

def perform_iteration(model: DiagnosticEnsemble, dataloader, batch_size: int, device, sample_count=5):
    accuracies_meter = AverageEnsembleMeter(number_of_values=5)
    losses_meter = AverageEnsembleMeter(number_of_values=5)
    
    for messages, properties in dataloader:
        messages = messages.long().to(device)
        properties = properties.long().to(device)

        current_accuracies, current_losses = model.forward(messages, properties, sample_count)
        
        accuracies_meter.update(current_accuracies)
        losses_meter.update(current_losses, crash=False)

    return accuracies_meter, losses_meter

def generate_unique_name(length, vocabulary_size, seed, inference, step3):
    result = f'max_len_{length}_vocab_{vocabulary_size}_seed_{seed}'
    if inference:
        result += '_inference'
    
    if step3:
        result += '_step3'

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
        seed=args.seed,
        inference=args.inference,
        step3=args.step3)

    train_dataset = DiagnosticDataset(unique_name, DatasetType.Train)
    train_dataloader = data.DataLoader(train_dataset, num_workers=1, pin_memory=True, shuffle=True, batch_size=args.batch_size)

    validation_dataset = DiagnosticDataset(unique_name, DatasetType.Valid)
    validation_dataloader = data.DataLoader(validation_dataset, num_workers=1, pin_memory=True, shuffle=False, batch_size=args.batch_size)

    test_dataset = DiagnosticDataset(unique_name, DatasetType.Test)
    test_dataloader = data.DataLoader(test_dataset, num_workers=1, pin_memory=True, shuffle=False, batch_size=args.batch_size)

    diagnostic_model = DiagnosticEnsemble(
        num_classes_by_model=[3, 3, 2, 3, 3],
        batch_size=args.batch_size,
        device=device,
        embedding_size=args.embedding_size,
        num_hidden=args.hidden_size,
        learning_rate=args.lr)

    diagnostic_model.to(device)

    # Setup the loss and optimizer

    print(header)

    for epoch in range(args.max_epochs):

        # TRAIN
        diagnostic_model.train()
        perform_iteration(diagnostic_model, train_dataloader, args.batch_size, device, sample_count=args.training_sample_count)

        # VALIDATION

        diagnostic_model.eval()
        validation_accuracies_meter, validation_losses_meter = perform_iteration(diagnostic_model, validation_dataloader, args.batch_size, device)
        print_results(validation_accuracies_meter, validation_losses_meter, epoch, args.max_epochs, "validation")

    diagnostic_model.eval()
    test_accuracies_meter, test_losses_meter = perform_iteration(diagnostic_model, test_dataloader, args.batch_size, device)
    print_results(test_accuracies_meter, test_losses_meter, epoch, args.max_epochs, "test")

if __name__ == "__main__":
    baseline(sys.argv[1:])
