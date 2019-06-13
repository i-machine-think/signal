import pickle
import argparse
import sys
import torch
import os

import numpy as np

from helpers.game_helper import get_sender_receiver, get_trainer, get_training_data
from helpers.train_helper import TrainHelper
from helpers.file_helper import FileHelper
from helpers.metrics_helper import MetricsHelper


def parse_arguments(args):
    # Training settings
    parser = argparse.ArgumentParser(
        description="Training Sender/Receiver Agent on a task"
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--sender-path",
        type=str,
        required=True,
        metavar="S",
        help="Sender to be loaded",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        metavar="S",
        help="Sender to be loaded",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to be used. Pick from none/cpu/cuda. If default none is used automatic check will be done")
        
    parser.add_argument(
        "--max-length",
        type=int,
        default=10,
        metavar="N",
        help="max sentence length allowed for communication (default: 5)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=25,
        metavar="N",
        help="Size of vocabulary (default: 5)",
    )

    args = parser.parse_args(args)
    return args


def generate_messages_filename(max_length, vocab_size, seed, set_type):
    """
    Generates a filename from baseline params (see baseline.py)
    """
    name = f'max_len_{max_length}_vocab_{vocab_size}_seed_{seed}.{set_type}.messages.npy'
    return name


def baseline(args):

    args = parse_arguments(args)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_helper = TrainHelper(device)
    train_helper.seed_torch(seed=args.seed)

    # get sender and receiver models and save them
    # sender, receiver = get_sender_receiver(device, args)
    sender = torch.load(args.sender_path)

    model = get_trainer(sender, None, device, "raw")

    train_data, validation_data, test_data, _, _ = get_training_data(
        device=device,
        batch_size=1024,
        k=3,
        debugging=False,
        dataset_type="raw")

    model.to(device)

    train_messages = []
    validation_messages = []
    test_messages = []

    model.eval()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    for i, train_batch in enumerate(train_data):
        print(f'Train: {i}/{len(train_data)}       \r', end='')
        target, distractors = train_batch
        current_messages = model(target, distractors)
        train_messages.extend(current_messages.cpu().tolist())

    train_messages_filename = generate_messages_filename(args.max_length, args.vocab_size, args.seed, "train")
    train_messages_filepath = os.path.join(args.output_path,  train_messages_filename)
    np.save(train_messages_filepath, np.array(train_messages))

    for i, validation_batch in enumerate(validation_data):
        print(f'Validation: {i}/{len(validation_data)}       \r', end='')
        target, distractors = validation_batch
        current_messages = model(target, distractors)
        validation_messages.extend(current_messages.cpu().tolist())

    validation_messages_filename = generate_messages_filename(args.max_length, args.vocab_size, args.seed, "validation")
    validation_messages_filepath = os.path.join(
        args.output_path, validation_messages_filename)
    np.save(validation_messages_filepath, np.array(validation_messages))

    for i, test_batch in enumerate(test_data):
        print(f'Test: {i}/{len(test_data)}       \r', end='')
        target, distractors = test_batch
        current_messages = model(target, distractors)
        test_messages.extend(current_messages.cpu().tolist())

    test_messages_filename = generate_messages_filename(args.max_length, args.vocab_size, args.seed, "test")
    test_messages_filepath = os.path.join(args.output_path, test_messages_filename)
    np.save(test_messages_filepath, np.array(test_messages))


if __name__ == "__main__":
    baseline(sys.argv[1:])
