import pickle
import argparse
import sys
import torch
import os

import numpy as np

from helpers.game_helper import get_sender_receiver, get_trainer
from helpers.dataloader_helper import get_shapes_dataloader
from helpers.train_helper import TrainHelper
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
        "--visual-module-path",
        type=str,
        required=True,
        metavar="S",
        help="Visual module to be loaded",
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
    # later added
    parser.add_argument(
        "--single-model",
        help="Use a single model (default: False)",
        action="store_true",
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
        "--greedy",
        help="Use argmax at prediction time instead of sampling (default: False)",
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

def generate_unique_filename(max_length, vocab_size, seed, inference, step3, set_type):
    name = f'max_len_{max_length}_vocab_{vocab_size}_seed_{seed}'
    if inference:
        name += '_inference'
    
    if step3:
        name += '_step3'

    name += f'.{set_type}'

    return name

def generate_messages_filename(max_length, vocab_size, seed, inference, step3, set_type):
    """
    Generates a filename from baseline params (see baseline.py)
    """
    name = f'{generate_unique_filename(max_length, vocab_size, seed, inference, step3, set_type)}.messages.npy'
    return name

def generate_indices_filename(max_length, vocab_size, seed, inference, step3, set_type):
    """
    Generates a filename from baseline params (see baseline.py)
    """
    name = f'{generate_unique_filename(max_length, vocab_size, seed, inference, step3, set_type)}.indices.npy'
    return name

def sample_messages_from_dataset(model, args, dataset, dataset_type):
    messages = []
    indices = []
    
    for i, batch in enumerate(dataset):
        print(f'{dataset_type}: {i}/{len(dataset)}       \r', end='')
        target, distractors, current_indices = batch
        
        current_target_indices = current_indices[:, 0].detach().cpu().tolist()
        indices.extend(current_target_indices)
        current_messages = model(target, distractors)
        messages.extend(current_messages.cpu().tolist())

    messages_filename = generate_messages_filename(args.max_length, args.vocab_size, args.seed, args.inference, args.step3, dataset_type)
    messages_filepath = os.path.join(args.output_path,  messages_filename)
    np.save(messages_filepath, np.array(messages))

    # Added lines to save properties as np.save as well
    # Separate file is made, similar to generate_messages_filename
    indices_filename = generate_indices_filename(args.max_length, args.vocab_size, args.seed, args.inference, args.step3, dataset_type)
    indices_filepath = os.path.join(args.output_path, indices_filename)
    np.save(indices_filepath, np.array(indices))

def baseline(args):
    args = parse_arguments(args)

    if not os.path.isfile(args.sender_path):
        raise Exception('Sender path is invalid')
        
    if not os.path.isfile(args.visual_module_path):
        raise Exception('Visual module path is invalid')

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_helper = TrainHelper(device)
    train_helper.seed_torch(seed=args.seed)

    # get sender and receiver models and save them
    sender = torch.load(args.sender_path, map_location=device)
    sender.greedy = True
    
    model = get_trainer(sender, device, inference_step=False, dataset_type="raw")
    
    model.visual_module = torch.load(
        args.visual_module_path,
        map_location=lambda storage, location: storage)

    train_data, validation_data, test_data = get_shapes_dataloader(
        device=device,
        batch_size=1024,
        k=3,
        debug=False,
        dataset_type="raw")

    model.to(device)

    model.eval()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    sample_messages_from_dataset(model, args, train_data, 'train')
    sample_messages_from_dataset(model, args, validation_data, 'validation')
    sample_messages_from_dataset(model, args, test_data, 'test')

if __name__ == "__main__":
    baseline(sys.argv[1:])
