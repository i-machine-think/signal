import pickle
import argparse
import sys
import torch
import os

import numpy as np

from helpers.game_helper import get_sender_receiver, get_trainer
from helpers.dataloader_helper import get_shapes_dataloader
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
        "--receiver-path",
        type=str,
        default=False,
        metavar="S",
        help="Receiver to be loaded",
    )

    args = parser.parse_args(args)
    return args

def generate_unique_filename(max_length, vocab_size, seed, set_type):
    name = f'max_len_{max_length}_vocab_{vocab_size}_seed_{seed}.{set_type}'
    return name

def generate_messages_filename(max_length, vocab_size, seed, set_type):
    """
    Generates a filename from baseline params (see baseline.py)
    """
    name = f'{generate_unique_filename(max_length, vocab_size, seed, set_type)}.messages.npy'
    return name

def generate_indices_filename(max_length, vocab_size, seed, set_type):
    """
    Generates a filename from baseline params (see baseline.py)
    """
    name = f'{generate_unique_filename(max_length, vocab_size, seed, set_type)}.indices.npy'
    return name

<<<<<<< HEAD
def save_iteration(model, args, dataset, meta_data, dataset_type):
    messages = []
    properties = []
    for i, batch in enumerate(dataset):
        print(f'{dataset_type}: {i}/{len(dataset)}       \r', end='')
        target, distractors, indices = batch
        current_messages = model(target, distractors)
        messages.extend(current_messages.cpu().tolist())
        # Added properties to be saved)
        current_properties = meta_data[indices[:,0]]
        properties.extend(current_properties.tolist())

    messages_filename = generate_messages_filename(args.max_length, args.vocab_size, args.seed, dataset_type)
    messages_filepath = os.path.join(args.output_path,  messages_filename)
    np.save(messages_filepath, np.array(messages))

    # Added lines to save properties as np.save as well
    # Seperate file is made, similar to generate_messages_filename
    properties_filename = generate_properties_filename(args.max_length, args.vocab_size, args.seed, dataset_type)
    properties_filepath = os.path.join(args.output_path, properties_filename)
    np.save(properties_filepath, np.array(properties))

=======
>>>>>>> 74407108ded883fb73dc48e85e068f76e30127e8

def baseline(args):

    args = parse_arguments(args)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_helper = TrainHelper(device)
    train_helper.seed_torch(seed=args.seed)

    # get sender and receiver models and save them
<<<<<<< HEAD
    sender, _ = get_sender_receiver(device, args)
    sender = torch.load(args.sender_path, map_location=device)
=======
    # sender, receiver = get_sender_receiver(device, args)
    sender = torch.load(args.sender_path)
>>>>>>> 74407108ded883fb73dc48e85e068f76e30127e8

    model = get_trainer(sender, None, device, "raw")

    train_data, validation_data, test_data = get_shapes_dataloader(
        device=device,
        batch_size=1024,
        k=3,
        debug=False,
        dataset_type="raw")

    model.to(device)

    train_messages = []
    train_indices = []

    validation_messages = []
    validation_indices = []

    test_messages = []
    test_indices = []

    model.eval()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

<<<<<<< HEAD
    save_iteration(model, args, train_data, train_meta_data, 'train')
    save_iteration(model, args, validation_data, valid_meta_data, 'validation')
    save_iteration(model, args, test_data, test_meta_data, 'test')

    # for i, train_batch in enumerate(train_data):
    #     print(f'Train: {i}/{len(train_data)}       \r', end='')
    #     target, distractors, indices = train_batch
    #     current_messages = model(target, distractors)
    #     train_messages.extend(current_messages.cpu().tolist())
    #     # Added properties to be saved)
    #     current_properties = train_meta_data[indices[:,0]]
    #     train_properties.extend(current_properties.tolist())

    # train_messages_filename = generate_messages_filename(args.max_length, args.vocab_size, args.seed, "train")
    # train_messages_filepath = os.path.join(args.output_path,  train_messages_filename)
    # np.save(train_messages_filepath, np.array(train_messages))

    # # Added lines to save properties as np.save as well
    # # Seperate file is made, similar to generate_messages_filename
    # train_properties_filename = generate_properties_filename(args.max_length, args.vocab_size, args.seed, "train")
    # train_properties_filepath = os.path.join(args.output_path, train_properties_filename)
    # np.save(train_properties_filepath, np.array(train_properties))

    # for i, validation_batch in enumerate(validation_data):
    #     print(f'Validation: {i}/{len(validation_data)}       \r', end='')
    #     target, distractors = validation_batch
    #     current_messages = model(target, distractors)
    #     validation_messages.extend(current_messages.cpu().tolist())

    # validation_messages_filename = generate_messages_filename(args.max_length, args.vocab_size, args.seed, "validation")
    # validation_messages_filepath = os.path.join(
    #     args.output_path, validation_messages_filename)
    # np.save(validation_messages_filepath, np.array(validation_messages))

    # for i, test_batch in enumerate(test_data):
    #     print(f'Test: {i}/{len(test_data)}       \r', end='')
    #     target, distractors = test_batch
    #     current_messages = model(target, distractors)
    #     test_messages.extend(current_messages.cpu().tolist())

    # test_messages_filename = generate_messages_filename(args.max_length, args.vocab_size, args.seed, "test")
    # test_messages_filepath = os.path.join(args.output_path, test_messages_filename)
    # np.save(test_messages_filepath, np.array(test_messages))
=======
    for i, train_batch in enumerate(train_data):
        print(f'Train: {i}/{len(train_data)}       \r', end='')
        target, distractors, indices = train_batch
        current_target_indices = indices[:, 0].detach().cpu().tolist()
        train_indices.extend(current_target_indices)
        current_messages = model(target, distractors)
        train_messages.extend(current_messages.cpu().tolist())

    train_messages_filename = generate_messages_filename(args.max_length, args.vocab_size, args.seed, "train")
    train_messages_filepath = os.path.join(args.output_path,  train_messages_filename)
    np.save(train_messages_filepath, np.array(train_messages))
    
    train_indices_filename = generate_indices_filename(args.max_length, args.vocab_size, args.seed, "train")
    train_indices_filepath = os.path.join(args.output_path, train_indices_filename)
    np.save(train_indices_filepath, np.array(train_indices))

    for i, validation_batch in enumerate(validation_data):
        print(f'Validation: {i}/{len(validation_data)}       \r', end='')
        target, distractors, indices = validation_batch
        current_target_indices = indices[:, 0].detach().cpu().tolist()
        validation_indices.extend(current_target_indices)
        current_messages = model(target, distractors)
        validation_messages.extend(current_messages.cpu().tolist())

    validation_messages_filename = generate_messages_filename(args.max_length, args.vocab_size, args.seed, "validation")
    validation_messages_filepath = os.path.join(
        args.output_path, validation_messages_filename)
    np.save(validation_messages_filepath, np.array(validation_messages))
    
    validation_indices_filename = generate_indices_filename(args.max_length, args.vocab_size, args.seed, "validation")
    validation_indices_filepath = os.path.join(args.output_path, validation_indices_filename)
    np.save(validation_indices_filepath, np.array(validation_indices))

    for i, test_batch in enumerate(test_data):
        print(f'Test: {i}/{len(test_data)}       \r', end='')
        target, distractors, indices = test_batch
        current_target_indices = indices[:, 0].detach().cpu().tolist()
        test_indices.extend(current_target_indices)
        current_messages = model(target, distractors)
        test_messages.extend(current_messages.cpu().tolist())

    test_messages_filename = generate_messages_filename(args.max_length, args.vocab_size, args.seed, "test")
    test_messages_filepath = os.path.join(args.output_path, test_messages_filename)
    np.save(test_messages_filepath, np.array(test_messages))
    
    test_indices_filename = generate_indices_filename(args.max_length, args.vocab_size, args.seed, "test")
    test_indices_filepath = os.path.join(args.output_path, test_indices_filename)
    np.save(test_indices_filepath, np.array(test_indices))
>>>>>>> 74407108ded883fb73dc48e85e068f76e30127e8


if __name__ == "__main__":
    baseline(sys.argv[1:])
