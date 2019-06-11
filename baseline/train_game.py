# Baseline setting in which there are only two agents
# - no evolution

import pickle
import argparse
import sys
import torch
import os

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
        default=5,
        metavar="N",
        help="max sentence length allowed for communication (default: 5)",
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
        default=5,
        metavar="N",
        help="Size of vocabulary (default: 5)",
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

    if args.debugging:
        args.iterations = 1000
        args.max_length = 5

    return args


def baseline(args):

    args = parse_arguments(args)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    file_helper = FileHelper()
    train_helper = TrainHelper(device)
    train_helper.seed_torch(seed=args.seed)

    model_name = train_helper.get_filename_from_baseline_params(args)
    run_folder = file_helper.get_run_folder(args.folder, model_name)

    metrics_helper = MetricsHelper(run_folder, args.seed)

    # get sender and receiver models and save them
    sender, receiver = get_sender_receiver(device, args)

    sender_file = file_helper.get_sender_path(run_folder)
    receiver_file = file_helper.get_receiver_path(run_folder)
    torch.save(sender, sender_file)
    torch.save(receiver, receiver_file)

    model = get_trainer(sender, receiver, device, args)

    if not os.path.exists(file_helper.model_checkpoint_path):
        print('No checkpoint exists. Saving model...\r')
        torch.save(model.visual_module, file_helper.model_checkpoint_path)
        print('No checkpoint exists. Saving model...Done')

    train_data, valid_data, test_data, valid_meta_data, valid_features = get_training_data(device, args)

    # dump arguments
    pickle.dump(args, open(f'{run_folder}/experiment_params.p', "wb"))

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    if not args.disable_print:
        # Print info
        print("----------------------------------------")
        print(
            "Model name: {} \n|V|: {}\nL: {}".format(
                model_name, args.vocab_size, args.max_length
            )
        )
        print(sender)
        print(receiver)
        print("Total number of parameters: {}".format(pytorch_total_params))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    i = 0
    current_patience = args.patience
    best_accuracy = -1.
    converged = False

    while i < args.iterations:
        for train_batch in train_data:
            print(f'{i}/{args.iterations}       \r', end='')

            loss, _ = train_helper.train_one_batch(
                model, train_batch, optimizer)

            if i % args.log_interval == 0:

                valid_loss_meter, valid_acc_meter, valid_entropy_meter, valid_messages, hidden_sender, hidden_receiver = train_helper.evaluate(
                    model, valid_data
                )

                if valid_acc_meter.avg < best_accuracy:
                    current_patience -= 1

                    if current_patience <= 0:
                        print('Model has converged. Stopping training...')
                        converged = True
                        break
                else:
                    best_accuracy = valid_acc_meter.avg
                    current_patience = args.patience

                metrics_helper.log_metrics(
                    model,
                    valid_meta_data,
                    valid_features,
                    valid_loss_meter,
                    valid_acc_meter,
                    valid_entropy_meter,
                    valid_messages, 
                    hidden_sender,
                    hidden_receiver,
                    loss,
                    i)

                # Skip for now
                if not args.disable_print:
                    print(
                        "{}/{} Iterations: val loss: {}, val accuracy: {}".format(
                            i,
                            args.iterations,
                            valid_loss_meter.avg,
                            valid_acc_meter.avg,
                        )
                    )

            i += 1
        
        if converged:
            break

    best_model = get_trainer(sender, receiver, device, args)
    state = torch.load(
        "{}/best_model".format(run_folder),
        map_location=lambda storage, location: storage,
    )
    best_model.load_state_dict(state)
    best_model.to(device)
    # Evaluate best model on test data
    _, test_acc_meter, _, test_messages, _, _ = train_helper.evaluate(
        best_model, test_data)
    if not args.disable_print:
        print("Test accuracy: {}".format(test_acc_meter.avg))

    # Update receiver and sender files with new state
    torch.save(best_model.sender, sender_file)
    torch.save(best_model.receiver, receiver_file)

    if args.dataset_type == "raw":
        best_model.to(torch.device("cpu"))
        torch.save(best_model.visual_module, file_helper.model_checkpoint_path)

    torch.save(test_messages, "{}/test_messages.p".format(run_folder))
    pickle.dump(
        test_acc_meter, open(
            "{}/test_accuracy_meter.p".format(run_folder), "wb")
    )

    return run_folder


if __name__ == "__main__":
    baseline(sys.argv[1:])
