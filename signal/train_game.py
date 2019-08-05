# Baseline setting in which there are only two agents
# - no evolution

import pickle
import argparse
import sys
import torch

# for logging of data
import os
import time
import csv
from itertools import zip_longest
import datetime

from helpers.game_helper import (
    get_sender_receiver,
    get_trainer,
    get_training_data,
    get_meta_data,
)
from helpers.train_helper import TrainHelper
from helpers.file_helper import FileHelper
from helpers.metrics_helper import MetricsHelper

from plotting import plot_data


def parse_arguments(args):
    # Training settings
    parser = argparse.ArgumentParser(
        description="Training Sender/Receiver Agent on a task"
    )
    parser.add_argument(
        "--single-model",
        help="Use a single model (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="features",
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
        "--lr",
        type=float,
        default=1e-3,
        metavar="N",
        help="Adam learning rate (default: 1e-3)",
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
        "--tau",
        type=float,
        default=1.2,
        help="temperature parameter for softmax distributions",
    )
    parser.add_argument(
        "--vqvae",
        help="switch for using vector quantization (default:False)",
        action="store_true",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.25,
        help="weighting factor for loss-terms 2 and 3 in VQ-VAE",
    )
    parser.add_argument(
        "--discrete_latent_number",
        type=int,
        default=25,
        help="Number of embedding vectors in the VQ-VAE case",
    )
    parser.add_argument(
        "--discrete_latent_dimension",
        type=int,
        default=25,
        help="dimension of embedding vectors in the VQ-VAE case",
    )
    parser.add_argument(
        "--discrete_communication",
        help="switch for communicating discretely in the vqvae case",
        action="store_true",
    )
    parser.add_argument(
        "--gumbel_softmax",
        help="switch for using straight-through gumbel_softmax in the vqvae-discrete_communication case",
        action="store_true",
    )
    parser.add_argument(
        "--rl",
        help="switch for using REINFORCE for training the sender",
        action="store_true",
    )
    parser.add_argument(
        "--entropy_coefficient",
        type=float,
        default=1.0,
        help="weighting factor for the entropy that's increased in RL",
    )
    parser.add_argument(
        "--myopic",
        help="switch for forgetting old hinge losses faster in the RL setting",
        action="store_true",
    )
    parser.add_argument(
        "--myopic_coefficient",
        type=float,
        default=0.1,
        help="coefficient for how much to update in the direction of new loss-term in myopic case",
    )

    # Arguments not specific to the training process itself
    parser.add_argument(
        "--debugging",
        help="Enable debugging mode (default: False)",
        action="store_true",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        metavar="N",
        help="number of iterations between logs (default: 200)",
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
        "--patience",
        type=int,
        default=10,
        help="Amount of epochs to check for not improved validation score before early stopping",
    )
    parser.add_argument(
        "--test-mode",
        help="Only run the saved model on the test set",
        action="store_true",
    )
    parser.add_argument(
        "--resume-training",
        help="Resume the training from the saved model state",
        action="store_true",
    )

    args = parser.parse_args(args)

    if args.debugging:
        args.iterations = 1000
        args.max_length = 5
        args.batch_size = 16

    return args


def save_model_state(
    model, checkpoint_path: str, epoch: int, iteration: int, best_score: int
):
    checkpoint_state = {}

    if model.sender:
        checkpoint_state["sender"] = model.sender.state_dict()

    if model.visual_module:
        checkpoint_state["visual_module"] = model.visual_module.state_dict()

    if model.receiver:
        checkpoint_state["receiver"] = model.receiver.state_dict()

    if model.diagnostic_receiver:
        checkpoint_state["diagnostic_receiver"] = model.diagnostic_receiver.state_dict(
        )

    if epoch:
        checkpoint_state["epoch"] = epoch

    if iteration:
        checkpoint_state["iteration"] = iteration

    if best_score:
        checkpoint_state["best_score"] = best_score

    torch.save(checkpoint_state, checkpoint_path)


def load_model_state(model, model_path):
    if not os.path.isfile(model_path):
        raise Exception(f'Model not found at "{model_path}"')

    checkpoint = torch.load(model_path)

    if "sender" in checkpoint.keys() and checkpoint["sender"]:
        model.sender.load_state_dict(checkpoint["sender"])

    if "visual_module" in checkpoint.keys() and checkpoint["visual_module"]:
        model.visual_module.load_state_dict(checkpoint["visual_module"])

    if "receiver" in checkpoint.keys() and checkpoint["receiver"]:
        model.receiver.load_state_dict(checkpoint["receiver"])

    if "diagnostic_receiver" in checkpoint.keys() and checkpoint["diagnostic_receiver"]:
        model.diagnostic_receiver.load_state_dict(
            checkpoint["diagnostic_receiver"])

    best_score = -1.0
    if "best_score" in checkpoint.keys() and checkpoint["best_score"]:
        best_score = checkpoint["best_score"]

    epoch = 0
    if "epoch" in checkpoint.keys() and checkpoint["epoch"]:
        epoch = checkpoint["epoch"]

    iteration = 0
    if "iteration" in checkpoint.keys() and checkpoint["iteration"]:
        iteration = checkpoint["iteration"]

    return epoch, iteration, best_score


def baseline(args):
    args = parse_arguments(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_helper = FileHelper()
    train_helper = TrainHelper(device)
    train_helper.seed_torch(seed=args.seed)

    model_name = train_helper.get_filename_from_baseline_params(args)
    run_folder = file_helper.get_run_folder(args.folder, model_name)

    metrics_helper = MetricsHelper(run_folder, args.seed)

    # get sender and receiver models and save them
    sender, receiver, diagnostic_receiver = get_sender_receiver(device, args)

    sender_file = file_helper.get_sender_path(run_folder)
    receiver_file = file_helper.get_receiver_path(run_folder)
    # torch.save(sender, sender_file)

    if receiver:
        torch.save(receiver, receiver_file)

    model = get_trainer(
        sender,
        device,
        args.dataset_type,
        receiver=receiver,
        diagnostic_receiver=diagnostic_receiver,
        vqvae=args.vqvae,
        rl=args.rl,
        entropy_coefficient=args.entropy_coefficient,
        myopic=args.myopic,
        myopic_coefficient=args.myopic_coefficient,
    )

    model_path = file_helper.create_unique_model_path(model_name)

    best_accuracy = -1.0
    epoch = 0
    iteration = 0

    if args.resume_training or args.test_mode:
        epoch, iteration, best_accuracy = load_model_state(model, model_path)
        print(
            f"Loaded model. Resuming from - epoch: {epoch} | iteration: {iteration} | best accuracy: {best_accuracy}"
        )

    if not os.path.exists(file_helper.model_checkpoint_path):
        print("No checkpoint exists. Saving model...\r")
        torch.save(model.visual_module, file_helper.model_checkpoint_path)
        print("No checkpoint exists. Saving model...Done")

    train_data, valid_data, test_data, valid_meta_data, _ = get_training_data(
        device=device,
        batch_size=args.batch_size,
        k=args.k,
        debugging=args.debugging,
        dataset_type=args.dataset_type,
    )

    train_meta_data, valid_meta_data, test_meta_data = get_meta_data()

    # dump arguments
    pickle.dump(args, open(f"{run_folder}/experiment_params.p", "wb"))

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
        if receiver:
            print(receiver)

        if diagnostic_receiver:
            print(diagnostic_receiver)

        print("Total number of parameters: {}".format(pytorch_total_params))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    current_patience = args.patience
    best_accuracy = -1.0
    converged = False

    start_time = time.time()

    if args.test_mode:
        test_loss_meter, test_acc_meter, _ = train_helper.evaluate(
            model, test_data, test_meta_data, device, args.rl
        )

        average_test_accuracy = test_acc_meter.avg
        average_test_loss = test_loss_meter.avg

        print(
            f"TEST results: loss: {average_test_loss} | accuracy: {average_test_accuracy}"
        )
        return

    iterations = []
    losses = []
    hinge_losses = []
    rl_losses = []
    entropies = []
    accuracies = []

    while iteration < args.iterations:
        for train_batch in train_data:
            print(f"{iteration}/{args.iterations}       \r", end="")

            # !!! This is the complete training procedure. Rest is only logging!
            _, _ = train_helper.train_one_batch(
                model, train_batch, optimizer, train_meta_data, device
            )

            if iteration % args.log_interval == 0:

                if not args.rl:
                    valid_loss_meter, valid_acc_meter, _, = train_helper.evaluate(
                        model, valid_data, valid_meta_data, device, args.rl
                    )
                else:
                    valid_loss_meter, hinge_loss_meter, rl_loss_meter, entropy_meter, valid_acc_meter, _ = train_helper.evaluate(
                        model, valid_data, valid_meta_data, device, args.rl
                    )

                new_best = False

                average_valid_accuracy = valid_acc_meter.avg

                if (
                    average_valid_accuracy < best_accuracy
                ):  # No new best found. May lead to early stopping
                    current_patience -= 1

                    if current_patience <= 0:
                        print("Model has converged. Stopping training...")
                        converged = True
                        break
                else:  # new best found. Is saved.
                    new_best = True
                    best_accuracy = average_valid_accuracy
                    current_patience = args.patience
                    save_model_state(model, model_path, epoch,
                                     iteration, best_accuracy)

                # Skip for now  <--- What does this comment mean? printing is not disabled, so this will be shown, right?
                if not args.disable_print:

                    if not args.rl:
                        print(
                            "{}/{} Iterations: val loss: {:.3f}, val accuracy: {:.3f}".format(
                                iteration,
                                args.iterations,
                                valid_loss_meter.avg,
                                valid_acc_meter.avg,
                            )
                        )
                    else:
                        print(
                            "{}/{} Iterations: val loss: {:.3f}, val hinge loss: {:.3f}, val rl loss: {:.3f}, val entropy: {:.3f}, val accuracy: {:.3f}".format(
                                iteration,
                                args.iterations,
                                valid_loss_meter.avg,
                                hinge_loss_meter.avg,
                                rl_loss_meter.avg,
                                entropy_meter.avg,
                                valid_acc_meter.avg,
                            )
                        )

                iterations.append(iteration)
                losses.append(valid_loss_meter.avg)
                if args.rl:
                    hinge_losses.append(hinge_loss_meter.avg)
                    rl_losses.append(rl_loss_meter.avg)
                    entropies.append(entropy_meter.avg)
                accuracies.append(valid_acc_meter.avg)

            iteration += 1
            if iteration >= args.iterations:
                break

        epoch += 1

        if converged:
            break

    # prepare writing of data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path.replace("/baseline", "")
    timestamp = str(datetime.datetime.now())
    filename = "output_data/vqvae_{}_rl_{}_dc_{}_gs_{}_dln_{}_dld_{}_beta_{}_entropy_coefficient_{}_myopic_{}_mc_{}_seed_{}_{}.csv".format(
        args.vqvae,
        args.rl,
        args.discrete_communication,
        args.gumbel_softmax,
        args.discrete_latent_number,
        args.discrete_latent_dimension,
        args.beta,
        args.entropy_coefficient,
        args.myopic,
        args.myopic_coefficient,
        args.seed,
        timestamp,
    )
    full_filename = os.path.join(dir_path, filename)

    # write data
    d = [iterations, losses, hinge_losses, rl_losses, entropies, accuracies]
    export_data = zip_longest(*d, fillvalue="")
    with open(full_filename, "w", encoding="ISO-8859-1", newline="") as myfile:
        wr = csv.writer(myfile)
        wr.writerow(
            ("iteration", "loss", "hinge loss", "rl loss", "entropy", "accuracy")
        )
        wr.writerows(export_data)
    myfile.close()

    # plotting
    print(filename)
    plot_data(filename, args)

    return run_folder


if __name__ == "__main__":
    baseline(sys.argv[1:])
