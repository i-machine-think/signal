# Baseline setting in which there are only two agents
# - no evolution

import pickle
import argparse
import sys
import torch
import os
import time

from torch.utils import data

from datasets.message_dataset import MessageDataset
from enums.dataset_type import DatasetType

from helpers.game_helper import get_sender_receiver, get_trainer, get_training_data, get_meta_data
from helpers.train_helper import TrainHelper
from helpers.file_helper import FileHelper
from helpers.metrics_helper import MetricsHelper

from models.image_receiver import ImageReceiver
from metrics.average_meter import AverageMeter

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

header = '  Time Epoch Iteration     Progress (%Epoch) |     Loss Accuracy | Best'
log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,| {:>8.6f} {:>8.6f} | {:>4s}'.split(','))

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
        default="raw",
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
        "--seed", type=int, default=13, metavar="S", help="random seed (default: 13)"
    )

    parser.add_argument(
        "--messages-seed",
        type=int,
        required=True,
        metavar="S",
        help="The seed which was used for sampling messages"
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
    # parser.add_argument(
    #     "--sender-path",
    #     type=str,
    #     required=True,
    #     metavar="S",
    #     help="Sender to be loaded",
    # )
    # parser.add_argument(
    #     "--visual-module-path",
    #     type=str,
    #     required=True,
    #     metavar="S",
    #     help="Visual module to be loaded",
    # )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to be used. Pick from none/cpu/cuda. If default none is used automatic check will be done")
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Amount of epochs to check for not improved validation score before early stopping",
    )
    parser.add_argument(
        "--test-mode",
        help="Only run the saved model on the test set",
        action="store_true"
    )

    args = parser.parse_args(args)

    return args

def save_image_grid(images, image_size, rows, iteration, appendix, images_path):
    sample_images = images.view(-1, 3, image_size, image_size)

    samples_grid = make_grid(sample_images, nrow=rows, normalize=True, pad_value=.5, padding=1).cpu().numpy().transpose(1,2,0)

    filepath = os.path.join(images_path, f'{appendix}_{iteration}.png')
    plt.imsave(filepath, samples_grid)

def perform_iteration(
    model,
    criterion,
    optimizer,
    messages,
    original_targets,
    iteration,
    device,
    images_path,
    generate_images):

    messages = messages.float().to(device)
    original_targets = original_targets.float().to(device)

    if model.training:
        optimizer.zero_grad()

    output = model.forward(messages)

    original_targets_vector = original_targets.view(original_targets.shape[0], -1)
    loss = criterion.forward(output, original_targets_vector)

    if model.training:
        loss.backward()
        optimizer.step()

    if generate_images:
        if iteration == 0:
            save_image_grid(original_targets_vector[0:100].detach().cpu(), 30, 10, iteration, 'original', images_path)

        save_image_grid(output[0:100].detach().cpu(), 30, 10, iteration, 'predicted', images_path)

    acc = torch.mean(torch.mean(((original_targets_vector > .5) == (output > .5)).float(), dim=1))

    return loss.item(), acc.item()

def evaluate(model, criterion, dataloader, iteration, device, images_path):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.eval()

    for i, (messages, original_targets) in enumerate(dataloader):
        loss, acc = perform_iteration(model, criterion, None, messages, original_targets, iteration, device, images_path, i==0)
        loss_meter.update(loss)
        acc_meter.update(acc)

    return loss_meter, acc_meter

def generate_unique_name(length, vocabulary_size, seed, inference, multi_task, multi_task_lambda):
    result = f'max_len_{length}_vocab_{vocabulary_size}_seed_{seed}'
    return result


def generate_model_name(length, vocabulary_size, messages_seed, training_seed):
    result = f'max_len_{length}_vocab_{vocabulary_size}_msg_seed_{messages_seed}_train_seed_{training_seed}'
    return result

def save_model(path, model, optimizer, iteration=None):
    torch_state = {'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict()
                   }
    if iteration:
        torch_state["iteration"] = iteration

    torch.save(torch_state, path)

def load_model(path, model, optimizer):
    if not os.path.exists(path):
        return 0

    print('Loading model from checkpoint...')

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    iteration = 0
    if checkpoint['iteration']:
        iteration = checkpoint['iteration']

    return iteration

def baseline(args):

    args = parse_arguments(args)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # file_helper = FileHelper()
    train_helper = TrainHelper(device)
    train_helper.seed_torch(seed=args.seed)

    model = ImageReceiver()
    model.to(device)

    unique_name = generate_unique_name(
        length=args.max_length,
        vocabulary_size=args.vocab_size,
        seed=args.messages_seed)

    model_name = generate_model_name(
        length=args.max_length,
        vocabulary_size=args.vocab_size,
        messages_seed=args.messages_seed,
        training_seed=args.seed)

    if not os.path.exists('results'):
        os.mkdir('results')

    reproducing_folder = os.path.join('results', 'reproducing-images')
    if not os.path.exists(reproducing_folder):
        os.mkdir(reproducing_folder)

    output_path = os.path.join(reproducing_folder, model_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    model_path = os.path.join(output_path, 'messages_receiver.p')

    print(f'loading messages using unique name: "{unique_name}"')

    train_dataset = MessageDataset(unique_name, DatasetType.Train)
    train_dataloader = data.DataLoader(train_dataset, num_workers=1, pin_memory=True, shuffle=True, batch_size=args.batch_size)

    validation_dataset = MessageDataset(unique_name, DatasetType.Valid)
    validation_dataloader = data.DataLoader(validation_dataset, num_workers=1, pin_memory=True, shuffle=False, batch_size=args.batch_size)

    test_dataset = MessageDataset(unique_name, DatasetType.Test)
    test_dataloader = data.DataLoader(test_dataset, num_workers=1, pin_memory=True, shuffle=False, batch_size=args.batch_size)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    # Print info
    print("----------------------------------------")
    # print(
    #     "Model name: {} \n|V|: {}\nL: {}".format(
    #         model_name, args.vocab_size, args.max_length
    #     )
    # )
    # print(sender)
    # print(visual_module)
    print(model)
    print("Total number of parameters: {}".format(pytorch_total_params))


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch = 0
    current_patience = args.patience
    best_accuracy = -1.
    converged = False

    start_time = time.time()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    images_path = os.path.join(output_path, 'samples')
    if not os.path.exists(images_path):
        os.mkdir(images_path)

    iteration = load_model(model_path, model, optimizer)

    if args.test_mode:
        test_images_path = os.path.join(images_path, 'test')
        if not os.path.exists(test_images_path):
            os.mkdir(test_images_path)

        test_loss_meter, test_acc_meter = evaluate(model, criterion, test_dataloader, 0, device, test_images_path)
        print(f'TEST results: loss: {test_loss_meter.avg} | accuracy: {test_acc_meter.avg}')
        return


    print(header)
    while iteration < args.iterations:
        for (messages, original_targets) in train_dataloader:
            print(f'{iteration}/{args.iterations}       \r', end='')

            model.train()

            _, _ = perform_iteration(model, criterion, optimizer, messages, original_targets, iteration, device, images_path, False)

            if iteration % args.log_interval == 0:
                valid_loss_meter, valid_acc_meter = evaluate(model, criterion, validation_dataloader, iteration, device, images_path)

                new_best = False
                average_valid_accuracy = valid_acc_meter.avg

                if average_valid_accuracy < best_accuracy:
                    current_patience -= 1

                    if current_patience <= 0:
                        print('Model has converged. Stopping training...')
                        converged = True
                        break
                else:
                    new_best = True
                    best_accuracy = average_valid_accuracy
                    current_patience = args.patience
                    save_model(model_path, model, optimizer, iteration)


                print(log_template.format(
                    time.time()-start_time,
                    epoch,
                    iteration,
                    1 + iteration,
                    args.iterations,
                    100. * (1+iteration) / args.iterations,
                    valid_loss_meter.avg,
                    valid_acc_meter.avg,
                    "BEST" if new_best else ""))

            iteration += 1

        epoch += 1

        if converged:
            break

if __name__ == "__main__":
    baseline(sys.argv[1:])
