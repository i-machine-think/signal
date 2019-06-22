# Baseline setting in which there are only two agents
# - no evolution

import pickle
import argparse
import sys
import torch
import os
import time

from helpers.game_helper import get_sender_receiver, get_trainer, get_training_data, get_meta_data
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
                        help="Disable printing", action="store_true"
    )                        
    parser.add_argument(
        "--device",
        type=str,
        help="Device to be used. Pick from none/cpu/cuda. If default none is used automatic check will be done")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Amount of epochs to check for not improved validation score before early stopping",
    )
    parser.add_argument(
        "--inference-step",
        action="store_true",
        help="Use inference step receiver model",
    )
    parser.add_argument(
        "--step3",
        help="Run with property specific distractors",
        action="store_true",
    )
    parser.add_argument(
        "--multi-task",
        help="Run multi-task approach training using both baseline and diagnostic classifiers approaches",
        action="store_true"
    )
    parser.add_argument(
        "--multi-task-lambda",
        type=float,
        default=0.5,
        help="Lambda value to be used to distinguish importance between baseline approach and the diagnostic classifiers approach",
    )

    args = parser.parse_args(args)

    if args.debugging:
        args.iterations = 1000
        args.max_length = 5
        args.batch_size = 16

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
    sender, baseline_receiver, diagnostic_receiver = get_sender_receiver(device, args)
    
    sender_file = file_helper.get_sender_path(run_folder)
    receiver_file = file_helper.get_receiver_path(run_folder)
    torch.save(sender, sender_file)
    
    if baseline_receiver:
        torch.save(baseline_receiver, receiver_file)

    model = get_trainer(
        sender,
        device,
        args.inference_step,
        args.multi_task,
        args.multi_task_lambda,
        args.dataset_type,
        args.step3,
        baseline_receiver=baseline_receiver,
        diagnostic_receiver=diagnostic_receiver)

    sender_path = file_helper.create_unique_sender_path(model_name)
    visual_module_path = file_helper.create_unique_visual_module_path(model_name)

    if not os.path.exists(file_helper.model_checkpoint_path):
        print('No checkpoint exists. Saving model...\r')
        torch.save(model.visual_module, file_helper.model_checkpoint_path)
        print('No checkpoint exists. Saving model...Done')

    train_data, valid_data, test_data, valid_meta_data, valid_features = get_training_data(
        device=device,
        batch_size=args.batch_size,
        k=args.k,
        debugging=args.debugging,
        dataset_type=args.dataset_type,
        step3=args.step3,)

    train_meta_data, valid_meta_data, test_meta_data = get_meta_data()

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
        if baseline_receiver:
            print(baseline_receiver)

        if diagnostic_receiver:
            print(diagnostic_receiver)

        print("Total number of parameters: {}".format(pytorch_total_params))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    epoch = 0
    iteration = 0
    current_patience = args.patience
    best_accuracy = -1.
    converged = False

    start_time = time.time()
    if args.multi_task:
        header = '  Time Epoch Iteration    Progress (%Epoch) | Loss-Avg  Acc-Avg | Loss-Base Acc-Base | Loss-Color Loss-Shape Loss-Size Loss-PosH Loss-PosW | Acc-Color Acc-Shape Acc-Size Acc-PosH Acc-PosW | Best'
        print(header)
        log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,| {:>8.6f} {:>7.6f} | {:>9.6f} {:>8.6f} | {:>10.6f} {:>10.6f} {:>9.6f} {:>9.6f} {:>9.6f} | {:>9.6f} {:>9.6f} {:>8.6f} {:>8.6f} {:>8.6f} | {:>4s}'.split(','))
    if args.inference_step:
        header = '  Time Epoch Iteration    Progress (%Epoch) | Loss-Avg  Acc-Avg | Loss-Color Loss-Shape Loss-Size Loss-PosH Loss-PosW | Acc-Color Acc-Shape Acc-Size Acc-PosH Acc-PosW | Best'
        print(header)
        log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,| {:>8.6f} {:>7.6f} | {:>10.6f} {:>10.6f} {:>9.6f} {:>9.6f} {:>9.6f} | {:>9.6f} {:>9.6f} {:>8.6f} {:>8.6f} {:>8.6f} | {:>4s}'.split(','))
    if args.step3:
        # The data is saved according to the following sequence [hp,vp,sh,co,si]
        # Thus it should be checked still, with the order in the print statements        
        header = '  Time Epoch Iteration    Progress (%Epoch) | Loss-Avg  Acc-Avg | Loss-PosH Loss-PosW Loss-Color Loss-Shape Loss-Size | Acc-PosH Acc-PosW Acc-Color Acc-Shape Acc-Size | Best'
        print(header)
        log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,| {:>8.6f} {:>7.6f} | {:>10.6f} {:>10.6f} {:>9.6f} {:>9.6f} {:>9.6f} | {:>9.6f} {:>9.6f} {:>8.6f} {:>8.6f} {:>8.6f} | {:>4s}'.split(','))


    while iteration < args.iterations:
        for train_batch in train_data:
            print(f'{iteration}/{args.iterations}       \r', end='')

            _, _ = train_helper.train_one_batch(
                model, train_batch, optimizer, train_meta_data, device, args.inference_step, args.multi_task)

            if iteration % args.log_interval == 0:

                valid_loss_meter, valid_acc_meter, _, = train_helper.evaluate(
                    model, valid_data, valid_meta_data, device, args.inference_step, args.multi_task, args.step3)

                new_best = False
                
                if args.multi_task:
                    average_valid_accuracy = args.multi_task_lambda * valid_acc_meter[0].avg + (1 - args.multi_task_lambda) * valid_acc_meter[1].avg
                else:
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
                    torch.save(model.sender, sender_path)
                    torch.save(model.visual_module, visual_module_path)

                # metrics_helper.log_metrics(
                #     model,
                #     valid_meta_data,
                #     valid_features,
                #     valid_loss_meter,
                #     valid_acc_meter,
                #     valid_entropy_meter,
                #     valid_messages, 
                #     hidden_sender,
                #     hidden_receiver,
                #     loss,
                #     i)

                # Skip for now
                if not args.disable_print:
                    if args.multi_task:
                        print(log_template.format(
                            time.time()-start_time,
                            epoch,
                            iteration,
                            1 + iteration,
                            args.iterations,
                            100. * (1+iteration) / args.iterations,
                            valid_loss_meter[0].avg,
                            valid_acc_meter[0].avg,
                            valid_loss_meter[1].avg,
                            valid_acc_meter[1].avg,
                            valid_loss_meter[0].averages[0],
                            valid_loss_meter[0].averages[1],
                            valid_loss_meter[0].averages[2],
                            valid_loss_meter[0].averages[3],
                            valid_loss_meter[0].averages[4],
                            valid_acc_meter[0].averages[0],
                            valid_acc_meter[0].averages[1],
                            valid_acc_meter[0].averages[2],
                            valid_acc_meter[0].averages[3],
                            valid_acc_meter[0].averages[4],
                            "BEST" if new_best else ""
                        ))
                    elif args.inference_step or args.step3:
                        print(log_template.format(
                            time.time()-start_time,
                            epoch,
                            iteration,
                            1 + iteration,
                            args.iterations,
                            100. * (1+iteration) / args.iterations,
                            valid_loss_meter.avg,
                            valid_acc_meter.avg,
                            valid_loss_meter.averages[0],
                            valid_loss_meter.averages[1],
                            valid_loss_meter.averages[2],
                            valid_loss_meter.averages[3],
                            valid_loss_meter.averages[4],
                            valid_acc_meter.averages[0],
                            valid_acc_meter.averages[1],
                            valid_acc_meter.averages[2],
                            valid_acc_meter.averages[3],
                            valid_acc_meter.averages[4],
                            "BEST" if new_best else ""
                        ))
                    else:
                        print(
                            "{}/{} Iterations: val loss: {}, val accuracy: {}".format(
                                iteration,
                                args.iterations,
                                valid_loss_meter.avg,
                                valid_acc_meter.avg,
                            )
                        )

            iteration += 1
        
        epoch += 1
        
        if converged:
            break

    return run_folder


if __name__ == "__main__":
    baseline(sys.argv[1:])
