import random
import numpy as np
import torch

from metrics.average_meter import AverageMeter

class TrainHelper():
    def train_one_batch(self, model, batch, optimizer):
        """
        Train for single batch
        """
        model.train()
        optimizer.zero_grad()

        target, distractors = batch
        loss, acc, messages = model(target, distractors)

        loss.backward()
        optimizer.step()

        return loss.item(), acc.item()

    def evaluate(self, model, data, return_softmax=False):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        entropy_meter = AverageMeter()
        hidden_sender, hidden_receiver = [], []
        messages, sentence_probabilities = [], []

        #########################################
        ############ DIAGNOSTIC CODE ############
        #########################################
        class_loss_meters = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
        #########################################

        model.eval()
        for d in data:
            if len(d) == 2:  # shapes
                target, distractors = d
                #########################################
                ############ DIAGNOSTIC CODE ############
                #########################################
                loss, acc, msg, h_s, h_r, entropy, sent_p, class_losses = model(target, distractors)
                #########################################

            if len(d) == 3:  # obverter task
                first_image, second_image, label = d
                loss, acc, msg, h_s, h_r, entropy, sent_p = model(
                    first_image, second_image, label
                )

            loss_meter.update(loss.item())
            acc_meter.update(acc.item())
            entropy_meter.update(entropy.item())

            #########################################
            ############ DIAGNOSTIC CODE ############
            #########################################
            for i, c in enumerate(class_loss_meters):
                c.update(class_losses[i].item())
            #########################################

            messages.append(msg)
            sentence_probabilities.append(sent_p)
            hidden_sender.append(h_s.detach().cpu().numpy())
            hidden_receiver.append(h_r.detach().cpu().numpy())

        hidden_sender = np.concatenate(hidden_sender)
        hidden_receiver = np.concatenate(hidden_receiver)

        #########################################
        ############ DIAGNOSTIC CODE ############
        #########################################
        classes = ['color', 'shape', 'size', 'pos_h', 'pos_w']
        print()
        for i,c in enumerate(classes):
            print(
                "Class {}: val accuracy: {}".format(
                    c,
                    class_loss_meters[i].avg,
                )
            )
        #########################################

        if return_softmax:
            return (
                loss_meter,
                acc_meter,
                entropy_meter,
                torch.cat(messages, 0),
                torch.cat(sentence_probabilities, 0),
                hidden_sender,
                hidden_receiver,
            )
        else:
            return (
                loss_meter,
                acc_meter,
                entropy_meter,
                torch.cat(messages, 0),
                hidden_sender,
                hidden_receiver,
            )

    def get_filename_from_baseline_params(self, params):
        """
        Generates a filename from baseline params (see baseline.py)
        """
        if params.name:
            return params.name

        name = params.task
        name += "_{}".format(params.dataset_type)
        name += "_e_{}".format(params.embedding_size)
        name += "_h_{}".format(params.hidden_size)
        name += "_lr_{}".format(params.lr)
        name += "_max_len_{}".format(params.max_length)
        if params.task == "shapes":
            name += "_k_{}".format(params.k)
        name += "_vocab_{}".format(params.vocab_size)
        name += "_seed_{}".format(params.seed)
        name += "_btch_size_{}".format(params.batch_size)
        if params.single_model:
            name += "_single_model"
        if params.greedy:
            name += "_greedy"
        if params.debugging:
            name += "_debug"
        if params.sender_path or params.receiver_path:
            name += "_loaded_from_path"
        if params.obverter_setup:
            name = "obverter_setup_with_" + name
        return name

    def seed_torch(self, seed=42):
        """
        Seed random, numpy and torch with same seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)