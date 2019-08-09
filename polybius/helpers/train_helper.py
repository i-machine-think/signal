import random
import numpy as np
import torch

from ..metrics.average_meter import AverageMeter

from ..models.full_model import FullModel


class TrainHelper:
    def __init__(self, device):
        self.device = device

    def train_one_batch(self, model: FullModel, batch, optimizer, meta_data, device):
        """
        Train for single batch
        """
        model.train()

        optimizer.zero_grad()

        target, distractors, indices, _ = batch

        md = None

        loss, losses, accuracies, _ = model.forward(target, distractors, md)

        loss.backward()
        optimizer.step()

        return losses, accuracies

    def evaluate(self, model, dataloader, valid_meta_data, device, rl):

        if not rl:
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
        else:
            combined_loss_meter = AverageMeter()
            hinge_loss_meter = AverageMeter()
            rl_loss_meter = AverageMeter()
            entropy_meter = AverageMeter()
            acc_meter = AverageMeter()

        messages = []

        model.eval()
        for batch in dataloader:
            target, distractors, indices, lkey = batch

            vmd = None

            _, loss_item, acc, msg = model.forward(target, distractors, vmd)

            if not rl:
                loss_meter.update(loss_item)
                acc_meter.update(acc)
            else:
                combined_loss, hinge_loss, rl_loss, entropy = loss_item
                combined_loss_meter.update(combined_loss)
                hinge_loss_meter.update(hinge_loss)
                rl_loss_meter.update(rl_loss)
                entropy_meter.update(entropy)
                acc_meter.update(acc)

            messages.append(msg)

        if not rl:
            return (loss_meter, acc_meter, torch.cat(messages, 0))
        else:
            return (
                combined_loss_meter,
                hinge_loss_meter,
                rl_loss_meter,
                entropy_meter,
                acc_meter,
                torch.cat(messages, 0),
            )

    def get_filename_from_baseline_params(self, params):
        """
        Generates a filename from baseline params (see baseline.py)
        """
        if params.name:
            return params.name

        name = params.dataset_type
        name += "_e_{}".format(params.embedding_size)
        name += "_h_{}".format(params.hidden_size)
        name += "_lr_{}".format(params.lr)
        name += "_max_len_{}".format(params.max_length)
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

        return name

    def seed_torch(self, seed=42):
        """
        Seed random, numpy and torch with same seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
