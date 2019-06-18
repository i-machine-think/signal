import random
import numpy as np
import torch

from metrics.average_meter import AverageMeter
from metrics.average_ensemble_meter import AverageEnsembleMeter

class TrainHelper():
    def __init__(self, device):
        self.device = device

    def train_one_batch(self, model, batch, optimizer, meta_data, device, inference_step):
        """
        Train for single batch
        """
        model.train()

        optimizer.zero_grad()

        target, distractors, indices = batch

        md = torch.tensor(meta_data[indices[:,0], :], device=device, dtype=torch.int64)
        loss, losses, accuracies, _ = model.forward(target, distractors, md)

        loss.backward()
        optimizer.step()

        return losses, accuracies

    def evaluate(self, model, dataloader, valid_meta_data, device, inference_step):
        
        if inference_step:
            loss_meter = AverageEnsembleMeter(5)
            acc_meter = AverageEnsembleMeter(5)
        else:
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()

        messages = []

        model.eval()
        for batch in dataloader:
            target, distractors, indices = batch
            
            vmd = torch.tensor(valid_meta_data[indices[:, 0], :], device=device, dtype=torch.int64)
            _, loss, acc, msg = model.forward(target, distractors, vmd)

            loss_meter.update(loss)
            acc_meter.update(acc)
            messages.append(msg)

        return (
            loss_meter,
            acc_meter,
            torch.cat(messages, 0)
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
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)