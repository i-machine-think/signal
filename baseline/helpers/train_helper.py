import random
import numpy as np
import torch

from metrics.average_meter import AverageMeter
from metrics.average_ensemble_meter import AverageEnsembleMeter

from models.shapes_trainer import ShapesTrainer

class TrainHelper():
    def __init__(self, device):
        self.device = device

    def train_one_batch(
        self,
        model: ShapesTrainer,
        batch,
        optimizer,
        meta_data,
        device,
        inference_step,
        multi_task):
        """
        Train for single batch
        """
        model.train()

        optimizer.zero_grad()

        target, distractors, indices, _ = batch
        # print('train',len(distractors))

        if inference_step or multi_task:
            md = torch.tensor(meta_data[indices[:,0], :], device=device, dtype=torch.int64)
        else:
            md = None
            
        loss, losses, accuracies, _ = model.forward(target, distractors, md)

        loss.backward()
        optimizer.step()

        return losses, accuracies

<<<<<<< HEAD
    def evaluate(self, model, dataloader, valid_meta_data, device, inference_step, step3 = False):
        
        if inference_step or step3:
=======
    def evaluate(self, model, dataloader, valid_meta_data, device, inference_step, multi_task):
        
        if multi_task:
            loss_meter = [AverageEnsembleMeter(5), AverageMeter()]
            acc_meter = [AverageEnsembleMeter(5), AverageMeter()]
        elif inference_step:
>>>>>>> 3d046c5a8c289a21d37370671c342b6ff4b3c92f
            loss_meter = AverageEnsembleMeter(5)
            acc_meter = AverageEnsembleMeter(5)
        # elif step3:
        #     loss_meter = AverageEnsembleMeter(5)#AverageMeter()
        #     acc_meter = AverageEnsembleMeter(5)
        else:
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()

        messages = []
        # batch_avg_accuracies = [0,0,0,0,0]
        # batch_avg_losses = [0,0,0,0,0]
        # batch_count = 0
        model.eval()
        for batch in dataloader:
<<<<<<< HEAD
            if step3:
                target, distractors, indices, lkey = batch
                # print('length', len(distractors))
                # print(len(distractors[0]),len(distractors[1]),len(distractors[2]),len(distractors[3]),len(distractors[4]))
            else:
                target, distractors, indices, _ = batch

            if inference_step:
=======
            target, distractors, indices = batch
            
            if inference_step or multi_task:
>>>>>>> 3d046c5a8c289a21d37370671c342b6ff4b3c92f
                vmd = torch.tensor(valid_meta_data[indices[:, 0], :], device=device, dtype=torch.int64)
            else:
                vmd = None

<<<<<<< HEAD
            # print('eval',len(distractors))

            # # if not step3:
            # n = np.random.randint(1)
            # if len(target) == 5:
            #     loss1, loss2, acc, msg = model.forward(target[n], distractors[n], vmd)
            #     lkey = torch.tensor(list(map(int, lkey)))

            #     lkey_stack = torch.stack([lkey == 0, lkey == 1, lkey == 2, lkey == 3, lkey == 4])
            #     class_acc = torch.sum(lkey_stack.float() * acc.float(), dim=1)/torch.sum(lkey_stack.float(),dim=1)
            #     class_acc = class_acc.numpy()
            #     acc = torch.mean(acc).item()
            #     acc = class_acc
            # else:
            loss1, loss2, acc, msg = model.forward(target, distractors, vmd)
            if step3:
                # The data is saved according to the following sequence [hp,vp,sh,co,si]
                # Thus it should be checked still, with the order in the print statements
                lkey = torch.tensor(list(map(int, lkey)))
                lkey_stack = torch.stack([lkey == 0, lkey == 1, lkey == 2, lkey == 3, lkey == 4])
                acc = (torch.sum(lkey_stack.float() * acc.float(), dim=1)/torch.sum(lkey_stack.float(),dim=1)).numpy()
                loss2 = (torch.sum(lkey_stack.float() * loss2.float(), dim=1)/torch.sum(lkey_stack.float(),dim=1)).detach().numpy()
                # class_acc = class_acc.numpy()
            elif not inference_step:
                acc = torch.mean(acc).item()
            # run for step3
            # calculations are done according to the five possible distractor sets
            # one for each class
            # else:
            #     loss2 = []
            #     acc = []
            #     for distractor in distractors:
            #         class_loss1, _, class_acc, msg = model.forward(target, distractor, vmd)
            #         loss2.append(class_loss1.item())
            #         acc.append(class_acc)
            #     loss2 = np.array(loss2)

            if inference_step or step3:
                loss_meter.update(loss2, crash=False)
            else:
                loss_meter.update(loss1)

            acc_meter.update(acc)
=======
            _, loss2, acc, msg = model.forward(target, distractors, vmd)

            if multi_task:
                loss_meter[0].update(loss2[0])
                loss_meter[1].update(loss2[1])

                acc_meter[0].update(acc[0])
                acc_meter[1].update(acc[1])
            else:
                loss_meter.update(loss2)
                acc_meter.update(acc)

>>>>>>> 3d046c5a8c289a21d37370671c342b6ff4b3c92f
            messages.append(msg)

            # batch_avg_accuracies += acc
            # batch_avg_losses += loss2
            # batch_count += 1
        # print('Batch avg acc.', batch_avg_accuracies/batch_count)
        # print('Batch avg loss', batch_avg_losses/batch_count)

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
        if params.inference_step:
            name += "_inference"
        if params.step3:
            name += "_step3"
        if params.multi_task:
            name += "_multi"
            if params.multi_task_lambda:
                name += f'_lambda_{params.multi_task_lambda}'
            
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