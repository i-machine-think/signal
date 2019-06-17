import random
import numpy as np
import torch

from metrics.average_meter import AverageMeter

#########################################
############ DIAGNOSTIC CODE ############
#########################################
PROPERTIES = ['Shape', 'Color', 'Size ', 'Pos_h', 'Pos_w']
#########################################

class TrainHelper():
    def __init__(self, device):
        self.device = device

    def train_one_batch(self, model, batch, optimizer, meta_data):
        """
        Train for single batch
        """
        model.train()
        optimizer.zero_grad()

        target, distractors, indices = batch

        # randomly selects class to update onto
        c = np.random.randint(meta_data.shape[1]) # class_property

        # losses = []
        # accuracies = []
        # for c in range(meta_data.shape[1]):
        md = torch.tensor(meta_data[indices[:,0],c])
        loss, acc, _ = model(target, distractors, md)
        # losses.append(loss)
        # accuracies.append(acc)

        loss.backward()
        optimizer.step()

        return loss.item(), acc.item()

    def evaluate(self, model, data, valid_meta_data, return_softmax=False):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        entropy_meter = AverageMeter()
        hidden_sender, hidden_receiver = [], []
        messages, sentence_probabilities = [], []

        #########################################
        ############ DIAGNOSTIC CODE ############
        #########################################
        # class_loss_meters = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
        #########################################

        c = 0 # for now only checks first class_property
        model.eval()
        for d in data:
            # NOTE, len==3 used to be 2, but due to diagnostic indices it is 3
            if len(d) == 3:  # shapes
                target, distractors, indices = d
                #########################################
                ############ DIAGNOSTIC CODE ############
                #########################################
                vmd = torch.tensor(valid_meta_data[indices[:,0],c]).long()
                loss, acc, msg = model(target, distractors, vmd) #, h_s, h_r, entropy, sent_p, class_losses, max_idx = model(target, distractors)
                #########################################

                #########################################
                ############ DIAGNOSTIC CODE ############
                #########################################
                # print('\nValid accuracy', torch.mean(acc).item())
                # predicted_indices = torch.stack([ind[max_idx[i].item()] for i, ind in enumerate(indices)])

                # pred_metas = valid_meta_data[predicted_indices]
                # true_metas = valid_meta_data[indices[:,0]]

                # pred_chunks = np.hsplit(pred_metas,5)
                # true_chunks = np.hsplit(true_metas,5)

                # for i, p in enumerate(PROPERTIES):
                #     prop_acc = np.sum(pred_chunks[i] == true_chunks[i], axis=1)
                    # print(p, 'accuracy', np.mean(np.where(prop_acc == 3, 1, 0)))
                #########################################

            # if len(d) == 3:  # obverter task
            #     first_image, second_image, label = d
            #     loss, acc, msg, h_s, h_r, entropy, sent_p = model(
            #         first_image, second_image, label
            #     )

            loss_meter.update(loss.item())
            acc_meter.update(acc.item())
            # entropy_meter.update(entropy.item())

            #########################################
            ############ DIAGNOSTIC CODE ############
            #########################################
            # for i, c in enumerate(class_loss_meters):
            #     c.update(class_losses[i].item())
            #########################################

            messages.append(msg)
            # sentence_probabilities.append(sent_p)
            # hidden_sender.append(h_s.detach().cpu().numpy())
            # hidden_receiver.append(h_r.detach().cpu().numpy())

        # hidden_sender = np.concatenate(hidden_sender)
        # hidden_receiver = np.concatenate(hidden_receiver)

        #########################################
        ############ DIAGNOSTIC CODE ############
        #########################################
        # classes = ['color', 'shape', 'size', 'pos_h', 'pos_w']
        # print()
        # for i,c in enumerate(classes):
        #     print(
        #         "Class {}: val accuracy: {}".format(
        #             c,
        #             class_loss_meters[i].avg,
        #         )
        #     )
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