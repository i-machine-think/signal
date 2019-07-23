import random

import torch
import torch.nn as nn

from .cnn import CNN

from .receiver import Receiver
from .sender import Sender

import numpy as np

class FullModel(nn.Module):
    def __init__(
            self,
            sender: Sender,
            device,
            receiver: Receiver = None,
            diagnostic_receiver= None,
            extract_features=False,
            vqvae=False,
            rl=False,
            entropy_coefficient=1.0,
            myopic=False,
            myopic_coefficient=0.1):
        super().__init__()

        self.sender = sender
        self.receiver = receiver
        self.diagnostic_receiver = diagnostic_receiver

        self.extract_features = extract_features
        if extract_features:
            self.visual_module = CNN(sender.hidden_size)

        self.device = device
        self.output_len = self.sender.output_len
        self.vqvae = vqvae
        self.rl = rl
        self.entropy_coefficient = entropy_coefficient
        self.n_baseline_updates = 0
        self.hinge_loss_baseline = 0
        self.myopic = myopic
        self.myopic_coefficient = myopic_coefficient


    def _pad(self, messages, seq_lengths):
        """
        Pads the messages using the sequence length
        and the eos token stored in sender
        """
        _, max_len = messages.shape[0], messages.shape[1]

        mask = torch.arange(max_len, device=self.device).expand(len(seq_lengths), max_len) < seq_lengths.unsqueeze(1)

        mask = mask.type(dtype=messages.dtype)
        messages = messages * mask.unsqueeze(2)

        # give full probability (1) to eos tag (used as padding in this case)
        messages[:, :, self.sender.eos_id] += (mask == 0).type(dtype=messages.dtype)

        return messages

    def update_baseline(self, value):
        # Compute the mean of the hinge losses seen so far.
        # Acts as a baseline for stabilizing RL.
        if self.n_baseline_updates==0:
            self.n_baseline_updates+=1
            self.hinge_loss_baseline = value.detach().item()
        else:
            if self.myopic:
                self.hinge_loss_baseline = (1-self.myopic_coefficient)*self.hinge_loss_baseline + self.myopic_coefficient*value.detach().item()
            else:
                self.n_baseline_updates += 1
                self.hinge_loss_baseline += (value.detach().item() - self.hinge_loss_baseline) / self.n_baseline_updates

    def forward(self, target, distractors, meta_data = None):
        batch_size = target.shape[0]

        target = target.to(self.device)
        distractors = [d.to(self.device) for d in distractors] # There is a list of distractors

        if self.extract_features:
            target = self.visual_module(target)  # This is the "f" function in the paper! No eta exists.
            distractors = [self.visual_module(d) for d in distractors]

        messages, lengths, entropy, _, _, loss_2_3, message_logits = self.sender.forward(
            hidden_state=target) # The first hidden state is the target, as in Referential Games paper.

        if not self.vqvae and not self.rl:
            messages = self._pad(messages, lengths) # If I understand correctly: After eos happens the first time, all later words in message are eos as well.

        if not self.diagnostic_receiver and not self.receiver:
            return messages

        final_loss = 0

        r_transform, _ = self.receiver.forward(messages=messages) # r_transform is the last hidden receiver state, which is then processed by some g (eta inverse), which here probably is the identity...

        hinge_loss = 0

        target = target.view(batch_size, 1, -1)
        r_transform = r_transform.view(batch_size, -1, 1)

        target_score = torch.bmm(target, r_transform).squeeze()  # scalars (over batch). Does a batch matrix multiplication

        all_scores = torch.zeros((batch_size, 1 + len(distractors)))

        target_index = 0
        all_scores[:, target_index] = target_score

        i = 0
        for d in distractors:
            if i == target_index:
                i += 1

            d = d.view(batch_size, 1, -1)
            d_score = torch.bmm(d, r_transform).squeeze()
            all_scores[:, i] = d_score
            hinge_loss += torch.max(
                torch.tensor(0.0, device=self.device), 1.0 -
                target_score + d_score
            ) # This creates the sum in equation (1) of the original paper!
            i += 1

        # Calculate accuracy
        all_scores = torch.exp(all_scores)
        _, max_idx = torch.max(all_scores, 1)

        accuracy = max_idx == target_index # model liegt "richtig", wenn die dot-product similarity zwischen target und Ergebnis größer ist als die mit allen Distractors.
        accuracy = accuracy.to(dtype=torch.float32)

        # print(type(torch.mean(hinge_loss)), type(hinge_loss), type(accuracy))
        # print((torch.mean(hinge_loss).shape), (hinge_loss.shape), (accuracy.shape))

        accuracy_mean = torch.mean(accuracy).item()
        hinge_mean_loss = torch.mean(hinge_loss) # without item, since it will be backpropagated

        if self.vqvae:
            # In the vqvae case, we add loss_2_3
            hinge_mean_loss += loss_2_3

        if self.rl:
            logit = torch.sum(message_logits, dim=1)
            entropy_mean = torch.mean(torch.sum(entropy, dim=1) / self.output_len)
            self.update_baseline(hinge_mean_loss)
            rl_mean_loss = torch.mean((hinge_loss.detach() - self.hinge_loss_baseline) * logit)
            final_loss = hinge_mean_loss + rl_mean_loss - self.entropy_coefficient*entropy_mean
            return final_loss, (final_loss.item(), hinge_mean_loss.item(), rl_mean_loss.item(), entropy_mean.item()), accuracy_mean, messages

        hinge_mean_loss_item = hinge_mean_loss.item()

        return hinge_mean_loss, hinge_mean_loss_item, accuracy_mean, messages
