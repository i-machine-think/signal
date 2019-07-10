import random

import torch
import torch.nn as nn

from .shapes_cnn import ShapesCNN

from .shapes_receiver import ShapesReceiver
from .messages_receiver import MessagesReceiver
from .shapes_sender import ShapesSender

import numpy as np

class ShapesTrainer(nn.Module):
    def __init__(
            self,
            sender: ShapesSender,
            device,
            inference_step,
            multi_task,
            multi_task_lambda,
            step3,
            baseline_receiver: ShapesReceiver = None,
            diagnostic_receiver: MessagesReceiver = None,
            extract_features=False,
            vqvae=False,
            beta=0.25):
        super().__init__()

        self.sender = sender
        self.baseline_receiver = baseline_receiver
        self.diagnostic_receiver = diagnostic_receiver

        self.extract_features = extract_features
        if extract_features:
            self.visual_module = ShapesCNN(sender.hidden_size)

        self.device = device
        self.vqvae = vqvae
        self.inference_step = inference_step
        self.step3 = step3
        self.multi_task = multi_task
        self.multi_task_lambda = multi_task_lambda
        self.vqvae = vqvae

    def _pad(self, messages, seq_lengths):
        """
        Pads the messages using the sequence length
        and the eos token stored in sender
        """
        _, max_len = messages.shape[0], messages.shape[1]

        mask = torch.arange(max_len, device=self.device).expand(len(seq_lengths), max_len) < seq_lengths.unsqueeze(1)

        if self.training:
            mask = mask.type(dtype=messages.dtype)
            messages = messages * mask.unsqueeze(2)

            # give full probability (1) to eos tag (used as padding in this case)
            messages[:, :, self.sender.eos_id] += (mask == 0).type(dtype=messages.dtype)
        else:
            # fill in the rest of message with eos
            messages = messages.masked_fill_(mask == 0, self.sender.eos_id)

        return messages

    def forward(self, target, distractors, meta_data = None):
        batch_size = target.shape[0]

        target = target.to(self.device)
        distractors = [d.to(self.device) for d in distractors] # There is a list of distractors

        if self.extract_features:
            target = self.visual_module(target)  # This is the "f" function in the paper! No eta exists.
            distractors = [self.visual_module(d) for d in distractors]

        messages, lengths, _, _, _, loss_2_3 = self.sender.forward(
            hidden_state=target) # The first hidden state is the target, as in Referential Games paper.

        if not self.vqvae:
            messages = self._pad(messages, lengths) # If I understand correctly: After eos happens the first time, all later words in message are eos as well.

        if not self.diagnostic_receiver and not self.baseline_receiver:
            return messages

        final_loss = 0
        if self.inference_step or self.multi_task:
            out = self.diagnostic_receiver.forward(messages, meta_data)

            loss = 0

            inference_accuracies = np.zeros((len(out),))
            inference_losses = np.zeros((len(out),))

            for i, out_property in enumerate(out):
                current_targets = meta_data[:, i]

                current_loss = nn.functional.cross_entropy(out_property, current_targets)

                loss += current_loss
                inference_losses[i] = current_loss.item()
                inference_accuracies[i] = torch.mean((torch.argmax(out_property, dim=1) == current_targets).float()).item()

            if not self.multi_task:
                return loss, inference_losses, inference_accuracies, messages

            final_loss = self.multi_task_lambda * loss

        if not self.inference_step or self.multi_task:
            r_transform, _ = self.baseline_receiver.forward(messages=messages) # r_transform is the last hidden receiver state, which is then processed by some g (eta inverse), which here probably is the identity...

            baseline_loss = 0

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
                baseline_loss += torch.max(
                    torch.tensor(0.0, device=self.device), 1.0 -
                    target_score + d_score
                ) # This creates the sum in equation (1) of the original paper!
                i += 1

            # Calculate accuracy
            all_scores = torch.exp(all_scores)
            _, max_idx = torch.max(all_scores, 1)

            accuracy = max_idx == target_index # model liegt "richtig", wenn die dot-product similarity zwischen target und Ergebnis größer ist als die mit allen Distractors.
            accuracy = accuracy.to(dtype=torch.float32)

            # print(type(torch.mean(baseline_loss)), type(baseline_loss), type(accuracy))
            # print((torch.mean(baseline_loss).shape), (baseline_loss.shape), (accuracy.shape))
            if self.step3:
                return torch.mean(baseline_loss), baseline_loss, accuracy, messages

            baseline_accuracy = torch.mean(accuracy).item()
            baseline_mean_loss = torch.mean(baseline_loss) # without item, since it will be backpropagated

            if self.vqvae:
                # In the vqvae case, we add loss_2_3
                baseline_mean_loss += loss_2_3

            baseline_loss = baseline_mean_loss.item()

            if not self.multi_task:
                return baseline_mean_loss, baseline_loss, baseline_accuracy, messages

            final_loss += (1 - self.multi_task_lambda) * baseline_mean_loss

            return final_loss, (inference_losses, baseline_loss), (inference_accuracies, baseline_accuracy), messages
