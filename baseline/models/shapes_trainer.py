import random

import torch
import torch.nn as nn

from .shapes_cnn import ShapesCNN

from .shapes_receiver import ShapesReceiver
from .shapes_sender import ShapesSender


class ShapesTrainer(nn.Module):
    def __init__(
            self,
            sender: ShapesSender,
            receiver: ShapesReceiver,
            device,
            inference_step,
            extract_features=False):
        super().__init__()

        self.sender = sender
        self.receiver = receiver

        self.extract_features = extract_features
        if extract_features:
            self.visual_module = ShapesCNN(sender.hidden_size)

        self.device = device

        self.inference_step = inference_step

    def _pad(self, messages, seq_lengths):
        """
        Pads the messages using the sequence length
        and the eos token stored in sender
        """
        _, max_len = messages.shape[0], messages.shape[1]

        mask = torch.arange(max_len, device=self.device).expand(len(seq_lengths), max_len) < seq_lengths.unsqueeze(1)

        if self.training and not self.inference_step:
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
        distractors = [d.to(self.device) for d in distractors]
        
        if self.extract_features:
            target = self.visual_module(target)
            distractors = [self.visual_module(d) for d in distractors]

        messages, lengths, _, _, _ = self.sender.forward(
            hidden_state=target)

        messages = self._pad(messages, lengths)

        if not self.receiver:
            return messages

        if self.inference_step:
            accuracies, losses = self.receiver.forward(messages, meta_data)
            return losses, accuracies, messages
        else:
            r_transform, _ = self.receiver.forward(messages=messages)

            loss = 0

            target = target.view(batch_size, 1, -1)
            r_transform = r_transform.view(batch_size, -1, 1)

            target_score = torch.bmm(target, r_transform).squeeze()  # scalar

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
                loss += torch.max(
                    torch.tensor(0.0, device=self.device), 1.0 -
                    target_score + d_score
                )
                i += 1

            # Calculate accuracy
            all_scores = torch.exp(all_scores)
            _, max_idx = torch.max(all_scores, 1)

            accuracy = max_idx == target_index
            accuracy = accuracy.to(dtype=torch.float32)

            # if self.training:
            return torch.mean(loss), torch.mean(accuracy).item(), messages
            # else:
            #     #########################################
            #     ############ DIAGNOSTIC CODE ############
            #     #########################################
            #     # losses = []
            #     # meta_predict_exp = torch.exp(r_transform).squeeze()
            #     # for m, n in enumerate(range(0, 15, 3)):
            #     #     _, max_idx_pred = torch.max(meta_predict_exp[:, n:n+3], 1)
            #     #     _, max_idx_target = torch.max(target.squeeze()[:, n:n+3], 1)
            #     #     losses.append(
            #     #         torch.eq(max_idx_target, max_idx_pred).double().mean())
            #     # #########################################

            #     return (
            #         torch.mean(loss),
            #         torch.mean(accuracy),
            #         messages,
            #         h_s,
            #         h_r,
            #         entropy,
            #         sent_p,
            #         losses,
            #         max_idx,
            #     )
