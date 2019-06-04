import torch
import torch.nn as nn

from .shapes_cnn import ShapesCNN

class ShapesTrainer(nn.Module):
    def __init__(self, sender, receiver, extract_features=False, device=None):
        super().__init__()

        self.sender = sender
        self.receiver = receiver

        self.extract_features = extract_features
        if extract_features:
            self.visual_module = ShapesCNN(sender.hidden_size)

        self.device = device
        if device is None:
            self.device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

    def _pad(self, messages, seq_lengths):
        """
        Pads the messages using the sequence length
        and the eos token stored in sender
        """
        batch_size, max_len = messages.shape[0], messages.shape[1]

        mask = torch.arange(max_len, device=self.device).expand(
            len(seq_lengths), max_len
        ) < seq_lengths.unsqueeze(1)

        if self.training:
            mask = mask.type(dtype=messages.dtype)
            messages = messages * mask.unsqueeze(2)
            # give full probability (1) to eos tag (used as padding in this case)
            messages[:, :, self.sender.eos_id] += (mask == 0).type(dtype=messages.dtype)
        else:
            # fill in the rest of message with eos
            messages = messages.masked_fill_(mask == 0, self.sender.eos_id)

        return messages

    def forward(self, target, distractors):
        batch_size = target.shape[0]

        target = target.to(self.device)
        distractors = [d.to(self.device) for d in distractors]

        if self.extract_features:
            target = self.visual_module(target)
            distractors = [self.visual_module(d) for d in distractors]

        messages, lengths, entropy, h_s, sent_p = self.sender(
            hidden_state=target, device=self.device
        )
        messages = self._pad(messages, lengths)
        r_transform, h_r = self.receiver(messages=messages, device=self.device)

        loss = 0

        target = target.view(batch_size, 1, -1)
        r_transform = r_transform.view(batch_size, -1, 1)

        target_score = torch.bmm(target, r_transform).squeeze()  # scalar

        all_scores = torch.zeros((batch_size, 1 + len(distractors)))
        all_scores[:, 0] = target_score

        for i, d in enumerate(distractors):
            d = d.view(batch_size, 1, -1)
            d_score = torch.bmm(d, r_transform).squeeze()
            all_scores[:, i + 1] = d_score
            loss += torch.max(
                torch.tensor(0.0, device=self.device), 1.0 - target_score + d_score
            )

        # Calculate accuracy
        all_scores = torch.exp(all_scores)
        _, max_idx = torch.max(all_scores, 1)

        accuracy = max_idx == 0
        accuracy = accuracy.to(dtype=torch.float32)

        if self.training:
            return torch.mean(loss), torch.mean(accuracy), messages
        else:
            return (
                torch.mean(loss),
                torch.mean(accuracy),
                messages,
                h_s,
                h_r,
                entropy,
                sent_p,
            )
