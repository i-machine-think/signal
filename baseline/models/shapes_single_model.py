import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from .shapes_meta_visual_module import ShapesMetaVisualModule
from .shapes_sender import ShapesSender

from helpers.utils_helper import UtilsHelper


class ShapesSingleModel(ShapesSender):
    def __init__(self, *args, **kwargs):
        kwargs["reset_params"] = False
        super().__init__(*args, **kwargs)

        self.utils_helper = UtilsHelper()

        self.output_module = ShapesMetaVisualModule(
            hidden_size=kwargs["hidden_size"],
            dataset_type=kwargs["dataset_type"],
            sender=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding, 0.0, 0.1)

        nn.init.constant_(self.linear_out.weight, 0)
        nn.init.constant_(self.linear_out.bias, 0)

        self.input_module.reset_parameters()
        self.output_module.reset_parameters()

        if type(self.rnn) is nn.LSTMCell:
            nn.init.xavier_uniform_(self.rnn.weight_ih)
            nn.init.orthogonal_(self.rnn.weight_hh)
            nn.init.constant_(self.rnn.bias_ih, val=0)
            # # cuDNN bias order: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
            # # add some positive bias for the forget gates [b_i, b_f, b_o, b_g] = [0, 1, 0, 0]
            nn.init.constant_(self.rnn.bias_hh, val=0)
            nn.init.constant_(
                self.rnn.bias_hh[self.hidden_size: 2 * self.hidden_size], val=1
            )

    def forward(self, hidden_state=None, messages=None, device=None, tau=1.2):
        """
        Merged version of Sender and Receiver
        """
        if device is None:
            # if torch.cuda.is_available() else "cpu")
            device = torch.device("cuda")

        if messages is None:
            hidden_state = self.input_module(hidden_state)
            state, batch_size = self._init_state(
                hidden_state, type(self.rnn), device)

            # Init output
            if self.training:
                output = [
                    torch.zeros(
                        (batch_size, self.vocab_size),
                        dtype=torch.float32,
                        device=device,
                    )
                ]
                output[0][:, self.sos_id] = 1.0
            else:
                output = [
                    torch.full(
                        (batch_size,),
                        fill_value=self.sos_id,
                        dtype=torch.int64,
                        device=device,
                    )
                ]

            # Keep track of sequence lengths
            initial_length = self.output_len + 1  # add the sos token
            seq_lengths = (
                torch.ones([batch_size], dtype=torch.int64, device=device)
                * initial_length
            )

            embeds = []  # keep track of the embedded sequence
            entropy = 0.0
            sentence_probability = torch.zeros(
                (batch_size, self.vocab_size), device=device
            )

            for i in range(self.output_len):
                if self.training:
                    emb = torch.matmul(output[-1], self.embedding)
                else:
                    emb = self.embedding[output[-1]]

                embeds.append(emb)
                state = self.rnn(emb, state)

                if type(self.rnn) is nn.LSTMCell:
                    h, c = state
                else:
                    h = state

                p = F.softmax(self.linear_out(h), dim=1)
                entropy += Categorical(p).entropy()

                if self.training:
                    token = self.utils_helper.calculate_gumbel_softmax(p, tau, hard=True)
                else:
                    sentence_probability += p.detach()
                    if self.greedy:
                        _, token = torch.max(p, -1)

                    else:
                        token = Categorical(p).sample()

                    if batch_size == 1:
                        token = token.unsqueeze(0)

                output.append(token)

                self._calculate_seq_len(
                    seq_lengths, token, initial_length, seq_pos=i + 1
                )

            return (
                torch.stack(output, dim=1),
                seq_lengths,
                torch.mean(entropy) / self.output_len,
                torch.stack(embeds, dim=1),
                sentence_probability,
            )

        else:
            batch_size = messages.shape[0]

            emb = (
                torch.matmul(messages, self.embedding)
                if self.training
                else self.embedding[messages]
            )

            # initialize hidden
            h = torch.zeros([batch_size, self.hidden_size], device=device)
            if self.cell_type == "lstm":
                c = torch.zeros([batch_size, self.hidden_size], device=device)
                h = (h, c)

            # make sequence_length be first dim
            seq_iterator = emb.transpose(0, 1)
            for w in seq_iterator:
                h = self.rnn(w, h)

            if self.cell_type == "lstm":
                h = h[0]  # keep only hidden state

            out = self.output_module(h)

            return out, emb
