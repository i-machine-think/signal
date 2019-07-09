import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

from .shapes_meta_visual_module import ShapesMetaVisualModule
from .darts_cell import DARTSCell
from .vector_quantization import VectorQuantization

from helpers.utils_helper import UtilsHelper

class ShapesSender(nn.Module):
    def __init__(
        self,
        vocab_size, # Specifies number of words in baseline setting. In VQ-VAE Setting: Dimension of embedding space.
        output_len, # called max_length in other files
        sos_id,
        device,
        eos_id=None,
        embedding_size=256,
        hidden_size=512,
        greedy=False,
        cell_type="lstm",
        genotype=None,
        dataset_type="meta",
        reset_params=True,
        inference_step=False,
        vqvae=False, # If True, use VQ instead of Gumbel Softmax
        discrete_latent_number=25 # Number of embedding vectors e_i in embedding table in vqvae setting
        ):

        super().__init__()
        self.vocab_size = vocab_size
        self.cell_type = cell_type
        self.output_len = output_len
        self.sos_id = sos_id
        self.utils_helper = UtilsHelper()
        self.device = device

        if eos_id is None:
            self.eos_id = sos_id
        else:
            self.eos_id = eos_id

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.greedy = greedy
        self.inference_step = inference_step

        if cell_type == "lstm":
            self.rnn = nn.LSTMCell(embedding_size, hidden_size)
        elif cell_type == "darts":
            self.rnn = DARTSCell(embedding_size, hidden_size, genotype)
        else:
            raise ValueError(
                "ShapesSender case with cell_type '{}' is undefined".format(cell_type)
            )

        self.embedding = nn.Parameter(
            torch.empty((vocab_size, embedding_size), dtype=torch.float32)
        )

        self.linear_out = nn.Linear(hidden_size, vocab_size) # from a hidden state to the vocab
        self.vqvae = vqvae

        if reset_params:
            self.reset_parameters()

        if self.vqvae:
            self.discrete_latent_number = discrete_latent_number

            self.e = nn.Parameter(
                torch.empty((self.discrete_latent_number, self.vocab_size), dtype=torch.float32)
            ) # The discrete embedding table
            self.vq = VectorQuantization()

    def reset_parameters(self):
        nn.init.normal_(self.embedding, 0.0, 0.1)
        nn.init.constant_(self.linear_out.weight, 0)
        nn.init.constant_(self.linear_out.bias, 0)
        if self.vqvae:
            nn.init.normal_(self.e, 0.0, 0.1)

        # self.input_module.reset_parameters()

        if type(self.rnn) is nn.LSTMCell:
            nn.init.xavier_uniform_(self.rnn.weight_ih)
            nn.init.orthogonal_(self.rnn.weight_hh)
            nn.init.constant_(self.rnn.bias_ih, val=0)
            # # cuDNN bias order: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
            # # add some positive bias for the forget gates [b_i, b_f, b_o, b_g] = [0, 1, 0, 0]
            nn.init.constant_(self.rnn.bias_hh, val=0)
            nn.init.constant_(
                self.rnn.bias_hh[self.hidden_size : 2 * self.hidden_size], val=1
            )

    def _init_state(self, hidden_state, rnn_type):
        """
            Handles the initialization of the first hidden state of the decoder.
            Hidden state + cell state in the case of an LSTM cell or
            only hidden state in the case of a GRU cell.
            Args:
                hidden_state (torch.tensor): The state to initialize the decoding with.
                rnn_type (type): Type of the rnn cell.
            Returns:
                state: (h, c) if LSTM cell, h if GRU cell
                batch_size: Based on the given hidden_state if not None, 1 otherwise
        """

        # h0
        if hidden_state is None:
            batch_size = 1
            h = torch.zeros([batch_size, self.hidden_size], device=self.device)
        else:
            batch_size = hidden_state.shape[0]
            h = hidden_state  # batch_size, hidden_size

        # c0
        if rnn_type is nn.LSTMCell:
            c = torch.zeros([batch_size, self.hidden_size], device=self.device)
            state = (h, c)
        else:
            state = h

        return state, batch_size

    def _calculate_seq_len(self, seq_lengths, token, initial_length, seq_pos):
        """
            Calculates the lengths of each sequence in the batch in-place.
            The length goes from the start of the sequece up until the eos_id is predicted.
            If it is not predicted, then the length is output_len + n_sos_symbols.
            Args:
                seq_lengths (torch.tensor): To keep track of the sequence lengths.
                token (torch.tensor): Batch of predicted tokens at this timestep.
                initial_length (int): The max possible sequence length (output_len + n_sos_symbols).
                seq_pos (int): The current timestep.
        """
        if self.training:
            max_predicted, vocab_index = torch.max(token, dim=1)
            mask = (vocab_index == self.eos_id) * (max_predicted == 1.0) # all words in batch that are "already done"
        else:
            mask = token == self.eos_id

        mask *= seq_lengths == initial_length
        import pdb; pdb.set_trace()
        seq_lengths[mask.nonzero()] = seq_pos + 1  # start always token appended. This tells the sequence to be smaller at the positions where the sentence already ended.

    def forward(self, tau=1.2, hidden_state=None):
        """
        Performs a forward pass. If training, use Gumbel Softmax (hard) for sampling, else use
        discrete sampling.
        Hidden state here represents the encoded image/metadata - initializes the RNN from it.
        """

        # hidden_state = self.input_module(hidden_state)
        state, batch_size = self._init_state(hidden_state, type(self.rnn))

        # Init output
        if not self.vqvae:
            if self.training:
                output = [ torch.zeros((batch_size, self.vocab_size), dtype=torch.float32, device=self.device)]
                output[0][:, self.sos_id] = 1.0
            else:
                output = [
                    torch.full(
                        (batch_size,),
                        fill_value=self.sos_id,
                        dtype=torch.int64,
                        device=self.device,
                    )
                ]
        else:
            # In vqvae case, there is no sos symbol, since all words come from the unordered embedding table.
            output = [ torch.zeros((batch_size, self.vocab_size), dtype=torch.float32, device=self.device)]

        # Keep track of sequence lengths
        initial_length = self.output_len + 1  # add the sos token
        seq_lengths = (
            torch.ones([batch_size], dtype=torch.int64, device=self.device) * initial_length
        ) # [initial_length, initial_length, ..., initial_length]. This gets reduced whenever it ends somewhere.

        embeds = []  # keep track of the embedded sequence
        entropy = 0.0
        sentence_probability = torch.zeros((batch_size, self.vocab_size), device=self.device)

        for i in range(self.output_len):
            if self.training or self.vqvae:
                emb = torch.matmul(output[-1], self.embedding)
            else:
                emb = self.embedding[output[-1]]

            embeds.append(emb)

            state = self.rnn.forward(emb, state)

            if type(self.rnn) is nn.LSTMCell:
                h, _ = state
            else:
                h = state

            if not self.vqvae:
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
                self._calculate_seq_len(seq_lengths, token, initial_length, seq_pos=i + 1)
            else:
                pre_quant = self.linear_out(h)
                token = self.vq(pre_quant, self.e)

            output.append(token)

        messages = torch.stack(output, dim=1)
        entropy_out = torch.mean(entropy) / self.output_len


        return (
            messages,
            seq_lengths,
            entropy_out,
            torch.stack(embeds, dim=1),
            sentence_probability,
        )
