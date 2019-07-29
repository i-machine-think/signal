import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

# this utils function is taken from https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/23
def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


class HardMax(torch.autograd.Function):
    """
    Takes a softmax vector and returns the hard max.
    With straight-through gradient.
    """

    @staticmethod
    def forward(ctx, softmax, max_indices, n_dims):

        _, max_indices[:] = torch.max(softmax, dim=1)
        hard_max = to_one_hot(torch.Tensor(max_indices), n_dims)

        return hard_max

    @staticmethod
    def backward(ctx, grad_hard_max):
        return grad_hard_max, None, None


class VectorQuantization(torch.autograd.Function):
    """
    A function that compares the input of the forward pass to the embedding table.
    returns the closest embedding vector.
    Backward pass is straight-through.
    Inspired by VQ-VAE (van den Oord et al., 2018).
    Implementation adapted from https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/functions.py.
    """

    @staticmethod
    def forward(ctx, pre_quant, e, indices):
        distance_computer = EmbeddingtableDistances(e)

        distances = distance_computer.forward(pre_quant)
        _, indices[:] = torch.min(
            distances, dim=1
        )  # indices lists, for each vector in the batch pre_quant, the index of the closest codeword

        return e[indices]

    @staticmethod
    def backward(ctx, grad_e):
        return grad_e, None, None


class EmbeddingtableDistances(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.register_buffer("e", e)  # The embedding table from VQ-VAE

    def forward(self, pre_quant):
        # Use ||a - b||^2 = ||a||^2 + ||b||^2 - 2ab for computation of distances.
        # Square computation:
        e_sq = torch.sum(self.e ** 2, dim=1)
        pre_quant_sq = torch.sum(pre_quant ** 2, dim=1, keepdim=True)

        # Compute the distances to the codebook e
        distances = torch.addmm(
            e_sq + pre_quant_sq, pre_quant, self.e.t(), alpha=-2.0, beta=1.0
        )
        return distances
