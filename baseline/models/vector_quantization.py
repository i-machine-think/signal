import torch
import torch.nn as nn


class VectorQuantization(torch.autograd.Function):
    """
    A function that compares the input of the forward pass to the embedding table.
    returns the closest embedding vector.
    Furthermore returns a loss-term: ||sg(pre_quant) - e||^2 + beta*||pre_quant - sg(e)||^2
    Backward pass is straight-through.
    Inspired by VQ-VAE (van den Oord et al., 2018).
    Implementation adapted from https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/functions.py.
    """
    @staticmethod
    def forward(ctx, pre_quant, e, beta, indices):
        # Use ||a - b||^2 = ||a||^2 + ||b||^2 - 2ab for computation of distances.
        # Square computation:
        e_sq = torch.sum(e ** 2, dim=1)
        pre_quant_sq = torch.sum(pre_quant ** 2, dim=1, keepdim=True)

        # Compute the distances to the codebook e
        distances = torch.addmm(e_sq + pre_quant_sq,
            pre_quant, e.t(), alpha=-2.0, beta=1.0)

        _, indices[:] = torch.min(distances, dim=1) # indices lists, for each vector in the batch pre_quant, the index of the closest codeword

        return e[indices]

    @staticmethod
    def backward(ctx, grad_e):
        return grad_e, None, None, None
