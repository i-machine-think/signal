import torch
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical


class UtilsHelper():
    def calculate_gumbel_softmax(self, probs, tau, hard):
        """ Computes sampling from the Gumbel Softmax (GS) distribution
        Args:
            probs (torch.tensor): probabilities of shape [batch_size, n_classes] 
            tau (float): temperature parameter for the GS
            hard (bool): discretize if True
        """

        rohc = RelaxedOneHotCategorical(tau, probs)
        y = rohc.rsample()

        if hard:
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y

        return y
        # return torch.argmax(y, dim=-1).long()