import torch
import torch.nn as nn
import torch.nn.functional as F


class DARTSCell(nn.Module):
    def __init__(self, ninp, nhid, genotype, init_range=0.04):
        super(DARTSCell, self).__init__()
        self.nhid = nhid
        self.genotype = genotype

        # In genotype is None when doing arch search
        steps = len(self.genotype.recurrent)
        self._W0 = nn.Parameter(
            torch.Tensor(ninp + nhid, 2 *
                         nhid).uniform_(-init_range, init_range)
        )
        self._Ws = nn.ParameterList(
            [
                nn.Parameter(
                    torch.Tensor(
                        nhid, 2 * nhid).uniform_(-init_range, init_range)
                )
                for i in range(steps)
            ]
        )

        # initialization
        nn.init.xavier_uniform_(self._W0)
        for p in self._Ws:
            nn.init.xavier_uniform_(p)

    def _get_activation(self, name):
        if name == "tanh":
            f = torch.tanh
        elif name == "relu":
            f = F.relu
        elif name == "sigmoid":
            f = torch.sigmoid
        elif name == "identity":
            def f(x): return x
        else:
            raise NotImplementedError
        return f

    def _compute_init_state(self, x, h_prev):
        xh_prev = torch.cat([x, h_prev], dim=-1)
        c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h_prev + c0 * (h0 - h_prev)
        return s0

    def forward(self, x, h_prev):
        s0 = self._compute_init_state(x, h_prev)

        states = [s0]
        for i, (name, pred) in enumerate(self.genotype.recurrent):
            s_prev = states[pred]
            ch = s_prev.mm(self._Ws[i])
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()
            fn = self._get_activation(name)
            h = fn(h)
            s = s_prev + c * (h - s_prev)
            states += [s]
        output = torch.mean(
            torch.stack([states[i] for i in self.genotype.concat], -1), -1
        )
        return output
