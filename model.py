import numpy as np
import torch
from torch import nn

class SingleAreaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SingleAreaRNN, self).__init__()

        # define network size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # define weights
        self.wi = nn.Parameter(
            torch.randn(input_size, hidden_size)/np.sqrt(input_size), requires_grad=True)
        self.wrec = nn.Parameter(
            torch.randn(hidden_size, hidden_size)/np.sqrt(hidden_size), requires_grad=True)
        self.wo = nn.Parameter(
            torch.randn(hidden_size, output_size)/np.sqrt(hidden_size), requires_grad=True)

        # define hyperparameters
        self.activation = torch.tanh
        self.alpha = 0.1    # dt / tau
        self.noise = 0.02

    def forward(self, x: torch.Tensor, return_hidden=False):
        # init network states
        h = torch.zeros(x.shape[0], self.hidden_size
                        ).to(self.wi.device)
        out = torch.zeros(x.shape[0], x.shape[1], self.output_size
                          ).to(self.wo.device)

        if return_hidden:
            hs = [h.detach()]
        x_ = x.to(self.wi.device)
        for i in range(x.shape[1]):
            h = (1 - self.alpha) * h + self.alpha*(
                x_[:, i] @ self.wi \
                + self.activation(h) @ self.wrec \
                + torch.randn(h.shape, device=self.wi.device) * self.noise
            )
            if return_hidden:
                hs.append(h.detach())
            out[:,i] = self.activation(h) @ self.wo
        if return_hidden:
            return out, torch.stack(hs, dim=1)
        else:
            return out

class TwoAreaRNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: torch.Tensor, return_hidden=False):
        raise NotImplementedError('Not implemented yet')

if __name__ == '__main__':
    from data import gen_data
    input_size = 5
    hidden_size = 100
    output_size = 2
    model = SingleAreaRNN(5, 100, 2)
    x, y, metadata = gen_data(100)
    output, hs = model(torch.from_numpy(x), return_hidden=True)
