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

    def get_final_state(self, x: torch.Tensor, h: torch.Tensor):
        x_ = x.to(self.wi.device)
        for i in range(x.shape[1]):
            h = (1 - self.alpha) * h + self.alpha*(
                x_[:, i] @ self.wi \
                + self.activation(h) @ self.wrec \
                + torch.randn(h.shape, device=self.wi.device) * self.noise
            )
        return h


class TwoAreaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoAreaRNN, self).__init__()

        # define network size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # define weights for area 1 (stimulus processing)
        self.wi_stim = nn.Parameter(
            torch.randn(3, hidden_size) / np.sqrt(3), requires_grad=True)  # first 3 inputs (fixation + 2 stimuli)
        self.wrec11 = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size), requires_grad=True)

        # define weights for area 2 (context processing)
        self.wi_ctx = nn.Parameter(
            torch.randn(2, hidden_size) / np.sqrt(2), requires_grad=True)  # last 2 inputs (context)
        self.wrec22 = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size), requires_grad=True)

        # define inter-area weights
        self.wrec12 = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size), requires_grad=True)
        self.wrec21 = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size), requires_grad=True)

        # output weights from area 2
        self.wo = nn.Parameter(
            torch.randn(hidden_size, output_size) / np.sqrt(hidden_size), requires_grad=True)

        # define hyperparameters
        self.activation = torch.tanh
        self.alpha1 = 0.1  # dt / tau for area 1
        self.alpha2 = 0.1  # dt / tau for area 2
        self.noise = 0.02

    def forward(self, x: torch.Tensor, return_hidden=False):
        # init network states for both areas
        h1 = torch.zeros(x.shape[0], self.hidden_size).to(self.wi_stim.device)
        h2 = torch.zeros(x.shape[0], self.hidden_size).to(self.wi_stim.device)
        out = torch.zeros(x.shape[0], x.shape[1], self.output_size).to(self.wo.device)

        if return_hidden:
            h1s = [h1.detach()]
            h2s = [h2.detach()]

        x_ = x.to(self.wi_stim.device)
        for i in range(x.shape[1]):
            # update area 1 (stimulus processing)
            h1 = (1 - self.alpha1) * h1 + self.alpha1 * (
                    x_[:, i, :3] @ self.wi_stim  # only stimulus-related inputs
                    + self.activation(h1) @ self.wrec11
                    + self.activation(h2) @ self.wrec12
                    + torch.randn(h1.shape, device=self.wi_stim.device) * self.noise
            )

            # update area 2 (context processing)
            h2 = (1 - self.alpha2) * h2 + self.alpha2 * (
                    x_[:, i, 3:] @ self.wi_ctx  # only context-related inputs
                    + self.activation(h1) @ self.wrec21
                    + self.activation(h2) @ self.wrec22
                    + torch.randn(h2.shape, device=self.wi_stim.device) * self.noise
            )

            if return_hidden:
                h1s.append(h1.detach())
                h2s.append(h2.detach())

            # output is based on area 2
            out[:, i] = self.activation(h2) @ self.wo

        if return_hidden:
            return out, (torch.stack(h1s, dim=1), torch.stack(h2s, dim=1))
        else:
            return out

    def get_final_state(self, x: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor):
        x_ = x.to(self.wi_stim.device)
        for i in range(x.shape[1]):
            h1 = (1 - self.alpha1) * h1 + self.alpha1 * (
                    x_[:, i, :3] @ self.wi_stim  # only stimulus-related inputs
                    + self.activation(h1) @ self.wrec11
                    + self.activation(h2) @ self.wrec12
                    + torch.randn(h1.shape, device=self.wi_stim.device) * self.noise
            )

            h2 = (1 - self.alpha2) * h2 + self.alpha2 * (
                    x_[:, i, 3:] @ self.wi_ctx  # only context-related inputs
                    + self.activation(h1) @ self.wrec21
                    + self.activation(h2) @ self.wrec22
                    + torch.randn(h2.shape, device=self.wi_stim.device) * self.noise
            )
        return h1, h2


if __name__ == '__main__':
    from data import gen_data
    input_size = 5
    hidden_size = 100
    output_size = 2
    model = SingleAreaRNN(5, 100, 2)
    x, y, metadata = gen_data(100)
    output, hs = model(torch.from_numpy(x), return_hidden=True)
