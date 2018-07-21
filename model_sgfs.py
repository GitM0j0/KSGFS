import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

class mlp(nn.Module):
    """
    "Standard" MLP with support with goodfellow's backprop trick
    """
    def __init__(self, architecture, nl=F.relu):
        super(mlp, self).__init__()

        self.nl = nl
        self.hidden_layers = nn.ModuleList([nn.Linear(architecture[i-1], architecture[i], bias=False) for i in range(1, len(architecture) - 1)])
        #nonlinearities = nn.ModuleList([nn.ReLU() for _ in self.hidden_layers])
        #interleaved = chain.from_iterable(zip(self.hidden_layers, nonlinearities))
        self.output_layer = nn.Linear(architecture[-2], architecture[-1], bias=False)


    def forward(self, x):
        """
        Forward pass that returns also returns
        * the activations (H) and
        * the linear combinations (Z)
        of each layer, to be able to use the trick from [1].

        Args:
            - x : The inputs of the network
        Returns:
            - logits
            - activations at each layer (including the inputs)
            - linear combinations at each layer
        > [1] EFFICIENT PER-EXAMPLE GRADIENT COMPUTATIONS
        > by Ian Goodfellow
        > https://arxiv.org/pdf/1510.01799.pdf
        """
        x = x.view(x.size(0), -1)

        # Save the model inputs, which are considered the activations of the 0'th layer.
        activations = [x]
        preactivations = []

        for l in self.hidden_layers:
            preactivation = l(x)
            x = self.nl(preactivation)

            # Save the activations and linear combinations from this layer.
            activations.append(x)
            preactivation.retain_grad()
            preactivation.requires_grad_(True)
            preactivations.append(preactivation)

        logits = self.output_layer(x)

        logits.retain_grad()
        logits.requires_grad_(True)
        preactivations.append(logits)

        return logits, activations, preactivations
