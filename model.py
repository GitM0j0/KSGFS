import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class shallow_network(nn.Module):

    def __init__(self):
        super(shallow_network, self).__init__()
        self.model = nn.Sequential(
                      nn.Linear(784, 400, bias=False),
                      nn.ReLU(),
                      nn.Linear(400, 400, bias=False),
                      nn.ReLU(),
                      nn.Linear(400, 400, bias=False),
                      nn.ReLU(),
                      nn.Linear(400,10, bias=False)
                    ).to(device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x

class network_projected(nn.Module):
    def __init__(self):
        super(network_projected, self).__init__()
        self.model = nn.Sequential(
                      nn.Linear(20, 30, bias=False),
                      nn.ReLU(),
                      nn.Linear(30, 30, bias=False),
                      nn.ReLU(),
                      nn.Linear(30, 10, bias=False)
                    ).to(device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x
