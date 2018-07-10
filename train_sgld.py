import torch
import torchvision
import os

from itertools import chain
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
#from numpy.random import RandomState

#import sgld
from sgld import model, optim

import mnist

### Utils

#def log_gaussian(x, precision):
#    # Normal distribution N(0, 1/precision)
#    return -0.5 * (torch.log(torch.tensor(2.) * np.pi / precision) + precision * x.pow(2))
#
#def gaussian_prior(x, precision):
#    res = 0
#    for p in x:
#        res += torch.sum(log_gaussian(p, precision))
#    return res


###


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

batch_size = 500
dataset_size = 60000
#lr = 1e-5
#lr_decayEpoch = 20

weight_decay = 0.001
a =  1.
b = 1e04
gamma = 0.55

#precision = 1.

train_loader, test_loader = mnist.get_mnist()

kind = "classifier"
network = model.shallow_network()
criterion = nn.CrossEntropyLoss(size_average=True)

#optim = sgld_alt.optim.sgld(network, lr, weight_decay, lr_decayEpoch, batch_size, dataset_size)
optim = optim.sgld(network, a, b, gamma, weight_decay, batch_size, dataset_size)

for epoch in range(10):
    running_loss = 0
    for x, y in iter(train_loader):
        x = x.view(x.size(0), -1)
        target = y if kind == "classifier" else x

        network.zero_grad()
        output = network(x)
        #log_likelihood = criterion(output, target)
        #log_prior = sum([torch.sum(gaussian_prior(p, precision)) for p in network.parameters()])
        #loss = log_prior - train_data.train_data.size(0) / batch_size * log_likelihood
        loss = criterion(output, target)
        loss.backward()
        optim.step(epoch)

        running_loss += loss * batch_size / train_data.train_data.size(0)
        #running_loss += loss
        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = torch.sum(prediction.eq(target)).float()/batch_size

    print("Epoch {:d} - loss: {:.4f} - acc: {:.4f}".format(epoch, running_loss, accuracy))

    with torch.autograd.no_grad():
        test_metric = 0
        for x, y in iter(test_loader):
            x = x.view(x.size(0), -1)
            output = network(x)
            if kind == "classifier":
                test_metric += 100 * (output.argmax(1) == y).float().sum() / test_data.test_data.size(0)
            else:
                raise ValueError("Unreachable")
        print("\ttest: ", test_metric)
