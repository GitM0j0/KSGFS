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
from sgld import optim

import model

import mnist


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


#lr = 1e-5
#lr_decayEpoch = 20
batch_size = 500
num_workers = 5

lambda_ = 0.001
a =  1.
b = 100.
gamma = 0.55


train_loader, test_loader = mnist.get_mnist(batch_size, num_workers)
dataset_size = 60000


network = model.shallow_network()
criterion = nn.CrossEntropyLoss(size_average=True)

#optim = sgld_alt.optim.sgld(network, lr, lambda_, lr_decayEpoch, batch_size, dataset_size)
optim = optim.sgld(network, a, b, gamma, lambda_, batch_size, dataset_size)

for epoch in range(10):
    running_loss = 0
    for x, y in iter(train_loader):
        x = x.view(x.size(0), -1)

        network.zero_grad()
        output = network(x)
        loss = criterion(output, y)
        loss.backward()
        optim.step()

        # TO DO: update
        running_loss += loss * batch_size / dataset_size
        prediction = output.data.max(1)[1]
        accuracy = torch.sum(prediction.eq(y)).float()/batch_size

    print("Epoch {:d} - loss: {:.4f} - acc: {:.4f}".format(epoch, running_loss, accuracy))

    with torch.autograd.no_grad():
        test_metric = 0
        for x, y in iter(test_loader):
            x = x.view(x.size(0), -1)
            output = network(x)
            test_metric += 100 * (output.argmax(1) == y).float().sum() / 10000
            prediction = output.data.max(1)[1]
            accuracy = torch.sum(prediction.eq(y)).float()/1000

        print("\ttest: {:.4}  - acc: {:.4}".format( test_metric, accuracy))
