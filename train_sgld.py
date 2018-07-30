import torch
import torchvision
import os

from itertools import chain
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

from sgld import optim

import model

import mnist


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Model parameter
lambda_ = 1e-3
lr = 5e-1

batch_size = 500
dataset_size=60000


train_data = torchvision.datasets.MNIST(root=os.environ.get("DATASETS_PATH", "~/datasets"), train=True,
                                         download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=5)

test_data = torchvision.datasets.MNIST(root=os.environ.get("DATASETS_PATH", "~/datasets"), train=False,
                                        download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)


network = model.shallow_network()
criterion = nn.CrossEntropyLoss(size_average=True)

#optim = sgld_alt.optim.sgld(network, lr, lambda_, lr_decayEpoch, batch_size, dataset_size)
#optim = optim.sgld(network, a, b, gamma, lambda_, batch_size, dataset_size)
#optim = optim.sgld(network, lr, lambda_, batch_size, dataset_size)
#optim = optim.sgld(network.parameters(), lr, lr_decay=0.999, lr_reset_period=0, dataset_size=len(train_loader.dataset))
optim = optim.sgld(network, lr, lambda_, dataset_size)

losses_sgld = []

for epoch in range(100):
    running_loss = 0
    for x, y in iter(train_loader):
        x = x.view(x.size(0), -1)

        network.zero_grad()
        output = network(x)
        loss = criterion(output, y)
        loss.backward()
        optim.step()

        losses_sgld.append(loss)

        running_loss += loss * batch_size / dataset_size
        prediction = output.data.max(1)[1]
        accuracy = torch.sum(prediction.eq(y)).float()/batch_size

    print("Epoch {:d} - loss: {:.4f} - acc: {:.4f}".format(epoch, running_loss, accuracy))

    with torch.autograd.no_grad():
        test_metric = 0
        testLoss_running = 0
        for x, y in iter(test_loader):
            x = x.view(x.size(0), -1)
            output = network(x)
            test_loss = criterion(output, y)
            testLoss_running += test_loss * 1000. / 10000
            test_metric += 100 * (output.argmax(1) == y).float().sum() / 10000
        testLoss_sgld.append(testLoss_running)
        print("\ttest: {:.4}".format( test_metric))
