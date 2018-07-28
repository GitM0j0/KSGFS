import torch
import torchvision

import os

from torch import nn
from torchvision import transforms
import torch.nn.functional as F


from sgfs import optim
import model_sgfs

import mnist






if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

batch_size = 2000
#lr = 1e-5
#lr_decayEpoch = 20
num_workers = 5

lambda_ = 0.001
epsilon =  1.

train_data = torchvision.datasets.MNIST(root=os.environ.get("DATASETS_PATH", "~/datasets"), train=True,
                                         download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=5)

test_data = torchvision.datasets.MNIST(root=os.environ.get("DATASETS_PATH", "~/datasets"), train=False,
                                        download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

train_size = 60000
test_size = 10000

# model_sgfs: preactivation and activation matrices required
network = model_sgfs.mlp([20,30,10],nl=F.sigmoid)
criterion = F.binary_cross_entropy_with_logits

random_matrix = torch.randn(784,20)


optim = optim.sgfs(network, epsilon, lambda_, batch_size, train_size)

for epoch in range(10):
    running_loss = 0
    for x, y in iter(train_loader):
        x = x.view(x.size(0), -1)
        x_projection = x.mm(random_matrix)
        one_hot = torch.zeros(batch_size,10).scatter_(1,y.view(-1,1).long(),1)

        network.zero_grad()
        output, nonLinearities, preactivations = network(x_projection)
        loss = criterion(output, one_hot)
        preactivation_grads = torch.autograd.grad(loss, preactivations)
        # according to Goodfellow (see optim.py)
        # https://arxiv.org/pdf/1510.01799.pdf
        optim.element_backward(nonLinearities, preactivation_grads)
        optim.emp_fisher()


        optim.step()
        #TO DO: update
        running_loss += loss * batch_size / train_size
        prediction = output.data.max(1)[1]
        accuracy = torch.sum(prediction.eq(y)).float()/batch_size

    print("Epoch {:d} - loss: {:.4f} - acc: {:.4f}".format(epoch, running_loss, accuracy))

    with torch.autograd.no_grad():
        test_metric = 0
        for x, y in iter(test_loader):
            x = x.view(x.size(0), -1)
            x_projection = x.mm(random_matrix)
            output, _, _ = network(x_projection)
            test_metric += 100 * (output.argmax(1) == y).float().sum() / test_size


        print("\ttest: ", test_metric)
