import torch

from torch import nn
import torch.nn.functional as F


from sgfs import optim
import model_sgfs

import mnist






if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

batch_size = 500
#lr = 1e-5
#lr_decayEpoch = 20
num_workers = 5

lambda_ = 0.001
epsilon =  2.

### Change B

#precision = 1.

train_loader, test_loader = mnist.get_mnist(batch_size, num_workers)

train_size = len(train_loader.dataset)
test_size = len(test_loader.dataset)

#network = model.shallow_network()
network = model_sgfs.mlp([784,400,400,10])
criterion = F.binary_cross_entropy_with_logits


optim = optim.sgfs(network, epsilon, lambda_, batch_size, train_size)

for epoch in range(1):
    running_loss = 0
    for x, y in iter(train_loader):
        x = x.view(x.size(0), -1)
        one_hot = torch.zeros(batch_size,10).scatter_(1,y.view(-1,1).long(),1)

        network.zero_grad()
        output, nonLinearities, preactivations = network(x)
        loss = criterion(output, one_hot)
        preactivation_grads = torch.autograd.grad(loss, preactivations)
        optim.element_backward(nonLinearities, preactivation_grads)
        optim.emp_fisher()
        #loss = criterion(output, y)
        #loss.backward()


        optim.step()
        #TO DO: update
        #running_loss += loss * batch_size / train_size
        prediction = output.data.max(1)[1]
        accuracy = torch.sum(prediction.eq(y)).float()/batch_size

    print("Epoch {:d} - loss: {:.4f} - acc: {:.4f}".format(epoch, running_loss, accuracy))

    with torch.autograd.no_grad():
        test_metric = 0
        for x, y in iter(test_loader):
            x = x.view(x.size(0), -1)
            output, _, _ = network(x)
            test_metric += 100 * (output.argmax(1) == y).float().sum() / test_size


        print("\ttest: ", test_metric)
