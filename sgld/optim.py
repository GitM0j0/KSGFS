# Load required libraries
import torch
import torchvision

from torch import nn
from torch.distributions import Normal

import numpy as np


class sgld(object):
    def __init__(self,net, a, b, gamma, lambda_, batch_size, dataset_size):
        self.net = net
        self.n = batch_size
        self.N = dataset_size
        self.linear_layers = [m for m in self.net.modules() if isinstance (m, nn.Linear)]
        #self.lr_init = lr
        self.a = a
        self.b = b
        self.gamma = gamma
        #self.lr_decayEpoch = lr_decayEpoch
        self.lambda_ = lambda_

    def step(self, epoch=0):
        for l in self.linear_layers:
            weight_grad = l.weight.grad
            grad_logPost = (-float(self.N) / self.n * weight_grad).add_(self.lambda_, l.weight.data)
            # Exponential LR decay
            #learning_rate = self.lr_init * (2**(-epoch // self.lr_decayEpoch))
            learning_rate = self.a * (self.b + epoch) ** (-self.gamma)

            size = weight_grad.size()
            noise = Normal(
                torch.zeros(size),
                torch.ones(size) * np.sqrt(learning_rate)
                )
            # Mini-batch updates
            # theta_(t+1) = theta_t - eta_t * 0.5 *( grad(log p(theta_t)) + N/n sum(grad(log p(x_t|theta_t)))) + eta_t
            # with eta_t ~ N(0, eta_t)

            #update = learning_rate * 0.5 * self.N / self.n * weight_grad + noise.sample()
            update = learning_rate * 0.5 * grad_logPost + noise.sample()
            l.weight.data.add_(-update)
