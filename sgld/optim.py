# Load required libraries
import torch
import torchvision

from torch import nn

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
        self.t = 1


    def step(self):
        for l in self.linear_layers:
            weight_grad = l.weight.grad

            # Mini-batch updates according to Ahn et al. (2012)
            # theta_(t+1) = theta_t - epsilon_t * 0.5 *( grad(log p(theta_t)) + N/n sum(grad(log p(x_t|theta_t)))) + eta_t
            # with eta_t ~ N(0, epsilon_t)

            # According to Ahn et al. (2012)
            learning_rate = self.a * (self.b + self.t) ** (-self.gamma)
            noise = torch.randn_like(weight_grad) * (learning_rate/self.N) ** 0.5
            update = (learning_rate * 0.5 * weight_grad).add_(self.lambda_ / self.N, l.weight.data).add_(noise)

            l.weight.data.add_(-update)
        self.t += 1
