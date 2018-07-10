# Load required libraries
import torch
import torchvision

from torch import nn
from torch.distributions import Normal

import numpy as np


class sgfs(object):
    def __init__(self,net, B, a, b, gamma, lambda_, batch_size, dataset_size):
        self.net = net
        self.n = batch_size
        self.N = dataset_size
        self.gamma = np.float(dataset_size + batch_size) / batch_size
        self.linear_layers = [m for m in self.net.modules() if isinstance (m, nn.Linear)]
        self.B = B
        #self.lr_init = lr
        #self.lr_decayEpoch = lr_decayEpoch
        self.lambda_ = lambda_

    def step(self, epoch=0):
        for l in self.linear_layers:
            weight_grad = l.weight.grad
            mean_weight_grad = torch.mean(weight_grad,0)
            diff_grad = weight_grad - mean_weight_grad
            cov_grads = 1 / (self.n - 1) * diff_grad.transpose(1,0).mm(diff_grad)

            #weight_grad.add_(self.lambda_, l.weight.data)
            # Exponential LR decay
            #learning_rate = self.lr_init * (2**(-epoch // self.lr_decayEpoch))
            learning_rate = self.a * (self.b + epoch) ** (-self.gamma)


            I_hat = (1 - 1. / epoch) * I_hat + 1. / epoch * cov_grads

            mat = self.gamma * I_hat + (4. / learning_rate) * self.B
            l, u = torch.symeig(mat)
            mat_inv = (u * ((l + ) ** (-1))).mm(u.transpose(1,0))
            ### INVERSE

            size=weight_grad.size()

            noise = Normal(
                torch.zeros(size),
                torch.ones(size) * np.sqrt((4. /learning_rate) * self.B)
                )
            # Mini-batch updates
            # theta_(t+1) = theta_t + 2 * (gamma * I_hat + N * grad_avg(theta_t; X_t))^⁻1 * ( grad(log p(theta_t)) + N * grad_avg(theta_t) + eta_t)
            # with eta_t ~ N(0, 4 * B / eta_t)

            update = mat_inv.mm(2. * (self.N * mean_weight_grad).add_(self.lambda_, l.weight.data) + noise.sample())
            l.weight.data.add_(update)
