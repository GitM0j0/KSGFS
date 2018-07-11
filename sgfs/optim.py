# Load required libraries
import torch
import torchvision

from torch import nn
from torch.distributions import Normal

import numpy as np


class sgfs(object):
    def __init__(self,net, B, a, b, nu, tau, batch_size, dataset_size):
        self.net = net
        self.n = batch_size
        self.N = dataset_size
        self.a = a
        self.b = b
        self.nu = nu
        self.gamma = np.float(dataset_size + batch_size) / batch_size
        self.linear_layers = [m for m in self.net.modules() if isinstance (m, nn.Linear)]
        self.B = B
        #self.lr_init = lr
        #self.lr_decayEpoch = lr_decayEpoch
        self.tau = tau
        self.I_hat = dict()

        ### DIMENSION not correctS

    def step(self, epoch=0):
        for l in self.linear_layers:
            if epoch == 0:
                self.I_hat[l] = torch.zeros(l.weight.data.size(1),l.weight.data.size(1))

            weight_grad = l.weight.grad
            mean_weight_grad = torch.mean(weight_grad, 0)
            diff_grad = weight_grad - mean_weight_grad
            cov_scores = 1 / (self.n - 1) * diff_grad.transpose(1,0).mm(diff_grad)

            #weight_grad.add_(self.tau, l.weight.data)
            # Exponential LR decay
            #learning_rate = self.lr_init * (2**(-epoch // self.lr_decayEpoch))
            learning_rate = self.a * (self.b + epoch) ** (-self.nu)


            self.I_hat[l] = (1 - 1. / (epoch+1)) * self.I_hat[l] + 1. / (epoch+1) * cov_scores

            mat = self.gamma * self.I_hat[l] + (4. / learning_rate) * self.B
            #Expensive inversion
            mat_inv = mat.inverse()

            # B needs to be changed in training file
            B_ch = torch.potrf(B)
            noise = (2. * learning_rate ** (-0.5) * B_ch).mm(torch.randn_like(weight_grad))
            print(noise.size())


            #noise = torch.randn_like(weight_grad) * (4. / (learning_rate) * self.B) ** 0.5

            # Mini-batch updates
            # theta_(t+1) = theta_t + 2 * (gamma * I_hat + N * grad_avg(theta_t; X_t))^⁻1 * ( grad(log p(theta_t)) + N * grad_avg(theta_t) + eta_t)
            # with eta_t ~ N(0, 4 * B / eta_t)

            update = mat_inv.mm(2. * (self.N * mean_weight_grad).add_(self.tau, l.weight.data).add_(noise))
            l.weight.data.add_(update)
