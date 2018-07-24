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

            # Mini-batch updates in Ahn et al. (2012)
            # theta_(t+1) = theta_t - epsilon_t * 0.5 *( grad(log p(theta_t)) + N/n sum(grad(log p(x_t|theta_t)))) + eta_t
            # with eta_t ~ N(0, epsilon_t)


            # According to Ahn et al. (2012) without averaging (original paper) - NOT WORKING!!!
            # grad_logPost = (float(self.N) * weight_grad).add_(self.lambda_, l.weight.data)
            #print(grad_logPost)
            learning_rate = self.a * (self.b + self.t) ** (-self.gamma)
            # noise = torch.randn_like(weight_grad) * (learning_rate) ** (0.5)
            # update = (learning_rate * 0.5 * grad_logPost).add_(noise)

            # According to Marceau-Caron/Ollivier (2017) with averaging - WORKS!
            # Uses modified learning_rate lr = 2 * epsilon / N
            grad_logPost = (weight_grad).add_(self.lambda_ /self.N, l.weight.data)
            #print(grad_logPost)
            # Exponential LR decay
            # According to Ahn et al. (2012)
            #learning_rate = self.a * (self.b + self.t) ** (-self.gamma)
            # According to Li et al.(2016) - Modification of required parameters!
            #learning_rate = self.lr_init * (2**(-epoch // self.lr_decayEpoch))

            noise = torch.randn_like(weight_grad) * (2*learning_rate/self.N) ** 0.5



            update = (learning_rate * weight_grad).add_(self.lambda_ / self.N, l.weight.data)#.add_(noise)
            #update = (learning_rate * 0.5 * grad_logPost).add_(2 * noise.sample() / self.N)
            l.weight.data.add_(-update)
        self.t += 1
