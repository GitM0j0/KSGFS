from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class sgld(object):

    #def __init__(self, network, a, b, gamma, lambda_, dataset_size):
    def __init__(self, network, lr, lambda_, batch_size, dataset_size):
        self.network = network
        self.n = batch_size
        self.N = dataset_size
        self.linear_layers = [m for m in self.network.modules() if isinstance (m, nn.Linear)]
        self.lr_init = lr
        # self.a = a
        # self.b = b
        # self.gamma = gamma
        #self.lr_decayEpoch = lr_decayEpoch
        self.lambda_ = lambda_
        self.t = 1.





    def step(self,):
        # print(self.t)
        #learning_rate = self.lr_init * 10 ** -(self.t // 1000)
        learning_rate = self.lr_init * 0.5 ** (self.t // 10000)
        # learning_rate = self.a * (self.b + self.t) ** -self.gamma
        for l in self.linear_layers:
            likelihood_grad = l.weight.grad
            prior_grad = l.weight.data
            if l.bias is not None:
                bias_grad = l.bias.grad
                likelihood_grad = torch.cat((likelihood_grad, bias_grad.unsqueeze(1)), 1)
                prior_grad = torch.cat((prior_grad, l.bias.data.unsqueeze(1)), 1)

            likelihood_grad *= float(self.N) / self.n


            # posterior_grad = likelihood_grad.add(self.lambda_ / self.N , prior_grad)
            posterior_grad = likelihood_grad.add(self.lambda_, prior_grad)
            # noise = torch.randn_like(posterior_grad) * math.sqrt(learning_rate / self.N)
            noise = torch.randn_like(posterior_grad) * math.sqrt(learning_rate)
            update = (learning_rate * posterior_grad).add_(noise)

            if l.bias is not None:
                l.weight.data.add_(-update[:, :-1])
                l.bias.data.add_(-update[:, -1])
            else:
                l.weight.data.add_(-update)
        self.t +=1
