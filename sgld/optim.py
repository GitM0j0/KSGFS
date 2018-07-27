from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class sgld(object):

    def __init__(self, network, lr, lambda_, dataset_size=1):
        self.network = network
        self.N = dataset_size
        self.linear_layers = [m for m in self.network.modules() if isinstance (m, nn.Linear)]
        self.lr_init = lr
        #self.a = a
        #self.b = b
        #self.gamma = gamma
        #self.lr_decayEpoch = lr_decayEpoch
        self.lambda_ = lambda_
        self.t = 1





    def step(self,):
        learning_rate = self.lr_init * 10 ** -(self.t // 100)

        for l in self.linear_layers:
            weight_grad = l.weight.grad.add(self.lambda_ / self.N, l.weight.data)
            noise = torch.randn_like(weight_grad) * math.sqrt(learning_rate / self.N)
            update = (learning_rate * 0.5 * weight_grad).add(noise)
            l.weight.data.add_(-update)
        self.t +=1
