from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class psgld(object):
    ### pSGLD with RMSProp as preconditioner

    #def __init__(self, network, a, b, gamma, lambda_, dataset_size):
    def __init__(self, network, lr, alpha, lambda_, dataset_size):
        self.network = network
        self.N = dataset_size
        self.linear_layers = [m for m in self.network.modules() if isinstance (m, nn.Linear)]
        self.lr_init = lr
        self.alpha = alpha
        # self.a = a
        # self.b = b
        # self.gamma = gamma
        #self.lr_decayEpoch = lr_decayEpoch
        self.lambda_ = lambda_
        self.t = 1.

        self.square_avg = dict()





    def step(self,):
        #learning_rate = self.lr_init * 10 ** -(self.t // 1000)
        learning_rate = self.lr_init * 0.5 ** (self.t // 100000)
        # learning_rate = self.a * (self.b + self.t) ** -self.gamma
        for l in self.linear_layers:
            weight_grad = (l.weight.grad).add(self.lambda_ / self.N , l.weight.data)
            if self.t == 1:
                self.square_avg[l] = torch.zeros_like(weight_grad)

            self.square_avg[l].mul_(self.alpha).addcmul_(1 - self.alpha, l.weight.grad, l.weight.grad)
            avg = self.square_avg[l].sqrt().add_(1e-8)
            # print(avg.size())
            # avg_ch = torch.potrf(avg, upper=False)
            noise = torch.randn_like(weight_grad)


            l.weight.data.addcdiv_(-learning_rate * 0.5, weight_grad, avg).addcdiv_(math.sqrt(learning_rate / self.N), noise, avg.sqrt())

        self.t +=1



'''
state.history = rmsd*state.history + (1-rmsd)*grad.^2;
pcder=(eps + sqrt(state.history));
grad = lr* grad ./ pcder + sqrt(2*lr./pcder).*randn(size(grad))/opts.N ;
'''
