from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class psgld(object):
    ### pSGLD with RMSProp as preconditioner

    #def __init__(self, network, a, b, gamma, lambda_, dataset_size):
    def __init__(self, network, lr, alpha, lambda_, batch_size, dataset_size):
        self.network = network
        self.n = batch_size
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
        learning_rate = self.lr_init * 10 ** -(self.t // 50000)
        # learning_rate = self.a * (self.b + self.t) ** -self.gamma
        for l in self.linear_layers:
            likelihood_grad = l.weight.grad
            prior_grad = l.weight.data
            if l.bias is not None:
                likelihood_grad = torch.cat((likelihood_grad, l.bias.grad.unsqueeze(1)), 1)
                prior_grad = torch.cat((prior_grad, l.bias.data.unsqueeze(1)), 1)

            likelihood_grad *= float(self.N) / self.n

            # posterior_grad = (likelihood_grad).add(self.lambda_ / self.N , prior_grad)
            posterior_grad = likelihood_grad.add(self.lambda_, prior_grad)

            if self.t == 1:
                self.square_avg[l] = torch.zeros_like(posterior_grad)

            self.square_avg[l].mul_(self.alpha).addcmul_(1. - self.alpha, likelihood_grad, likelihood_grad)
            avg = self.square_avg[l].sqrt().add_(1e-8)
            # print(avg.size())
            # avg_ch = torch.potrf(avg, upper=False)
            noise = torch.randn_like(posterior_grad)


            # update = (learning_rate * 0.5 * torch.div(posterior_grad, avg)).addcdiv_(math.sqrt(learning_rate / self.N), noise, avg.sqrt())
            update = (learning_rate * 0.5 * torch.div(posterior_grad, avg)).addcdiv_(math.sqrt(learning_rate), noise, avg.sqrt())
            # update = (learning_rate * torch.div(posterior_grad, avg)).addcdiv_(math.sqrt(2*learning_rate), noise, avg.sqrt())  / self.N
            #update = (learning_rate * 0.5 * torch.div(posterior_grad, avg)).add(math.sqrt(learning_rate / self.N), torch.div(noise, avg.sqrt()))


            if l.bias is not None:
                l.weight.data.add_(-1, update[:, :-1])
                l.bias.data.add_(-1, update[:, -1])
            else:
                l.weight.data.add_(-1, update)

        self.t +=1
