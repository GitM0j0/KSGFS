# Load required libraries
import torch
import torchvision

from torch import nn
from torch.distributions import Normal

import numpy as np


class sgfs(object):
    def __init__(self,net, epsilon,lambda_, batch_size, dataset_size):
        self.net = net
        self.n = batch_size
        self.N = dataset_size
        self.epsilon =epsilon
        self.gamma = np.float(dataset_size + batch_size) / batch_size
        self.linear_layers = [m for m in self.net.modules() if isinstance (m, nn.Linear)]
        #self.lr_init = lr
        #self.lr_decayEpoch = lr_decayEpoch
        self.lambda_ = lambda_
        self.t = 1
        self.I_hat = dict()
        self.grad_mean = dict()


        self.grads_per_element = dict()


    def element_backward(self, activations, preactivation_grads):
        for i, l in enumerate(self.linear_layers):
            G, X = preactivation_grads[i], activations[i]
            if len(G.shape) < 2:
                G = G.unsqueeze(1)

            #G *= G.shape[0] # if the function is an average

            #gradients.append(torch.bmm(G.unsqueeze(2), X.unsqueeze(1)))
            self.grads_per_element[l] = torch.bmm(G.unsqueeze(2), X.unsqueeze(1))
    		#gradients.append(G)

        #return gradients

    def emp_fisher(self):
        for l in self.linear_layers:
            self.grad_mean[l] = self.grads_per_element[l].sum(0) / self.n
            diff_grads = self.grads_per_element[l] - self.grad_mean[l]
            scale = (self.n - 1) ** (-1)
            cov_scores = torch.einsum('bij,bkj->ik', (diff_grads, diff_grads)) * scale
            if self.t == 1:
                self.I_hat[l] = torch.zeros(l.weight.data.size(0),l.weight.data.size(0))
            else:
                self.I_hat[l] = (1 - 1. / self.t) * self.I_hat[l] + (1. / self.t) * cov_scores
            #print(self.I_hat[l].size())





    def step(self):
        for l in self.linear_layers:
            # if self.t == 1:
            #     self.I_hat[l] = torch.zeros(l.weight.data.size(0),l.weight.data.size(0))

            # weight_grad = l.weight.grad

            #mean_weight_grad = torch.mean(weight_grad,0).view(1,-1) # correct dimension
            # mean_weight_grad = l.weight.grad / self.n
            # diff_grad = weight_grad - mean_weight_grad
            # cov_scores = 1 / (self.n - 1) * diff_grad.mm(diff_grad.transpose(1,0))
            #cov_scores = 1 / (self.n - 1) * diff_grad.mm(diff_grad.transpose(1,0))
            #print(np.sum(np.cov(weight_grad.numpy())-(1 / (self.n-1) * diff_grad.mm(diff_grad.transpose(1,0))).numpy()))


            # SGFS-d update
    #
            # SGFS-f update
            #self.I_hat[l] = (1 - 1. / self.t) * self.I_hat[l] + (1. / self.t) * cov_scores

            # According to Ahn et al. (2012): B \propto N*I_hat
            if self.t < 50:
                B = torch.eye(self.I_hat[l].size(0))
            else:
                B = self.N* self.I_hat[l]
                #B = self.I_hat[l]
            #B = torch.eye(self.grad_mean[l].size(0))
            #print(torch.prod(torch.diag(B)))
            #B = self.N * self.I_hat[l]
            #print(torch.diag(B))

            mat = self.gamma * self.N * self.I_hat[l]  + (4. / self.epsilon * B)
            #mat = self.gamma * self.I_hat[l]  + (4. / self.epsilon * B)
            mat_inv = mat.inverse()

            # B needs to be changed in training file
            B_ch = torch.potrf(B)
            noise = (2. * self.epsilon ** (-0.5) * B_ch).mm(torch.randn_like(self.grad_mean[l]))
            # noise = (2. * (self.epsilon * self.N) ** (-0.5) * B_ch).mm(torch.randn_like(self.grad_mean[l]))
            #prior = self.lambda_ / self.N * l.weight.data


            #noise = torch.randn_like(weight_grad) * (4. / (learning_rate) * self.B) ** 0.5

            # Mini-batch updates
            # theta_(t+1) = theta_t + 2 * (gamma * I_hat + N * grad_avg(theta_t; X_t))^⁻1 * ( grad(log p(theta_t)) + N * grad_avg(theta_t) + eta_t)
            # with eta_t ~ N(0, 4 * B / eta_t)
            update = mat_inv.mm(2. * (self.N * mean_weight_grad).add_(self.lambda_, l.weight.data).add_(noise))
            #update = 2. * mat_inv.mm(noise.add_(self.lambda_ / self.N, l.weight.data).add_(self.grad_mean[l]))
            #update = 2. * mat_inv.mm(noise.add_(2 * prior).add_(self.N, mean_weight_grad))

            l.weight.data.add_(-update)
        self.t += 1
