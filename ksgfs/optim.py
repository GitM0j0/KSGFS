from enum import Enum

import scipy.linalg as la

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class BackpropMode(Enum):
    STANDARD = 0
    CURVATURE = 1


class KSGFS(object):

    def __init__(self, net, criterion, batch_size, dataset_size, eta=1., v=0., lambda_=1e-3, epsilon=2., l2=1e-3, stochastic=False, invert_np=True,
                 invert_every=1):
        if not isinstance(criterion, (nn.CrossEntropyLoss, nn.BCEWithLogitsLoss, nn.MSELoss)):
            raise ValueError("Unrecognized loss:", criterion.__class__.__name__)


        self.net = net
        self.criterion = criterion
        self.stochastic = stochastic
        self.invert_np = invert_np
        self.invert_every = invert_every
        self.inversion_counter = -1


        self.n = batch_size
        self.N = dataset_size
        self.gamma = np.float(dataset_size + batch_size) / batch_size
        self.learning_rate = 2. / (self.gamma * (1. + 4. / epsilon))
        self.noise_factor = 2. * math.sqrt(self.gamma / (self.N * epsilon))

        self.eta = eta
        self.v = v
        self.lambda_ = lambda_
        self.l2 = l2
        self.epsilon = epsilon

        self.mode = BackpropMode.STANDARD

        self.linear_layers = [m for m in self.net.modules() if isinstance(m, nn.Linear)]

        self.input_covariances = dict()
        self.preactivation_fishers = dict()
        self.preactivations = dict()
        self.preactivation_fisher_inverses = dict()
        self.input_covariance_inverses = dict()

        self.t = 1.

        self._add_hooks_to_net()

    def update_curvature(self, x):
        self.mode = BackpropMode.CURVATURE

        output = self.net(x)
        preacts = [self.preactivations[l] for l in self.linear_layers]
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            p = F.softmax(output, 1).detach()
            label_sample = torch.multinomial(p, 1, out=p.new(p.size(0)).long()).squeeze()
            loss_fun = F.cross_entropy
        elif isinstance(self.criterion, (nn.BCEWithLogitsLoss, nn.MSELoss)):
            p = output.detach()
            label_sample = torch.bernoulli(p, out=p.new(p.size()))
            loss_fun = lambda x, y, **kwargs: F.mse_loss(x, y, **kwargs).sum(1)
        else:
            raise NotImplemented

        instance_loss = loss_fun(output, label_sample, reduce=False)

        if self.stochastic:
            noise = torch.randn(instance_loss.size(0))
            l = instance_loss.dot(noise)
        else:
            l = instance_loss.sum()

        preact_grads = torch.autograd.grad(l, preacts)
        scale = p.size(0) ** -0.5
        for pg, mod in zip(preact_grads, self.linear_layers):
            preact_fisher = pg.transpose(1, 0).mm(pg).detach() * scale
            self._update_factor(self.preactivation_fishers, mod, preact_fisher)

        self.mode = BackpropMode.STANDARD

        self.inversion_counter += 1
        if self.inversion_counter % self.invert_every == 0:
            self.inversion_counter = 0
            self.invert_curvature()

    def invert_curvature(self):
        self._invert_all(self.preactivation_fishers, self.preactivation_fisher_inverses)
        self._invert_all(self.input_covariances, self.input_covariance_inverses)

    def _invert_all(self, d, inv_dict):
        for mod, mat in d.items():
            if self.invert_np:
                l, u = map(mat.new, la.eigh(mat.numpy()))
            else:
                l, u = torch.symeig(mat)
            inv = (u * ((l + self.lambda_) ** -1)).mm(u.transpose(1, 0))
            inv_dict[mod] = inv

    def _linear_forward_hook(self, mod, inputs, output):
        if self.mode == BackpropMode.CURVATURE:
            self.preactivations[mod] = output
            inp = inputs[0]
            scale = output.size(0) ** -0.5
            if mod.bias is not None:
                inp = torch.cat((inp, inp.new(inp.size(0), 1).fill_(1)), 1)
            input_cov = inp.transpose(1, 0).mm(inp).detach() * scale
            self._update_factor(self.input_covariances, mod, input_cov)

    def _update_factor(self, d, mod, mat):
        if mod not in d or (self.v == 0 and self.eta == 1):
            d[mod] = mat
        else:
            d[mod] = self.v * d[mod] + self.eta * mat

    def step(self, closure=None):
        for l in self.linear_layers:
            weight_grad = l.weight.grad
            if l.bias is not None:
                bias_grad = l.bias.grad
                weight_grad = torch.cat((weight_grad, bias_grad.unsqueeze(1)), 1)

            noise = torch.randn_like(weight_grad)

            # Matrix sqrt!!! Using cholesky factors fails!!!! #Possibly numerical issue
            eps = 1e-5 * 1. ** -(self.t // 5)
            q_ch = torch.potrf(self.input_covariances[l].add_(eps, torch.eye(self.input_covariances[l].size(1))))
            f_ch = torch.potrf(self.preactivation_fishers[l].add_(eps, torch.eye(self.preactivation_fishers[l].size(0))))
            noise_scaled = f_ch.mm(noise).mm(q_ch)

            weight_grad.add_(self.lambda_ / self.N, l.weight.data).add_(self.noise_factor, noise_scaled)

            q_inv = self.input_covariance_inverses[l]
            f_inv = self.preactivation_fisher_inverses[l]
            update = self.learning_rate * f_inv.mm(weight_grad).mm(q_inv)
            #update = f_inv.mm(weight_grad).mm(q_inv)

#             if l.bias is not None:
#                 l.weight.data.add_(-update[:, :-1])
#                 l.bias.data.add_(-update[:, -1])
#             else:
#                 l.weight.data.add_(-update)
            l.weight.data.add_(-update)
        self.t += 1

    def _add_hooks_to_net(self):
        def register_hook(m):
            if isinstance(m, nn.Linear):
                m.register_forward_hook(self._linear_forward_hook)

        self.net.apply(register_hook)
