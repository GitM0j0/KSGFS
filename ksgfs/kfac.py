# Load required libraries
import torch
import torchvision

from torch import nn

import numpy as np

from enum import Enum
import math

import torch.nn.functional as F



class BackpropMode(Enum):
    STANDARD = 0
    CURVATURE = 1


class KSGFS(torch.optim.Optimizer):

    def __init__(self, param_to_module_map, output, lr=1., l2=0., eta=1., v=0., damping=1e-3,
                 stochastic=False, empirical_fisher=False):
        if output not in ("categorical", "binary", "gaussian"):
            raise ValueError("Received output='{}', but expected one of ('categorical', 'binary', 'gaussian')".format(
                output
            ))
        defaults = dict(lr=lr, l2=l2, eta=eta, v=v, damping=damping)
        super(KFAC, self).__init__(param_to_module_map.keys(), defaults)

        self.stochastic = stochastic
        self.empirical_fisher = empirical_fisher
        self.param_to_module_map = param_to_module_map
        self.output = output
        self.mode = BackpropMode.STANDARD


        # self.network = network
        # self.n = batch_size
        # self.N = dataset_size
        # self.gamma = np.float(dataset_size + batch_size) / batch_size

        # self.noise_factor = 2. / np.sqrt(self.gamma / (self.N * self.epsilon))
        # self.learning_rate = 2. / (self.gamma * (1. + 4. / self.epsilon))

        if output == "binary":
            self.default_loss_fn = F.binary_cross_entropy_with_logits
        elif output == "categorical":
            self.default_loss_fn = F.cross_entropy
        elif output == "gaussian":
            raise NotImplementedError
        else:
            raise ValueError("Unreachable!")

        self._register_hooks()

    def step(self, closure):
        if closure is None:
            raise ValueError("Must provide closure calculating the loss")

        self.update_curvature(closure)

        closure(size_average=True, stochastic=False, empirical=True)
        for pg in self.param_groups:
            lr = pg["lr"]
            l2 = pg["l2"]
            damping = pg["damping"]

            for param in pg["params"]:
                state = self.state[param]

                grad = param.grad.data
                grad.add_(l2 * param.data)

                noise = torch.randn_like(grad)

                q = state["input_covariance"]
                f = state["activation_fisher"]

                m = q.size(0)
                n = f.size(0)
                omega = n * torch.trace(q) / (m * torch.trace(f))

                reg_q = q + math.sqrt(l2 + damping) * omega * torch.eye(m)
                reg_f = f + math.sqrt(l2 + damping) / omega * torch.eye(n)

                mm1 = torch.gesv(grad.t(), reg_q)[0].t()
                update = torch.gesv(mm1, reg_f)[0]
                param.data.add_(-lr * update)

    def update_curvature(self, closure):
        self.mode = BackpropMode.CURVATURE
        closure(size_average=False, stochastic=self.stochastic, empirical=self.empirical_fisher)
        self.mode = BackpropMode.STANDARD

    def make_closure(self, net, x, y, loss_fn=None):
        def closure(size_average, stochastic, empirical=False):
            self.zero_grad()
            output = net(x)
            if empirical:
                target = y
                calc_loss = loss_fn if loss_fn is not None else self.default_loss_fn
            elif self.output == "binary":
                target = torch.bernoulli(F.sigmoid(output.detach())).long()
                calc_loss = loss_fn if loss_fn is not None else F.binary_cross_entropy_with_logits
            elif self.output == "categorical":
                target = torch.multinomial(F.softmax(output.detach(), 1), 1).long().squeeze()
                calc_loss = loss_fn if loss_fn is not None else F.cross_entropy
            elif self.output == "gaussian":
                raise NotImplementedError
            else:
                raise ValueError("Unreachable")

            losses = calc_loss(output, target, reduce=False)
            if stochastic:
                eps = torch.randn_like(losses)
                loss = torch.sum(losses * eps)
            else:
                loss = torch.sum(losses)

            if size_average:
                loss /= losses.size(0)

            loss.backward()

        return closure

    def _register_hooks(self):
        for pg in self.param_groups:
            for param in pg["params"]:
                module = self.param_to_module_map[param]
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        raise ValueError("Linear module can't have bias. Use bayestorch.nn.HomLinear instead.")
                    self._register_linear_hook(pg, param, module)
                else:
                    raise NotImplementedError

    def _register_linear_hook(self, group, param, module):
        def forward_hook(mod, inputs, output):
            def backward_hook(grad):
                if self.mode == BackpropMode.CURVATURE:
                    inp = inputs[0]
                    # if isinstance(mod, HomLinear) and inp.size(-1) != mod.in_features:
                    #     inp = torch.cat((inp, inp.new_ones(*inp.size()[:-1], 1)), -1)
                    n = inp.size(0)
                    self._update_curvature_factor(group, param, "input_covariance", inp.t().mm(inp) / n)
                    self._update_curvature_factor(group, param, "activation_fisher", grad.t().mm(grad) / n)

            output.register_hook(backward_hook)
        module.register_forward_hook(forward_hook)

    def _update_curvature_factor(self, group, param, key, factor):
        state = self.state[param]
        if key not in state:
            state[key] = factor
        else:
            v = group["v"]
            eta = group["eta"]

            state[key] = v * state[key] + eta * factor

    @classmethod
    def get_map(cls, net, criterion, **kwargs):
        param_to_module_map = dict()
        for module in net.modules():
            for param in module.parameters():
                param_to_module_map[param] = module
        return cls(param_to_module_map, criterion, **kwargs)
