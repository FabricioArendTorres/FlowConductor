import argparse
import os
import time
import math
import numpy as np

import torch
from torch.nn.utils.parametrize import is_parametrized

from enflows.nn.nets import activations
from enflows.nn.nets.lipschitz_dense import LipschitzDenseLayer
from enflows.nn.nets.lipschitz import scaled_spectral_norm_induced, scaled_spectral_norm_powerits
from enflows.nn.nets.extended_basic_nets import ExtendedSequential, ExtendedLinear

from siren_pytorch import Siren

coeff = 0.9

ACTIVATION_FNS = {
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh,
    'elu': torch.nn.ELU,
    'selu': torch.nn.SELU,
    'fullsort': activations.FullSort,
    'maxmin': activations.MaxMin,
    'swish': activations.Swish,
    'LeakyLSwish': activations.LeakyLSwish,
    'CLipSwish': activations.CLipSwish,
    'lcube': activations.LipschitzCube,
}

import torch.nn as nn


def exists(val):
    return val is not None


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x) / self.w0


class SirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1., c=6., is_first=False, use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None

        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class LipschitzDenseNetBuilder:
    def __init__(self,
                 input_channels,
                 densenet_depth,
                 densenet_growth=16,
                 context_features=0,
                 activation_function=activations.Swish,
                 learnable_concat=True,
                 lip_coeff=0.98,
                 n_lipschitz_iters=1):
        self.input_channels = input_channels
        self.densenet_growth = densenet_growth
        self.densenet_depth = densenet_depth
        self.act_fun = activation_function
        self.learnable_concat = learnable_concat
        self.lip_coeff = lip_coeff
        self.n_lipschitz_iters = n_lipschitz_iters

        self.context_features = context_features if context_features is not None else 0
        self.wrapper = lambda network: scaled_spectral_norm_induced(network,
                                                                    n_power_iterations=self.n_lipschitz_iters,
                                                                    domain=2, codomain=2)

        self.set_activation(self.act_fun)

    @property
    def activation(self):
        return self.act_fun

    def set_activation(self, activation):
        # Change growth size for CLipSwish:
        if isinstance(self.act_fun, activations.CLipSwish)  or isinstance(
                self.act_fun, activations.CSin):
            assert self.densenet_growth % 2 == 0, "Select an even densenet growth size for CLipSwish!"
            self.output_channels = self.densenet_growth // 2
        else:
            self.output_channels = self.densenet_growth
        self.act_fun = activation

    def build_network(self) -> ExtendedSequential:
        nnet = []
        total_in_channels = self.input_channels + self.context_features
        for i in range(self.densenet_depth):
            part_net = []

            part_net.append(
                self.wrapper(ExtendedLinear(total_in_channels, self.output_channels))
            )
            part_net.append(self.activation)
            nnet.append(
                LipschitzDenseLayer(
                    ExtendedSequential(*part_net),
                    learnable_concat=self.learnable_concat,
                    lip_coeff=self.lip_coeff
                )
            )

            total_in_channels += self.densenet_growth
        nnet.append(
            self.wrapper(ExtendedLinear(total_in_channels, self.input_channels))
        )
        return ExtendedSequential(*nnet)


class LipschitzFCNNBuilder:
    def __init__(self,
                 units,
                 activation_function=activations.Swish,
                 lip_coeff=0.98,
                 n_lipschitz_iters=1):
        self.units = units
        self.act_fun = activation_function
        self.lip_coeff = lip_coeff
        self.n_lipschitz_iters = n_lipschitz_iters

        self.wrapper = lambda network: scaled_spectral_norm_induced(network,
                                                                    n_power_iterations=self.n_lipschitz_iters,
                                                                    coeff=self.lip_coeff,
                                                                    domain=2, codomain=2)

    @property
    def activation(self):
        return self.act_fun

    def build_network(self) -> torch.nn.Sequential:
        nnet = []
        for i in range(1, len(self.units) - 1):
            part_net = []
            part_net.append(
                self.wrapper(SirenLayer(self.units[i - 1], self.units[i], w0=30))
            )
            # part_net.append(self.activation)
            nnet.append(
                torch.nn.Sequential(*part_net),
            )

        nnet.append(
            self.wrapper(torch.nn.Linear(self.units[-2], self.units[-1]))
        )

        return torch.nn.Sequential(*nnet)
