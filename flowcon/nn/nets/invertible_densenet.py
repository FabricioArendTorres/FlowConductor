import abc
import logging
import math
from typing import *
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pprint import pformat

from flowcon.nn.nets import activations
from flowcon.nn.nets.extended_basic_nets import ExtendedSequential, ExtendedLinear
from flowcon.nn.nets.spectral_norm import scaled_spectral_norm
from flowcon.nn.nets.lipschitz_dense import LipschitzDenseLayer
from flowcon.nn.nets import MLP

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
    'csin': activations.CSin,
}

logger = logging.getLogger()

class _DenseNet(torch.nn.Module):
    def __init__(self,
                 dimension,
                 densenet_depth: int = 2,
                 densenet_growth: int = 16,
                 activation_function: Union[str, Callable] = "CLipSwish",
                 lip_coeff: float = 0.98,
                 n_lipschitz_iters: int = 5
                 ):
        super().__init__()

        self.dimension = dimension
        self.densenet_depth = densenet_depth
        self.densenet_growth = densenet_growth

        self.lip_coeff = lip_coeff
        self.n_lipschitz_iters = n_lipschitz_iters

        assert n_lipschitz_iters > 0, "n_lipschitz_iters must be > 0"
        assert lip_coeff > 0, "lip_coeff must be > 0"

        if isinstance(activation_function, str):
            assert activation_function in ACTIVATION_FNS.keys(), f"Activation function {activation_function} not found."
            self.activation = ACTIVATION_FNS[activation_function]()
        else:
            self.activation = activation_function

        self.output_channels = self.calc_output_channels(self.activation,
                                                         self.densenet_growth)

    def spectral_normalization(self, network):
        return scaled_spectral_norm(network,
                                    n_power_iterations=self.n_lipschitz_iters,
                                    domain=2, codomain=2, coeff=self.lip_coeff)

    def build_densenet(self, total_in_channels, densenet_depth, densenet_growth, learnable_concat=True,
                       include_last_layer=True) -> Tuple[ExtendedSequential, int]:
        nnet = []
        for i in range(densenet_depth):
            part_net = []

            part_net.append(
                self.spectral_normalization(torch.nn.Linear(total_in_channels, self.output_channels))
            )
            part_net.append(self.activation)
            nnet.append(
                LipschitzDenseLayer(
                    ExtendedSequential(*part_net),
                    learnable_concat=learnable_concat,
                    lip_coeff=self.lip_coeff
                )
            )

            total_in_channels += densenet_growth

        if include_last_layer:
            nnet.append(
                self.spectral_normalization(ExtendedLinear(total_in_channels, self.dimension))
            )
        if include_last_layer:
            total_in_channels = 1

        return ExtendedSequential(*nnet), total_in_channels

    @staticmethod
    def calc_output_channels(activation, densenet_growth):
        # Change growth size for CLipSwish:
        if hasattr(activation, "_does_concat") and activation._does_concat: #isinstance(activation, activations.CLipSwish) or isinstance(activation, activations.CSin):
            assert densenet_growth % 2 == 0, "Select an even densenet growth size for CLipSwish!"
            output_channels = densenet_growth // 2
        else:
            output_channels = densenet_growth
        return output_channels


    @classmethod
    def factory(cls,
                condition_input=False,
                condition_lastlayer=False,
                condition_multiplicative=False,
                **kwargs):

        if not (condition_input or condition_lastlayer or condition_multiplicative):
            lipschitz_network = DenseNet
        elif condition_input and not (condition_lastlayer or condition_multiplicative):
            lipschitz_network = InputConditionalDenseNet

        elif condition_lastlayer and not (condition_input or condition_multiplicative):
            lipschitz_network = LastLayerConditionalDenseNet

        elif condition_multiplicative and not (condition_input or condition_lastlayer):
            lipschitz_network = LastLayerConditionalDenseNet

        elif (condition_input and condition_lastlayer) and not condition_multiplicative:
            lipschitz_network = MixedConditionalDenseNet

        elif (condition_multiplicative and condition_input) and not condition_lastlayer:
            lipschitz_network = MultiplicativeAndInputConditionalDenseNet
        else:
            raise NotImplementedError("This combination of conditions for a Lipschitz Network is not implemented .")

        return lambda: lipschitz_network(**kwargs)

    @abstractmethod
    def forward(self, x, context=None):
        pass


class DenseNet(_DenseNet):
    """
    Provides a Lipschitz contiuous network g(x)  with a fixed lipschitz constant.
    """

    def __init__(self,
                 dimension,
                 densenet_depth: int = 2,
                 densenet_growth: int = 16,
                 activation_function: Union[str, Callable] = "CLipSwish",
                 lip_coeff: float = 0.98,
                 n_lipschitz_iters: int = 5,
                 **kwargs):
        super().__init__(dimension=dimension,
                         densenet_depth=densenet_depth,
                         densenet_growth=densenet_growth,
                         activation_function=activation_function,
                         lip_coeff=lip_coeff,
                         n_lipschitz_iters=n_lipschitz_iters)

        if len(kwargs) > 0:
            logger.warning("Unused kwargs for {}: {}".format(self.__class__.__name__, pformat(kwargs)))
        self.dense_net, self.densenet_final_layer_dim = self.build_densenet(self.dimension,
                                                                            self.densenet_depth,
                                                                            self.densenet_growth)

    def forward(self, x, context=None):
        assert context is None, "Context not supported for this Class."
        return self.dense_net(x)


class InputConditionalDenseNet(_DenseNet):
    """
    Provides a Lipschitz contiuous network g(x;t) = h(concat[x, f(c)]) with a fixed lipschitz constant.
    The network h(x;t) is a DenseNet and Lipschitz Continuous w.r.t. x.
    the network f provides an embedding of the context.
    """

    def __init__(self, dimension, context_features,
                 densenet_depth,
                 densenet_growth=16,
                 c_embed_hidden_sizes=(128, 128, 10),
                 activation_function=activations.Swish,
                 lip_coeff=0.98,
                 n_lipschitz_iters=5,
                 **kwargs
                 ):
        super().__init__(dimension=dimension,
                         densenet_depth=densenet_depth,
                         densenet_growth=densenet_growth,
                         activation_function=activation_function,
                         lip_coeff=lip_coeff,
                         n_lipschitz_iters=n_lipschitz_iters)
        if len(kwargs) > 0:
            logger.warning("Unused kwargs for class '{}': \n {}".format(self.__class__.__name__, pformat(kwargs)))

        self.context_features = context_features
        self.c_embed_hidden_sizes = c_embed_hidden_sizes

        self.bn = torch.nn.BatchNorm1d(self.context_features)
        self.dense_net, self.densenet_final_layer_dim = self.build_densenet(
            total_in_channels=self.dimension + self.c_embed_hidden_sizes[-1],
            densenet_growth=densenet_growth,
            densenet_depth=densenet_depth,
            include_last_layer=True)

        self.context_embedding_net = MLP((self.context_features,), (self.c_embed_hidden_sizes[-1],),
                                         hidden_sizes=self.c_embed_hidden_sizes,
                                         activation=torch.nn.SiLU())

    def forward(self, inputs, context=None):
        context = self.bn(context)
        context_embedding = self.context_embedding_net(context)
        concat_inputs = torch.cat([inputs, context_embedding], -1)
        outputs = self.dense_net(concat_inputs)
        return outputs


class MultiplicativeAndInputConditionalDenseNet(_DenseNet):
    """
    Provides a Lipschitz contiuous network g(x;c) = φ(c) h(x;t) with a fixed lipschitz constant.
    The network h(x;c) is an InputConditionalDenseNet and Lipschitz Continuous w.r.t. x, and φ(c) \in (0,1), making g Lipschitz Continuous w.r.t. x.
    """

    def __init__(self, dimension, context_features,
                 densenet_depth,
                 densenet_growth=16,
                 c_embed_hidden_sizes=(128, 128, 10),
                 m_embed_hidden_sizes=(128, 128),
                 activation_function=activations.Swish,
                 lip_coeff=0.98,
                 n_lipschitz_iters=5,
                 **kwargs
                 ):
        super().__init__(dimension=dimension,
                         densenet_depth=densenet_depth,
                         densenet_growth=densenet_growth,
                         activation_function=activation_function,
                         lip_coeff=lip_coeff,
                         n_lipschitz_iters=n_lipschitz_iters)
        if len(kwargs) > 0:
            logger.warning("Unused kwargs for class '{}': \n {}".format(self.__class__.__name__, pformat(kwargs)))

        self.context_features = context_features
        self.c_embed_hidden_sizes = c_embed_hidden_sizes
        self.m_embed_hidden_sizes = m_embed_hidden_sizes

        self.bn = torch.nn.BatchNorm1d(self.context_features)
        self.dense_net, self.densenet_final_layer_dim = self.build_densenet(
            total_in_channels=self.dimension + self.c_embed_hidden_sizes[-1],
            densenet_growth=densenet_growth,
            densenet_depth=densenet_depth,
            include_last_layer=True)

        self.factor_net = MLP((self.context_features,), (1,),
                              hidden_sizes=self.m_embed_hidden_sizes,
                              activation=torch.nn.SiLU())

        self.embedding = MLP((self.context_features,), (self.c_embed_hidden_sizes[-1],),
                             hidden_sizes=self.c_embed_hidden_sizes,
                             activation=torch.nn.SiLU())

    def forward(self, inputs, context=None):
        context = self.bn(context)
        factor = self.factor_net(context)
        context_embedding = self.embedding(context)
        concat_inputs = torch.cat([inputs, context_embedding], -1)
        outputs = torch.nn.functional.tanh(factor) * self.dense_net(concat_inputs)
        return outputs


class MultiplicativeConditionalDenseNet(_DenseNet):
    """
    Provides a Lipschitz contiuous network g(x;c) = φ(c) h(x;t) with a fixed lipschitz constant.
    The network h(x;c) is an InputConditionalDenseNet and Lipschitz Continuous w.r.t. x, and φ(c) \in (0,1), making g Lipschitz Continuous w.r.t. x.
    """

    def __init__(self, dimension, context_features,
                 densenet_depth,
                 densenet_growth=16,
                 c_embed_hidden_sizes=(128, 128, 10),
                 m_embed_hidden_sizes=(128, 128),
                 activation_function=activations.Swish,
                 lip_coeff=0.98,
                 n_lipschitz_iters=5,
                 **kwargs
                 ):
        super().__init__(dimension=dimension,
                         densenet_depth=densenet_depth,
                         densenet_growth=densenet_growth,
                         activation_function=activation_function,
                         lip_coeff=lip_coeff,
                         n_lipschitz_iters=n_lipschitz_iters)
        if len(kwargs) > 0:
            logger.warning("Unused kwargs for class '{}': \n {}".format(self.__class__.__name__, pformat(kwargs)))

        self.context_features = context_features
        self.c_embed_hidden_sizes = c_embed_hidden_sizes
        self.m_embed_hidden_sizes = m_embed_hidden_sizes

        self.bn = torch.nn.BatchNorm1d(self.context_features)
        self.dense_net, self.densenet_final_layer_dim = self.build_densenet(
            total_in_channels=self.dimension + self.c_embed_hidden_sizes[-1],
            densenet_growth=densenet_growth,
            densenet_depth=densenet_depth,
            include_last_layer=True)

        self.factor_net = MLP((self.context_features,), (1,),
                              hidden_sizes=self.m_embed_hidden_sizes,
                              activation=torch.nn.SiLU())

    def forward(self, inputs, context=None):
        context = self.bn(context)
        factor = self.factor_net(context)
        outputs = torch.nn.functional.tanh(factor) * self.dense_net(inputs)
        return outputs


class LastLayerAttention(torch.nn.Module):
    def __init__(self,
                 dimension,
                 context_features,
                 value_dim,
                 hidden_sizes=(64, 64),
                 activation=activations.Swish()):
        super().__init__()
        self.dimension = dimension
        self.context_features = context_features

        self.hidden_sizes = hidden_sizes

        self.value_dim = value_dim
        self.activation = activation

        self.bias_net = MLP((self.context_features,), (self.dimension,),
                            hidden_sizes=self.hidden_sizes,
                            activation=self.activation)

        self.weight_network = MLP((self.context_features,), (self.dimension, self.value_dim),
                                  hidden_sizes=self.hidden_sizes,
                                  activation=self.activation)

    def attention(self, context, values):
        presoftmax = self.weight_network(context)  # .view(-1, self.dimension, self.num_heads)
        post_softmax = torch.nn.functional.softmax(presoftmax, dim=-1)
        weights = torch.bmm(post_softmax, values).squeeze()
        return weights + self.bias_net(context)


class LastLayerConditionalDenseNet(_DenseNet):
    """
    Provides a Lipschitz contiuous network g(x;c) = h(x;t) with a fixed lipschitz constant.
    The network h(x;c) is a DenseNet.
    Let xϵR^d.
    For a unconditional DenseNet the last layer is given by Az, where zϵR^k with k>d is a
    higher dimensional embedding and AϵR^(d\times k) is a matrix with spectral norm <1.
    For making this last layer conditional, we instead parameterize A(c) with a hypernetwork.
    We further pass each row through a softmax (making A row stochastic), such that the Lipschitz constant of remains unchanged.
    """

    def __init__(self, dimension, context_features,
                 densenet_depth,
                 densenet_growth=16,
                 last_layer_hidden_sizes=(64, 64),
                 activation_function=activations.Swish,
                 lip_coeff=0.98,
                 n_lipschitz_iters=5,
                 **kwargs
                 ):
        super().__init__(dimension=dimension,
                         densenet_depth=densenet_depth,
                         densenet_growth=densenet_growth,
                         activation_function=activation_function,
                         lip_coeff=lip_coeff,
                         n_lipschitz_iters=n_lipschitz_iters)
        if len(kwargs) > 0:
            logger.warning("Unused kwargs for class '{}': \n {}".format(self.__class__.__name__, pformat(kwargs)))

        self.context_features = context_features
        self.last_layer_hidden_sizes = last_layer_hidden_sizes

        self.bn = torch.nn.BatchNorm1d(self.context_features)
        self.dense_net, self.densenet_final_layer_dim = self.build_densenet(total_in_channels=self.dimension,
                                                                            densenet_growth=densenet_growth,
                                                                            densenet_depth=densenet_depth,
                                                                            include_last_layer=False)

        self.custom_attention = LastLayerAttention(dimension=self.dimension,
                                                   context_features=self.context_features,
                                                   value_dim=self.densenet_final_layer_dim,
                                                   hidden_sizes=self.last_layer_hidden_sizes)

    def forward(self, inputs, context=None):
        context = self.bn(context)
        values_weights = self.dense_net(inputs).unsqueeze(-1)
        weights = self.custom_attention.attention(context, values_weights)
        return weights


class MixedConditionalDenseNet(_DenseNet):
    """
    Combination of LastLayerConditionalDenseNet and InputConditionalDenseNet,
    i.e. both the first and last layer of the densenet are conditional.
    """

    def __init__(self, dimension, context_features,
                 densenet_depth,
                 densenet_growth=16,
                 last_layer_hidden_sizes=(64, 64),
                 c_embed_hidden_sizes=(32, 32, 10),
                 activation_function=activations.Swish,
                 lip_coeff=0.98,
                 n_lipschitz_iters=5,
                 **kwargs
                 ):
        super().__init__(dimension=dimension,
                         densenet_depth=densenet_depth,
                         densenet_growth=densenet_growth,
                         activation_function=activation_function,
                         lip_coeff=lip_coeff,
                         n_lipschitz_iters=n_lipschitz_iters)

        if len(kwargs) > 0:
            logger.warning("Unused kwargs for class '{}': \n {}".format(self.__class__.__name__, pformat(kwargs)))

        self.context_features = context_features
        self.c_embed_hidden_sizes = c_embed_hidden_sizes
        self.last_layer_hidden_sizes = last_layer_hidden_sizes

        self.output_channels = self.calc_output_channels(self.activation,
                                                         self.densenet_growth)
        self.bn = torch.nn.BatchNorm1d(self.context_features)
        self.dense_net, self.densenet_final_layer_dim = self.build_densenet(
            total_in_channels=self.dimension + self.c_embed_hidden_sizes[-1],
            densenet_growth=densenet_growth,
            densenet_depth=densenet_depth,
            include_last_layer=False)

        self.custom_attention = LastLayerAttention(dimension=self.dimension,
                                                   context_features=self.context_features,
                                                   value_dim=self.densenet_final_layer_dim,
                                                   hidden_sizes=self.last_layer_hidden_sizes)

        self.context_embedding_net = MLP((self.context_features,), (self.c_embed_hidden_sizes[-1],),
                                         hidden_sizes=self.c_embed_hidden_sizes,
                                         activation=torch.nn.SiLU())

    def forward(self, inputs, context=None):
        context = self.bn(context)
        context_embedding = self.context_embedding_net(context)
        concat_inputs = torch.cat([inputs, context_embedding], -1)
        values_weights = self.dense_net(concat_inputs).unsqueeze(-1)
        weights = self.custom_attention.attention(context, values_weights)
        return weights

# class SirenLayer(nn.Module):
#     def __init__(self, dim_in, dim_out, w0=1., c=6., is_first=False, use_bias=True, activation=None):
#         super().__init__()
#         self.dim_in = dim_in
#         self.is_first = is_first
#
#         weight = torch.zeros(dim_out, dim_in)
#         bias = torch.zeros(dim_out) if use_bias else None
#         self.init_(weight, bias, c=c, w0=w0)
#
#         self.weight = nn.Parameter(weight)
#         self.bias = nn.Parameter(bias) if use_bias else None
#
#         self.activation = Sine(w0) if activation is None else activation
#
#     def init_(self, weight, bias, c, w0):
#         dim = self.dim_in
#
#         w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
#         weight.uniform_(-w_std, w_std)
#
#         if exists(bias):
#             bias.uniform_(-w_std, w_std)
#
#     def forward(self, x):
#         out = torch.nn.functional.linear(x, self.weight, self.bias)
#         out = self.activation(out)
#         return out
