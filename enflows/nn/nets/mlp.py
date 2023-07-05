"""Implementations multi-layer perceptrons."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable

from enflows.nn.nets.nets_util import Sine, init_weights_normal, init_weights_selu, init_weights_elu, \
    init_weights_xavier, gen_sine_init, first_layer_sine_init


class MLP(nn.Module):
    """A standard multi-layer perceptron."""

    def __init__(
            self,
            in_shape,
            out_shape,
            hidden_sizes,
            activation=F.relu,
            activate_output=False,
    ):
        """
        Args:
            in_shape: tuple, list or torch.Size, the shape of the input.
            out_shape: tuple, list or torch.Size, the shape of the output.
            hidden_sizes: iterable of ints, the hidden-layer sizes.
            activation: callable, the activation function.
            activate_output: bool, whether to apply the activation to the output.
        """
        super().__init__()
        self._in_shape = torch.Size(in_shape)
        self._out_shape = torch.Size(out_shape)
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._activate_output = activate_output

        if len(hidden_sizes) == 0:
            raise ValueError("List of hidden sizes can't be empty.")

        self._input_layer = nn.Linear(np.prod(in_shape), hidden_sizes[0])
        self._hidden_layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:])
            ]
        )
        self._output_layer = nn.Linear(hidden_sizes[-1], np.prod(out_shape))

    def forward(self, inputs):
        if inputs.shape[1:] != self._in_shape:
            raise ValueError(
                "Expected inputs of shape {}, got {}.".format(
                    self._in_shape, inputs.shape[1:]
                )
            )

        inputs = inputs.reshape(-1, np.prod(self._in_shape))
        outputs = self._input_layer(inputs)
        outputs = self._activation(outputs)

        for hidden_layer in self._hidden_layers:
            outputs = hidden_layer(outputs)
            outputs = self._activation(outputs)

        outputs = self._output_layer(outputs)
        if self._activate_output:
            outputs = self._activation(outputs)
        outputs = outputs.reshape(-1, *self._out_shape)

        return outputs


class FCBlock(torch.nn.Module):
    """
    Fully Connected Block, that also supports sine activations (they need a specific initialization)
    """

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 num_blocks,
                 outermost_activation: Callable = torch.nn.Identity,
                 activation: str = "tanh", **kwargs):
        super().__init__()

        nls_and_inits = {'sine': (Sine(kwargs.get("sine_frequency", 7)),
                                  gen_sine_init(kwargs.get("sine_frequency", 7)),
                                  first_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus': (nn.Softplus(), init_weights_normal, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None)}

        nl, self.weight_init, first_layer_init = nls_and_inits[activation]

        self.net = []
        self.net.append(torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features), nl
        ))

        for i in range(num_blocks):
            self.net.append(torch.nn.Sequential(
                torch.nn.Linear(hidden_features, hidden_features), nl  # , nn.BatchNorm1d(hidden_features)
            ))

        if outermost_activation:
            self.net.append(torch.nn.Sequential(
                torch.nn.Linear(hidden_features, out_features), outermost_activation()
            ))
        else:
            self.net.append(torch.nn.Sequential(
                torch.nn.Linear(hidden_features, out_features)
            ))

        self.net = torch.nn.Sequential(*self.net)

        self.net.apply(self.weight_init)
        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords):
        return self.net(coords)
