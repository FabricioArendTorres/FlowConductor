import numpy as np
import torch
from torch.nn import functional as F

from flowcon.transforms.base import Transform
from flowcon.transforms import made as made_module
from flowcon.transforms.splines.cubic import cubic_spline
from flowcon.transforms.splines.linear import linear_spline
from flowcon.transforms.splines.quadratic import (
    quadratic_spline,
    unconstrained_quadratic_spline,
)
from flowcon.transforms.splines import rational_quadratic
from flowcon.transforms.splines.rational_quadratic import (
    rational_quadratic_spline,
    unconstrained_rational_quadratic_spline,
)
from flowcon.utils import torchutils
from flowcon.transforms.UMNN import *
from flowcon.transforms.adaptive_sigmoids import DeepSigmoidModule
from flowcon.transforms.autoregressive.autoregressive import AutoregressiveTransform


class MaskedDeepSigmoidTransform(AutoregressiveTransform):
    """An unconstrained monotonic neural networks autoregressive layer that transforms the variables.
        """

    class DeepSigmoidMadeModule(DeepSigmoidModule):
        def forward(self, inputs, context=None) -> torch.Tensor:
            raise NotImplementedError("Do not directly use this class.")

    def __init__(
            self,
            features,
            hidden_features,
            n_sigmoids=30,
            context_features=None,
            num_blocks=2,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
    ):
        self.features = features
        self.n_sigmoids = n_sigmoids


        made = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        super().__init__(made)
        self.deep_sigmoid_module = self.DeepSigmoidMadeModule(n_sigmoids=n_sigmoids, eps=3e-5,
                                                              num_inverse_iterations=50)


    def _output_dim_multiplier(self):
        return 3 * self.n_sigmoids

    def _elementwise_forward(self, inputs, autoregressive_params):
        ds_params = autoregressive_params.view(inputs.shape[0], self.features, self._output_dim_multiplier())
        outputs, logabsdet = self.deep_sigmoid_module.forward_given_params(inputs, dsparams=ds_params/5)
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        raise NotImplementedError("..")
        #
        # forward_fun = lambda inputs, context: self.deep_sigmoid_module.forward_given_params(inputs, dsparams=ds_params)
        # ds_params = autoregressive_params.view(inputs.shape[0], self.features, self._output_dim_multiplier())
        #
        # x, logabsdet = self.deep_sigmoid_module.inverse(inputs=inputs, context=None,
        #                                                 forward_function=forward_fun)
        # return x, logabsdet
