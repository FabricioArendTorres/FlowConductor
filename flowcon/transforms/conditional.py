"""Implementations of autoregressive transforms."""

import numpy as np
import torch
from torch.nn import functional as F

from flowcon.transforms.base import Transform
from flowcon.transforms.splines.linear import linear_spline
from flowcon.transforms.splines import rational_quadratic
from flowcon.transforms.splines.rational_quadratic import (
    rational_quadratic_spline,
    unconstrained_rational_quadratic_spline,
)
from flowcon.transforms.UMNN import *

from flowcon.nn.nets import ResidualNet, MLP
from flowcon.utils import torchutils
from flowcon.transforms.orthogonal import ParametrizedHouseHolder, _apply_batchwise_transforms_nodet
from flowcon.transforms.adaptive_sigmoids import SumOfSigmoids
from typing import *


class ConditionalTransform(Transform):
    """Transforms each input variable with an invertible transformation, conditioned on some given input.
    """

    def __init__(self,
                 features,
                 hidden_features=64,
                 context_features=1,
                 num_blocks=2,
                 use_residual_blocks=True,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_batch_norm=False,
                 conditional_net: torch.nn.Module = None, ):

        super(ConditionalTransform, self).__init__()
        self.features = features

        if conditional_net is not None:
            assert isinstance(conditional_net, torch.nn.Module)
            self.conditional_net = conditional_net
        else:

            self.conditional_net = self.set_default_network(activation, context_features, dropout_probability,
                                                            hidden_features, num_blocks,
                                                            use_batch_norm, use_residual_blocks)

    def set_default_network(self, activation, context_features, dropout_probability, hidden_features, num_blocks,
                            use_batch_norm, use_residual_blocks):
        if use_residual_blocks:
            conditional_net = ResidualNet(in_features=context_features,
                                          out_features=self._num_parameters(),
                                          hidden_features=hidden_features,
                                          activation=activation,
                                          num_blocks=num_blocks,
                                          dropout_probability=dropout_probability,
                                          use_batch_norm=use_batch_norm
                                          )
        else:
            conditional_net = MLP(in_shape=(context_features,),
                                  out_shape=(self._num_parameters(),),
                                  hidden_sizes=[hidden_features] * num_blocks)
            if dropout_probability > 1e-12:
                raise NotImplementedError("No dropout for MLP")
            if use_batch_norm:
                raise NotImplementedError("No batch norm for MLP")
        return conditional_net

    def _num_parameters(self):
        return self.features * self._output_dim_multiplier()

    def forward(self, inputs, context=None):
        if context is None:
            raise TypeError("Conditional transforms require a context.")
        conditional_params = self.conditional_net(context)
        outputs, logabsdet = self._forward_given_params(inputs, conditional_params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if context is None:
            raise TypeError("Conditional transforms require a context.")
        conditional_params = self.conditional_net(context)
        outputs, logabsdet = self._inverse_given_params(inputs, conditional_params)
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _forward_given_params(self, inputs, autoregressive_params):
        raise NotImplementedError()

    def _inverse_given_params(self, inputs, autoregressive_params):
        raise NotImplementedError()


class AffineConditionalTransform(ConditionalTransform):
    def __init__(
            self,
            features,
            hidden_features,
            context_features,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
    ):
        super(AffineConditionalTransform, self).__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )

    def _output_dim_multiplier(self):
        return 2

    def _forward_given_params(self, inputs, conditional_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            conditional_params
        )
        # scale = torch.sigmoid(unconstrained_scale + 2.0) + self._epsilon
        scale = F.softplus(unconstrained_scale) + self._epsilon
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _inverse_given_params(self, inputs, conditional_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            conditional_params
        )
        # scale = torch.sigmoid(unconstrained_scale + 2.0) + self._epsilon
        scale = F.softplus(unconstrained_scale) + self._epsilon
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, conditional_params):
        conditional_params = conditional_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        unconstrained_scale = conditional_params[..., 0]
        shift = conditional_params[..., 1]
        return unconstrained_scale, shift


class ConditionalShiftTransform(ConditionalTransform):
    def     __init__(
            self,
            features,
            hidden_features,
            context_features,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            **kwargs
    ):
        self.features = features
        super(ConditionalShiftTransform, self).__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            **kwargs
        )

    def _output_dim_multiplier(self):
        return 1

    def _forward_given_params(self, inputs, conditional_params):
        shift = self._unconstrained_shift(
            conditional_params
        )
        # scale = torch.sigmoid(unconstrained_scale + 2.0) + self._epsilon
        outputs = inputs + shift.view(inputs.shape)
        logabsdet = inputs.new_zeros(inputs.shape[0])
        return outputs, logabsdet

    def _inverse_given_params(self, inputs, conditional_params):
        shift = self._unconstrained_shift(
            conditional_params
        )
        # scale = torch.sigmoid(unconstrained_scale + 2.0) + self._epsilon
        outputs = (inputs - shift.view(inputs.shape))
        logabsdet = inputs.new_zeros(inputs.shape[0])
        return outputs, logabsdet

    def _unconstrained_shift(self, conditional_params):
        # split_idx = autoregressive_params.size(1) // 2
        # unconstrained_scale = autoregressive_params[..., :split_idx]
        # shift = autoregressive_params[..., split_idx:]
        # return unconstrained_scale, shift
        conditional_params = conditional_params.view(
            -1, self.features
        )
        shift = conditional_params  # [..., None]
        return shift


class ConditionalScaleTransform(ConditionalTransform):
    def __init__(
            self,
            features,
            hidden_features,
            context_features,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            **kwargs
    ):
        self.features = features
        super(ConditionalScaleTransform, self).__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            **kwargs
        )
        self.eps = 1e-5

    def _output_dim_multiplier(self):
        return 1


    def _forward_given_params(self, inputs, conditional_params):
        scale = self._constrained_scale(
            conditional_params
        )
        outputs = inputs * scale.view(inputs.shape)
        logabsdet = torch.log(scale).sum(-1).view(inputs.shape[0])

        return outputs, logabsdet

    def _inverse_given_params(self, inputs, conditional_params):
        scale = self._constrained_scale(
            conditional_params
        )
        outputs = inputs / scale.view(inputs.shape)
        logabsdet = -torch.log(scale).sum(-1).view(inputs.shape[0])

        return outputs, logabsdet

    def _constrained_scale(self, conditional_params):
        # split_idx = autoregressive_params.size(1) // 2
        # unconstrained_scale = autoregressive_params[..., :split_idx]
        # shift = autoregressive_params[..., split_idx:]
        # return unconstrained_scale, shift
        conditional_params = conditional_params.view(
            -1, self.features
        )
        scale = torch.nn.functional.softplus(conditional_params) + self.eps
        return scale


class ConditionalLUTransform(ConditionalTransform):
    def __init__(
            self,
            features,
            hidden_features,
            context_features,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            eps=1e-7
    ):
        super(ConditionalLUTransform, self).__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )

        self.eps = eps

        self.lower_indices = np.tril_indices(features, k=-1)
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)

        self.softplus = torch.nn.Softplus()
        self.diag_entries = torch.nn.Parameter(torch.eye(features), requires_grad=False)
        self.default_pivot = torch.nn.Parameter(torch.arange(1, self.features + 1, dtype=torch.int32).unsqueeze(0),
                                                requires_grad=False)
        self.scale_non_diag = torch.nn.Parameter(- 2 * torch.ones(()), requires_grad=True)

    def _output_dim_multiplier(self):
        return self.features

    def _forward_given_params(self, inputs, conditional_params):
        lower, upper = self._create_lower_upper(
            conditional_params
        )
        outputs = upper @ inputs.unsqueeze(-1)
        outputs = (lower @ outputs).view(inputs.shape)
        logabsdet = upper.diagonal(0, -1, -2).log().sum(-1)  # ...#inputs.new_ones(inputs.shape[0])
        return outputs, logabsdet

    def _inverse_given_params(self, inputs, conditional_params):
        lower, upper = self._create_lower_upper(
            conditional_params
        )
        outputs = torch.linalg.lu_solve(torch.tril(lower, -1) + upper,
                                        torch.broadcast_to(self.default_pivot, inputs.shape),
                                        inputs.unsqueeze(-1))

        logabsdet = -upper.diagonal(0, -1, -2).log().sum(-1)  # ...#inputs.new_ones(inputs.shape[0])
        return outputs.view(inputs.shape), logabsdet

    def _unconstrained_entries(self, conditional_params):
        conditional_params = conditional_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        return conditional_params

    def _create_lower_upper(self, conditional_params):
        unconstrained_matrix_vals = self._unconstrained_entries(conditional_params)

        lower = torch.nn.functional.softplus(self.scale_non_diag) * torch.tril(unconstrained_matrix_vals, diagonal=-1) + self.diag_entries
        upper_diag = torch.diag_embed(self.softplus(unconstrained_matrix_vals.diagonal(0, -1, -2)) + self.eps)
        upper = torch.nn.functional.softplus(self.scale_non_diag) * torch.triu(unconstrained_matrix_vals, diagonal=1) + upper_diag
        return lower, upper


class ConditionalRotationTransform(ConditionalTransform):
    def _output_dim_multiplier(self):
        pass

    def __init__(
            self,
            features,
            hidden_features,
            context_features,
            num_blocks=1,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
    ):
        assert features == 2, "Only available for 2D rotations."

        super(ConditionalRotationTransform, self).__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )

    def _num_parameters(self):
        return 1

    def build_matrix(self, conditional_params):
        theta = conditional_params
        m1 = torch.cos(theta)
        m2 = -torch.sin(theta)
        m3 = -m2
        m4 = m1
        matrix = torch.concatenate([m1, m2, m3, m4], -1).view(-1, self.features, self.features)
        return matrix

    def _forward_given_params(self, inputs: torch.Tensor, conditional_params):
        matrix = self.build_matrix(conditional_params)

        outputs = (matrix @ inputs.unsqueeze(-1)).squeeze()
        logabsdet = inputs.new_zeros(inputs.shape[0])  # ...#inputs.new_ones(inputs.shape[0])
        return outputs, logabsdet

    def _inverse_given_params(self, inputs, conditional_params):
        matrix = self.build_matrix(conditional_params)

        outputs = (torch.transpose(matrix, dim0=-2, dim1=-1) @ inputs.unsqueeze(-1)).squeeze()
        logabsdet = inputs.new_zeros(inputs.shape[0])  # ...#inputs.new_ones(inputs.shape[0])
        return outputs, logabsdet


class ConditionalOrthogonalTransform(ConditionalTransform):
    def __init__(
            self,
            features,
            hidden_features,
            context_features,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
    ):
        super(ConditionalOrthogonalTransform, self).__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )

    def _output_dim_multiplier(self):
        return self.features

    def _forward_given_params(self, inputs, conditional_params):
        householder = self._get_matrices(conditional_params)
        outputs, logabsdet = householder.forward(inputs)
        return outputs, logabsdet

    def _inverse_given_params(self, inputs, conditional_params):
        householder = self._get_matrices(conditional_params)
        outputs, logabsdet = householder.inverse(inputs)

        return outputs.squeeze(), logabsdet

    def _get_matrices(self, conditional_params) -> ParametrizedHouseHolder:
        q_vectors = self._unconstrained_entries(
            conditional_params
        )
        householder = ParametrizedHouseHolder(q_vectors)
        return householder

    def _unconstrained_entries(self, conditional_params):
        conditional_params = conditional_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        return conditional_params


class ConditionalSVDTransform(ConditionalTransform):
    def __init__(
            self,
            features,
            hidden_features,
            context_features,
            use_bias=True,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            eps=1e-3,
            lipschitz_constant_limit=None
    ):
        self.use_bias = use_bias

        super(ConditionalSVDTransform, self).__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )
        self.eps = eps
        self.lipschitz_constant = lipschitz_constant_limit

        self._epsilon = 1e-2

    def _output_dim_multiplier(self):
        multiplier_orthogonal_matrices = self.features * 2
        multiplier_diagonal_matrix = 1

        multiplier_bias = 1 if self.use_bias else 0

        return multiplier_orthogonal_matrices + multiplier_diagonal_matrix + multiplier_bias

    def _forward_given_params(self, inputs, conditional_params):
        householder_U, diag_entries_S, householder_Vt, bias = self._get_matrices(conditional_params)

        VtX, _ = householder_Vt.forward(inputs)
        SVtX = VtX * diag_entries_S
        USVtX, _ = householder_U.forward(SVtX)
        outputs = USVtX + bias if self.use_bias else USVtX

        logabsdet = diag_entries_S.log().sum(-1)  # |det(SVD)| = product of singular values

        return outputs, logabsdet

    def _inverse_given_params(self, inputs, conditional_params):
        householder_U, diag_entries_S, householder_Vt, bias = self._get_matrices(conditional_params)

        y = inputs - bias if self.use_bias else inputs
        Uty, _ = householder_U.inverse(y)
        SinvUty = Uty / diag_entries_S
        outputs, _ = householder_Vt.inverse(SinvUty)

        logabsdet = - diag_entries_S.log().sum(-1)  # |det(SVD)| = product of singular values

        return outputs.squeeze(), logabsdet

    def _get_matrices(self, conditional_params):
        q_vectors_U, q_vectors_V, diag_entries_S_unconstrained, bias = self._unconstrained_params(
            conditional_params
        )
        householder_U = ParametrizedHouseHolder(q_vectors_U)
        householder_Vt = ParametrizedHouseHolder(q_vectors_V)
        if self.lipschitz_constant is not None:
            diag_entries_S = torch.sigmoid(diag_entries_S_unconstrained) * (
                    self.lipschitz_constant - self.eps) + self.eps
        else:
            diag_entries_S = torch.exp(diag_entries_S_unconstrained) + self.eps

        return householder_U, diag_entries_S, householder_Vt, bias

    def _unconstrained_params(self, conditional_params):
        output_shapes = [self.features ** 2, self.features ** 2, self.features, self.features]

        if self.use_bias:
            q_vectors_U, q_vectors_Vt, diag_entries_S, bias = torch.split(conditional_params, output_shapes, -1)
        else:
            q_vectors_U, q_vectors_Vt, diag_entries_S = torch.split(conditional_params, output_shapes[:-1], -1)
            bias = None

        return q_vectors_U.view(-1, self.features, self.features), q_vectors_Vt.view(-1, self.features,
                                                                                     self.features), diag_entries_S, bias


class ConditionalUMNNTransform(ConditionalTransform):
    """An unconstrained monotonic neural networks autoregressive layer that transforms the variables.

        Reference:
        > A. Wehenkel and G. Louppe, Unconstrained Monotonic Neural Networks, NeurIPS2019.

        ---- Specific arguments ----
        integrand_net_layers: the layers dimension to put in the integrand network.
        cond_size: The embedding size for the conditioning factors.
        nb_steps: The number of integration steps.
        solver: The quadrature algorithm - CC or CCParallel. Both implements Clenshaw-Curtis quadrature with
        Leibniz rule for backward computation. CCParallel pass all the evaluation points (nb_steps) at once, it is faster
        but requires more memory.
        """

    def __init__(
            self,
            features,
            hidden_features,
            context_features=None,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            integrand_net_layers=[50, 50, 50],
            cond_size=20,
            nb_steps=20,
            solver="CCParallel",
    ):
        self.cond_size = cond_size

        super().__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )
        self.transformer = MonotonicNormalizer(integrand_net_layers, cond_size, nb_steps, solver)

    def _output_dim_multiplier(self):
        return self.cond_size

    def _forward_given_params(self, inputs, conditional_params):
        z, jac = self.transformer(inputs, conditional_params.reshape(inputs.shape[0], inputs.shape[1], -1))
        log_det_jac = jac.log().sum(1)
        return z, log_det_jac

    def _inverse_given_params(self, inputs, conditional_params):
        x = self.transformer.inverse_transform(inputs,
                                               conditional_params.reshape(inputs.shape[0], inputs.shape[1], -1))
        z, jac = self.transformer(x, conditional_params.reshape(inputs.shape[0], inputs.shape[1], -1))
        log_det_jac = -jac.log().sum(1)
        return x, log_det_jac


class PiecewiseLinearConditionalTransform(ConditionalTransform):
    def __init__(
            self,
            num_bins,
            features,
            hidden_features,
            context_features=None,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
    ):
        self.num_bins = num_bins

        super().__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )

    def _output_dim_multiplier(self):
        return self.num_bins

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_pdf = autoregressive_params.view(
            batch_size, self.features, self._output_dim_multiplier()
        )

        outputs, logabsdet = linear_spline(
            inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse,
            left=-4.0, right=4.0, bottom=-4.0, top=4.0
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _forward_given_params(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _inverse_given_params(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)


class ConditionalPiecewiseRationalQuadraticTransform(ConditionalTransform):
    def __init__(
            self,
            features,
            hidden_features,
            context_features=None,
            num_bins=10,
            tails=None,
            tail_bound=1.0,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            min_bin_width=rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
            min_bin_height=rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
            min_derivative=rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        super().__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins: 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins:]

        if hasattr(self.conditional_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.conditional_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.conditional_net.hidden_features)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {"left": -1.2, "right": 1.2, "bottom": -1.2, "top": 1.2}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            enable_identity_init=True,
            **spline_kwargs
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _forward_given_params(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _inverse_given_params(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)


class ConditionalSumOfSigmoidsTransform(ConditionalTransform):
    def __init__(
            self,
            features,
            hidden_features,
            context_features=None,
            n_sigmoids=10,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
    ):
        self.n_sigmoids = n_sigmoids
        super().__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )

    def _output_dim_multiplier(self):
        return 3 * self.n_sigmoids + 1

    def _forward_given_params(self, inputs, autoregressive_params):
        transformer = SumOfSigmoids(n_sigmoids=self.n_sigmoids, features=self.features,
                                    raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                          self._output_dim_multiplier()))

        z, logabsdet = transformer(inputs)
        return z, logabsdet

    def _inverse_given_params(self, inputs, autoregressive_params):
        transformer = SumOfSigmoids(n_sigmoids=self.n_sigmoids, features=self.features,
                                    raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                          self._output_dim_multiplier()))
        x, logabsdet = transformer.inverse(inputs)
        return x, logabsdet


class ConditionalPlanarTransform(ConditionalTransform):
    def __init__(
            self,
            features,
            hidden_features,
            context_features,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
    ):
        super(ConditionalPlanarTransform, self).__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )

        self.softplus = torch.nn.Softplus()
        self.diag_entries = torch.nn.Parameter(torch.eye(features, device="cuda"), requires_grad=False)

        self.default_pivot = torch.nn.Parameter(torch.arange(1, self.features + 1, dtype=torch.int32).unsqueeze(0),
                                                requires_grad=False)

    def _num_parameters(self):
        return self.features * self._output_dim_multiplier() + self._constant_dim_addition()

    def _output_dim_multiplier(self):
        return 2

    def _constant_dim_addition(self):
        return 1

    def _forward_given_params(self, inputs, conditional_params):
        u_, w_, b_ = self._create_uwb(
            conditional_params
        )
        pre_act = torch.bmm(inputs.view(-1, 1, self.features), torch.transpose(w_, dim0=-2, dim1=-1)).squeeze() + b_.squeeze()
        outputs = inputs + (u_ * torch.tanh(pre_act).view(-1, 1, 1)).squeeze()

        # calc logabsdet
        psi = (1 - torch.tanh(pre_act) ** 2).view(-1, 1, 1) * w_
        abs_det = (1 + torch.bmm(u_, torch.transpose(psi, dim0=-2, dim1=-1))).abs()
        logabsdet = torch.log(1e-7 + abs_det).squeeze()
        return outputs, logabsdet

    def _inverse_given_params(self, inputs, conditional_params):
        raise NotImplementedError()

    def get_u_hat(self, _u, _w):
        """Enforce w^T u >= -1. When using h(.) = tanh(.), this is a sufficient condition
        for invertibility of the transformation f(z). See Appendix A.1.
        """
        wtu = torch.bmm(_u, torch.transpose(_w, dim0=-2, dim1=-1))

        m_wtu = -1 + torch.log(1 + torch.exp(wtu))
        return _u + (m_wtu - wtu) * _w / torch.norm(_w, p=2, dim=-1, keepdim=True) ** 2
        # self.u.data = (
        #         self.u + (m_wtu - wtu) * self.w / torch.norm(self.w, p=2, dim=1) ** 2
        # )

    def _create_uwb(self, conditional_params):
        _b = conditional_params[..., -1:]
        unconstrained_matrix_vals = conditional_params[..., :-1].view(
            -1, self.features, self._output_dim_multiplier()
        )
        _u, _w = unconstrained_matrix_vals[..., 0], unconstrained_matrix_vals[..., 1]
        _u = _u[:, None, :]
        _w = _w[:, None, :]
        _u = self.get_u_hat(_u, _w)
        return _u, _w, _b


def h(x):
    return torch.tanh(x)


def dh_dx(x):
    return 1 - torch.tanh(x) ** 2


class ConditionalSylvesterTransform(ConditionalTransform):
    def __init__(
            self,
            features: int,
            hidden_features: int,
            context_features: int,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            eps=1e-3
    ):

        self.eps = eps

        self.features = features
        self.__output_splits = self._output_splits()
        super(ConditionalSylvesterTransform, self).__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )
        self.features = torch.tensor(features)
        self.features = torch.nn.Parameter(self.features, requires_grad=False)
        self.reverse_idx = torch.nn.Parameter(torch.arange(self.features - 1, -1, -1), requires_grad=False)
        self.triu_mask = torch.nn.Parameter(
            torch.triu(torch.ones(self.features, self.features).unsqueeze(0), diagonal=1), requires_grad=False)
        self.identity = torch.nn.Parameter(torch.eye(features, features).unsqueeze(0), requires_grad=False)

    def _output_splits(self) -> List[int]:
        # R1, R2
        # splits = [self.n_diag_entries + self.n_triangular_entries] * 2
        splits = [self.features ** 2, self.features]
        # Q via ParameterizedHouseholder
        splits += [self.features ** 2]
        # b
        splits += [self.features]

        # R1_diag, R1_triu, R2_diag, R2_triu, q_unconstrained, b
        return splits

    # def _num_parameters(self):

    def _num_parameters(self):
        return sum(self.__output_splits)

    def _forward_given_params(self, inputs, conditional_params):
        R1, R2, Q, bias = self._create_mats(
            conditional_params
        )

        logabsdet, outputs = self.forward_jit(Q, R1, R2, bias, inputs)
        return outputs, logabsdet

    @staticmethod
    @torch.jit.script
    def forward_jit(orth_Q: torch.Tensor, triu_R1: torch.Tensor, triu_R2: torch.Tensor, bias: torch.Tensor,
                    inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Qtz = torch.transpose(orth_Q, dim0=-2, dim1=-1) @ inputs.unsqueeze(-1)
        # Qtz = Q_orthogonal.inverse(inputs)[0].unsqueeze(-1)  # (n, d, 1)
        RQtz = triu_R1 @ Qtz  # (n, d, 1)
        preact = RQtz + bias
        # QR2act = Q_orthogonal.forward((R2 @ h(preact)).squeeze())[0]
        QR2act = orth_Q @ (triu_R2 @ h(preact))
        outputs = inputs + QR2act.squeeze()
        # calc logabsdet
        deriv_act = dh_dx(preact).squeeze()
        R_sq_diag = (torch.diagonal(triu_R1, dim1=-2, dim2=-1) * torch.diagonal(triu_R2, dim1=-2,
                                                                                dim2=-1)).squeeze()  # (n, d)
        diag = R_sq_diag.new_ones(inputs.shape[-1]) + deriv_act * R_sq_diag
        logabsdet = torch.log(diag).sum(-1)
        return logabsdet, outputs

    def _inverse_given_params(self, inputs, conditional_params):
        raise NotImplementedError()

    def _create_mats(self, conditional_params) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        R_full, R2_diag, q_unconstrained, b = torch.split(conditional_params, self.__output_splits, -1)

        R1, R2 = self._create_upper(R_full.view(-1, self.features, self.features), R2_diag, self.triu_mask)
        b: torch.Tensor = b.view(-1, self.features, 1)
        q_vectors: torch.Tensor = q_unconstrained.view(-1, self.features, self.features)

        # Q_householder = ParametrizedHouseHolder(q_vectors=q_vectors)
        # Q = Q_householder.matrix()
        Q = self.build_orthogonal_matrix(q_vectors)
        return R1, R2, Q, b

    def build_orthogonal_matrix(self, q_vectors: torch.Tensor):
        # identity = torch.repeat_interleave(identity[None, ...], conditional_params.shape[0], 0)
        outputs1 = _apply_batchwise_transforms_nodet(self.identity[:, 0, :], q_vectors[:, self.reverse_idx, :])
        outputs2 = _apply_batchwise_transforms_nodet(self.identity[:, 1, :], q_vectors[:, self.reverse_idx, :])
        Q = torch.stack([outputs1, outputs2], 1)
        return Q

    @staticmethod
    @torch.jit.script
    def _create_upper(full_matr_r, diag_vals, triu_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_1 = full_matr_r * triu_mask
        masked_2 = torch.transpose(full_matr_r, dim0=-2, dim1=-1) * triu_mask

        diag_1 = torch.diag_embed(torch.tanh(torch.diagonal(full_matr_r, dim1=-2, dim2=-1)), dim1=-2, dim2=-1)
        diag_2 = torch.diag_embed(torch.tanh(diag_vals), dim1=-2, dim2=-1)

        R1 = masked_1 + diag_1
        R2 = masked_2 + diag_2

        return R1, R2
