"""Implementations of various coupling layers."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Iterable

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.nn.functional import softplus

from flowcon.transforms.base import Transform
from flowcon.transforms.monotonic import (
    PiecewiseCubicCDF,
    PiecewiseLinearCDF,
    PiecewiseQuadraticCDF,
    PiecewiseRationalQuadraticCDF,
)
from flowcon.transforms.monotonic.MonotonicNormalizer import MonotonicNormalizer
from flowcon.transforms.monotonic.splines.util import cubic, linear, quadratic, rational_quadratic
from flowcon.utils import torchutils


class CouplingTransform(Transform):
    """A base class for coupling layers. Supports 2D inputs (NxD), as well as 4D inputs for
    images (NxCxHxW). For images the splitting is done on the channel dimension, using the
    provided 1D mask."""

    def __init__(
        self,
        mask: torch.Tensor | ArrayLike,
        transform_net_create_fn: Callable[[int, int], torch.nn.Module],
        unconditional_transform: Callable[[int], Transform] | None = None,
    ):
        """
        Constructor.

        Args:
            mask: a 1-dim tensor, tuple or list. It indexes inputs as follows:
                * If `mask[i] > 0`, `input[i]` will be transformed.
                * If `mask[i] <= 0`, `input[i]` will be passed unchanged.
        """
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError("Mask must be a 1-dim tensor.")
        if mask.numel() <= 0:
            raise ValueError("Mask can't be empty.")

        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)

        self.register_buffer("identity_features", features_vector.masked_select(mask <= 0))
        self.register_buffer("transform_features", features_vector.masked_select(mask > 0))

        assert self.num_identity_features + self.num_transform_features == self.features

        self.transform_net = transform_net_create_fn(
            self.num_identity_features,
            self.num_transform_features * self._transform_dim_multiplier(),
        )

        if unconditional_transform is None:
            self.unconditional_transform = None
        else:
            self.unconditional_transform = unconditional_transform(self.num_identity_features)

    @property
    def num_identity_features(self) -> int:
        return len(self.identity_features)

    @property
    def num_transform_features(self) -> int:
        return len(self.transform_features)

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError(
                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
            )

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet = self._coupling_transform_forward(
            inputs=transform_split, transform_params=transform_params
        )

        if self.unconditional_transform is not None:
            identity_split, logabsdet_identity = self.unconditional_transform(
                identity_split, context
            )
            logabsdet += logabsdet_identity

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split

        return outputs, logabsdet

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError(
                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
            )

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        logabsdet = 0.0
        if self.unconditional_transform is not None:
            identity_split, logabsdet = self.unconditional_transform.inverse(
                identity_split, context
            )

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet_split = self._coupling_transform_inverse(
            inputs=transform_split, transform_params=transform_params
        )
        logabsdet += logabsdet_split

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features] = identity_split
        outputs[:, self.transform_features] = transform_split

        return outputs, logabsdet

    def _transform_dim_multiplier(self) -> int:
        """Number of features to output for each transform dimension."""
        raise NotImplementedError()

    def _coupling_transform_forward(
        self, inputs: torch.Tensor, transform_params: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the coupling transform."""
        raise NotImplementedError()

    def _coupling_transform_inverse(
        self, inputs: torch.Tensor, transform_params: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of the coupling transform."""
        raise NotImplementedError()


class UMNNCouplingTransform(CouplingTransform):
    """An unconstrained monotonic neural networks coupling layer that transforms the variables.

    Reference:
    > A. Wehenkel and G. Louppe, Unconstrained Monotonic Neural Networks, NeurIPS2019.

    ---- Specific arguments ----
        integrand_net_layers: the layers dimension to put in the integrand network.
        cond_size: The embedding size for the conditioning factors.
        nb_steps: The number of integration steps.
        solver: The quadrature algorithm - CC or CCParallel.
        Both implements Clenshaw-Curtis quadrature with
        Leibniz rule for backward computation. CCParallel pass all the evaluation points (nb_steps)
        at once, it is faster
        but requires more memory.

    """

    def __init__(
        self,
        mask: torch.Tensor,
        transform_net_create_fn: Callable[[int, int], torch.nn.Module],
        integrand_net_layers: Iterable[int] = [50, 50, 50],
        cond_size: int = 20,
        nb_steps: int = 20,
        solver: str = "CCParallel",
        apply_unconditional_transform: bool = False,
    ):
        if apply_unconditional_transform:

            def unconditional_transform(features: int) -> Transform:
                return MonotonicNormalizer(  # type: ignore
                    integrand_net_layers, 0, nb_steps, solver
                )
        else:
            unconditional_transform = None  # type: ignore
        self.cond_size = cond_size
        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

        self.transformer = MonotonicNormalizer(integrand_net_layers, cond_size, nb_steps, solver)

    def _transform_dim_multiplier(self) -> int:
        return self.cond_size

    def _coupling_transform_forward(
        self, inputs: torch.Tensor, transform_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(inputs.shape) == 2:
            z, jac = self.transformer(
                inputs, transform_params.reshape(inputs.shape[0], inputs.shape[1], -1)
            )
            log_det_jac = jac.log().sum(1)
            return z, log_det_jac
        else:
            B, C, H, W = inputs.shape
            z, jac = self.transformer(
                inputs.permute(0, 2, 3, 1).reshape(-1, inputs.shape[1]),
                transform_params.permute(0, 2, 3, 1).reshape(-1, 1, transform_params.shape[1]),
            )
            log_det_jac = jac.log().reshape(B, -1).sum(1)
            return z.reshape(B, H, W, C).permute(0, 3, 1, 2), log_det_jac

    def _coupling_transform_inverse(
        self, inputs: torch.Tensor, transform_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(inputs.shape) == 2:
            x = self.transformer.inverse_transform(
                inputs, transform_params.reshape(inputs.shape[0], inputs.shape[1], -1)
            )
            _, jac = self.transformer(
                x, transform_params.reshape(inputs.shape[0], inputs.shape[1], -1)
            )
            log_det_jac = -jac.log().sum(1)
            return x, log_det_jac
        else:
            B, C, H, W = inputs.shape
            x = self.transformer.inverse_transform(
                inputs.permute(0, 2, 3, 1).reshape(-1, inputs.shape[1]),
                transform_params.permute(0, 2, 3, 1).reshape(-1, 1, transform_params.shape[1]),
            )
            _, jac = self.transformer(
                x, transform_params.permute(0, 2, 3, 1).reshape(-1, 1, transform_params.shape[1])
            )
            log_det_jac = -jac.log().reshape(B, -1).sum(1)
            return x.reshape(B, H, W, C).permute(0, 3, 1, 2), log_det_jac


def DEFAULT_SCALE_ACTIVATION(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x + 2) + 1e-3


def GENERAL_SCALE_ACTIVATION(x: torch.Tensor) -> torch.Tensor:
    return (softplus(x) + 1e-3).clamp(0, 3)


class AffineCouplingTransform(CouplingTransform):
    """An affine coupling layer that scales and shifts part of the variables.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.

    The user should supply `scale_activation`, the final activation function in the neural network
    producing the scale tensor.
    Two options are predefined in the class.
    `DEFAULT_SCALE_ACTIVATION` preserves backwards compatibility but only produces scales <= 1.001.
    `GENERAL_SCALE_ACTIVATION` produces scales <= 3, which is more useful in general applications.
    """

    # DEFAULT_SCALE_ACTIVATION = lambda x: torch.sigmoid(x + 2) + 1e-3
    # GENERAL_SCALE_ACTIVATION = lambda x: (softplus(x) + 1e-3).clamp(0, 3)

    def __init__(
        self,
        mask: torch.Tensor,
        transform_net_create_fn: Callable[[int, int], torch.nn.Module],
        unconditional_transform: Callable[[int], Transform] | None = None,
        scale_activation: Callable[[torch.Tensor], torch.Tensor] = DEFAULT_SCALE_ACTIVATION,
    ):
        self.scale_activation = scale_activation
        super().__init__(mask, transform_net_create_fn, unconditional_transform)

    def _transform_dim_multiplier(self) -> int:
        return 2

    def _scale_and_shift(self, transform_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unconstrained_scale = transform_params[:, self.num_transform_features :, ...]
        shift = transform_params[:, : self.num_transform_features, ...]
        scale = self.scale_activation(unconstrained_scale)
        return scale, shift

    def _coupling_transform_forward(
        self, inputs: torch.Tensor, transform_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = inputs * scale + shift
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _coupling_transform_inverse(
        self, inputs: torch.Tensor, transform_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet


class AdditiveCouplingTransform(AffineCouplingTransform):
    """An additive coupling layer, i.e. an affine coupling layer without scaling.

    Reference:
    > L. Dinh et al., NICE:  Non-linear  Independent  Components  Estimation,
    > arXiv:1410.8516, 2014.
    """

    def _transform_dim_multiplier(self) -> int:
        return 1

    def _scale_and_shift(self, transform_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift = transform_params
        scale = torch.ones_like(shift)
        return scale, shift


class PiecewiseCouplingTransform(CouplingTransform):
    def _coupling_transform_forward(
        self, inputs: torch.Tensor, transform_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._coupling_transform(inputs, transform_params, inverse=False)

    def _coupling_transform_inverse(
        self, inputs: torch.Tensor, transform_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._coupling_transform(inputs, transform_params, inverse=True)

    def _coupling_transform(
        self, inputs: torch.Tensor, transform_params: torch.Tensor, inverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.dim() == 4:
            b, c, h, w = inputs.shape
            # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
            transform_params = transform_params.reshape(b, c, -1, h, w).permute(0, 1, 3, 4, 2)
        elif inputs.dim() == 2:
            b, d = inputs.shape
            # For 2D data, reshape transform_params from Bx(D*?) to BxDx?
            transform_params = transform_params.reshape(b, d, -1)

        outputs, logabsdet = self._piecewise_cdf(inputs, transform_params, inverse)

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _piecewise_cdf(
        self, inputs: torch.Tensor, transform_params: torch.Tensor, inverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class PiecewiseLinearCouplingTransform(PiecewiseCouplingTransform):
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """

    def __init__(
        self,
        mask: torch.Tensor,
        transform_net_create_fn: Callable[[int, int], torch.nn.Module],
        num_bins: int = 10,
        tails: str | None = None,
        tail_bound: float = 1.0,
        apply_unconditional_transform: bool = False,
        img_shape: list[int] | None = None,
    ):
        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound

        if apply_unconditional_transform:

            def unconditional_transform(features: int) -> PiecewiseLinearCDF:
                _shape = img_shape if img_shape else []
                return PiecewiseLinearCDF(
                    shape=[features] + _shape,
                    num_bins=num_bins,
                    tails=tails,
                    tail_bound=tail_bound,
                )
        else:
            unconditional_transform = None  # type: ignore

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

    def _transform_dim_multiplier(self) -> int:
        return self.num_bins

    def _piecewise_cdf(
        self, inputs: torch.Tensor, transform_params: torch.Tensor, inverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unnormalized_pdf = transform_params

        if self.tails is None:
            return linear.linear_spline(
                inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse
            )
        else:
            return linear.unconstrained_linear_spline(
                inputs=inputs,
                unnormalized_pdf=unnormalized_pdf,
                inverse=inverse,
                tails=self.tails,
                tail_bound=self.tail_bound,
            )


class PiecewiseQuadraticCouplingTransform(PiecewiseCouplingTransform):
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """

    def __init__(
        self,
        mask: torch.Tensor,
        transform_net_create_fn: Callable[[int, int], torch.nn.Module],
        num_bins: int = 10,
        tails: str | None = None,
        tail_bound: float = 1.0,
        apply_unconditional_transform: bool = False,
        img_shape: list[int] | None = None,
        min_bin_width: float = quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = quadratic.DEFAULT_MIN_BIN_HEIGHT,
    ):
        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height

        if apply_unconditional_transform:

            def unconditional_transform(features: int) -> PiecewiseQuadraticCDF:
                _shape = img_shape if img_shape else []
                return PiecewiseQuadraticCDF(
                    shape=[features] + _shape,
                    num_bins=num_bins,
                    tails=tails,
                    tail_bound=tail_bound,
                    min_bin_width=min_bin_width,
                    min_bin_height=min_bin_height,
                )
        else:
            unconditional_transform = None  # type: ignore

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 2 - 1
        else:
            return self.num_bins * 2 + 1

    def _piecewise_cdf(
        self, inputs: torch.Tensor, transform_params: torch.Tensor, inverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins :]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)

        if self.tails is None:
            spline_fn = quadratic.quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = quadratic.unconstrained_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            **spline_kwargs,
        )


class PiecewiseCubicCouplingTransform(PiecewiseCouplingTransform):
    def __init__(
        self,
        mask: torch.Tensor,
        transform_net_create_fn: Callable[[int, int], torch.nn.Module],
        num_bins: int = 10,
        tails: str | None = None,
        tail_bound: float = 1.0,
        apply_unconditional_transform: bool = False,
        img_shape: list[int] | None = None,
        min_bin_width: float = cubic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = cubic.DEFAULT_MIN_BIN_HEIGHT,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.tails = tails
        self.tail_bound = tail_bound

        if apply_unconditional_transform:

            def unconditional_transform(features: int):
                _shape = img_shape if img_shape else []
                return PiecewiseCubicCDF(
                    shape=[features] + _shape,
                    num_bins=num_bins,
                    tails=tails,
                    tail_bound=tail_bound,
                    min_bin_width=min_bin_width,
                    min_bin_height=min_bin_height,
                )
        else:
            unconditional_transform = None  # type: ignore

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

    def _transform_dim_multiplier(self):
        return self.num_bins * 2 + 2

    def _piecewise_cdf(
        self, inputs: torch.Tensor, transform_params: torch.Tensor, inverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnorm_derivatives_left = transform_params[..., 2 * self.num_bins][..., None]
        unnorm_derivatives_right = transform_params[..., 2 * self.num_bins + 1][..., None]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)

        if self.tails is None:
            spline_fn = cubic.cubic_spline
            spline_kwargs = {}
        else:
            spline_fn = cubic.unconstrained_cubic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnorm_derivatives_left=unnorm_derivatives_left,
            unnorm_derivatives_right=unnorm_derivatives_right,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            **spline_kwargs,
        )


class PiecewiseRationalQuadraticCouplingTransform(PiecewiseCouplingTransform):
    def __init__(
        self,
        mask: torch.Tensor,
        transform_net_create_fn: Callable[[int, int], torch.nn.Module],
        num_bins: int = 10,
        tails: str | None = None,
        tail_bound: float = 1.0,
        apply_unconditional_transform: bool = False,
        img_shape: list[int] | None = None,
        min_bin_width: float = rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        if apply_unconditional_transform:

            def unconditional_transform(features: int) -> PiecewiseRationalQuadraticCDF:
                return PiecewiseRationalQuadraticCDF(
                    shape=[features] + (img_shape if img_shape else []),
                    num_bins=num_bins,
                    tails=tails,
                    tail_bound=tail_bound,
                    min_bin_width=min_bin_width,
                    min_bin_height=min_bin_height,
                    min_derivative=min_derivative,
                )
        else:
            unconditional_transform = None  # type: ignore

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

    def _transform_dim_multiplier(self) -> int:
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        else:
            return self.num_bins * 3 + 1

    def _piecewise_cdf(
        self, inputs: torch.Tensor, transform_params: torch.Tensor, inverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn("Inputs to the softmax are not scaled down: initialization might be bad.")

        if self.tails is None:
            spline_fn = rational_quadratic.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = rational_quadratic.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs,
        )
