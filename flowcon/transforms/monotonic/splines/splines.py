from typing import Iterable

import numpy as np
import torch
from torch import nn

from flowcon.transforms.base import (
    Transform,
)
from flowcon.utils import torchutils

from .util import cubic, linear, quadratic, rational_quadratic


class PiecewiseLinearCDF(Transform):
    def __init__(
        self,
        shape: Iterable[int],
        num_bins: int = 10,
        tails: str | None = None,
        tail_bound: float = 1.0,
    ):
        super().__init__()

        self.tail_bound = tail_bound
        self.tails = tails

        self.unnormalized_pdf = nn.Parameter(torch.randn(*shape, num_bins))

    def _spline(
        self, inputs: torch.Tensor, inverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]

        unnormalized_pdf = _share_across_batch(self.unnormalized_pdf, batch_size)

        if self.tails is None:
            outputs, logabsdet = linear.linear_spline(
                inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse
            )
        else:
            outputs, logabsdet = linear.unconstrained_linear_spline(
                inputs=inputs,
                unnormalized_pdf=unnormalized_pdf,
                inverse=inverse,
                tails=self.tails,
                tail_bound=self.tail_bound,
            )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._spline(inputs, inverse=False)

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._spline(inputs, inverse=True)


class PiecewiseQuadraticCDF(Transform):
    def __init__(
        self,
        shape: Iterable[int],
        num_bins: int = 10,
        tails: str | None = None,
        tail_bound: float = 1.0,
        min_bin_width: float = quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = quadratic.DEFAULT_MIN_BIN_HEIGHT,
    ):
        super().__init__()
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.tail_bound = tail_bound
        self.tails = tails

        self.unnormalized_widths = nn.Parameter(torch.randn(*shape, num_bins))
        if tails is None:
            self.unnormalized_heights = nn.Parameter(torch.randn(*shape, num_bins + 1))
        else:
            self.unnormalized_heights = nn.Parameter(torch.randn(*shape, num_bins - 1))

    def _spline(
        self, inputs: torch.Tensor, inverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]

        unnormalized_widths = _share_across_batch(self.unnormalized_widths, batch_size)
        unnormalized_heights = _share_across_batch(self.unnormalized_heights, batch_size)

        if self.tails is None:
            spline_fn = quadratic.quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = quadratic.unconstrained_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            **spline_kwargs,
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._spline(inputs, inverse=False)

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._spline(inputs, inverse=True)


class PiecewiseCubicCDF(Transform):
    def __init__(
        self,
        shape: Iterable[int],
        num_bins: int = 10,
        tails: str | None = None,
        tail_bound: float = 1.0,
        min_bin_width: float = cubic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = cubic.DEFAULT_MIN_BIN_HEIGHT,
    ):
        super().__init__()

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.tail_bound = tail_bound
        self.tails = tails

        self.unnormalized_widths = nn.Parameter(torch.randn(*shape, num_bins))
        self.unnormalized_heights = nn.Parameter(torch.randn(*shape, num_bins))
        self.unnorm_derivatives_left = nn.Parameter(torch.randn(*shape, 1))
        self.unnorm_derivatives_right = nn.Parameter(torch.randn(*shape, 1))

    def _spline(
        self, inputs: torch.Tensor, inverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]

        unnormalized_widths = _share_across_batch(self.unnormalized_widths, batch_size)
        unnormalized_heights = _share_across_batch(self.unnormalized_heights, batch_size)
        unnorm_derivatives_left = _share_across_batch(self.unnorm_derivatives_left, batch_size)
        unnorm_derivatives_right = _share_across_batch(self.unnorm_derivatives_right, batch_size)

        if self.tails is None:
            spline_fn = cubic.cubic_spline
            spline_kwargs = {}
        else:
            spline_fn = cubic.unconstrained_cubic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
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

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._spline(inputs, inverse=False)

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._spline(inputs, inverse=True)


class PiecewiseRationalQuadraticCDF(Transform):
    def __init__(
        self,
        shape: Iterable[int],
        num_bins: int = 10,
        tails: str | None = None,
        tail_bound: float = 1.0,
        identity_init: bool = False,
        min_bin_width: float = rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    ):
        super().__init__()

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        self.tail_bound = tail_bound
        self.tails = tails

        if isinstance(shape, int):
            shape = (shape,)
        if identity_init:
            self.unnormalized_widths = nn.Parameter(torch.zeros(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.zeros(*shape, num_bins))

            constant = np.log(np.exp(1 - min_derivative) - 1)
            num_derivatives = (num_bins - 1) if self.tails == "linear" else (num_bins + 1)
            self.unnormalized_derivatives = nn.Parameter(
                constant * torch.ones(*shape, num_derivatives)
            )
        else:
            self.unnormalized_widths = nn.Parameter(torch.rand(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.rand(*shape, num_bins))

            num_derivatives = (num_bins - 1) if self.tails == "linear" else (num_bins + 1)
            self.unnormalized_derivatives = nn.Parameter(torch.rand(*shape, num_derivatives))

    def _spline(
        self, inputs: torch.Tensor, inverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]

        unnormalized_widths = _share_across_batch(self.unnormalized_widths, batch_size)
        unnormalized_heights = _share_across_batch(self.unnormalized_heights, batch_size)
        unnormalized_derivatives = _share_across_batch(self.unnormalized_derivatives, batch_size)

        if self.tails is None:
            spline_fn = rational_quadratic.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = rational_quadratic.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
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

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._spline(inputs, inverse=False)

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._spline(inputs, inverse=True)


def _share_across_batch(params: torch.Tensor, batch_size: int) -> torch.Tensor:
    return params[None, ...].expand(batch_size, *params.shape)
