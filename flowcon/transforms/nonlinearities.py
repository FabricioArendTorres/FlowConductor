"""Implementations of invertible non-linearities."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn
from torch.nn import functional as F

from flowcon.transforms.base import (
    InputOutsideDomain,
    Inverse,
    Sequential,
    Transform,
)
from flowcon.utils import torchutils


class Exp(Transform):
    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = torch.exp(inputs)
        logabsdet = torchutils.sum_except_batch(inputs, num_batch_dims=1)

        return outputs, logabsdet

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.min(inputs) <= 0.0:
            raise InputOutsideDomain()

        outputs = torch.log(inputs)
        logabsdet = -torchutils.sum_except_batch(outputs, num_batch_dims=1)

        return outputs, logabsdet


class Tanh(Transform):
    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = torch.tanh(inputs)
        logabsdet = torch.log(1 - outputs**2)
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.min(inputs) <= -1 or torch.max(inputs) >= 1:
            raise InputOutsideDomain()
        outputs = 0.5 * torch.log((1 + inputs) / (1 - inputs))
        logabsdet = -torch.log(1 - inputs**2)
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet


class LogTanh(Transform):
    """Tanh with unbounded output.

    Constructed by selecting a cut_point, and replacing values to the right of cut_point
    with alpha * log(beta * x), and to the left of -cut_point with -alpha * log(-beta *
    x). alpha and beta are set to match the value and the first derivative of tanh at
    cut_point."""

    def __init__(self, cut_point: int = 1):
        if cut_point <= 0:
            raise ValueError("Cut point must be positive.")
        super().__init__()

        self.cut_point = cut_point
        self.inv_cut_point = np.tanh(cut_point)

        self.alpha = (1 - np.tanh(np.tanh(cut_point))) / cut_point
        self.beta = np.exp((np.tanh(cut_point) - self.alpha * np.log(cut_point)) / self.alpha)

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_right = inputs > self.cut_point
        mask_left = inputs < -self.cut_point
        mask_middle = ~(mask_right | mask_left)

        outputs = torch.zeros_like(inputs)
        outputs[mask_middle] = torch.tanh(inputs[mask_middle])
        outputs[mask_right] = self.alpha * torch.log(self.beta * inputs[mask_right])
        outputs[mask_left] = self.alpha * -torch.log(-self.beta * inputs[mask_left])

        logabsdet = torch.zeros_like(inputs)
        logabsdet[mask_middle] = torch.log(1 - outputs[mask_middle] ** 2)
        logabsdet[mask_right] = torch.log(self.alpha / inputs[mask_right])
        logabsdet[mask_left] = torch.log(-self.alpha / inputs[mask_left])
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)

        return outputs, logabsdet

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_right = inputs > self.inv_cut_point
        mask_left = inputs < -self.inv_cut_point
        mask_middle = ~(mask_right | mask_left)

        outputs = torch.zeros_like(inputs)
        outputs[mask_middle] = 0.5 * torch.log(
            (1 + inputs[mask_middle]) / (1 - inputs[mask_middle])
        )
        outputs[mask_right] = torch.exp(inputs[mask_right] / self.alpha) / self.beta
        outputs[mask_left] = -torch.exp(-inputs[mask_left] / self.alpha) / self.beta

        logabsdet = torch.zeros_like(inputs)
        logabsdet[mask_middle] = -torch.log(1 - inputs[mask_middle] ** 2)
        logabsdet[mask_right] = -np.log(self.alpha * self.beta) + inputs[mask_right] / self.alpha
        logabsdet[mask_left] = -np.log(self.alpha * self.beta) - inputs[mask_left] / self.alpha
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)

        return outputs, logabsdet


class LeakyReLU(Transform):
    def __init__(self, negative_slope: float = 1e-2):
        if negative_slope <= 0:
            raise ValueError("Slope must be positive.")
        super().__init__()
        # self.device = device
        self.negative_slope = negative_slope
        self.log_negative_slope = torch.nn.Parameter(
            torch.log(torch.as_tensor(self.negative_slope))
        )  # .to(device)

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = F.leaky_relu(inputs, negative_slope=self.negative_slope)
        mask = torch.as_tensor(inputs < 0, device=inputs.device)
        logabsdet = self.log_negative_slope * mask
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = F.leaky_relu(inputs, negative_slope=(1 / self.negative_slope))
        mask = torch.as_tensor(inputs < 0, device=inputs.device)
        logabsdet = -self.log_negative_slope * mask
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet


class Sigmoid(Transform):
    def __init__(self, temperature: float = 1, eps: float = 1e-6, learn_temperature: bool = False):
        super().__init__()
        self.eps = eps
        # if learn_temperature:
        self.temperature = nn.Parameter(
            torch.Tensor([temperature]), requires_grad=learn_temperature
        )
        # else:
        #     temperature = torch.Tensor([temperature])
        #     self.register_buffer("temperature", temperature)

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.temperature * inputs
        outputs = torch.sigmoid(inputs)
        logabsdet = torchutils.sum_except_batch(
            torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs)
        )
        return outputs, logabsdet

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()

        inputs = torch.clamp(inputs, self.eps, 1 - self.eps)

        outputs = (1 / self.temperature) * (torch.log(inputs) - torch.log1p(-inputs))
        logabsdet = -torchutils.sum_except_batch(
            torch.log(self.temperature)
            - F.softplus(-self.temperature * outputs)
            - F.softplus(self.temperature * outputs)
        )
        return outputs, logabsdet


class Softplus(Transform):
    def __init__(self, threshold: int = 20, eps: float = 0.0):
        super().__init__()

        self.eps = eps
        self.softplus = torch.nn.Softplus(beta=1, threshold=threshold)
        self.log_sigmoid = torch.nn.LogSigmoid()

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.softplus(inputs) + self.eps
        logabsdet = self.log_sigmoid(inputs).sum(-1)
        return outputs, logabsdet

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = inputs - self.eps
        outputs = torch.where(inputs > self.softplus.threshold, inputs, inputs.expm1().log())
        logabsdet = -torch.log(-torch.expm1(-inputs)).sum(-1)
        return outputs, logabsdet


class Logit(Inverse):
    def __init__(self, temperature: float = 1, eps: float = 1e-6):
        super().__init__(Sigmoid(temperature=temperature, eps=eps))


class GatedLinearUnit(Transform):
    def __init__(self):
        super().__init__()

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert context is not None
        gate = torch.sigmoid(context)
        # return inputs * (1 + gate), torch.log(torch.ones_like(gate) + gate).reshape(-1)
        return inputs * gate, torch.log(gate).reshape(-1)

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert context is not None
        gate = torch.sigmoid(context)
        # return inputs / (1 + gate), - torch.log(torch.ones_like(gate) + gate).reshape(-1)
        return inputs / gate, -torch.log(gate).reshape(-1)


class CauchyCDF(Transform):
    def __init__(self):
        super().__init__()

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = (1 / np.pi) * torch.atan(inputs) + 0.5
        logabsdet = torchutils.sum_except_batch(-np.log(np.pi) - torch.log(1 + inputs**2))
        return outputs, logabsdet

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()

        outputs = torch.tan(np.pi * (inputs - 0.5))
        logabsdet = -torchutils.sum_except_batch(-np.log(np.pi) - torch.log(1 + outputs**2))
        return outputs, logabsdet


class CauchyCDFInverse(Inverse):
    def __init__(self):
        super().__init__(CauchyCDF())


class CompositeCDFTransform(Sequential):
    def __init__(self, squashing_transform: Transform, cdf_transform: Transform):
        super().__init__(
            [
                squashing_transform,
                cdf_transform,
                Inverse(squashing_transform),
            ]
        )


class ExtendedSoftplus(torch.nn.Module):
    """
    Combination of a (shifted and scaled) softplus and the same softplus flipped around the origin

    Softplus(scale * (x-shift)) - Softplus(-scale * (x + shift))

    Linear outside of origin, flat around origin.
    """

    def __init__(self, features: int, shift: torch.Tensor | ArrayLike | None = None):
        self.features = features
        super(ExtendedSoftplus, self).__init__()
        if shift is None:
            self.shift = torch.nn.Parameter(torch.ones(1, features) * 3, requires_grad=True)
        elif isinstance(shift, torch.Tensor):
            self.shift = shift.reshape(-1, features)
        else:
            self.shift = torch.nn.Parameter(torch.tensor(shift), requires_grad=True)

        self._softplus = torch.nn.Softplus()

    def get_shift(self) -> torch.Tensor:
        return self._softplus(self.shift) + 1e-1

    def softplus(self, x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return self._softplus(x - shift)

    def softminus(self, x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return -self._softplus(-(x + shift))

    def diag_jacobian_pos(self, x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # (b e^(b x))/(e^(a b) + e^(b x))
        return torch.exp(x) / (torch.exp(shift) + torch.exp(x))

    def log_diag_jacobian_pos(self, x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # -log(e^(a b) + e^(b x)) + b x + log(b)
        log_jac = -torch.logaddexp(shift, x) + x
        return log_jac

    def diag_jacobian_neg(self, x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(-(shift + x))

    def log_diag_jacobian_neg(self, x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return -self._softplus(shift + x)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs = inputs.requires_grad_()
        shift = self.get_shift()
        outputs = self.softplus(inputs, shift) + self.softminus(inputs, shift)
        # ref_batch_jacobian = torchutils.batch_jacobian(outputs, inputs)
        # ref_logabsdet = torchutils.logabsdet(ref_batch_jacobian)
        # breakpoint()
        diag_jacobian = torch.logaddexp(
            self.log_diag_jacobian_pos(inputs, shift),
            self.log_diag_jacobian_neg(inputs, shift),
        )
        return outputs, diag_jacobian  # torch.log(diag_jacobian).sum(-1)
