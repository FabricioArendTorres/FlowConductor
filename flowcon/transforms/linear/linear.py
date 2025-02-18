"""Implementations of linear transforms."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

import flowcon.utils.typechecks as check
from flowcon.transforms.base import Transform
from flowcon.utils import torchutils

__all__ = ["Linear", "NaiveLinear", "ScalarScale", "ScalarShift"]


class LinearCache(nn.Module):
    """
    Stores the cache of a linear transform using PyTorch buffers.
    Makes use of buffers to support torch.compile.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.empty(0), persistent=False)
        self.register_buffer("inverse", torch.empty(0), persistent=False)
        self.register_buffer("logabsdet", torch.empty(0), persistent=False)

    def reset_cache(self):
        """
        Resets cached values.
        """
        self.weight = torch.empty(0)
        self.inverse = torch.empty(0)
        self.logabsdet = torch.empty(0)

    def is_weight_empty(self) -> bool:
        return self.weight.numel() == 0

    def is_inverse_empty(self) -> bool:
        return self.inverse.numel() == 0

    def is_logabsdet_empty(self) -> bool:
        return self.logabsdet.numel() == 0


class Linear(Transform):
    """Abstract base class for linear transforms that parameterize a weight matrix."""

    def __init__(self, n_features: int, using_cache: bool = False):
        if not check.is_positive_int(n_features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()

        self.n_features = n_features
        self.bias = nn.Parameter(torch.zeros(n_features))

        # Caching flag and values.
        self.using_cache = using_cache
        self.cache = LinearCache()

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training and self.using_cache:
            self._update_forward_cache()
            outputs = F.linear(inputs, self.cache.weight, self.bias)
            logabsdet = self.cache.logabsdet * outputs.new_ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.forward_no_cache(inputs)

    def _update_forward_cache(self) -> None:
        """
        Check if the caches for the weight matrix and logabsdet are empty,
        and if so, recompute and update them.
        """
        if self.cache.is_weight_empty() and self.cache.is_logabsdet_empty():
            self.cache.weight, self.cache.logabsdet = self.weight_and_logabsdet()
        elif self.cache.is_weight_empty():
            self.cache.weight = self.weight()
        elif self.cache.is_logabsdet_empty():
            self.cache.logabsdet = self.logabsdet()

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training and self.using_cache:
            self._update_inverse_cache()
            outputs = F.linear(inputs - self.bias, self.cache.inverse)
            logabsdet = (-self.cache.logabsdet) * outputs.new_ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.inverse_no_cache(inputs)

    def _update_inverse_cache(self) -> None:
        if self.cache.is_inverse_empty() and self.cache.is_logabsdet_empty():
            self.cache.inverse, self.cache.logabsdet = (
                self.weight_inverse_and_forwardlogabsdet()
            )

        elif self.cache.is_inverse_empty():
            self.cache.inverse = self.weight_inverse()

        elif self.cache.is_logabsdet_empty():
            self.cache.logabsdet = self.logabsdet()

    def train(self, mode: bool = True):
        if mode:
            # If training again, invalidate cache.
            self.cache.reset_cache()
        return super().train(mode)

    def use_cache(self, mode: bool = True) -> None:
        if not check.is_bool(mode):
            raise TypeError("Mode must be boolean.")
        self.using_cache = mode

    def weight_and_logabsdet(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        To be overridden by subclasses if it is more efficient to compute the weight matrix
        and its logabsdet together.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Weight matrix, and logabsdet.
        """
        return self.weight(), self.logabsdet()

    def weight_inverse_and_forwardlogabsdet(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        To be overridden by subclasses if it is more efficient to compute the weight matrix
        inverse and weight matrix logabsdet together.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Inverse of the weight matrix, and logabsdet of the forward transform.
        """
        return self.weight_inverse(), self.logabsdet()

    def forward_no_cache(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies `forward` method without using the cache."""
        raise NotImplementedError()

    def inverse_no_cache(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies `inverse` method without using the cache."""
        raise NotImplementedError()

    def weight(self) -> torch.Tensor:
        """Returns the weight matrix."""
        raise NotImplementedError()

    def weight_inverse(self) -> torch.Tensor:
        """Returns the inverse weight matrix."""
        raise NotImplementedError()

    def logabsdet(self) -> torch.Tensor:
        """Returns the log absolute determinant of the weight matrix."""
        raise NotImplementedError()


class NaiveLinear(Linear):
    """A general linear transform that uses an unconstrained weight matrix.

    This transform explicitly computes the log absolute determinant in the forward direction
    and uses a linear solver in the inverse direction.

    Both forward and inverse directions have a cost of O(D^3), where D is the dimension
    of the input.

    Only supports inputs of shape [mb_size, num_features], i.e. 2-dimensional tensors.
    """

    def __init__(
        self,
        n_features: int,
        orthogonal_initialization: bool = True,
        using_cache: bool = False,
    ):
        """
        Constructor.

        Parameters
        ----------
        num_features : int
            Number of features in the input, i.e. dimension of the matrix.
        orthogonal_initialization : bool, optional
            if True initialize weights to be a random orthogonal matrix, by default True.
        using_cache : bool, optional
            Whether to use cache in non-training mode, by default False
        """
        super().__init__(n_features, using_cache)

        if orthogonal_initialization:
            self._weight = nn.Parameter(torchutils.random_orthogonal(n_features))
        else:
            self._weight = nn.Parameter(torch.empty(n_features, n_features))
            stdv = 1.0 / np.sqrt(n_features)
            init.uniform_(self._weight, -stdv, stdv)

    def forward_no_cache(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cost:
            output = O(D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs

        Parameters
        ----------
        inputs : torch.Tensor
            Input values.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Transformed values, logabsdet of forward transform.
        """
        batch_size = inputs.shape[0]
        outputs = F.linear(inputs, self._weight, self.bias)
        logabsdet_T = torchutils.logabsdet(self._weight)
        logabsdet_T = logabsdet_T * outputs.new_ones(batch_size)
        return outputs, logabsdet_T

    def inverse_no_cache(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cost:
            output = O(D^3 + D^2N)
            logabsdet = O(D^3)
        where:
            D = num of features
            N = num of inputs

        Parameters
        ----------
        inputs : torch.Tensor
            Input values

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Transformed values, logabsdet of inverse transform.
        """
        batch_size = inputs.shape[0]
        outputs = inputs - self.bias
        # LU-decompose the weights and solve for the outputs.
        lu, lu_pivots = cast(
            tuple[torch.Tensor, torch.Tensor], torch.linalg.lu_factor(self._weight)
        )
        outputs = cast(
            torch.Tensor, torch.linalg.lu_solve(lu, lu_pivots, outputs.t()).t()
        )
        # The linear-system solver returns the LU decomposition of the weights, which we
        # can use to obtain the log absolute determinant directly.
        logabsdet_Tinv = -torch.sum(torch.log(torch.abs(torch.diag(lu))))
        logabsdet_Tinv = logabsdet_Tinv * outputs.new_ones(batch_size)
        return outputs, logabsdet_Tinv

    def weight(self) -> torch.Tensor:
        """
        Cost:
        weight = O(1)

        Returns
        -------
        torch.Tensor
            Weight matrix, tensor of shape [self.num_features, self.num_features]
        """
        return self._weight

    def weight_inverse(self) -> torch.Tensor:
        """
        Cost:
            inverse = O(D^3)
        where:
            D = num of features

        Returns
        -------
        torch.Tensor
            Inverse of the Weight matrix, tensor of shape [self.num_features, self.num_features]
        """
        return torch.inverse(self._weight)

    def weight_inverse_and_forwardlogabsdet(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cost:
            inverse = O(D^3)
            logabsdet = O(D)
        where:
            D = num of features

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Inverse of the Weight matrix, tensor of shape [self.num_features, self.num_features].
            Logabsdet of the inverse transform, scalar tensor.
        """
        # If both weight inverse and logabsdet are needed, it's cheaper to compute both together.
        identity = torch.eye(self.n_features, self.n_features)
        # LU-decompose the weights and solve for the outputs.
        lu, lu_pivots = cast(tuple[torch.Tensor, torch.Tensor], torch.lu(self._weight))
        weight_inv = torch.lu_solve(identity, lu, lu_pivots)
        logabsdet = torch.sum(torch.log(torch.abs(torch.diag(lu))))
        return weight_inv, logabsdet

    def logabsdet(self):
        """Cost:
            logabsdet = O(D^3)
        where:
            D = num of features
        """
        return torchutils.logabsdet(self._weight)


class ScalarScale(Transform):
    """
    A transformation that scales the input tensor element-wise by a trainable scalar.

    This transformation applies the function:

        T(x) = scale * x, where scale = softplus(scale_latent) + eps,

    where `scale_latent` is a trainable parameter, and `eps` is a small positive value
    added for numerical stability.

    Attributes
    ----------
    _scale : torch.nn.Parameter
        Trainable parameter that determines the scaling factor.
    eps : torch.Tensor
        A small constant added to the scale for numerical stability.


    """

    def __init__(self, scale: float = 1.0, trainable: bool = True, eps: float = 1e-5):
        """
        Constructor.

        Parameters
        ----------
        scale : float, optional
            Initial value for the scaling factor. Must be greater than 1e-6, by default 1.0
        trainable : bool, optional
            Whether the scale parameter is trainable, by default True
        eps : _type_, optional
            A small positive value added to the scale for stability.
            Must be non-negative. Default is 1e-5.
        """
        super().__init__()
        assert np.all(scale > 1e-6), "Scale too small.."
        assert np.all(eps >= 0.0), "The eps must be non-negative"

        # inverse of softplus
        scale_latent = scale + np.log(-np.expm1(-scale))

        self._scale = nn.Parameter(
            torch.tensor(scale_latent, dtype=torch.get_default_dtype()),
            requires_grad=trainable,
        )
        self.register_buffer("eps", torch.tensor(eps), persistent=True)

    @property
    def scale(self) -> torch.Tensor:
        """
        Computes the current scale value as `softplus(_scale) + eps`.

        Returns
        -------
        torch.Tensor
            A positive scale value, scalar tensor.
        """
        return torch.nn.functional.softplus(self._scale) + self.eps

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.scale * inputs

        logabsdet = (
            inputs.new_ones(inputs.shape[0])
            * torch.log(self.scale).sum()
            * np.sum(inputs.shape[1:])
        )
        return outputs, logabsdet

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = inputs * (1.0 / self.scale)
        logabsdet = (
            -inputs.new_ones(inputs.shape[0])
            * torch.log(self.scale).sum()
            * np.sum(inputs.shape[1:])
        )
        return outputs, logabsdet


class ScalarShift(Transform):
    """
    A transformation that shifts the input tensor element-wise by a trainable scalar.

    This transformation applies:

        T(x) = x + s,

    where `s` is a trainable parameter representing the shift.

    Parameters
    ----------
    shift : float, optional
        Initial value for the shift parameter. Default is 0.0.
    trainable : bool, optional
        Whether the shift parameter is trainable. Default is True.

    Attributes
    ----------
    shift : torch.nn.Parameter
        Trainable parameter that determines the shift value.

    """

    def __init__(self, shift: float = 0.0, trainable: bool = True):
        super().__init__()
        self.shift = nn.Parameter(
            torch.tensor(shift, dtype=torch.get_default_dtype()),
            requires_grad=trainable,
        )

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = inputs + self.shift
        return outputs, inputs.new_zeros(inputs.shape[0])

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = inputs - self.shift
        return outputs, inputs.new_zeros(inputs.shape[0])
