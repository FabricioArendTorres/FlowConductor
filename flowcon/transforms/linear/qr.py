from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from flowcon.transforms.linear.linear import Linear
from flowcon.transforms.linear.orthogonal import (
    OrthogonalBase,
    OrthogonalCaley,
    OrthogonalHouseholder,
    OrthogonalHouseholderGEQR,
)

__all__ = ["QRLinear"]


class QRLinear(Linear):
    """A linear module using the QR decomposition for the weight matrix."""

    def __init__(
        self,
        n_features: int,
        using_cache: bool = False,
        orthogonal_parametrization: str = "OrthogonalCaley",
    ):
        super().__init__(n_features, using_cache)

        # Parameterization for R
        self.upper_indices = np.triu_indices(n_features, k=1)
        self.diag_indices = np.diag_indices(n_features)
        n_triangular_entries = ((n_features - 1) * n_features) // 2
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.log_upper_diag = nn.Parameter(torch.zeros(n_features))

        # Parameterization for Q
        self.orthogonal: OrthogonalBase = {
            "OrthogonalCaley": OrthogonalCaley,
            "OrthogonalHouseholderGEQR": OrthogonalHouseholderGEQR,
            "OrthogonalHouseholder": OrthogonalHouseholder,
        }[orthogonal_parametrization](n_features=n_features)

        self._initialize()

    def _initialize(self) -> None:
        stdv = 1.0 / np.sqrt(self.n_features)
        init.uniform_(self.upper_entries, -stdv, stdv)
        init.uniform_(self.log_upper_diag, -stdv, stdv)
        init.constant_(self.bias, 0.0)

    def _create_upper(self) -> torch.Tensor:
        upper = self.upper_entries.new_zeros(self.n_features, self.n_features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = torch.exp(self.log_upper_diag)
        return upper

    def forward_no_cache(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cost:
            output = O(D^2N + KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        upper = self._create_upper()

        outputs = F.linear(inputs, upper)
        outputs, _ = self.orthogonal(outputs)  # Ignore logabsdet as we know it's zero.
        outputs += self.bias

        logabsdet = self.logabsdet() * outputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def inverse_no_cache(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cost:
            output = O(D^2N + KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        upper = self._create_upper()
        outputs = inputs - self.bias
        outputs, _ = self.orthogonal.inverse(outputs)  # Ignore logabsdet since we know it's zero.
        outputs = cast(torch.Tensor, torch.linalg.solve_triangular(upper, outputs.t(), upper=True))
        outputs = outputs.t()
        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * outputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def weight(self) -> torch.Tensor:
        """Cost:
            weight = O(KD^2)
        where:
            K = num of householder transforms
            D = num of features
        """
        upper = self._create_upper()
        weight, _ = self.orthogonal(upper.t())
        return weight.t()

    def weight_inverse(self) -> torch.Tensor:
        """Cost:
            inverse = O(D^3 + KD^2)
        where:
            K = num of householder transforms
            D = num of features
        """
        upper = self._create_upper()
        identity = torch.eye(self.n_features, self.n_features)
        upper_inv = cast(torch.Tensor, torch.linalg.solve_triangular(upper, identity, upper=True))
        weight_inv, _ = self.orthogonal(upper_inv)
        return weight_inv

    def logabsdet(self) -> torch.Tensor:
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(self.log_upper_diag)
