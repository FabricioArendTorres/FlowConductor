from typing import cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from flowcon.transforms.linear import Linear


class LULinear(Linear):
    """A linear transform where we parameterize the LU decomposition of the weights."""

    def __init__(
        self,
        num_features: int,
        using_cache: bool = False,
        identity_init: bool = False,
        eps: float = 1e-3,
    ):
        super().__init__(num_features, using_cache)

        self.eps = eps

        self._raw_matrix = nn.Parameter(torch.empty(num_features, num_features))

        self.lower_indices = np.tril_indices(num_features, k=-1)
        self.upper_indices = np.triu_indices(num_features, k=1)
        self.diag_indices = np.diag_indices(num_features)

        self.register_buffer("eye", torch.eye(num_features), persistent=True)

        self._initialize(identity_init)

    def _initialize(self, identity_init: bool):
        # inverse softplus to ensure 1-diagonal in softplus transformed diagonal
        raw_diagonal_constant = np.log(np.exp(1 - self.eps) - 1)

        with torch.no_grad():
            if identity_init:
                init.eye_(self._raw_matrix)
                self._raw_matrix.fill_diagonal_(raw_diagonal_constant)
                init.zeros_(self.bias)

            else:
                self._raw_matrix.data.copy_(
                    torch.randn_like(self._raw_matrix) * 1e-2
                )  # Add small noise
                raw_diagonal_constant = np.log(np.expm1(1 - self.eps))
                self._raw_matrix.fill_diagonal_(raw_diagonal_constant)
                init.constant_(self.bias, 1e-3)

    def get_lower_upper(self):
        # lower triangular values + ones on diagonal
        lower = torch.tril(self._raw_matrix, diagonal=-1) + self.eye
        # upper triangular values with zeros on diagonal
        upper = torch.triu(self._raw_matrix, diagonal=1)
        # set diagonal
        # preserves gradient flow, does not create new values, looks nice
        # i did not find a better solution..
        upper.diagonal()[:] = self.upper_diag_flat

        return lower, upper

    def forward_no_cache(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self.get_lower_upper()
        outputs = F.linear(inputs, upper)
        outputs = F.linear(outputs, lower, self.bias)
        logabsdet = self.logabsdet() * inputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def inverse_no_cache(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self.get_lower_upper()
        outputs = inputs - self.bias
        outputs = cast(
            torch.Tensor,
            torch.linalg.solve_triangular(
                lower, outputs.t(), upper=False, unitriangular=True
            ),
        )
        outputs = cast(
            torch.Tensor,
            torch.linalg.solve_triangular(
                upper, outputs, upper=True, unitriangular=False
            ),
        )
        outputs = outputs.t()

        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def weight(self) -> torch.Tensor:
        """Cost:
            weight = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self.get_lower_upper()
        return lower @ upper

    def weight_inverse(self) -> torch.Tensor:
        """Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self.get_lower_upper()
        lower_inverse = cast(
            torch.Tensor,
            torch.linalg.solve_triangular(
                lower, self.eye, upper=False, unitriangular=True
            ),
        )
        weight_inverse = cast(
            torch.Tensor,
            torch.linalg.solve_triangular(
                upper, lower_inverse, upper=True, unitriangular=False
            ),
        )
        return weight_inverse

    @property
    def upper_diag_flat(self) -> torch.Tensor:
        return F.softplus(self._raw_matrix.diagonal()) + self.eps

    def logabsdet(self) -> torch.Tensor:
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(torch.log1p(self.upper_diag_flat - 1))
