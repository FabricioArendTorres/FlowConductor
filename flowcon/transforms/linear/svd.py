import numpy as np
import torch
from torch import nn
from torch.nn import init

from flowcon.transforms.linear.linear import Linear
from flowcon.transforms.linear.orthogonal import OrthogonalCaley


class SVDLinear(Linear):
    """A linear module using the SVD decomposition for the weight matrix."""

    def __init__(
        self,
        n_features: int,
        using_cache: bool = False,
        identity_init: bool = True,
        eps: float = 1e-3,
    ):
        super().__init__(n_features, using_cache)

        # minimum value for diagonal
        self.eps = eps
        # First orthogonal matrix (U).
        self.orthogonal_1 = OrthogonalCaley(n_features=n_features)

        # Logs of diagonal entries of the diagonal matrix (S).
        self.unconstrained_diagonal = nn.Parameter(torch.zeros(n_features))

        # Second orthogonal matrix (V^T).
        self.orthogonal_2 = OrthogonalCaley(n_features=n_features)

        self.identity_init = identity_init
        self._initialize()

    @property
    def diagonal(self):
        # return torchutils.map_to_0_1_stable(
        #     self.unconstrained_diagonal, epsilon=self.eps
        # )
        return torch.nn.functional.softplus(self.unconstrained_diagonal) + self.eps

    @property
    def log_diagonal(self):
        return torch.log(self.diagonal)

    def _initialize(self):
        init.zeros_(self.bias)
        if self.identity_init:
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_diagonal, constant)
        else:
            constant = np.log(np.exp(1 - self.eps) - 1)
            stdv = 1.0 / np.sqrt(self.n_features)
            constant += torch.randn_like(constant) * stdv
            init.constant_(self.unconstrained_diagonal, constant)

            init.uniform_(self.unconstrained_diagonal, -stdv, stdv)

    def forward_no_cache(self, inputs: torch.Tensor):
        """Cost:
            output = O(KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        outputs, _ = self.orthogonal_2(inputs)  # Ignore logabsdet as we know it's zero.
        outputs *= self.diagonal
        outputs, _ = self.orthogonal_1(
            outputs
        )  # Ignore logabsdet as we know it's zero.
        outputs += self.bias

        logabsdet_T = self.logabsdet() * outputs.new_ones(outputs.shape[0])

        return outputs, logabsdet_T

    def inverse_no_cache(self, inputs: torch.Tensor):
        """Cost:
            output = O(KDN)
            logabsdet = O(D)
        where:
            K = num of householder transforms
            D = num of features
            N = num of inputs
        """
        outputs = inputs - self.bias
        outputs, _ = self.orthogonal_1.inverse(
            outputs
        )  # Ignore logabsdet since we know it's zero.
        outputs /= self.diagonal
        outputs, _ = self.orthogonal_2.inverse(
            outputs
        )  # Ignore logabsdet since we know it's zero.
        logabsdet_Tinv = -self.logabsdet()
        logabsdet_Tinv = logabsdet_Tinv * outputs.new_ones(outputs.shape[0])
        return outputs, logabsdet_Tinv

    def weight(self):
        """Cost:
            weight = O(KD^2)
        where:
            K = num of householder transforms
            D = num of features
        """
        diagonal = torch.diag(self.diagonal)
        weight, _ = self.orthogonal_2.inverse(diagonal)
        weight, _ = self.orthogonal_1(weight.t())
        return weight.t()

    def weight_inverse(self):
        """Cost:
            inverse = O(KD^2)
        where:
            K = num of householder transforms
            D = num of features
        """
        diagonal_inv = torch.diag(torch.reciprocal(self.diagonal))
        weight_inv, _ = self.orthogonal_1(diagonal_inv)
        weight_inv, _ = self.orthogonal_2.inverse(weight_inv.t())
        return weight_inv.t()

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(self.log_diagonal)
