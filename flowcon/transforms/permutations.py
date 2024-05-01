"""Implementations of permutation-like transforms."""

import torch

from flowcon.transforms.base import Transform
import flowcon.utils.typechecks as check
import numpy as np


class Permutation(Transform):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation, dim=1):
        if permutation.ndimension() != 1:
            raise ValueError("Permutation must be a 1D tensor.")
        if not check.is_positive_int(dim):
            raise ValueError("dim must be a positive integer.")

        super().__init__()
        self._dim = dim
        self.register_buffer("_permutation", permutation)

    @property
    def _inverse_permutation(self):
        return torch.argsort(self._permutation)

    @staticmethod
    def _permute(inputs, permutation, dim):
        if dim >= inputs.ndimension():
            raise ValueError("No dimension {} in inputs.".format(dim))
        if inputs.shape[dim] != len(permutation):
            raise ValueError(
                "Dimension {} in inputs must be of size {}.".format(
                    dim, len(permutation)
                )
            )
        batch_size = inputs.shape[0]
        outputs = torch.index_select(inputs, dim, permutation)
        logabsdet = inputs.new_zeros(batch_size)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._permute(inputs, self._permutation, self._dim)

    def inverse(self, inputs, context=None):
        return self._permute(inputs, self._inverse_permutation, self._dim)


class RandomPermutation(Permutation):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, features, dim=1):
        if not check.is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(torch.randperm(features), dim)


class ReversePermutation(Permutation):
    """Reverses the elements of the input. Only works with 1D inputs."""

    def __init__(self, features, dim=1):
        if not check.is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(torch.arange(features - 1, -1, -1), dim)


class FillTriangular(Transform):
    def __init__(self, features: int = None, matrix_dimension: int = None):
        super().__init__()

        if (features is not None) and (matrix_dimension is None):
            self.features = features
            self.matrix_dim = self.calc_matrix_dimension(features)
        elif (features is None) and (matrix_dimension is not None):
            self.features = self.calc_n_ltri(matrix_dimension)
            self.matrix_dim = matrix_dimension
        else:
            raise ValueError("Provide either 'features' or 'full_matrix_dimension', but not both.")

        self.lower_indices = np.tril_indices(self.matrix_dim, k=0)

    @staticmethod
    def calc_matrix_dimension(n_ltri_entries):
        assert n_ltri_entries > 0, f"Dimension must be positive, but is {n_ltri_entries}"
        temp = 1 + 8 * n_ltri_entries
        assert np.square(
            np.floor(np.sqrt(temp))) == temp, "invalid dimension: can't be mapped to lower triangular matrix"
        matrix_dim = int((-1 + np.floor(np.sqrt(temp))) // 2)
        return matrix_dim

    @staticmethod
    def calc_n_ltri(matrix_dim):
        return int((matrix_dim * (matrix_dim + 1)) / 2)

    def forward(self, inputs, context=None):
        assert inputs.shape[-1] == self.features

        mb = inputs.shape[0]
        outputs = inputs.new_zeros((mb, self.matrix_dim, self.matrix_dim))
        outputs[:, self.lower_indices[0], self.lower_indices[1]] = inputs

        logabsdet = inputs.new_zeros(mb)

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        assert inputs.shape[-2:] == (self.matrix_dim, self.matrix_dim)

        outputs = inputs[:, self.lower_indices[0], self.lower_indices[1]]

        logabsdet = inputs.new_zeros(inputs.shape[0])

        return outputs, logabsdet
