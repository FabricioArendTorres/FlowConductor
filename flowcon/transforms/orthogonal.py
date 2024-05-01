"""Implementations of orthogonal transforms."""

import torch
from torch import nn

from flowcon.transforms.base import Transform
from flowcon.utils import torchutils
import flowcon.utils.typechecks as check


class HouseholderSequence(Transform):
    """A sequence of Householder transforms.

    This class can be used as a way of parameterizing an orthogonal matrix.
    """

    def __init__(self, features, num_transforms):
        """Constructor.

        Args:
            features: int, dimensionality of the input.
            num_transforms: int, number of Householder transforms to use.

        Raises:
            TypeError: if arguments are not the right type.
        """
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        if not check.is_positive_int(num_transforms):
            raise TypeError("Number of transforms must be a positive integer.")

        super().__init__()
        self.features = features
        self.num_transforms = num_transforms
        # TODO: are randn good initial values?
        # these vectors are orthogonal to the hyperplanes through which we reflect
        # self.q_vectors = nets.Parameter(torch.randn(num_transforms, features))
        # self.q_vectors = nets.Parameter(torch.eye(num_transforms // 2, features))
        import numpy as np

        def tile(a, dim, n_tile):
            if a.nelement() == 0:
                return a
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))

            order_index = torch.Tensor(
                np.concatenate(
                    [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
                )
            ).long()
            return torch.index_select(a, dim, order_index)

        qv = tile(torch.eye(num_transforms // 2, features), 0, 2)

        if np.mod(num_transforms, 2) != 0:  # odd number of transforms, including 1
            qv = torch.cat((qv, torch.zeros(1, features)))
            qv[-1, num_transforms // 2] = 1
        self.q_vectors = nn.Parameter(qv)

    def forward(self, inputs, context=None):
        # return self._apply_transforms(inputs, self.q_vectors)
        return _apply_batchwise_transforms(inputs, self.q_vectors)

    def inverse(self, inputs, context=None):
        # Each householder transform is its own inverse, so the total inverse is given by
        # simply performing each transform in the reverse order.
        reverse_idx = torch.arange(self.num_transforms - 1, -1, -1)
        return _apply_batchwise_transforms(inputs, self.q_vectors[reverse_idx])

    def matrix(self):
        """Returns the orthogonal matrix that is equivalent to the total transform.

        Costs O(KD^2), where:
        - K is number of transforms
        - D is dimensionality of inputs

        Returns:
            A Tensor of shape [D, D].
        """
        identity = torch.eye(self.features, self.features, device=self.q_vectors.device)
        outputs, _ = self.inverse(identity)
        return outputs



class ParametrizedHouseHolder(Transform):
    """A sequence of Householder transforms.

    This class can be used as a way of parameterizing an orthogonal matrix.
    """

    def __init__(self, q_vectors):
        """Constructor.

        Args:
            features: int, dimensionality of the input.
            num_transforms: int, number of Householder transforms to use.

        Raises:
            TypeError: if arguments are not the right type.
        """

        super().__init__()
        self.features = q_vectors.shape[-1]
        self.num_transforms = q_vectors.shape[-2]
        self.q_vectors = q_vectors
        self.reverse_idx = torch.arange(self.num_transforms - 1, -1, -1).to(q_vectors.device)

    def forward(self, inputs, context=None):
        return _apply_batchwise_transforms(inputs, self.q_vectors)

    def inverse(self, inputs, context=None):
        # Each householder transform is its own inverse, so the total inverse is given by
        # simply performing each transform in the reverse order.
        return _apply_batchwise_transforms(inputs, self.q_vectors[..., self.reverse_idx, :])

    def matrix(self) -> torch.Tensor:
        """Returns the orthogonal matrix that is equivalent to the total transform.

        Costs O(KD^2), where:
        - K is number of transforms
        - D is dimensionality of inputs

        Returns:
            A Tensor of shape [D, D].
        """

        identity = torch.eye(self.features, self.features).to(self.q_vectors.device)
        if len(self.q_vectors.shape) > 2:
            identity = torch.repeat_interleave(identity[None, ...], self.q_vectors.shape[0], 0)
        # identity_flat = identity.view(self.features * self.q_vectors.shape[0], self.features)
        # outputs, _ = self.inverse(identity)
        outputs = []
        for i in range(self.features):
            outputs_, _ = self.inverse(identity[..., i, :])
            outputs.append(outputs_.unsqueeze(-2))
        outputs = torch.concatenate(outputs, -2)
        return outputs


@torch.jit.script
def _apply_batchwise_transforms_nodet(inputs, q_vectors):
    """Apply the sequence of transforms parameterized by given q_vectors to inputs.

    Costs O(KDN), where:
    - K is number of transforms
    - D is dimensionality of inputs
    - N is number of inputs

    Args:
        inputs: Tensor of shape [N, D]
        q_vectors: Tensor of shape [K, D]

    Returns:
        A tuple of:
        - A Tensor of shape [N, D], the outputs.
        - A Tensor of shape [N], the log absolute determinants of the total transform.
    """
    squared_norms = torch.sum(q_vectors ** 2, dim=-1)
    outputs = inputs
    for i in range(q_vectors.shape[-2]):
        q_vector = q_vectors[..., i, :]
        squared_norm = squared_norms[..., i].unsqueeze(-1)
        temp = torchutils.batchwise_dot_prod(outputs, q_vector)  # Inner product.
        temp = temp.unsqueeze(-1) * ((2.0 / squared_norm) * q_vector)
        outputs = outputs - temp

    return outputs

@torch.jit.script
def _apply_batchwise_transforms(inputs, q_vectors):
    """Apply the sequence of transforms parameterized by given q_vectors to inputs.

    Costs O(KDN), where:
    - K is number of transforms
    - D is dimensionality of inputs
    - N is number of inputs

    Args:
        inputs: Tensor of shape [N, D]
        q_vectors: Tensor of shape [K, D]

    Returns:
        A tuple of:
        - A Tensor of shape [N, D], the outputs.
        - A Tensor of shape [N], the log absolute determinants of the total transform.
    """
    outputs = _apply_batchwise_transforms_nodet(inputs, q_vectors)
    batch_size = outputs.shape[0]
    logabsdet = inputs.new_zeros(batch_size)
    return outputs, logabsdet

