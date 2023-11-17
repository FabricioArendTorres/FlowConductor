from typing import Union

from enflows.distributions.base import Distribution
from enflows.utils import torchutils
import torch
from torch import distributions
from torch.distributions.utils import broadcast_all
from numbers import Number


class BoxUniform(distributions.Independent):
    def __init__(
            self,
            low: Union[torch.Tensor, float],
            high: Union[torch.Tensor, float],
            reinterpreted_batch_ndims: int = 1,
    ):
        """Multidimensionqal uniform distribution defined on a box.
        
        A `Uniform` distribution initialized with e.g. a parameter vector low or high of length 3 will result in a /batch/ dimension of length 3.
        A log_prob evaluation will then output three numbers, one for each of the independent Uniforms in the batch.
        Instead, a `BoxUniform` initialized in the same way has three /event/ dimensions, and returns a scalar log_prob corresponding to whether the evaluated point is in the box defined by low and high or outside.
    
        Refer to torch.distributions.Uniform and torch.distributions.Independent for further documentation.
    
        Args:
            low (Tensor or float): lower range (inclusive).
            high (Tensor or float): upper range (exclusive).
            reinterpreted_batch_ndims (int): the number of batch dims to
                                             reinterpret as event dims.
        """

        super().__init__(
            distributions.Uniform(low=low, high=high), reinterpreted_batch_ndims
        )


class MG1Uniform(distributions.Uniform):
    def log_prob(self, value):
        return super().log_prob(self._to_noise(value))

    def sample(self, sample_shape=torch.Size()):
        return self._to_parameters(super().sample(sample_shape))

    def _to_parameters(self, noise):
        A_inv = torch.tensor([[1.0, 1, 0], [0, 1, 0], [0, 0, 1]])
        return noise @ A_inv

    def _to_noise(self, parameters):
        A = torch.tensor([[1.0, -1, 0], [0, 1, 0], [0, 0, 1]])
        return parameters @ A


class Uniform(Distribution):
    def __init__(self,
                 low: Union[torch.Tensor, float] = 0.,
                 high: Union[torch.Tensor, float] = 1.,
                 reinterpreted_batch_ndims=1):
        super().__init__()

        if isinstance(low, Number) and isinstance(high, Number):
            self.low, self.high = broadcast_all(torch.tensor([low]), torch.tensor([high]))
        else:
            self.low, self.high = broadcast_all(low, high)
        shape = self.low.size()
        self._shape = shape
        # self.register_buffer("_shape", shape)

        self.dist = distributions.Independent(
            distributions.Uniform(low=low, high=high), reinterpreted_batch_ndims=reinterpreted_batch_ndims
        )

    def _log_prob(self, value, context):
        return self.dist.log_prob(value)

    def _sample(self, num_samples, context=None, batch_size=None):

        if context is None:
            samples = self.dist.sample((num_samples, *self._shape))
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = self.dist.sample((context_size * num_samples, *self._shape))
            samples = torchutils.split_leading_dim(samples, [context_size, num_samples])

        return samples


class LotkaVolterraOscillating:
    def __init__(self):
        mean = torch.log(torch.tensor([0.01, 0.5, 1, 0.01]))
        sigma = 0.5
        covariance = sigma ** 2 * torch.eye(4)
        self._gaussian = distributions.MultivariateNormal(
            loc=mean, covariance_matrix=covariance
        )
        self._uniform = BoxUniform(low=-5 * torch.ones(4), high=2 * torch.ones(4))
        self._log_normalizer = -torch.log(
            torch.erf((2 - mean) / sigma) - torch.erf((-5 - mean) / sigma)
        ).sum()

    def log_prob(self, value):
        unnormalized_log_prob = self._gaussian.log_prob(value) + self._uniform.log_prob(
            value
        )

        return self._log_normalizer + unnormalized_log_prob

    def sample(self, sample_shape=torch.Size()):
        num_remaining_samples = sample_shape[0]
        samples = []
        while num_remaining_samples > 0:
            candidate_samples = self._gaussian.sample((num_remaining_samples,))

            uniform_log_prob = self._uniform.log_prob(candidate_samples)

            accepted_samples = candidate_samples[~torch.isinf(uniform_log_prob)]
            samples.append(accepted_samples.detach())

            num_accepted = (~torch.isinf(uniform_log_prob)).sum().item()
            num_remaining_samples -= num_accepted

        # Aggregate collected samples.
        samples = torch.cat(samples)

        # Make sure we have the right amount.
        samples = samples[: sample_shape[0], ...]
        assert samples.shape[0] == sample_shape[0]

        return samples
