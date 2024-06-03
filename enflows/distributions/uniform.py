from typing import Union

import torch
from torch import distributions
from enflows.distributions.base import Distribution
from enflows.utils import torchutils


class BoxUniform(distributions.Independent):
    def __init__(
        self,
        low: Union[torch.Tensor, float],
        high: Union[torch.Tensor, float],
        reinterpreted_batch_ndims: int = 1,
    ):
        """Multidimensionqal uniform distribution defined on a box.
        
        A `Uniform` distribution initialized with e.g. a parameter vector low or high of length 3 will result in a /batch/ dimension of length 3. A log_prob evaluation will then output three numbers, one for each of the independent Uniforms in the batch. Instead, a `BoxUniform` initialized in the same way has three /event/ dimensions, and returns a scalar log_prob corresponding to whether the evaluated point is in the box defined by low and high or outside. 
    
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


class Uniform(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape, low, high):
        super().__init__()
        self._shape = torch.Size(shape)
        self._low = low
        self._high = high

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        assert torch.all(inputs.le(self._high)) and torch.all(inputs.ge(self._low))

        log_prob = - torch.log(self._high - self._low) * inputs.shape[-1] # self._shape

        return inputs.new_ones(inputs.shape[:1]) * log_prob

    def _sample(self, num_samples, context):
        if context is None:
            samples = torch.rand((num_samples, *self._shape), device=self._low.device)
            return self._low + samples * (self._high - self._low)
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = torch.rand((context_size * num_samples, *self._shape), device=self._low.device)
            samples = self._low + samples * (self._high - self._low)
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

class MultimodalUniform(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape, low, high, n_modes):
        super().__init__()
        self._shape = torch.Size(shape)
        self._low = low
        self._high = high
        self._n_modes = n_modes
        self.build_means()

    def build_means(self):
        self.means = torch.linspace(self._low.item(), self._high.item(), self._n_modes * 2, device=self._low.device)[::2]
        assert self.means.shape[0] == self._n_modes
        self.scale = (self._high - self._low) / (self._n_modes * 2 - 1)


    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        assert torch.all(inputs.le(self._high)) and torch.all(inputs.ge(self._low))

        log_prob = inputs.new_ones(inputs.shape[:1]) * -12
        # first we create a mask to checks if in each dimension the samples are within the range specified by self.means
        mask = torch.any( (inputs.unsqueeze(-1) > self.means) * (inputs.unsqueeze(-1) < self.means+self.scale), dim=-1)
        # then the mask checks if all dimensions are within that range
        mask = torch.all(mask, dim=-1)
        log_prob[mask] = - torch.log(self.scale * self._n_modes)

        return log_prob

    def _sample(self, num_samples, context):
        assert num_samples % self._n_modes == 0
        if context is None:
            samples = torch.rand((num_samples, *self._shape), device=self._low.device)
            samples = samples.reshape(self._n_modes, num_samples//self._n_modes, *self._shape) * self.scale + self.means.reshape(-1,1,1)
            return samples.reshape(num_samples, *self._shape)
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = torch.rand((context_size * num_samples, *self._shape), device=self._low.device)
            samples = samples.reshape(context_size, self._n_modes, num_samples//self._n_modes, *self._shape) * self.scale + self.means.reshape(1,-1,1,1)
            return samples.reshape(context_size, num_samples, *self._shape)


import numpy as np
import scipy as sp
from enflows.transforms.injective.utils import logabsdet_sph_to_car, cartesian_to_spherical_torch, spherical_to_cartesian_torch

class UniformSphere(Distribution):
    """Uniform distribution on a (d+1)-sphere. Probabilities are defined over d angles"""

    def __init__(self, shape, all_positive=False):
        super().__init__()
        self._shape = torch.Size(shape)
        self.radius = 1.
        self.all_positive = all_positive
        self.compute_log_surface()
        self.register_buffer("_log_z", torch.tensor(self.log_surface_area, dtype=torch.float64), persistent=False)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        radius = torch.ones_like(inputs[:,:1]) * self.radius
        jacobian = logabsdet_sph_to_car(torch.cat((inputs, radius), dim=-1))
        return jacobian - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            samples = torch.randn(num_samples, *self._shape, device=self._log_z.device)
            add_dim = torch.randn(num_samples, 1, device=self._log_z.device)
            samples = torch.cat((samples, add_dim), dim=-1)
            samples /= torch.norm(samples, dim=-1).reshape(-1, 1)
            samples *= self.radius
            if self.all_positive:
                samples = samples.abs()
            samples = cartesian_to_spherical_torch(samples)[:,:-1]
            assert len(samples.shape) == 2

            return samples
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = torch.randn(context_size * num_samples, *self._shape, device=context.device)
            add_dim = torch.randn(context_size * num_samples, 1, device=context.device)
            samples = torch.cat((samples, add_dim), dim=-1)
            samples /= torch.norm(samples, dim=-1).reshape(-1, 1)
            samples *= self.radius
            if self.all_positive:
                samples = samples.abs()
            samples = cartesian_to_spherical_torch(samples)[:, :-1]

            assert len(samples.shape) == 2

            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def compute_log_surface(self):
        dim = self._shape[-1]
        log_const_1 = np.log(2) + 0.5 * dim * np.log(np.pi)
        log_const_2 = (dim - 1) * np.log(self.radius)
        log_const_3 = - sp.special.loggamma(0.5 * dim)
        log_const_4 = - np.log(2)*dim if self.all_positive else 0.0

        self.log_surface_area = log_const_1 + log_const_2 + log_const_3 + log_const_4


class UniformSimplex(Distribution):
    """Uniform distribution on a d-simplex. is then transformed to polar coordinates over d angles"""

    def __init__(self, shape, extend_star_like=False, factor=1):
        super().__init__()
        self._shape = torch.Size(shape)
        self.cart_dim = self._shape[-1]+1
        self.factor = factor
        self.extend_star_like = extend_star_like
        self.log_surface_area = self.__compute_log_surface()
        self.register_buffer("_log_z", torch.tensor(self.log_surface_area, dtype=torch.float64), persistent=False)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        radius = torch.ones_like(inputs[:, :1])
        #TODO optimize such that only one transformation is required
        cartesian = spherical_to_cartesian_torch(torch.cat((inputs, radius), dim=-1))
        radius_l1 = 1. / cartesian.norm(p=1, dim=-1, keepdim=True)
        jacobian = logabsdet_sph_to_car(torch.cat((inputs, radius_l1), dim=-1))
        return jacobian - self._log_z

    def _sample(self, num_samples, context):
        if context is not None:
            nnum_samples = num_samples * context.shape[0]
            ddevice = context.device
        else:
            nnum_samples = num_samples
            ddevice = self._log_z.device

        # create the samples that are on the simplex. see Donald B. Rubin, The Bayesian bootstrap Ann. Statist. 9, 1981, 130-134
        samples = torch.rand((nnum_samples, self.cart_dim-1), device=ddevice)
        samples = torch.concat([
            torch.zeros((nnum_samples,1), device=ddevice),
            torch.sort(samples, dim=-1)[0],
            torch.ones((nnum_samples,1), device=ddevice),
        ], dim=-1)
        samples = samples[...,1:] - samples[...,:-1]  # we omit the multiplication with the basis (identity)

        if self.extend_star_like:
            # move samples randomly to other simplexes
            #raise NotImplementedError("moving to other quadrants not yet implemented")
            signs = torch.randn(samples.shape, device=ddevice)
            signs = torch.sign(signs)
            samples = samples * signs

        samples = cartesian_to_spherical_torch(samples)[:,:-1]

        if context is not None:
            samples = torchutils.split_leading_dim(samples, [context.shape[0], num_samples])
        return samples

    def __compute_log_surface(self):
        # uses the specific way the vertices are chosen for an efficient formula
        # see https://en.wikipedia.org/wiki/Simplex#Volume for the vertices e_i
        # d = self.cart_dim
        # sqrt_det = np.sqrt(d + 1)
        # factorial = (d*d+d) / 2
        # all_quandrants = np.log(2) * d if self.extend_star_like else 0.0
        # return sqrt_det - factorial + all_quandrants

        # side_length = np.sqrt(2.)
        # log_num = self.cart_dim * np.log(side_length) + 0.5 * np.log(self.cart_dim)
        # log_den = 2 * sp.special.gammaln(self.cart_dim)
        log_const = - sp.special.loggamma(self.cart_dim)
        all_quadrants = np.log(2) * self.cart_dim if self.extend_star_like else 0.0

        return log_const + all_quadrants

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
