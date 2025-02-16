"""Implementations of Normal distributions."""

from typing import Optional
import numpy as np
import torch
from torch import nn

from flowcon.distributions.base import BaseDistribution
from flowcon.utils import torchutils


class StandardNormal(BaseDistribution):
    """A multivariate Normal 𝒩(µ=0,  Σ=I), i.e. with zero mean and unit covariance."""

    def __init__(self, dim: int):
        """
        Constructor.

        Parameters
        ----------
        dim : int
            Dimension of the random variable.
            Samples of this distribution will be of shape [num_samples, dim].
            Likelihood evaluation will assume input of shape [batch_size, dim].
        """
        super().__init__(dim=dim)

        self.register_buffer(
            "_log_z",
            torch.tensor(
                0.5 * self.dim * np.log(2 * np.pi), dtype=torch.get_default_dtype()
            ),
            persistent=True,
        )

    def log_prob(self, inputs: torch.Tensor) -> torch.Tensor:
        assert len(inputs.shape) == 2, "Expected tensor of length 2."
        assert inputs.shape[1] == self.dim, (
            f"Expected input of shape [None, {self.dim}]"
        )

        neg_energy = -0.5 * torchutils.sum_except_batch(inputs**2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples: int) -> torch.Tensor:
        samples = torch.randn(num_samples, self.dim, device=self._log_z.device)
        return samples


class DiagonalNormal(BaseDistribution):
    """
    A diagonal multivariate Normal 𝒩(µ,  Σ=σI) with optionally trainable parameters.
    This should be redundant to a transform with elementwise shift and / or scales,
    but is added nonetheless as a utility.
    """

    def __init__(
        self,
        dim: int,
        mean: Optional[torch.Tensor] = None,
        log_std: Optional[torch.Tensor] = None,
        trainable_mean: bool = True,
        trainable_log_std: bool = True,
    ):
        """
        Constructor.


        Parameters
        ----------
        dim : int
            Dimension of Distribution.
        mean : torch.Tensor, optional
            Optional initial mean of the DiagonalNormal, by default None.
            Initialized as zero if None.
            Needs to be squeezable to shape [dim].
        log_std : torch.Tensor, optional
            Optional initial log_std of the DiagonalNormal, by default None.
            Initialized as zero if None.
            Needs to be squeezable to shape [dim].
        trainable_mean : bool, optional
            Whether the mean is trainable, by default True.
        trainable_log_std : bool, optional
            Whether the log_std is trainable, by default True.
        """
        super().__init__(dim=dim)

        if mean is None:
            mean = torch.zeros(self.dim)
        else:
            mean = torch.squeeze(mean)

        if log_std is None:
            log_std = torch.zeros(self.dim)
        else:
            log_std = torch.squeeze(log_std)

        assert len(mean.shape) == 1 and mean.shape[0] == self.dim
        assert len(log_std.shape) == 1 and log_std.shape[0] == self.dim

        self._mean = nn.Parameter(
            mean.reshape(1, self.dim), requires_grad=trainable_mean
        )
        self._log_std = nn.Parameter(
            log_std.reshape(1, self.dim), requires_grad=trainable_log_std
        )

        self.register_buffer(
            "_log_z",
            torch.tensor(
                0.5 * self.dim * np.log(2 * np.pi), dtype=torch.get_default_dtype()
            ),
            persistent=True,
        )

    def log_prob(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == self.dim

        # Compute parameters.
        means = self._mean
        log_stds = self._log_std

        # Compute log prob.
        norm_inputs = (inputs - means) * torch.exp(-log_stds)
        log_prob = -0.5 * torchutils.sum_except_batch(norm_inputs**2, num_batch_dims=1)
        log_prob -= torchutils.sum_except_batch(log_stds, num_batch_dims=1)
        log_prob -= self._log_z
        return log_prob

    def _sample(self, num_samples: int) -> torch.Tensor:
        samples = torch.randn(num_samples, self.dim, device=self._log_z.device)
        return (samples * torch.exp(self._log_std)) + self._mean
