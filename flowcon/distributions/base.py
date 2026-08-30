"""Definition of the base class for the base distributions of the flow."""

from typing import Tuple

import torch
from torch import nn

import flowcon.utils.typechecks as check


class BaseDistribution(nn.Module):
    """
    Base class for all Distribution objects.
    A Distribution provides the density and samples from a `dim`-dimensional jointly distributed random variable.
    Samples within the Batch dimension are always assumed to be independent.

    Furthermore, Objects of this class are not conditional distributions.
    If you desire a conditional distribution, do so via defining a transform with an embedding network.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()

        if not check.is_positive_int(dim):
            raise TypeError("Dimension of BaseDistribution must be a positive integer.")
        self._dim = dim

    def sample_and_log_prob(self, num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples from the distribution together with their log probability.

        Parameters
        ----------
        num_samples : int, optional
             Number of samples to generate, by default 1

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            * Samples of shape [num_samples, self.dim]
            * Log probabilities of the Samples of shape [num_samples].
        """

        if not check.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        samples = self._sample(num_samples)
        log_prob = self.log_prob(samples)

        return samples, log_prob

    def log_prob(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log-probability of the input under this distribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape [samples, dim]

        Returns
        -------
        torch.Tensor
            Output probabilities of shape [samples]
        """
        raise NotImplementedError()

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Generates samples from the distribution.
        Samples are generated in batches.

        Parameters
        ----------
        num_samples : int
            number of samples per batch, by default 1.

        Returns
        -------
        torch.Tensor
            Samples of shape [batch_size, num_samples, self.dim].

        Raises
        ------
        TypeError
                If num_samples or batch_size is not a positive integer.
        """
        if not check.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        return self._sample(num_samples=num_samples)

    def _sample(self, num_samples: int) -> torch.Tensor:
        """
        Generates samples from the distribution.
        Must be implemented by subclasses.


        Parameters
        ----------
        num_samples : int
            Number of samples to generate.

        Returns
        -------
        torch.Tensor
            Samples as a tensor of shape [num_samples, dim].
        """
        raise NotImplementedError()

    @property
    def dim(self) -> int:
        """
        Returns the dimension of this distribution.

        Returns
        -------
        int
            Dimension as a positive valued integer.

        """
        return self._dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x)
