from typing import Optional

import torch.nn
from torch.nn import Module

from flowcon.distributions import BaseDistribution
from flowcon.transforms import Transform
from flowcon.utils import torchutils

__all__ = ["Flow"]


class Flow(Module):
    """Base class for all flow objects."""

    def __init__(
        self,
        transform: Transform,
        distribution: BaseDistribution,
        embedding_net: Optional[Module] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        transform : Transform
            A `Transform` object, it transforms data into noise.
        distribution : Distribution
            A `AutoregressiveTransform` object, the base distribution of the flow that
                generates the noise.
        embedding_net : _type_, optional
            A `nn.Module` which has trainable parameters to encode the
                context (condition) and is trained jointly with the flow, by default None.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution

        if embedding_net is not None:
            assert isinstance(embedding_net, torch.nn.Module), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self._embedding_net = embedding_net
        else:
            self._embedding_net = torch.nn.Identity()

    def _log_prob(self, inputs, context):
        embedded_context = self._embedding_net(context)
        z, logabsdet = self._transform(inputs, context=embedded_context)
        log_prob = self._distribution.log_prob(z)
        return log_prob + logabsdet

    def _sample(self, num_samples, context):
        embedded_context = self._embedding_net(context)
        if self._context_used_in_base:
            noise = self._distribution.sample(num_samples, context=embedded_context)
        else:
            repeat_noise = self._distribution.sample(
                num_samples * embedded_context.shape[0]
            )
            noise = torch.reshape(
                repeat_noise, (embedded_context.shape[0], -1, repeat_noise.shape[1])
            )

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, _ = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        embedded_context = self._embedding_net(context)
        if self._context_used_in_base:
            noise, log_prob = self._distribution.sample_and_log_prob(
                num_samples, context=embedded_context
            )
        else:
            noise, log_prob = self._distribution.sample_and_log_prob(num_samples)

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _ = self._transform(inputs, context=self._embedding_net(context))
        return noise
