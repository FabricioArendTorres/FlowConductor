from typing import Optional, Tuple

import torch.nn
from torch.nn import Module

from flowcon.distributions import BaseDistribution
from flowcon.transforms import Transform
from flowcon.utils import torchutils

__all__ = ["Flow", "ConditionalFlow"]


class Flow(Module):
    """
    A generic Normalizing Flow model.

    This class implements an unconditional Normalizing Flow, which models a joint
    probability distribution of random variables without conditioning on external inputs.
    It consists of a bijective transformation that maps data points to a base distribution,
    typically a simple distribution such as a Gaussian.

    The log-probability of a data point is computed using the change of variables formula,
    and sampling is performed by drawing from the base distribution and applying the inverse
    transformation.

    Notation:
        - `mb_size` refers to the minibatch size (e.g., `mb_size=10` for 10 samples).
        - `dim_z` is the dimensionality of the base distribution.
        - `X` represents the data distribution, with samples `x`.
        - `Z` represents the (latent) base distribution, with samples `z`.
        - `T` is the bijective transformation:
        - `T: X → Z` corresponds to `transform.forward(x)`.
        - `T⁻¹: Z → X` corresponds to `transform.inverse(z)`.

    Attributes
    ----------
    transform : Transform
        A bijective transformation mapping data x to latent space z.
    base_distribution : BaseDistribution
        The base probability distribution for Z.
    """

    def __init__(self, transform: Transform, distribution: BaseDistribution):
        """
        Constructor.

        Parameters
        ----------
        transform : Transform
            Trainable Bijection implementing T: X->Z and T^{-1}:Z->X.
            That is, it that transforms from the data domain (domain(X))
            to the domain of the base distribution (domain(Z)).

            The output shape of transform.forward must always (!) be of shape [mb_dim, dim_z], with dim
            being the dimension of the base distribution.

            T: X->Z is implemented via calling transform.forward and used for evaluating the log-likelihood of the flow.
            T^{-1}: Z->X is implemented via calling transform.inverse and used for generating samples.

            Ensure that depending on the use case, you mainly call the efficient direction of the transform.
            That is, for density estimation, you want T to be efficient.
            For variational inference, you want T^{-1} to be efficient.

        distribution : Distribution
            Fixed distribution
                generates the noise.
        """
        super().__init__()
        self._transform = transform
        self._base_distribution = distribution

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log_probability of the flow by mapping an input x to z,
        evaluating the log_probability of the base distribution at z, and then
        shift it according to the logabsdet of the transform.

        Returns ln(p(x)) = ln(p_{base}(T(x))) + ln|det JT(x)|.

        Makes use of the forward transform to map from x to z.

        Parameters
        ----------
        x : torch.Tensor
            Untransformed values in the data space. Of shape [mb_size, ...].

        Returns
        -------
        torch.Tensor
            Log-probability of x under the distribution parameterized by this Flow object.
            Of shape [mb_size].
        """
        z, logabsdet_T = self._transform(x)
        log_prob_z = self._base_distribution.log_prob(z)
        return log_prob_z + logabsdet_T

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generates samples from the flow by sampling z from the base distribution
        and transforming it to x with the inverse transform.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.

        Returns
        -------
        torch.Tensor
            (samples_x, log_prob), with samples_x being of shape [num_samples, ...],
            and log_prob of shape [num_samples].
        """
        samples_z = self._base_distribution.sample(num_samples)
        samples_x, _ = self._transform.inverse(samples_z)
        return samples_x

    def sample_and_log_prob(
        self, num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Generates samples from the flow, together with their log probabilities.

        For flows, this is slighly more efficient that calling `sample` and `log_prob` separately.
        TODO: Currently this only avoids evaluating the base logprob,
              but still computes all the logabsdet values.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (samples_x, log_prob), with samples_x being of shape [num_samples, ...],
            and log_prob of shape [num_samples].
        """
        assert num_samples > 0, f"Positive number of samples expected, {num_samples=}."

        samples_z, log_prob_z = self._base_distribution.sample_and_log_prob(num_samples)
        samples_x, logabsdet_Tinv = self._transform.inverse(
            samples_z,
        )
        # reciprocal in log space
        logabsdet_T = -logabsdet_Tinv

        return samples_x, log_prob_z + logabsdet_T

    @property
    def base_distribution(self) -> BaseDistribution:
        """
        Getter for the base distribution used by this flow object.

        Returns
        -------
        BaseDistribution
            The base distribution passed during initialization.
        """
        return self._base_distribution

    @property
    def transform(self) -> Transform:
        """
        Getter for the bijective transform used by this flow object.

        Returns
        -------
        Transform
            The bijective transform passed during initialization.
        """
        return self._transform


class ConditionalFlow(Module):
    """
    Base class for all conditional flow objects.
    That is, the probabilities of this flows are conditioned on an input, the so-called context.
    Internally, this is done by passing the context to the individual transforms of the flow.
    Optionally, one may also specify an "context_embedder", a torch Module (e.g. Neural Network)
    which transforms the context before passing it to the individual layers.
    """

    def __init__(
        self,
        transform: Transform,
        distribution: BaseDistribution,
        context_embedder: Optional[Module] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        transform : Transform
            A `Transform` object, it transforms data into noise.
        distribution : Distribution
            A `BaseDistribution` object, the base distribution of the flow that
                generates the noise.
        embedding_net : torch.nn.Module, optional
            A `nn.Module` which has trainable parameters to encode the
                context (condition) and is trained jointly with the flow, by default None.
                If not passed, the context will be directly passed to the individual transforms.
        """
        super().__init__()
        self._transform = transform
        self._base_distribution = distribution

        if context_embedder is not None:
            self._context_embedder = context_embedder
        else:
            self._context_embedder = torch.nn.Identity()

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """

        Evaluate the log_probability of the flow by mapping an input x to z,
        evaluating the log_probability of the base distribution at z, and then
        shift it according to the logabsdet of the transform.

        Returns ln(p(x|c)) = ln(p_{base}(T(x|c))) + ln|det JT(x)|.

        Makes use of the forward transform to map from x to z.

        Parameters
        ----------
        x : torch.Tensor
            Untransformed values in the data space. Of shape [mb_size, ...].
        context : torch.Tensor
            Context of shape [mb_size, ...]

        Returns
        -------
        torch.Tensor
            Log-probability of x conditioned on the context, p(x|c),
             under the distribution parameterized by this Flow object.
             Of shape [mb_size].
        """
        embedded_context = self._context_embedder(context)
        self.assert_embedded_context_shape(context, embedded_context)
        z, logabsdet = self._transform(x, context=embedded_context)
        log_prob = self._base_distribution.log_prob(z)
        return log_prob + logabsdet

    def sample(self, context: torch.Tensor) -> torch.Tensor:
        """
        Generates samples of X conditioned on the context
        from the flow by sampling z from the base distribution
        and transforming it to x with the inverse transform.
        The inverse transform is conditioned on the (embedded) context.
        One sample is generated for each context row (assuming context of shape [mb_size, ...]).

        Parameters
        ----------
        context : torch.Tensor
            Context of shape [mb_size, ...]

        Returns
        -------
        torch.Tensor
            Samples p(x|context) of shape [mb_size, ...]
        """
        assert context is not None, "Passed context is None."
        # `embedded_context` has shape `[mb_size, ...]`
        embedded_context = self._context_embedder(context)
        self.assert_embedded_context_shape(context, embedded_context)
        # `sample_z` has shape `[mb_size, dim_z]`
        samples_z = self._base_distribution.sample(embedded_context.shape[0])
        # `samples_x` has shape `[mb_size, ...]`
        samples_x, _ = self._transform.inverse(samples_z, context=embedded_context)

        return samples_x

    def sample_multi(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        """
        Similar to `ConditionalFlow.sample`, but creates multiple samples per context.

        Generates samples of X conditioned on the context
        from the flow by sampling z from the base distribution
        and transforming it to x with the inverse transform.
        The inverse transform is conditioned on the (embedded) context.
        `num_samples` samples are generated for each context row (assuming context of shape [mb_size, ...]).

        Parameters
        ----------
        num_samples : int
            Number of samples to be drawn for each context row.
        context : torch.Tensor
            Context of shape [mb_size, ...]

        Returns
        -------
        torch.Tensor
            Samples p(x|context) of shape [mb_size, num_samples, ...]
        """
        # shape [mb_size, ...]
        embedded_context = self._context_embedder(context)
        self.assert_embedded_context_shape(context, embedded_context)
        mb_size = embedded_context.shape[0]

        # just do repeated independent sampling - the context will never affect the base distribution.
        # shape [mb_size*num_samples, dim_z]
        repeat_samples_z = self._base_distribution.sample(num_samples * mb_size)

        # Repeat the context dimension with sample dimension in order to apply the transform.
        # shape [mb_size*num_samples, ...]
        embedded_context = torchutils.repeat_rows(
            embedded_context, num_reps=num_samples
        ).contiguous()

        samples, _ = self._transform.inverse(repeat_samples_z, context=embedded_context)

        # Split the context dimension from sample dimension.
        # shape [mb_size, num_samples, ...]
        samples = torchutils.split_leading_dim(samples, shape=[mb_size, num_samples])

        return samples

    def sample_and_log_prob(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples `p(x|context)` from the flow by sampling z from the base distribution
        and transforming it to x with the inverse transform.

        The inverse transform is conditioned on the (embedded) context.
        One sample is generated for each context row (assuming context of shape [mb_size, ...]).

        For flows, this is slighly more efficient that calling `sample` and `log_prob` separately.
        TODO: Currently this only avoids evaluating the base logprob,
              but still computes all the logabsdet values.

        Parameters
        ----------
        context : torch.Tensor
            Context of shape [mb_size, ...]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (samples_x, log_prob), with samples_x being of shape [num_samples, ...],
            and log_prob of shape [num_samples].
        """

        # `embedded_context` has shape `[mb_size, ...]`
        embedded_context = self._context_embedder(context)
        self.assert_embedded_context_shape(context, embedded_context)
        # `sample_z` has shape `[mb_size, dim_z]`
        # `log_prob_z` has shape `mb_size`.
        samples_z, log_prob_z = self._base_distribution.sample_and_log_prob(
            embedded_context.shape[0]
        )
        # `samples_x` has shape `[mb_size, ...]`
        samples_x, logabsdet_Tinv = self._transform.inverse(
            samples_z, context=embedded_context
        )

        # reciprocal in log space
        logabsdet_T = -logabsdet_Tinv
        return samples_x, log_prob_z + logabsdet_T

    def sample_and_log_prob_multi(
        self, num_samples: int, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Similar to `ConditionalFlow.sample_and_log_prob`, but creates multiple samples per context.

        Generates samples `p(x|context)` from the flow by sampling z from the base distribution
        and transforming it to x with the inverse transform.

        The inverse transform is conditioned on the (embedded) context.
        Multiple (i.e. num_samples) samples are generated for each context row (assuming context of shape [mb_size, ...]).

        For flows, this is slighly more efficient that calling `sample` and `log_prob` separately.
        TODO: Currently this only avoids evaluating the base logprob,
              but still computes all the logabsdet values.

        Parameters
        ----------
        context : torch.Tensor
            Context of shape [mb_size, ...]

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (samples_x, log_prob), with samples_x being of shape [num_samples, ...],
            and log_prob of shape [num_samples].
        """

        # shape [mb_size, ...]
        embedded_context = self._context_embedder(context)
        self.assert_embedded_context_shape(context, embedded_context)
        mb_size = embedded_context.shape[0]

        # just do repeated independent sampling - the context will never affect the base distribution.
        # shape [mb_size*num_samples, dim_z]
        repeat_samples_z, log_prob_z = self._base_distribution.sample_and_log_prob(
            num_samples * mb_size
        )

        # Repeat the context dimension with sample dimension in order to apply the transform.
        # shape [mb_size*num_samples, ...]
        embedded_context = torchutils.repeat_rows(
            embedded_context, num_reps=num_samples
        ).contiguous()

        samples_x, logabsdet_Tinv = self._transform.inverse(
            repeat_samples_z, context=embedded_context
        )

        # Split the context dimension from sample dimension.
        # shape [mb_size, num_samples, ...]
        samples_x = torchutils.split_leading_dim(
            samples_x, shape=[mb_size, num_samples]
        )

        # reciprocal in log space
        logabsdet_T = -logabsdet_Tinv
        log_prob_x = log_prob_z + logabsdet_T

        log_prob_x = torchutils.split_leading_dim(
            log_prob_x, shape=[mb_size, num_samples]
        )

        return samples_x, log_prob_x

    @staticmethod
    def assert_embedded_context_shape(
        context: torch.Tensor, embedded_context: torch.Tensor
    ) -> None:
        assert context is not None, "Passed context is None."
        assert embedded_context is not None, "Transformed embbedded_context is None."
        assert embedded_context.shape[0] == context.shape[0], (
            "First axis (minibatch axis) should be unchanged,"
            + f"but {embedded_context.shape[0]=} != {context.shape[0]=}"
        )

    @property
    def base_distribution(self) -> BaseDistribution:
        """
        Getter for the base distribution used by this flow object.

        Returns
        -------
        BaseDistribution
            The base distribution passed during initialization.
        """
        return self._base_distribution

    @property
    def transform(self) -> Transform:
        """
        Getter for the bijective transform used by this flow object.

        Returns
        -------
        Transform
            The bijective transform passed during initialization.
        """
        return self._transform

    @property
    def context_embedder(self) -> Module:
        """
        Getter for the context embedder used by this flow object.

        Returns
        -------
        Module
            The module used for embedding/transforming context before passing it as a condition to the
            bijective Transform.
        """
        return self._context_embedder
