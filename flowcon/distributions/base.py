"""Basic definitions for the distributions module."""

import torch
from torch import nn, optim

from flowcon.utils import torchutils
import flowcon.utils.typechecks as check


class NoMeanException(Exception):
    """Exception to be thrown when a mean function doesn't exist."""

    pass


class Distribution(nn.Module):
    """Base class for all distribution objects."""

    def forward(self, *args):
        raise RuntimeError("Forward method cannot be called for a Distribution object.")

    def log_prob(self, inputs, context=None):
        """Calculate log probability under the distribution.

        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of context items."
                )
        return self._log_prob(inputs, context)

    def _log_prob(self, inputs, context):
        raise NotImplementedError()

    def sample(self, num_samples, context=None, batch_size=None):
        """Generates samples from the distribution. Samples can be generated in batches.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored. 
                     Should have shape [context_size, ...], where ... represents a (context) feature 
                     vector of arbitrary shape. This will generate num_samples for each context item 
                     provided. The overall shape of the samples will then be 
                     [context_size, num_samples, ...].
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.

        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given, where ... represents a feature
            vector of arbitrary shape.
        """
        if not check.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        if context is not None:
            context = torch.as_tensor(context)

        if batch_size is None:
            return self._sample(num_samples, context)

        else:
            if not check.is_positive_int(batch_size):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)

    def _sample(self, num_samples, context):
        raise NotImplementedError()

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the distribution together with their log probability.

        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored. 
                     Should have shape [context_size, ...], where ... represents a (context) feature 
                     vector of arbitrary shape. This will generate num_samples for each context item 
                     provided. The overall shape of the samples will then be 
                     [context_size, num_samples, ...].
        Returns:
            A tuple of:
                * A Tensor containing the samples, with shape [num_samples, ...] if context is None,
                  or [context_size, num_samples, ...] if context is given, where ... represents a 
                  feature vector of arbitrary shape.
                * A Tensor containing the log probabilities of the samples, with shape
                  [num_samples, features if context is None, or [context_size, num_samples, ...] if
                  context is given.
        """
        samples = self.sample(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to call log_prob.
            samples = torchutils.merge_leading_dims(samples, num_dims=2)
            context = torchutils.repeat_rows(context, num_reps=num_samples)
            assert samples.shape[0] == context.shape[0]

        log_prob = self.log_prob(samples, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = torchutils.split_leading_dim(log_prob, shape=[-1, num_samples])

        return samples, log_prob

    def mean(self, context=None):
        if context is not None:
            context = torch.as_tensor(context)
        return self._mean(context)

    def _mean(self, context):
        raise NoMeanException()

    def sample_maximum(self, num_samples, context=None, its=1, opt="LBFGS", *args):
        """calls sample_maps and returns the maximizing x and associated log prob for each context.

        In general it is recommended to use simulated annealing to recover the global maximum.
        returns max_x, max_logprob
        The shapes are [context_size, 1, dim] and [context_size, 1] if context is not None
                else [dim] and []
        """
        xs, log_ps = self.sample_maxima(num_samples, context, its, opt, *args)
        if context is not None:
            index = torch.argmax(log_ps, dim=-1, keepdim=True)
            # the following unsqueezing over multiple dimensions is to accommodate the variable shape of the space of x
            n_extra_dims = (xs.ndim - 2)
            index_unsqueezed = index.view(tuple(index.shape) + (1,) * n_extra_dims)
            max_x = torch.take_along_dim(xs, index_unsqueezed, dim=1)

            max_logprob = torch.take_along_dim(log_ps, index, dim=1)
        else:
            index = torch.argmax(log_ps)
            max_x, max_logprob = xs[index], log_ps[index]
        return max_x, max_logprob

    def sample_maxima(self, num_samples, context=None, its=1, opt="LBFGS", *args):
        """Takes a number of samples and maximizes their log prob.
        this can be used to approximately sample the maxima of a multimodal distribution.

        Args:
            num_samples: The number of samples per context.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.
            its: The number of optimization steps
            opt: The optimizer to use in the torch.optim package in str format.
            args: The arguments for the optimizer

        Returns:
            A `Tensor` of shape [batch_size, ...] representing samples xs, the associated log_prob.
        """
        initial_sample = self._sample(num_samples, context)
        if context is not None:
            initial_sample = torchutils.merge_leading_dims(initial_sample, num_dims=2)
            context = torchutils.repeat_rows(context, num_reps=num_samples)
        initial_sample = torch.nn.parameter.Parameter(initial_sample, requires_grad=True)
        optimizer = getattr(optim, opt)([initial_sample], *args)

        def closure():
            optimizer.zero_grad()
            neg_log_prob = -self.log_prob(initial_sample, context).mean()
            neg_log_prob.backward()
            return neg_log_prob

        for _ in range(its):
            optimizer.step(closure)

        if context is not None:
            return (torchutils.split_leading_dim(initial_sample.data, shape=[-1, num_samples]),
                    torchutils.split_leading_dim(self.log_prob(initial_sample.data, context), shape=[-1, num_samples]))
        else:
            return initial_sample.data, self.log_prob(initial_sample.data, context)
