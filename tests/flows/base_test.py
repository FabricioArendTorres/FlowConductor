"""Tests for the basic flow definitions."""

from math import prod

import pytest
import torch
import torchtestcase

from flowcon.distributions.normal import StandardNormal
from flowcon.flows import base
from flowcon.transforms.base import Sequential
from flowcon.transforms.linear.standard import AffineScalarTransform
from flowcon.transforms.reshape import FlattenTransform


class FlowTest(torchtestcase.TorchTestCase):
    def test_log_prob(self):
        batch_size = 10
        input_shape = [2, 3, 4]
        flow = base.Flow(
            transform=Sequential(
                (AffineScalarTransform(scale=2.0), FlattenTransform())
            ),
            distribution=StandardNormal(prod(input_shape)),
        )
        inputs = torch.randn(batch_size, *input_shape)
        log_prob = flow.log_prob(inputs)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([batch_size]))

    def test_sample(self):
        num_samples = 10
        input_shape = [2, 3, 4]
        flow = base.Flow(
            transform=Sequential(
                (AffineScalarTransform(scale=2.0), FlattenTransform())
            ),
            distribution=StandardNormal(prod(input_shape)),
        )
        flow.log_prob(torch.rand(1, *input_shape))
        samples = flow.sample(num_samples)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([num_samples] + input_shape))

    @pytest.mark.expensive
    def test_logprob_compiled(self):
        num_samples = 10
        input_shape = [2]
        flow = base.Flow(
            transform=Sequential((AffineScalarTransform(scale=2.0),)),
            distribution=StandardNormal(prod(input_shape)),
        )
        logprob_compiled = torch.compile(flow.log_prob)

        torch.random.manual_seed(1234)
        rand_val = torch.rand(num_samples, *input_shape)
        ll_ref = flow.log_prob(rand_val)
        ll_comp = logprob_compiled(rand_val)

        torch.testing.assert_close(ll_ref, ll_comp)

    @pytest.mark.expensive
    def test_sample_compiled(self):
        num_samples = 10
        input_shape = [2]
        flow = base.Flow(
            transform=Sequential((AffineScalarTransform(scale=2.0),)),
            distribution=StandardNormal(prod(input_shape)),
        )
        torch.random.manual_seed(1234)
        sample_compiled = torch.compile(flow.sample)
        samples = flow.sample(num_samples)
        samples_c = sample_compiled(num_samples)

        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(samples_c, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([num_samples] + input_shape))
        self.assertEqual(samples_c.shape, samples.shape)

    def test_sample_and_log_prob(self):
        num_samples = 10
        input_shape = [2]
        flow = base.Flow(
            transform=AffineScalarTransform(scale=2.0),
            distribution=StandardNormal(prod(input_shape)),
        )
        samples, log_prob_1 = flow.sample_and_log_prob(num_samples)
        log_prob_2 = flow.log_prob(samples)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob_1, torch.Tensor)
        self.assertIsInstance(log_prob_2, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([num_samples] + input_shape))
        self.assertEqual(log_prob_1.shape, torch.Size([num_samples]))
        self.assertEqual(log_prob_2.shape, torch.Size([num_samples]))
        self.assertEqual(log_prob_1, log_prob_2)


class ConditionalFlowTest(torchtestcase.TorchTestCase):
    def setUp(self):
        super().setUp()
        self.mb_size = 10
        self.context_shape = [5, 6]
        self.context = torch.randn(self.mb_size, *self.context_shape)

    def test_log_prob(self):
        input_shape = [2, 3, 4]
        flow = base.ConditionalFlow(
            transform=Sequential(
                (AffineScalarTransform(scale=2.0), FlattenTransform())
            ),
            distribution=StandardNormal(prod(input_shape)),
        )
        inputs = torch.randn(self.mb_size, *input_shape)
        log_prob = flow.log_prob(inputs, context=self.context)
        with pytest.raises(AssertionError):
            log_prob = flow.log_prob(inputs, context=None)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([self.mb_size]))

    def test_sample(self):
        num_samples = 10
        input_shape = [2, 3, 4]
        flow = base.ConditionalFlow(
            transform=Sequential(
                (AffineScalarTransform(scale=2.0), FlattenTransform())
            ),
            distribution=StandardNormal(prod(input_shape)),
        )
        x = torch.rand(self.mb_size, *input_shape)
        flow.log_prob(x, context=self.context)

        samples = flow.sample(context=self.context)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([self.mb_size] + input_shape))
        samples2 = flow.sample_multi(num_samples, context=self.context)
        self.assertEqual(
            samples2.shape,
            torch.Size([self.mb_size, num_samples] + input_shape),
        )

        with pytest.raises(AssertionError):
            flow.sample(context=None)
        with pytest.raises(AssertionError):
            flow.sample_multi(num_samples=10, context=None)

    @pytest.mark.expensive
    def test_logprob_compiled(self):
        num_samples = 10
        input_shape = [2]

        flow = base.ConditionalFlow(
            transform=Sequential((AffineScalarTransform(scale=2.0),)),
            distribution=StandardNormal(prod(input_shape)),
        )
        logprob_compiled = torch.compile(flow.log_prob)

        torch.random.manual_seed(1234)
        rand_val = torch.rand(num_samples, *input_shape)
        ll_ref = flow.log_prob(rand_val, self.context)
        ll_comp = logprob_compiled(rand_val, self.context)

        torch.testing.assert_close(ll_ref, ll_comp)

    @pytest.mark.expensive
    def test_sample_compiled(self):
        num_samples = 10
        input_shape = [2]
        flow = base.ConditionalFlow(
            transform=Sequential((AffineScalarTransform(scale=2.0),)),
            distribution=StandardNormal(prod(input_shape)),
        )
        torch.random.manual_seed(1234)
        sample_compiled = torch.compile(flow.sample)
        samples = flow.sample(context=self.context)
        samples_c = sample_compiled(context=self.context)

        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(samples_c, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([self.mb_size] + input_shape))
        self.assertEqual(samples_c.shape, samples.shape)

        samples2 = flow.sample_multi(num_samples, context=self.context)
        samples2_compiled = torch.compile(flow.sample_multi)(
            num_samples, context=self.context
        )

        self.assertEqual(
            samples2.shape,
            samples2_compiled.shape,
        )

    def test_sample_and_log_prob_with_context(self):
        input_dim = 2 * 3 * 4
        context_shape = [5, 6]
        flow = base.ConditionalFlow(
            transform=AffineScalarTransform(scale=2.0),
            distribution=StandardNormal(input_dim),
        )
        context = torch.randn(self.mb_size, *context_shape)
        samples, log_prob = flow.sample_and_log_prob(context=self.context)

        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([self.mb_size, input_dim]))
        self.assertEqual(log_prob.shape, torch.Size([self.mb_size]))

    def test_sample_and_log_prob_with_context_multi(self):
        num_samples = 10
        input_dim = 2 * 3 * 4
        flow = base.ConditionalFlow(
            transform=AffineScalarTransform(scale=2.0),
            distribution=StandardNormal(input_dim),
        )
        samples_multi, log_prob_multi = flow.sample_and_log_prob_multi(
            num_samples=num_samples, context=self.context
        )
        self.assertEqual(
            samples_multi.shape, torch.Size([self.mb_size, num_samples, input_dim])
        )
        self.assertEqual(log_prob_multi.shape, torch.Size([self.mb_size, num_samples]))


if __name__ == "__main__":
    pytest.main()
