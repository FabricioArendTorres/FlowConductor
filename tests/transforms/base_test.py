"""Tests for the basic transform definitions."""

import numpy as np
import pytest
import torch

from flowcon.transforms import Inverse, base, conditional
from flowcon.transforms.linear import standard
from tests.transforms import transform_test
from tests.transforms.transform_test import TransformTest


class CompositeTransformTest(TransformTest):
    def test_forward(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)
        transforms = [
            standard.AffineScalarTransform(scale=2.0),
            standard.IdentityTransform(),
            standard.AffineScalarTransform(scale=0.25),
        ]
        composite = base.Sequential(transforms)
        reference = standard.AffineScalarTransform(scale=0.5)
        outputs, logabsdet = composite(inputs)
        outputs_ref, logabsdet_ref = reference(inputs)
        self.assert_tensor_is_good(outputs, [batch_size] + shape)
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)

    def test_inverse(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)
        transforms = [
            standard.AffineScalarTransform(scale=2.0),
            standard.IdentityTransform(),
            standard.AffineScalarTransform(scale=0.25),
        ]
        composite = base.Sequential(transforms)
        reference = standard.AffineScalarTransform(scale=0.5)
        outputs, logabsdet = composite.inverse(inputs)
        outputs_ref, logabsdet_ref = reference.inverse(inputs)
        self.assert_tensor_is_good(outputs, [batch_size] + shape)
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)


class MultiscaleCompositeTransformTest(TransformTest):
    def create_transform(self, shape, split_dim=1):
        mct = base.MultiscaleSequential(num_transforms=4, split_dim=split_dim)
        for transform in [
            standard.AffineScalarTransform(scale=2.0),
            standard.AffineScalarTransform(scale=4.0),
            standard.AffineScalarTransform(scale=0.5),
            standard.AffineScalarTransform(scale=0.25),
        ]:
            shape = mct.add_transform(transform, shape)

        return mct

    def test_forward(self):
        batch_size = 5
        for shape in [(32, 4, 4), (64,), (65,)]:
            with self.subTest(shape=shape):
                inputs = torch.ones(batch_size, *shape)
                transform = self.create_transform(shape)
                outputs, logabsdet = transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size] + [np.prod(shape)])
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_forward_bad_shape(self):
        shape = (8,)
        with self.assertRaises(ValueError):
            transform = self.create_transform(shape)

    def test_forward_bad_split_dim(self):
        batch_size = 5
        shape = [32]
        inputs = torch.randn(batch_size, *shape)
        with self.assertRaises(ValueError):
            transform = self.create_transform(shape, split_dim=2)

    def test_inverse_not_flat(self):
        batch_size = 5
        shape = [32, 4, 4]
        inputs = torch.randn(batch_size, *shape)
        transform = self.create_transform(shape)
        with self.assertRaises(ValueError):
            transform.inverse(inputs)

    def test_forward_inverse_are_consistent(self):
        batch_size = 5
        for shape in [(32, 4, 4), (64,), (65,), (21,)]:
            with self.subTest(shape=shape):
                transform = self.create_transform(shape)
                inputs = torch.randn(batch_size, *shape).view(batch_size, -1)
                self.assert_forward_inverse_are_consistent(Inverse(transform), inputs)


class InverseTransformTest(TransformTest):
    def test_forward(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)
        transform = base.Inverse(standard.AffineScalarTransform(scale=2.0))
        reference = standard.AffineScalarTransform(scale=0.5)
        outputs, logabsdet = transform(inputs)
        outputs_ref, logabsdet_ref = reference(inputs)
        self.assert_tensor_is_good(outputs, [batch_size] + shape)
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)

    def test_inverse(self):
        batch_size = 10
        shape = [2, 3, 4]
        inputs = torch.randn(batch_size, *shape)
        transform = base.Inverse(standard.AffineScalarTransform(scale=2.0))
        reference = standard.AffineScalarTransform(scale=0.5)
        outputs, logabsdet = transform.inverse(inputs)
        outputs_ref, logabsdet_ref = reference.inverse(inputs)
        self.assert_tensor_is_good(outputs, [batch_size] + shape)
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)


class ConditionalTransformTest(TransformTest):
    def setUp(self):
        self.features = 3
        self.batch_size = 10

        self.random_input = torch.randn((self.batch_size, self.features))

        self.transform = conditional.ConditionalTransform(
            features=self.features, conditional_net=torch.nn.Identity()
        )

    def test_no_condition(self):
        with self.assertRaises(expected_exception=TypeError) as cm:
            self.transform.forward(self.random_input)

        with self.assertRaises(expected_exception=TypeError) as cm:
            self.transform.inverse(self.random_input)


class MockTransform(base.Transform):
    def __init__(self, allow_context=True):
        super().__init__()
        self._inverted = False
        self._allow_context = allow_context

    def forward(self, inputs, context=None):
        if not self._allow_context and (context is not None):
            raise RuntimeError(f"No context allowed, but {context=}.")
        elif self._allow_context and context is None:
            raise RuntimeError(f"Context expected, but {context=}.")

        _sum = torch.sum(context, -1, keepdim=True) if context is not None else 0
        return inputs + 1 + _sum, torch.zeros(inputs.shape[0])

    def inverse(self, inputs, context=None):
        if not self._allow_context and (context is not None):
            raise RuntimeError(f"No context allowed, but {context=}.")
        elif self._allow_context and context is None:
            raise RuntimeError(f"Context expected, but {context=}.")
        _sum = torch.sum(context, -1, keepdim=True) if context is not None else 0

        return inputs - 1 - _sum, torch.zeros(inputs.shape[0])


class RemoveContextTransformTest(transform_test.ConditionalTransformTest):
    def setUp(self):
        super().setUp()
        self.features = 3
        self.batch_size = 10
        self.random_context = torch.randn(self.batch_size, 5)
        self.random_input = torch.randn((self.batch_size, self.features))
        transforms_with_context1 = base.Sequential(
            [
                MockTransform(allow_context=True),
                MockTransform(allow_context=True),
            ]
        )
        transforms_with_context2 = base.Sequential(
            [
                MockTransform(allow_context=True),
                MockTransform(allow_context=True),
            ]
        )
        transforms_wo_context = base.Sequential(
            [MockTransform(allow_context=False), MockTransform(allow_context=False)]
        )
        transforms_wo_context2 = base.Sequential(
            [MockTransform(allow_context=False), MockTransform(allow_context=False)]
        )
        self.transform = base.Sequential(
            [
                transforms_with_context1,
                base.RemoveContext(transforms_wo_context),
                transforms_with_context2,
                base.RemoveContext(transforms_wo_context2),
            ]
        )

    def test_nocontext_passed(self):
        # the testing logic is in the mock class
        outputs, logabsdet = self.transform.forward(
            self.random_input, context=self.random_context
        )
        outputs_inv, logabsdet_inv = self.transform.inverse(
            self.random_input, context=self.random_context
        )
        with pytest.raises(RuntimeError):
            outputs, logabsdet = self.transform.forward(self.random_input)
            expected_outputs, expected_logabsdet = self.transform.inverse(
                self.random_input
            )

    def test_inverse(self):
        self.assert_conditional_forward_inverse_are_consistent(
            self.transform, self.random_input, self.random_context
        )


if __name__ == "__main__":
    pytest.main()
