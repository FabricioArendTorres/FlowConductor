"""Tests for the LU linear transforms."""

import unittest

import torch
from flowcon.transforms.conditional import ConditionalScaleTransform
from flowcon.utils import torchutils
from tests.transforms.transform_test import ConditionalTransformTest
import pytest
from parameterized import parameterized, parameterized_class


@parameterized_class(('batch_size', 'context_features', 'features'), [
    (10, 2, 1),
    (2, 2, 4),
    (10, 2, 2),
    (10, 2, 3),
    (10, 4, 3),
    (1, 4, 3),
    (1, 9, 1),
])
class ConditionalScaleTest(ConditionalTransformTest):
    def setUp(self):
        self.hidden_features = 32
        self.transform = ConditionalScaleTransform(features=self.features, hidden_features=self.hidden_features,
                                                   context_features=self.context_features)

    def test_forward(self):
        random_input = torch.randn(self.batch_size, self.features)
        random_context = torch.randn((self.batch_size, self.context_features))

        outputs, logabsdet = self.transform.forward(random_input, context=random_context)

        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])
        self.assert_jacobian_correct_context(transform=self.transform, inputs=random_input,context=random_context)


    def test_inverse(self):
        random_input = torch.randn(self.batch_size, self.features)
        random_context = torch.randn((self.batch_size, self.context_features))

        outputs, logabsdet = self.transform.inverse(random_input, random_context)

        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])
        self.assert_jacobian_correct_context(transform=self.transform, inputs=random_input,context=random_context)

    def test_forward_inverse_are_consistent(self):
        inputs = torch.randn(self.batch_size, self.features)
        random_context = torch.randn((self.batch_size, self.context_features))
        self.assert_conditional_forward_inverse_are_consistent(self.transform, inputs, random_context)


if __name__ == "__main__":
    unittest.main()
