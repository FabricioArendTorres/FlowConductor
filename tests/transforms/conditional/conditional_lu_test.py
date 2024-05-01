"""Tests for the LU linear transforms."""

import unittest

import torch
from flowcon.transforms import lu
from flowcon.transforms.conditional import ConditionalLUTransform
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
class LULinearTest(ConditionalTransformTest):
    def setUp(self):
        self.hidden_features = 32
        self.transform = ConditionalLUTransform(features=self.features, hidden_features=self.hidden_features,
                                       context_features=self.context_features)

    @pytest.mark.skip(reason="utility function")
    def calc_intermediate_vals(self, random_context):
        conditional_params = self.transform.conditional_net(random_context)
        lower, upper = self.transform._create_lower_upper(conditional_params)
        weight = lower @ upper
        weight_inverse = torch.inverse(weight)
        return conditional_params, lower, upper, weight, weight_inverse

    def test_batch_matrices(self):

        random_context = torch.randn((self.batch_size, self.context_features))
        conditional_params, lower, upper, weight, weight_inverse = self.calc_intermediate_vals(random_context)
        for i in range(self.batch_size):
            self.assertEqual(lower[i].shape, (self.features, self.features))
            torch.testing.assert_close(lower[i] @ upper[i], weight[i])
            self.assert_tensor_equal(torch.inverse(weight[i]), weight_inverse[i])

    def test_forward(self):
        inputs = torch.randn(self.batch_size, self.features)
        random_context = torch.randn((self.batch_size, self.context_features))
        conditional_params, lower, upper, weight, weight_inverse = self.calc_intermediate_vals(random_context)

        outputs, logabsdet = self.transform.forward(inputs, context=random_context)

        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])

        for i in range(self.batch_size):
            outputs_ref = torch.matmul(weight[i],inputs[i])
            logabsdet_ref = torchutils.logabsdet(weight[i])

            torch.testing.assert_close(outputs[i], outputs_ref)
            torch.testing.assert_close(logabsdet[i], logabsdet_ref)


    def test_inverse(self):
        inputs = torch.randn(self.batch_size, self.features)
        random_context = torch.randn((self.batch_size, self.context_features))
        conditional_params, lower, upper, weight, weight_inverse = self.calc_intermediate_vals(random_context)

        outputs, logabsdet = self.transform.inverse(inputs, random_context)

        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])

        for i in range(self.batch_size):
            outputs_ref = weight_inverse[i] @ inputs[i]
            logabsdet_ref1 = torchutils.logabsdet(weight_inverse[i])
            logabsdet_ref2 = -torchutils.logabsdet(weight[i])

            torch.testing.assert_close(outputs[i], outputs_ref)

            torch.testing.assert_close(logabsdet_ref1, logabsdet_ref1)
            torch.testing.assert_close(logabsdet[i], logabsdet_ref1)
            torch.testing.assert_close(logabsdet[i], logabsdet_ref2)


    def test_forward_inverse_are_consistent(self):
        inputs = torch.randn(self.batch_size, self.features)
        random_context = torch.randn((self.batch_size, self.context_features))
        self.assert_conditional_forward_inverse_are_consistent(self.transform, inputs, random_context)


if __name__ == "__main__":
    unittest.main()
