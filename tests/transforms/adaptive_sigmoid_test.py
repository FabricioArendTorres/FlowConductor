"""Tests for the LU linear transforms."""

import unittest

import torch

from flowcon.transforms.adaptive_sigmoids import SumOfSigmoids, DeepSigmoid
from tests.transforms.transform_test import TransformTest
from parameterized import parameterized_class


@parameterized_class(('batch_size', 'features', 'n_sigmoids'), [
    (10, 2, 3),
    (2, 4, 3),
    (10, 2, 30),
    (16, 3, 340),
    (10, 20, 10),
    (1, 3, 1),
    (1, 1, 1),
    (10, 1, 3),
])
class AdaptiveSigmoidTest(TransformTest):
    def setUp(self):
        torch.manual_seed(1234)
        self.inputs = torch.randn(self.batch_size, self.features)
        self.transform = SumOfSigmoids(features=self.features, n_sigmoids=self.n_sigmoids, iterations_bisection_inverse=30)
        self.eps = 1e-5

    def test_forward(self):
        outputs, logabsdet = self.transform.forward(self.inputs)

        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])

    def test_logabsdet(self):
        self.assert_jacobian_correct(transform=self.transform, inputs=self.inputs)
        outputs, _ = self.transform.forward(self.inputs)
        self.assert_inverse_jacobian_correct(transform=self.transform, outputs=outputs.detach())

    def test_forward_inverse_are_consistent(self):
        self.assert_forward_inverse_are_consistent(self.transform, self.inputs)

    def test_new_parameterized(self):
        raw_params = self.transform.get_raw_params()
        transform_parametrized = SumOfSigmoids(features=self.features, n_sigmoids=self.n_sigmoids,
                                               raw_params=raw_params)
        outputs, logabsdet = self.transform.forward(self.inputs)
        outputs_parametrized, logabsdet_parametrized = transform_parametrized.forward(self.inputs)
        self.assertEqual(outputs, outputs_parametrized)
        self.assertEqual(logabsdet, logabsdet_parametrized)
        self.assertEqual(raw_params, transform_parametrized.get_raw_params())

    def test_random_parameterized(self):
        raw_params = torch.randn_like(self.transform.get_raw_params())
        transform_parametrized = SumOfSigmoids(features=self.features, n_sigmoids=self.n_sigmoids,
                                               raw_params=raw_params)
        outputs, logabsdet = self.transform.forward(self.inputs)

        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])

        self.assert_jacobian_correct(transform=transform_parametrized, inputs=self.inputs)

    def test_inverse_large_values(self):
        data_large_values = torch.randn_like(self.inputs) * 10 + 200
        data_large_negative_values = -(torch.randn_like(self.inputs) * 10 + 200)

        outputs_negative, logabsdet_negative = self.transform.forward(data_large_negative_values)
        outputs, logabsdet = self.transform.forward(data_large_values)
        rec_data_large_values, _ = self.transform.inverse(outputs)
        rec_data_large_negative_values, _ = self.transform.inverse(outputs_negative)

        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])
        self.assert_tensor_is_good(outputs_negative, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet_negative, [self.batch_size])

        self.assert_tensor_equal(data_large_values, rec_data_large_values)
        self.assert_tensor_equal(data_large_negative_values, rec_data_large_negative_values)



if __name__ == "__main__":
    unittest.main()
