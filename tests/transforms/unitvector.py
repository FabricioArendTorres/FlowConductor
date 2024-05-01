"""Tests for the LU linear transforms."""

import unittest

import torch

from flowcon.transforms import UnitVector, InverseTransform
from flowcon.utils import torchutils
from tests.transforms.transform_test import TransformTest
from flowcon.utils import torchutils
from parameterized import parameterized_class


@parameterized_class(('batch_size', 'features'), [
    (10, 2),
    (2, 4),
    (10, 2),
    (16, 3),
    (10, 20),
    (1, 3),
])
class UnitVectorTest(TransformTest):
    def setUp(self):
        # self.features = 2
        self.transform = UnitVector(features=self.features)
        # self.batch_size = 10
        self.inputs = torch.randn(self.batch_size, self.features).requires_grad_(True)

        self.eps = 5e-5

    def test_forward(self):
        outputs, logabsdet = self.transform.forward(self.inputs)

        self.assert_tensor_is_good(outputs, [self.batch_size, self.features + 1])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])

        logabsdet_ref = torchutils.batch_JTJ_logabsdet(inputs=self.inputs, outputs=outputs).view(-1)

        self.assertEqual(logabsdet, logabsdet_ref)

    def test_invalid_inputs(self):
        with self.assertRaises(Exception) as context:
            inputs = torch.randn(self.batch_size, self.features - 1)
            self.transform.forward(inputs)

        with self.assertRaises(Exception) as context:
            inputs = torch.randn(self.batch_size, self.features + 1)
            self.transform.forward(inputs)

        with self.assertRaises(Exception) as context:
            inputs = torch.randn(self.batch_size, self.features) + 1
            self.transform.inverse(inputs)

    def test_inverse(self):
        inputs = torch.randn(self.batch_size, self.features).requires_grad_(True)
        outputs, logabsdet_forward = self.transform(inputs)
        # logabsdet_forward_ref = torchutils.batch_JTJ_logabsdet(inputs=inputs, outputs=outputs)

        outputs = outputs.detach().requires_grad_(True)
        inputs_rec, logabsdet_inverse = self.transform.inverse(outputs)

        self.assert_tensor_is_good(inputs_rec, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet_inverse, [self.batch_size])

        # logabsdet_inverse_ref = torchutils.batch_JTJ_logabsdet(inputs=outputs, outputs=inputs_rec)
        #
        # jacs = torchutils.batch_jacobian(inputs_rec, outputs)
        # logabsdet_ref_2 = 0.5 * torch.slogdet(torch.linalg.inv(torch.bmm(jacs, torch.transpose(jacs, -2, -1))))[1]
        # breakpoint()
        # self.assertEqual(logabsdet_inverse, logabsdet_inverse_ref)

    def test_forward_inverse_are_consistent(self):
        inputs = self.inputs
        self.assert_forward_inverse_are_consistent(InverseTransform(self.transform), inputs)


if __name__ == "__main__":
    unittest.main()
