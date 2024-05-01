"""Tests for permutations."""

import unittest

import torch

from flowcon.transforms import permutations, FillTriangular, InverseTransform
from tests.transforms.transform_test import TransformTest
from parameterized import parameterized_class


class PermutationTest(TransformTest):
    def test_forward(self):
        batch_size = 10
        features = 100
        inputs = torch.randn(batch_size, features)
        permutation = torch.randperm(features)
        transform = permutations.Permutation(permutation)
        outputs, logabsdet = transform(inputs)
        self.assert_tensor_is_good(outputs, [batch_size, features])
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, inputs[:, permutation])
        self.assertEqual(logabsdet, torch.zeros([batch_size]))

    def test_inverse(self):
        batch_size = 10
        features = 100
        inputs = torch.randn(batch_size, features)
        permutation = torch.randperm(features)
        transform = permutations.Permutation(permutation)
        temp, _ = transform(inputs)
        outputs, logabsdet = transform.inverse(temp)
        self.assert_tensor_is_good(outputs, [batch_size, features])
        self.assert_tensor_is_good(logabsdet, [batch_size])
        self.assertEqual(outputs, inputs)
        self.assertEqual(logabsdet, torch.zeros([batch_size]))

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 100
        inputs = torch.randn(batch_size, features)
        transforms = [
            permutations.Permutation(torch.randperm(features)),
            permutations.RandomPermutation(features),
            permutations.ReversePermutation(features),
        ]
        for transform in transforms:
            with self.subTest(transform=transform):
                self.assert_forward_inverse_are_consistent(transform, inputs)


@parameterized_class(('batch_size', 'matrix_dim'), [
    (10, 2),
    (2, 4),
    (10, 2),
    (16, 3),
    (10, 20),
    (1, 3),
])
class FillTriangularTest(TransformTest):
    def setUp(self):
        self.features = FillTriangular.calc_n_ltri(matrix_dim=self.matrix_dim)

        self.inputs = torch.randn(self.batch_size, self.features)
        self.transform = FillTriangular(features=self.features)

        self.eps = 1e-5

    def test_forward(self):
        outputs, logabsdet = self.transform(self.inputs)

        self.assert_tensor_is_good(outputs, [self.batch_size, self.matrix_dim, self.matrix_dim])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])
        self.assertEqual(logabsdet, torch.zeros([self.batch_size]))
        triu = torch.triu(outputs, diagonal=1)
        self.assertEqual(torch.zeros_like(triu), triu)

        with self.assertRaises(Exception) as context:
            self.transform(torch.randn(self.batch_size, self.features + 1))

        with self.assertRaises(Exception) as context:
            self.transform(torch.randn(self.batch_size, self.features - 1))

        with self.assertRaises(Exception) as context:
            self.transform(torch.randn(self.batch_size, self.features, self.features))

    def test_inverse(self):
        outputs, logabsdet = self.transform(self.inputs)

        inputs_rec, logabsdet_inv = self.transform.inverse(outputs)

        self.assert_tensor_is_good(inputs_rec, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet_inv, [self.batch_size])
        self.assertEqual(logabsdet_inv, torch.zeros([self.batch_size]))

        with self.assertRaises(Exception) as context:
            self.transform(torch.randn(self.batch_size, self.matrix_dim - 1, self.matrix_dim))

        with self.assertRaises(Exception) as context:
            self.transform(torch.randn(self.batch_size, self.matrix_dim - 1, self.matrix_dim - 1))

        with self.assertRaises(Exception) as context:
            self.transform(torch.randn(self.batch_size, self.matrix_dim + 1, self.matrix_dim + 1))

    def test_forward_inverse_are_consistent(self):
        self.assert_forward_inverse_are_consistent(self.transform, self.inputs)


if __name__ == "__main__":
    unittest.main()
