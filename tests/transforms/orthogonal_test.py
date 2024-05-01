"""Tests for the orthogonal transforms."""

import unittest

import torch

from flowcon.transforms import orthogonal
from flowcon.utils import torchutils
from tests.transforms.transform_test import TransformTest

class BatchwiseParameterizedHouseholderTest(TransformTest):
    def setUp(self):
        self.features = 5
        self.batch_size = 10
        self.num_householder_transforms = 2
        self.seed = 1236


        self.ref_householder = orthogonal.HouseholderSequence(self.features, self.num_householder_transforms)
        self.eps = 1e-05

        torch.manual_seed(self.seed)

    def test_forward(self):
        batch_q_vectors = torch.randn(self.batch_size, self.num_householder_transforms, self.features)
        transform = orthogonal.ParametrizedHouseHolder(batch_q_vectors)

        inputs = torch.randn(self.batch_size, self.features)
        outputs, logabsdet = transform.forward(inputs)
        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])

        for i in range(self.batch_size):
            ref_householder = orthogonal.ParametrizedHouseHolder(batch_q_vectors[i])
            ref_output, ref_logabsdet = ref_householder.forward(inputs)
            self.assertEqual(outputs[i], ref_output[i])
            self.assertEqual(logabsdet[i], ref_logabsdet[i])


    def test_inverse(self):
        batch_q_vectors = torch.randn(self.batch_size, self.num_householder_transforms, self.features)
        transform = orthogonal.ParametrizedHouseHolder(batch_q_vectors)

        inputs = torch.randn(self.batch_size, self.features)
        outputs, logabsdet = transform.inverse(inputs)
        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])

        for i in range(self.batch_size):
            ref_householder = orthogonal.ParametrizedHouseHolder(batch_q_vectors[i])
            ref_output, ref_logabsdet = ref_householder.inverse(inputs)
            self.assertEqual(outputs[i], ref_output[i])
            self.assertEqual(logabsdet[i], ref_logabsdet[i])


    def test_matrix(self):
        batch_q_vectors = torch.randn(self.batch_size, self.num_householder_transforms, self.features)
        transform = orthogonal.ParametrizedHouseHolder(batch_q_vectors)

        matrices = transform.matrix()

        self.assert_tensor_is_good(matrices, [self.batch_size, self.features, self.features])

        self.eps = 1e-05
        for i in range(self.batch_size):
            ref_householder = orthogonal.ParametrizedHouseHolder(batch_q_vectors[i])
            self.assertEqual(matrices[i], ref_householder.matrix())

        identity = torch.eye(self.features, self.features)
        identity = torch.repeat_interleave(identity[None, ...], self.batch_size, 0)
        self.assertEqual(matrices @ torch.transpose(matrices, dim0=-2, dim1=-1), identity)

    def test_forward_inverse_are_consistent(self):
        self.eps = 1e-5

        batch_q_vectors = torch.randn(self.batch_size, self.num_householder_transforms, self.features)
        transform = orthogonal.ParametrizedHouseHolder(batch_q_vectors)

        inputs = torch.randn(self.batch_size, self.features)
        self.assert_forward_inverse_are_consistent(transform, inputs)

class ParameterizedHouseholderTest(TransformTest):
    def setUp(self):
        self.features = 5
        self.batch_size = 10
        self.num_householder_transforms = 2
        self.seed = 1236

        torch.manual_seed(self.seed)
        self.ref_householder = orthogonal.HouseholderSequence(self.features, self.num_householder_transforms)
        self.ref_householder.q_vectors = torch.nn.Parameter(torch.randn_like(self.ref_householder.q_vectors))

        self.eps = 1e-5


    def test_forward(self):
        transform = orthogonal.ParametrizedHouseHolder(self.ref_householder.q_vectors)
        inputs = torch.randn(self.batch_size, self.features)

        outputs, logabsdet = transform.forward(inputs)
        outputs_ref, logabsdet_ref = self.ref_householder.forward(inputs)
        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])

        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)

    def test_inverse(self):
        transform = orthogonal.ParametrizedHouseHolder(self.ref_householder.q_vectors)
        inputs = torch.randn(self.batch_size, self.features)

        outputs, logabsdet = transform.inverse(inputs)
        outputs_ref, logabsdet_ref = self.ref_householder.inverse(inputs)
        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])

        self.assertEqual(outputs, outputs_ref)
        self.assertEqual(logabsdet, logabsdet_ref)



    def test_matrix(self):
        transform = orthogonal.ParametrizedHouseHolder(self.ref_householder.q_vectors)
        self.assertEqual(transform.matrix(), self.ref_householder.matrix())

    def test_forward_inverse_are_consistent(self):
        transform = orthogonal.ParametrizedHouseHolder(self.ref_householder.q_vectors)

        inputs = torch.randn(self.batch_size, self.features)
        self.assert_forward_inverse_are_consistent(transform, inputs)


class HouseholderSequenceTest(TransformTest):
    def test_forward(self):
        features = 100
        batch_size = 50

        for num_transforms in [1, 2, 11, 12]:
            with self.subTest(num_transforms=num_transforms):
                transform = orthogonal.HouseholderSequence(
                    features=features, num_transforms=num_transforms
                )
                matrix = transform.matrix()
                inputs = torch.randn(batch_size, features)
                outputs, logabsdet = transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, features])
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.eps = 1e-5
                self.assertEqual(outputs, inputs @ matrix.t())
                self.assertEqual(
                    logabsdet, torchutils.logabsdet(matrix) * torch.ones(batch_size)
                )

    def test_inverse(self):
        features = 100
        batch_size = 50

        for num_transforms in [1, 2, 11, 12]:
            with self.subTest(num_transforms=num_transforms):
                transform = orthogonal.HouseholderSequence(
                    features=features, num_transforms=num_transforms
                )
                matrix = transform.matrix()
                inputs = torch.randn(batch_size, features)
                outputs, logabsdet = transform.inverse(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, features])
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.eps = 1e-5
                self.assertEqual(outputs, inputs @ matrix)
                self.assertEqual(
                    logabsdet, torchutils.logabsdet(matrix) * torch.ones(batch_size)
                )

    def test_matrix(self):
        features = 100

        for num_transforms in [1, 2, 11, 12]:
            with self.subTest(num_transforms=num_transforms):
                transform = orthogonal.HouseholderSequence(
                    features=features, num_transforms=num_transforms
                )
                matrix = transform.matrix()
                self.assert_tensor_is_good(matrix, [features, features])
                self.eps = 1e-5
                self.assertEqual(matrix @ matrix.t(), torch.eye(features, features))
                self.assertEqual(matrix.t() @ matrix, torch.eye(features, features))
                self.assertEqual(matrix.t(), torch.inverse(matrix))
                det_ref = torch.tensor(1.0 if num_transforms % 2 == 0 else -1.0)
                self.assertEqual(matrix.det(), det_ref)

    def test_forward_inverse_are_consistent(self):
        features = 100
        batch_size = 50
        inputs = torch.randn(batch_size, features)
        transforms = [
            orthogonal.HouseholderSequence(
                features=features, num_transforms=num_transforms
            )
            for num_transforms in [1, 2, 11, 12]
        ]
        self.eps = 1e-5
        for transform in transforms:
            with self.subTest(transform=transform):
                self.assert_forward_inverse_are_consistent(transform, inputs)


if __name__ == "__main__":
    unittest.main()
