import unittest

import torch

from flowcon.transforms.conditional import ConditionalOrthogonalTransform
from flowcon.utils import torchutils
from tests.transforms.transform_test import ConditionalTransformTest

from parameterized import parameterized_class


@parameterized_class(('batch_size', 'context_features', 'features'), [
    (10, 2, 1),
    (2, 2, 4),
    (10, 2, 2),
    (10, 2, 3),
    (10, 4, 3),
    (1, 4, 3),
    (1, 9, 1),
    (10, 2, 1),
    (2, 2, 4),
    (10, 2, 2),
    (10, 2, 3),
    (10, 4, 3),
    (1, 4, 3),
    (1, 9, 1),
])
class ConditionalOrthogonalTest(ConditionalTransformTest):
    def setUp(self):
        self.features = 3
        self.hidden_features = 32
        self.batch_size = 10
        self.context_features = 15
        self.lipschitz_constant = None
        torch.manual_seed(1234)

        self.random_context = torch.randn((self.batch_size, self.context_features))
        self.random_input = torch.randn((self.batch_size, self.features))

        self.transform = ConditionalOrthogonalTransform(features=self.features, hidden_features=self.hidden_features,
                                                        context_features=self.context_features)

        self.eps = 1e-5

    def test_matrix_properties(self):
        conditional_params = self.transform.conditional_net(self.random_context)
        householder_transform = self.transform._get_matrices(conditional_params)
        Q_mb = householder_transform.matrix()

        self.assert_tensor_is_good(Q_mb, shape=[self.batch_size, self.features, self.features])

        householder_QQT = Q_mb @ torch.transpose(Q_mb, dim0=-2, dim1=-1)

        # orthogonality
        for i in range(self.batch_size):
            self.assertEqual(householder_QQT[i], Q_mb[i] @ Q_mb[i].T)
            self.assertEqual(householder_QQT[i], torch.eye(self.features))


    def test_forward(self):
        outputs, logabsdet = self.transform.forward(self.random_input, self.random_context)

        conditional_params = self.transform.conditional_net(self.random_context)
        householder_transform = self.transform._get_matrices(conditional_params)
        Q_mb = householder_transform.matrix()

        self.assert_tensor_is_good(outputs, shape=[self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, shape=[self.batch_size])
        self.assertEqual(logabsdet, torch.zeros(self.batch_size))

        for i in range(self.batch_size):
            output_ref = Q_mb[i] @ self.random_input[i]
            self.assertEqual(outputs[i], output_ref)
            self.assertEqual(torchutils.logabsdet(Q_mb[i]), logabsdet[i])

    def test_inverse(self):
        outputs, logabsdet = self.transform.inverse(self.random_input, self.random_context)

        conditional_params = self.transform.conditional_net(self.random_context)
        householder_transform = self.transform._get_matrices(conditional_params)
        Q_mb = householder_transform.matrix()

        self.assert_tensor_is_good(outputs, shape=[self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, shape=[self.batch_size])
        self.assertEqual(logabsdet, torch.zeros(self.batch_size))

        for i in range(self.batch_size):
            output_ref = Q_mb[i].T @ self.random_input[i]
            self.assertEqual(outputs[i], output_ref)
            self.assertEqual(torchutils.logabsdet(Q_mb[i].T), logabsdet[i])


    def test_forward_inverse_are_consistent(self):
        self.assert_conditional_forward_inverse_are_consistent(self.transform, self.random_input, self.random_context)


if __name__ == "__main__":
    unittest.main()
