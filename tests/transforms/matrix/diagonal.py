import unittest

import torch

from flowcon.transforms import InverseTransform, TransformDiagonal, Exp, CompositeTransform, Sigmoid, ScalarScale, \
    ScalarShift, Softplus, TransformDiagonalSoftplus, TransformDiagonalExponential
from tests.transforms.transform_test import TransformTest
from parameterized import parameterized_class
from flowcon.utils import torchutils

fancy_exp_transform = CompositeTransform([Sigmoid(),
                                          ScalarShift(1e-4, trainable=False),
                                          ScalarScale(scale=80., trainable=False),
                                          Exp(),
                                          ScalarShift(1e-5, trainable=False)])

fancy_softplus_transform = CompositeTransform([Sigmoid(),
                                               ScalarShift(1e-4, trainable=False),
                                               ScalarScale(scale=80., trainable=False),
                                               Softplus(),
                                               ScalarShift(1e-5, trainable=False)])


@parameterized_class(('batch_size', 'matrix_dim', 'diag_transform'), [
    (10, 2, Exp()),
    (2, 4, fancy_exp_transform),
    (10, 2, fancy_exp_transform),
    (16, 3, fancy_softplus_transform),
    (10, 20, "exp"),
    (1, 3, "softplus"),
    (50, 17, "softplus"),
])
class TransformDiagonalTest(TransformTest):
    def setUp(self):
        self._inputs = torch.randn(self.batch_size, self.matrix_dim * self.matrix_dim).requires_grad_(True)
        self.inputs = self._inputs.view(-1, self.matrix_dim, self.matrix_dim)
        if self.diag_transform == "exp":
            self.transform = TransformDiagonalExponential(N=self.matrix_dim)
        elif self.diag_transform == "softplus":
            self.transform = TransformDiagonalSoftplus(N=self.matrix_dim)
        else:
            self.transform = TransformDiagonal(N=self.matrix_dim, diag_transformation=self.diag_transform)
        self.eps = 2e-5

    def test_forward(self):
        outputs, logabsdet = self.transform(self.inputs)

        self.assert_tensor_is_good(outputs, [self.batch_size, self.matrix_dim, self.matrix_dim])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])

        logabsdet_ref = torchutils.logabsdet(torchutils.batch_jacobian(outputs.view(self.batch_size, -1), self._inputs))
        self.assertEqual(logabsdet, logabsdet_ref)

        self.assertGreater(torch.diagonal(outputs, dim1=-2, dim2=-1).detach(),
                           torch.zeros_like(torch.diagonal(outputs, dim1=-2, dim2=-1)).detach())

    def test_inverse(self):
        outputs, logabsdet = self.transform(self.inputs)
        _outputs = outputs.view(self.batch_size, -1).detach().requires_grad_(True)

        inputs_rec, logabsdet_inv = self.transform.inverse(_outputs.view(-1, self.matrix_dim, self.matrix_dim))

        self.assert_tensor_is_good(inputs_rec, [self.batch_size, self.matrix_dim, self.matrix_dim])
        self.assert_tensor_is_good(logabsdet_inv, [self.batch_size])

        logabsdet_inv_ref = torchutils.logabsdet(
            torchutils.batch_jacobian(inputs_rec.view(self.batch_size, -1), _outputs))
        self.assertEqual(logabsdet_inv, logabsdet_inv_ref)

        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))
        self.assertEqual(inputs_rec, self.inputs)

        self.assertGreater(torch.diagonal(outputs, dim1=-2, dim2=-1).detach(),
                           torch.zeros_like(torch.diagonal(outputs, dim1=-2, dim2=-1)).detach())

    def test_forward_inverse_are_consistent(self):
        self.assert_forward_inverse_are_consistent(self.transform, self.inputs)
