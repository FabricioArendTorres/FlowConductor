import unittest

import torch

from flowcon.transforms.conditional import ConditionalSVDTransform
from flowcon.utils import torchutils
from tests.transforms.transform_test import ConditionalTransformTest

from parameterized import parameterized_class



@parameterized_class(('batch_size', 'context_features', 'features', 'lipschitz_constant', "use_residual_blocks"), [
    (10, 2, 1, None, True),
    (2, 2, 4, None, True),
    (10, 2, 2, None, True),
    (10, 2, 3, None, True),
    (10, 4, 3, None, True),
    (1, 4, 3, None, False),
    (1, 9, 1, None, True),
    (10, 2, 1, 0.9, True),
    (2, 2, 4, 0.9, True),
    (10, 2, 2, 0.9, True),
    (10, 2, 3, 0.9, False),
    (10, 4, 3, 0.9, True),
    (1, 4, 3, 0.9, True),
    (1, 9, 1, 0.9, False),
])
class ConditionalSVDTest(ConditionalTransformTest):
    def setUp(self):
        self.features = 3
        self.hidden_features = 32
        self.batch_size = 10
        self.context_features = 15
        self.lipschitz_constant = None
        torch.manual_seed(1234)

        self.random_context = torch.randn((self.batch_size, self.context_features))
        self.random_input = torch.randn((self.batch_size, self.features))

        self.transform = ConditionalSVDTransform(features=self.features, hidden_features=self.hidden_features,
                                                 context_features=self.context_features,
                                                 lipschitz_constant_limit=self.lipschitz_constant,
                                                 use_residual_blocks=self.use_residual_blocks)

        self.eps = 1e-5

    def test_matrix_properties(self):
        conditional_params = self.transform.conditional_net(self.random_context)
        householder_U, diag_entries_S, householder_Vt, bias = self.transform._get_matrices(conditional_params)
        householder_U_matr, householder_Vt_matr = householder_U.matrix(), householder_Vt.matrix()

        self.assert_tensor_is_good(householder_U_matr, shape=[self.batch_size, self.features, self.features])
        self.assert_tensor_is_good(householder_Vt_matr, shape=[self.batch_size, self.features, self.features])
        self.assert_tensor_is_good(diag_entries_S, shape=[self.batch_size, self.features])
        self.assert_tensor_is_good(bias, shape=[self.batch_size, self.features])

        # positive diagonal
        self.assert_tensor_greater(diag_entries_S.detach(), torch.zeros_like(diag_entries_S))

        householder_UUT = householder_U_matr @ torch.transpose(householder_U_matr, dim0=-2, dim1=-1)
        householder_VVT = torch.transpose(householder_Vt_matr, dim0=-2, dim1=-1) @ householder_Vt_matr

        # orthogonality
        for i in range(self.batch_size):
            self.assertEqual(householder_UUT[i], householder_U_matr[i] @ householder_U_matr[i].T)
            self.assertEqual(householder_VVT[i], householder_Vt_matr[i].T @ householder_Vt_matr[i])

            self.assertEqual(householder_UUT[i], torch.eye(self.features))
            self.assertEqual(householder_VVT[i], torch.eye(self.features))

    def test_forward(self):
        outputs, logabsdet = self.transform.forward(self.random_input, self.random_context)

        conditional_params = self.transform.conditional_net(self.random_context)
        householder_U, diag_entries_S, householder_Vt, bias = self.transform._get_matrices(conditional_params)
        householder_U_matr, householder_Vt_matr = householder_U.matrix(), householder_Vt.matrix()

        self.assert_tensor_is_good(outputs, shape=[self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, shape=[self.batch_size])

        for i in range(self.batch_size):
            A = householder_U_matr[i] @ (diag_entries_S[i] * (householder_Vt_matr[i]))
            output_ref = householder_U_matr[i] @ (diag_entries_S[i] * (householder_Vt_matr[i] @ self.random_input[i])) + \
                         bias[i]

            self.assertEqual(outputs[i], output_ref)
            self.assertEqual(logabsdet[i], torchutils.logabsdet(A))

    def test_inverse(self):
        outputs, logabsdet = self.transform.inverse(self.random_input, self.random_context)

        conditional_params = self.transform.conditional_net(self.random_context)
        householder_U, diag_entries_S, householder_Vt, bias = self.transform._get_matrices(conditional_params)
        U, Vt = householder_U.matrix(), householder_Vt.matrix()

        UtX, _ = householder_U.inverse(self.random_input)

        self.assert_tensor_is_good(outputs, shape=[self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, shape=[self.batch_size])

        for i in range(self.batch_size):
            A = U[i] @ (diag_entries_S[i] * (Vt[i]))

            input_debiased = (self.random_input[i] - bias[i])

            UtX_ref = (U[i].T @ input_debiased)
            Spinv_UtX_ref = (torch.diag(1. / diag_entries_S[i]) @ UtX_ref)
            output_ref = Vt[i].T @ Spinv_UtX_ref

            self.assertEqual(outputs[i], output_ref)
            self.assertEqual(logabsdet[i], -torchutils.logabsdet(A))

    def test_forward_inverse_are_consistent(self):
        self.assert_conditional_forward_inverse_are_consistent(self.transform, self.random_input, self.random_context)


if __name__ == "__main__":
    unittest.main()
