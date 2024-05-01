import torch
import numpy as np

from tests.transforms.transform_test import TransformTest
from parameterized import parameterized_class
from flowcon.utils import torchutils
from flowcon.transforms import CholeskyOuterProduct, FillTriangular, InverseTransform

torch.set_default_dtype(torch.float64)


def make_spd_matrix(n_dim, *, random_state=None):
    """
    https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/datasets/_samples_generator.py#L1523
    """
    generator = np.random.default_rng(random_state)
    A = generator.uniform(size=(n_dim, n_dim))
    U, _, Vt = np.linalg.svd(np.dot(A.T, A))
    X = np.dot(np.dot(U, 1.0 + np.diag(generator.uniform(size=n_dim))), Vt)
    X = X + np.eye(X.shape[0])
    return 0.5 * (X + X.T)


def tril_flatten(tril):
    N = tril.size(-1)
    indicies = torch.tril_indices(N, N)
    indicies = N * indicies[0] + indicies[1]
    return tril.flatten(-2)[..., indicies]


def _tri_flatten(tri, indicies_func):
    N = tri.size(-1)
    indicies = indicies_func(N, N)
    indicies = N * indicies[0] + indicies[1]
    return tri.flatten(-2)[..., indicies]


def tril_flatten(tril, offset=0):
    return _tri_flatten(tril, lambda N, M: torch.tril_indices(N, M, offset=offset))


def triu_flatten(triu, offset=0):
    return _tri_flatten(triu, lambda N, M: torch.triu_indices(N, M, offset=offset))


@parameterized_class(('batch_size', 'matrix_dim'), [
    (10, 53),
    (2, 4),
    (10, 53),
    (16, 3),
    (10, 20),
    (1, 3),
    (50, 17),
])
class TransformOuterCholeskyTest(TransformTest):

    def gen_matrices(self):
        matrix_list = []
        L_list = []
        for i in range(self.batch_size):
            matrix = make_spd_matrix(self.matrix_dim)
            L = np.linalg.cholesky(matrix)
            matrix = torchutils.np_to_tensor(matrix)
            L = torchutils.np_to_tensor(L)
            L_list.append(L)
            matrix_list.append(matrix)
        self.matrices = torch.stack(matrix_list, 0)
        self.Ls = torch.stack(L_list, 0)
        self.filltril = FillTriangular(matrix_dimension=self.matrix_dim)

    def setUp(self):
        self.gen_matrices()
        # self._inputs = self.Ls[:, torch.tril_indices(self.matrix_dim, self.matrix_dim)]
        self._inputs = tril_flatten(self.Ls).requires_grad_(True)
        self.inputs = self.filltril(self._inputs)[0]
        self.transform = CholeskyOuterProduct(N=self.matrix_dim)

    def test_forward(self):
        self.eps = 1e-5

        outputs, logabsdet = self.transform(self.inputs)
        self.transform.check_pos_def(outputs)

        self.assert_tensor_is_good(outputs, [self.batch_size, self.matrix_dim, self.matrix_dim])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])
        #
        self.assertEqual(outputs, self.matrices)
        logabsdet_ref = torchutils.logabsdet(
            torchutils.batch_jacobian(tril_flatten(outputs).view(self.batch_size, -1), self._inputs))
        self.assertEqual(logabsdet, logabsdet_ref)

    def test_inverse(self):
        self.eps = 1e-5
        inputs_rec, logabsdet_inv = self.transform.inverse(self.matrices)

        self.assert_tensor_is_good(inputs_rec, [self.batch_size, self.matrix_dim, self.matrix_dim])
        self.assert_tensor_is_good(logabsdet_inv, [self.batch_size])

        triu_rec = triu_flatten(inputs_rec, offset=1)
        self.assertEqual(triu_rec, torch.zeros_like(triu_rec))
        self.assertEqual(inputs_rec, self.Ls)

    def test_forward_inverse_are_consistent(self):
        self.eps = 1e-3
        self.assert_forward_inverse_are_consistent(self.transform, self.inputs)
