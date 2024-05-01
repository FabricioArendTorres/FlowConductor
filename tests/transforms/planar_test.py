"""Tests for the planar transforms."""

import unittest

import torch

from flowcon.transforms import PlanarTransform
from tests.transforms.transform_test import TransformTest
import numpy as np
import random

class PlanarTest(TransformTest):
    def setUp(self):
        self.features = 3
        self.transform = PlanarTransform(features=self.features)
        self.eps = 1e-5
        torch.random.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        self.batch_size = 10
        self.inputs = torch.randn(self.batch_size, self.features)


        # self.transform.enforce_u_condition()
        u = self.transform.u
        w = self.transform.w
        b = self.transform.b

    def test_shapes(self):
        self.assert_tensor_is_good(self.transform.u, [1, self.features])
        self.assert_tensor_is_good(self.transform.w, [1, self.features])
        self.assert_tensor_is_good(self.transform.b, [1])

    def test_enforce_condition(self):

        """Enforce w^T u >= -1. When using h(.) = tanh(.), this is a sufficient condition
        for invertibility of the transformation f(z). See Appendix A.1.
        """
        # setting with w^T u < -1
        self.transform.u.data = -1 * torch.abs(torch.randn(1, self.features).normal_(1, 0.1))
        self.transform.w.data = torch.abs(torch.randn(1, self.features).normal_(1, 0.1))

        # make sure of it
        wt_u = self.transform.w.T.squeeze() @ self.transform.u.squeeze()
        self.assert_tensor_is_good(wt_u, [])
        self.assert_tensor_less(wt_u.detach(), -1)

        # check again
        wt_u_enforced = self.transform.w.T.squeeze() @ self.transform.get_constrained_u().squeeze()
        self.assert_tensor_is_good(wt_u_enforced, [])
        self.assert_tensor_greater_equal(wt_u_enforced.detach(), -1)


    def test_forward(self):
        self.assert_jacobian_correct(inputs = self.inputs, transform=self.transform)


if __name__ == "__main__":
    unittest.main()
