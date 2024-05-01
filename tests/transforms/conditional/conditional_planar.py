import unittest

import nflows.transforms
import torch

from flowcon.transforms.conditional import ConditionalPlanarTransform
from flowcon.utils import torchutils
# from tests.transforms.transform_test import ConditionalTransformTest
from tests.transforms.transform_test import TransformTest

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
class ConditionalOrthogonalTest(TransformTest):
    def setUp(self):
        self.features = 3
        self.hidden_features = 32
        self.batch_size = 10
        self.context_features = 15
        self.lipschitz_constant = None
        torch.manual_seed(1234)

        self.random_context = torch.randn((self.batch_size, self.context_features))
        self.random_input = torch.randn((self.batch_size, self.features))

        self.transform = ConditionalPlanarTransform(features=self.features, hidden_features=self.hidden_features,
                                                    context_features=self.context_features)

        self.eps = 1e-6

    def test_forward(self):
        self.random_input.requires_grad_(True)
        outputs, logabsdet = self.transform.forward(self.random_input, self.random_context)

        self.assert_tensor_is_good(outputs, shape=[self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, shape=[self.batch_size])

        self.assert_jacobian_correct_context(transform=nflows.transforms.InverseTransform(self.transform),
                                             inputs=self.random_input,
                                             context=self.random_context)


if __name__ == "__main__":
    unittest.main()
