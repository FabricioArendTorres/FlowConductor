import torch
import unittest
from parameterized import parameterized_class

from flowcon.nn.nets import activations
from flowcon.nn.nets.invertible_densenet import DenseNet
from flowcon.transforms.lipschitz.iresblock import iResBlock
from tests.transforms.transform_test import TransformTest

torch.set_default_dtype(torch.float32)


@parameterized_class(('batch_size', 'features', 'n_sigmoids'), [
    (10, 2, 3),
    (2, 4, 3),
    (60, 4, 30),
    (16, 3, 340),
    (10, 20, 10),
    (1, 3, 1),
    (1, 1, 1),
    (10, 1, 3),
])
class TestLipschitzLayer(TransformTest):

    def setUp(self) -> None:
        torch.manual_seed(1234)

        self.coef = 0.97

        self.inputs = torch.randn(self.batch_size, self.features)

        densenet_builder = DenseNet.factory(dimension=self.features,
                                            densenet_depth=3,
                                            activation_function=activations.Sin(),
                                            lip_coeff=self.coef, )
        self.transform = iResBlock(densenet_builder.build_network(),
                                   brute_force=True,
                                   exact_trace=True,
                                   unbiased_estimator=True)
        self.eps = 5e-4

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


if __name__ == '__main__':
    unittest.main()
