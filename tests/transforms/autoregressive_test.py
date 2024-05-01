"""Tests for the autoregressive transforms."""

import unittest

import torch

from flowcon.transforms import autoregressive, InverseTransform
from flowcon.utils import torchutils
from tests.transforms.transform_test import TransformTest
from parameterized import parameterized_class

class MaskedAffineAutoregressiveTransformTest(TransformTest):
    def test_forward(self):
        batch_size = 10
        features = 20
        inputs = torch.randn(batch_size, features).requires_grad_(True)
        for use_residual_blocks, random_mask in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            with self.subTest(
                use_residual_blocks=use_residual_blocks, random_mask=random_mask
            ):
                transform = autoregressive.MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=30,
                    num_blocks=5,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                outputs, logabsdet = transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, features])
                self.assert_tensor_is_good(logabsdet, [batch_size])

                batch_jac = torchutils.batch_jacobian(outputs, inputs)
                upper_diags = torch.triu(batch_jac, diagonal=1)
                lower_diags = torch.tril(batch_jac)

                self.assertEqual(upper_diags, torch.zeros_like(upper_diags))
                self.assertNotEqual(lower_diags, torch.zeros_like(upper_diags))


    def test_inverse(self):
        batch_size = 10
        features = 20
        inputs = torch.randn(batch_size, features).requires_grad_(True)
        for use_residual_blocks, random_mask in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            with self.subTest(
                use_residual_blocks=use_residual_blocks, random_mask=random_mask
            ):
                transform = autoregressive.MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=30,
                    num_blocks=5,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                outputs, logabsdet = transform.inverse(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, features])
                self.assert_tensor_is_good(logabsdet, [batch_size])

                batch_jac = torchutils.batch_jacobian(outputs, inputs)
                upper_diags = torch.triu(batch_jac, diagonal=1)
                lower_diags = torch.tril(batch_jac)

                self.assertEqual(upper_diags, torch.zeros_like(upper_diags))
                self.assertNotEqual(lower_diags, torch.zeros_like(upper_diags))



    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.randn(batch_size, features)
        self.eps = 1e-6
        for use_residual_blocks, random_mask in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            with self.subTest(
                use_residual_blocks=use_residual_blocks, random_mask=random_mask
            ):
                transform = autoregressive.MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=30,
                    num_blocks=5,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedPiecewiseLinearAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-3

        transform = autoregressive.MaskedPiecewiseLinearAutoregressiveTransform(
            num_bins=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedAdaptiveSigmoidAutoregressiveTransformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-3
        inputs = inputs.requires_grad_(True)

        transform = autoregressive.MaskedSumOfSigmoidsTransform(
            features=features,
            hidden_features=30,
            num_blocks=1,
            use_residual_blocks=True,
        )

        outputs, logabsdet = transform.forward(inputs)
        self.assert_tensor_is_good(outputs, [batch_size, features])
        self.assert_tensor_is_good(logabsdet, [batch_size])


        self.assert_forward_inverse_are_consistent(transform, inputs)

        _, ref_logabsdet = torch.linalg.slogdet(torchutils.batch_jacobian(outputs, inputs))
        self.assert_jacobian_correct(transform, inputs)



class MaskedPiecewiseQuadraticAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-4

        transform = autoregressive.MaskedPiecewiseQuadraticAutoregressiveTransform(
            num_bins=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedUMNNAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-4

        transform = autoregressive.MaskedUMNNAutoregressiveTransform(
            cond_size=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.assert_forward_inverse_are_consistent(transform, inputs)


class MaskedPiecewiseCubicAutoregressiveTranformTest(TransformTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-3

        transform = autoregressive.MaskedPiecewiseCubicAutoregressiveTransform(
            num_bins=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.assert_forward_inverse_are_consistent(transform, inputs)




if __name__ == "__main__":
    unittest.main()
