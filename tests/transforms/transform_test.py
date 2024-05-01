import torch
import torchtestcase

from flowcon.transforms import base
from flowcon.utils import torchutils


class TransformTest(torchtestcase.TorchTestCase):
    """Base test for all transforms."""

    def assert_tensor_is_good(self, tensor, shape=None):
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertFalse(torch.isnan(tensor).any())
        self.assertFalse(torch.isinf(tensor).any())
        if shape is not None:
            self.assertEqual(tensor.shape, torch.Size(shape))

    def assert_forward_inverse_are_consistent(self, transform, inputs):
        inverse = base.InverseTransform(transform)
        identity = base.CompositeTransform([transform, inverse])
        outputs, logabsdet = identity(inputs)
        self.assert_tensor_is_good(outputs, shape=inputs.shape)
        self.assert_tensor_is_good(logabsdet, shape=inputs.shape[:1])
        self.assert_tensor_equal(outputs, inputs, msg=f"Max Abs Error of {(outputs - inputs).abs().max().item():.3E}")
        self.assert_tensor_equal(logabsdet, torch.zeros(inputs.shape[:1]),
                                 msg=f"Max Abs Error of {(logabsdet).abs().max().item():.3E}"
                                 )

    def assert_jacobian_correct(self, transform, inputs):
        inputs = inputs.requires_grad_(True)
        outputs, logabsdet = transform.forward(inputs)
        _, ref_logabsdet = torch.linalg.slogdet(torchutils.batch_jacobian(outputs, inputs))

        self.assert_tensor_is_good(logabsdet, shape=inputs.shape[:1])
        self.assert_tensor_is_good(ref_logabsdet, shape=inputs.shape[:1])
        self.assert_tensor_equal(logabsdet, ref_logabsdet,
                                 msg=f"Jacobian mismatch by max abs error={(logabsdet - ref_logabsdet).abs().max():.2e}.")


    def assert_jacobian_correct_context(self, transform, inputs, context):
        inputs = inputs.requires_grad_(True)
        outputs, logabsdet = transform.forward(inputs, context)
        _, ref_logabsdet = torch.linalg.slogdet(torchutils.batch_jacobian(outputs, inputs))

        self.assert_tensor_is_good(logabsdet, shape=inputs.shape[:1])
        self.assert_tensor_is_good(ref_logabsdet, shape=inputs.shape[:1])

        self.assert_tensor_equal(logabsdet, ref_logabsdet, msg="Jacobian mismatch.")

    def assert_inverse_jacobian_correct(self, transform, outputs):
        outputs = outputs.detach().requires_grad_(True)
        inputs_reconstructed, inv_logabsdet = transform.inverse(outputs)
        _, ref_inv_logabsdet = torch.linalg.slogdet(torchutils.batch_jacobian(inputs_reconstructed, outputs))
        self.assert_tensor_is_good(inv_logabsdet, shape=outputs.shape[:1])
        self.assert_tensor_is_good(ref_inv_logabsdet, shape=outputs.shape[:1])
        self.assert_tensor_equal(inv_logabsdet, ref_inv_logabsdet,
                                 msg=f"Jacobian mismatch by max abs error={(inv_logabsdet - ref_inv_logabsdet).abs().max():.2e}.")

    def assertNotEqual(self, first, second, msg=None):
        if (self._eps and (first - second).abs().max().item() < self._eps) or (
                not self._eps and torch.equal(first, second)
        ):
            self._fail_with_message(msg, "The tensors are _not_ different!")


class ConditionalTransformTest(TransformTest):
    """Base test for all transforms."""

    def assert_conditional_forward_inverse_are_consistent(self, transform, inputs, context):
        inverse = base.InverseTransform(transform)
        identity = base.CompositeTransform([inverse, transform])
        outputs, logabsdet = identity(inputs, context)

        self.assert_tensor_is_good(outputs, shape=inputs.shape)
        self.assert_tensor_is_good(logabsdet, shape=inputs.shape[:1])
        torch.testing.assert_close(outputs, inputs)
        torch.testing.assert_close(logabsdet, torch.zeros(inputs.shape[:1]))

    def assertNotEqual(self, first, second, msg=None):
        if (self._eps and (first - second).abs().max().item() < self._eps) or (
                not self._eps and torch.equal(first, second)
        ):
            self._fail_with_message(msg, "The tensors are _not_ different!")
