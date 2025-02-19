from numbers import Real
from typing import cast

import torch
import torchtestcase  # type: ignore

# from flowcon.transforms import splines
from flowcon.transforms.monotonic.splines.util import (
    rational_quadratic_spline,
    unconstrained_rational_quadratic_spline,
)


class RationalQuadraticSplineTest(torchtestcase.TorchTestCase):
    def test_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2, 3, 4]

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins)
        unnormalized_derivatives = torch.randn(*shape, num_bins + 1)

        def call_spline_fn(
            inputs: torch.Tensor, inverse: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return rational_quadratic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=inverse,
            )

        inputs = torch.rand(*shape)
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = cast(Real, 1e-3)
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_identity_init(self):
        num_bins = 10
        shape = [2, 3, 4]

        unnormalized_widths = torch.zeros(*shape, num_bins)
        unnormalized_heights = torch.zeros(*shape, num_bins)
        unnormalized_derivatives = torch.zeros(*shape, num_bins + 1)

        def call_spline_fn(inputs, inverse=False):
            return rational_quadratic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=inverse,
                enable_identity_init=True,
            )

        inputs = torch.rand(*shape)
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)

        self.eps = cast(Real, 1e-6)
        self.assertEqual(inputs, outputs)
        self.assertEqual(logabsdet, torch.zeros_like(logabsdet))

        inputs = torch.rand(*shape)
        outputs, logabsdet = call_spline_fn(inputs, inverse=True)

        self.assertEqual(inputs, outputs)
        self.assertEqual(logabsdet, torch.zeros_like(logabsdet))


class UnconstrainedRationalQuadraticSplineTest(torchtestcase.TorchTestCase):
    def test_forward_inverse_are_consistent(self):
        num_bins = 10
        shape = [2, 3, 4]

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins)
        unnormalized_derivatives = torch.randn(*shape, num_bins + 1)

        def call_spline_fn(inputs, inverse=False):
            return unconstrained_rational_quadratic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=inverse,
            )

        inputs = 3 * torch.randn(*shape)  # Note inputs are outside [0,1].
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = cast(Real, 1e-3)
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_forward_inverse_are_consistent_in_tails(self):
        num_bins = 10
        shape = [2, 3, 4]
        tail_bound = 1.0

        unnormalized_widths = torch.randn(*shape, num_bins)
        unnormalized_heights = torch.randn(*shape, num_bins)
        unnormalized_derivatives = torch.randn(*shape, num_bins + 1)

        def call_spline_fn(inputs, inverse=False):
            return unconstrained_rational_quadratic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=inverse,
            )

        inputs = torch.sign(torch.randn(*shape)) * (
            tail_bound + torch.rand(*shape)
        )  # Now *all* inputs are outside [-tail_bound, tail_bound].
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)
        inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

        self.eps = cast(Real, 1e-3)
        self.assertEqual(inputs, inputs_inv)
        self.assertEqual(logabsdet + logabsdet_inv, torch.zeros_like(logabsdet))

    def test_identity_init(self):
        num_bins = 10
        shape = [2, 3, 4]
        tail_bound = 1.0

        unnormalized_widths = torch.zeros(*shape, num_bins)
        unnormalized_heights = torch.zeros(*shape, num_bins)
        unnormalized_derivatives = torch.zeros(*shape, num_bins + 1)

        def call_spline_fn(inputs, inverse=False):
            return unconstrained_rational_quadratic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=inverse,
                enable_identity_init=True,
            )

        inputs = torch.sign(torch.randn(*shape)) * (
            tail_bound + torch.rand(*shape)
        )  # Now *all* inputs are outside [-tail_bound, tail_bound].
        outputs, logabsdet = call_spline_fn(inputs, inverse=False)

        self.eps = cast(Real, 1e-6)
        self.assertEqual(inputs, outputs)
        self.assertEqual(logabsdet, torch.zeros_like(logabsdet))

        inputs = torch.rand(*shape)
        outputs, logabsdet = call_spline_fn(inputs, inverse=True)

        self.assertEqual(inputs, outputs)
        self.assertEqual(logabsdet, torch.zeros_like(logabsdet))
